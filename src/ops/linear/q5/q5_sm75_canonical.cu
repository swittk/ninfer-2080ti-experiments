#include "ops/linear/q5/q5_sm75_canonical.h"
#include "core/device.h"

#include <cstdint>
#include <stdexcept>

namespace ninfer::ops::detail {
namespace {

bool supported_shape(std::int32_t rows, std::int32_t columns) noexcept {
    return (rows == 5120 && columns == 17408) || (rows == 12288 && columns == 5120);
}

// Both SM75 Q5 leaves use the same 16-row, group-major fragment layout.
// One CTA reconstructs one block directly into canonical storage, without
// mutating the source or copying the reconstructed planes back over it.
__global__ void canonical_from_fragment(const std::uint16_t* low,
                                         const std::uint32_t* high,
                                         const std::uint16_t* scales,
                                         std::uint16_t* output_low,
                                         std::uint8_t* output_high,
                                         std::uint16_t* output_scales, int groups) {
    const std::int64_t rb = blockIdx.x;
    const int thread = static_cast<int>(threadIdx.x);
    for (int item = thread; item < 16 * groups * 16; item += blockDim.x) {
        const int row = item / (groups * 16);
        const int rem = item - row * groups * 16;
        const int group = rem / 16;
        const int pair = rem % 16;
        const int lane = 4 * (row & 7) + (pair & 3);
        const std::int64_t source = ((rb * groups + group) * 32 + lane) * 8 +
                                    (pair >> 2) * 2 + (row >= 8 ? 1 : 0);
        output_low[rb * 16 * groups * 16 + item] = low[source];
    }
    for (int item = thread; item < 16 * groups * 8; item += blockDim.x) {
        const int row = item / (groups * 8);
        const int rem = item - row * groups * 8;
        const int group = rem / 8;
        const int byte = rem % 8;
        const int lane = 4 * (row & 7) + 2 * (byte & 1);
        const int shift = (byte >> 1) * 8 + (row >= 8 ? 4 : 0);
        const std::int64_t source = (rb * groups + group) * 32 + lane;
        output_high[rb * 16 * groups * 8 + item] = static_cast<std::uint8_t>(
            ((high[source] >> shift) & 15u) | (((high[source + 1] >> shift) & 15u) << 4));
    }
    for (int item = thread; item < 16 * groups; item += blockDim.x) {
        const int row = item / groups;
        const int group = item % groups;
        output_scales[rb * 16 * groups + item] = scales[(rb * groups + group) * 16 + row];
    }
}
} // namespace

std::size_t q5_sm75_canonical_workspace_bytes(std::int32_t rows, std::int32_t columns) {
    if (!supported_shape(rows, columns)) {
        throw std::invalid_argument("SM75 canonical Q5 copy requires a supported target shape");
    }
    const std::size_t elements = static_cast<std::size_t>(rows) * columns;
    return elements / 2 + elements / 8 + elements / 64 * sizeof(std::uint16_t);
}

Weight q5_sm75_canonical_copy(const Weight& source, WorkspaceArena& workspace,
                               cudaStream_t stream) {
    const std::size_t bytes = q5_sm75_canonical_workspace_bytes(source.n, source.k);
    if (source.qtype != QType::Q5G64_F16S || source.qdata == nullptr ||
        source.qhigh == nullptr || source.scales == nullptr ||
        (source.layout != QuantLayout::RowSplit &&
         source.layout != QuantLayout::RowSplitSm75Fragment)) {
        throw std::invalid_argument("SM75 canonical Q5 copy requires valid Q5G64 planes");
    }
    if (source.layout == QuantLayout::RowSplit) { return source; }
    const std::size_t elements = static_cast<std::size_t>(source.n) * source.k;
    const std::size_t low_bytes = elements / 2;
    const std::size_t high_bytes = elements / 8;
    const DeviceSpan storage = workspace.alloc_bytes(bytes);
    auto* base = static_cast<std::uint8_t*>(storage.data);
    Weight output = source;
    output.payload = base;
    output.payload_bytes = bytes;
    output.high_plane_bytes = high_bytes;
    output.qdata = base;
    output.qhigh = base + low_bytes;
    output.scales = base + low_bytes + high_bytes;
    output.layout = QuantLayout::RowSplit;
    canonical_from_fragment<<<source.n / 16, 256, 0, stream>>>(
        static_cast<const std::uint16_t*>(source.qdata),
        static_cast<const std::uint32_t*>(source.qhigh),
        static_cast<const std::uint16_t*>(source.scales),
        reinterpret_cast<std::uint16_t*>(base), base + low_bytes,
        reinterpret_cast<std::uint16_t*>(base + low_bytes + high_bytes), source.k / 64);
    CUDA_CHECK(cudaGetLastError());
    return output;
}
} // namespace ninfer::ops::detail
