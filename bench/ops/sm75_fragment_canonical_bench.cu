#include "core/arena.h"
#include "core/device.h"
#include "ops/linear/q5/q5_sm75_canonical.h"
#include "ops/linear_swiglu/q4/q4_linear_swiglu_a8_sm75.h"
#include "ops/linear_add/q5/q5_linear_add_a8_sm75.h"
#include "ops/gdn_input_proj/q4_q5/q5_a8_sm75.h"
#include "quantized_weight.cuh"
#include "t16_fixture.h"

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <stdexcept>

using namespace ninfer;

namespace {
__global__ void compare_bytes(const unsigned char* a, const unsigned char* b,
                              std::size_t size, unsigned* mismatches) {
    unsigned local = 0;
    for (std::size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < size;
         i += static_cast<std::size_t>(blockDim.x) * gridDim.x) {
        local += a[i] != b[i];
    }
    if (local) { atomicAdd(mismatches, local); }
}

void copy_planes(const Weight& weight, DeviceBuffer& out, std::size_t low,
                  std::size_t high, std::size_t scales, cudaStream_t stream) {
    auto* base = static_cast<unsigned char*>(out.p);
    CUDA_CHECK(cudaMemcpyAsync(base, weight.qdata, low, cudaMemcpyDeviceToDevice, stream));
    if (high) {
        CUDA_CHECK(cudaMemcpyAsync(base + low, weight.qhigh, high, cudaMemcpyDeviceToDevice, stream));
    }
    CUDA_CHECK(cudaMemcpyAsync(base + low + high, weight.scales, scales, cudaMemcpyDeviceToDevice, stream));
}

unsigned compare(const void* a, const void* b, std::size_t bytes, DeviceBuffer& count,
                 cudaStream_t stream) {
    CUDA_CHECK(cudaMemsetAsync(count.p, 0, sizeof(unsigned), stream));
    compare_bytes<<<256, 256, 0, stream>>>(static_cast<const unsigned char*>(a),
        static_cast<const unsigned char*>(b), bytes, static_cast<unsigned*>(count.p));
    CUDA_CHECK(cudaGetLastError());
    unsigned result = 0;
    CUDA_CHECK(cudaMemcpyAsync(&result, count.p, sizeof(result), cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    return result;
}

bool test_shape(std::int32_t rows, std::int32_t columns, bool q5,
                 std::uint32_t seed, cudaStream_t stream) {
    const auto elements = static_cast<std::size_t>(rows) * columns;
    const std::size_t low = elements / 2;
    const std::size_t high = q5 ? elements / 8 : 0;
    const std::size_t scales = elements / 64 * sizeof(std::uint16_t);
    const std::size_t total = low + high + scales;
    auto packed = bench::make_row_split_weight(q5 ? QType::Q5G64_F16S : QType::Q4G64_F16S,
                                               rows, columns, columns, {0x31, 0xa5, 0x3c00});
    bench::t16_fill_weight(packed.weight, rows, columns, q5, seed);
    DeviceBuffer original(total), fragment(total), count(sizeof(unsigned));
    copy_planes(packed.weight, original, low, high, scales, stream);
    WorkspaceArena canonical(total);
    const auto make_copy = [&](WorkspaceArena& ws) {
        return q5 ? ops::detail::q5_sm75_canonical_copy(packed.weight, ws, stream)
                  : ops::detail::q4_linear_swiglu_sm75_canonical_copy(packed.weight, ws, stream);
    };
    const Weight identity = make_copy(canonical);
    if (identity.qdata != packed.weight.qdata || canonical.used() != 0) {
        throw std::runtime_error("canonical identity path allocated or changed ownership");
    }
    const std::size_t scratch_bytes = !q5 ? ops::detail::q4_linear_swiglu_sm75_transform_scratch_bytes()
        : rows == 5120 ? ops::detail::q5_linear_add_sm75_transform_scratch_bytes()
                       : ops::detail::gdn_q5_sm75_transform_scratch_bytes();
    DeviceBuffer scratch(scratch_bytes);
    if (!q5) {
        ops::detail::q4_linear_swiglu_sm75_transform_layout(packed.weight,
            QuantLayout::RowSplitSm75Fragment, scratch.p, scratch.bytes, stream);
    } else if (rows == 5120) {
        ops::detail::q5_linear_add_sm75_transform_layout(packed.weight,
            QuantLayout::RowSplitSm75Fragment, scratch.p, scratch.bytes, stream);
    } else {
        ops::detail::gdn_q5_sm75_transform_layout(packed.weight,
            QuantLayout::RowSplitSm75Fragment, scratch.p, stream);
    }
    copy_planes(packed.weight, fragment, low, high, scales, stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));
    const Weight view = make_copy(canonical);
    const unsigned restored_errors = compare(view.payload, original.p, total, count, stream);
    unsigned source_errors = compare(packed.weight.qdata, fragment.p, low, count, stream);
    if (high) {
        source_errors += compare(packed.weight.qhigh,
            static_cast<const unsigned char*>(fragment.p) + low, high, count, stream);
    }
    source_errors += compare(packed.weight.scales,
        static_cast<const unsigned char*>(fragment.p) + low + high, scales, count, stream);
    if (view.layout != QuantLayout::RowSplit || view.payload_bytes != total ||
        view.qdata == packed.weight.qdata || packed.weight.layout != QuantLayout::RowSplitSm75Fragment ||
        canonical.peak_used() != total) {
        throw std::runtime_error("canonical view metadata or workspace bound failed");
    }
    bool rejected_short_workspace = false;
    try {
        WorkspaceArena too_small(DeviceSpan{canonical.base(), total - 1});
        (void)make_copy(too_small);
    } catch (const std::exception&) { rejected_short_workspace = true; }
    std::printf("RESULT rows=%d columns=%d q5=%d seed=%u restored_mismatches=%u "
                "source_mismatches=%u workspace=%zu short_workspace_rejected=%d\n",
                rows, columns, q5, seed, restored_errors, source_errors, total,
                rejected_short_workspace ? 1 : 0);
    return restored_errors == 0 && source_errors == 0 && rejected_short_workspace;
}
} // namespace

int main(int argc, char** argv) {
    const auto seed = argc > 1 ? static_cast<std::uint32_t>(std::strtoul(argv[1], nullptr, 10)) : 1234567u;
    if (seed == 0) { return 2; }
    try {
        cudaStream_t stream = nullptr;
        CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
        bool exact = test_shape(34816, 5120, false, seed, stream);
        exact = test_shape(5120, 17408, true, seed, stream) && exact;
        exact = test_shape(12288, 5120, true, seed, stream) && exact;
        CUDA_CHECK(cudaStreamDestroy(stream));
        return exact ? 0 : 2;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "sm75_fragment_canonical_bench: %s\n", error.what());
        return 1;
    }
}
