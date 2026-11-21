#include "core/arena.h"
#include "core/device.h"
#include "ninfer_bench_common.h"
#include "ops/linear_swiglu/q4/q4_linear_swiglu_a8_sm75.h"
#include "quantized_weight.cuh"
#include "t16_fixture.h"
#include <cstdlib>

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <exception>
#include <vector>

using namespace ninfer;

namespace {
constexpr std::int32_t kGateUpRows = 34816;
constexpr std::int32_t kOutputRows = 17408;
constexpr std::int32_t kHidden = 5120;
constexpr std::int32_t kTokens = 16;
constexpr std::size_t kFlushBytes = 256ULL << 20;
}

int main(int argc, char** argv) {
    const std::uint32_t seed = argc > 1 ? static_cast<std::uint32_t>(std::strtoul(argv[1], nullptr, 10)) : 1234567u;
    if (seed == 0) return 2;
    try {
        cudaStream_t stream = nullptr;
        CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
        DeviceBuffer flush(kFlushBytes);
        DeviceBuffer input = bench::make_bf16(static_cast<std::size_t>(kHidden) * kTokens);
        DeviceBuffer reference(static_cast<std::size_t>(kOutputRows) * kTokens * sizeof(std::uint16_t));
        DeviceBuffer candidate(reference.bytes);
        bench::PackedQuantizedWeight packed = bench::make_row_split_weight(
            QType::Q4G64_F16S, kGateUpRows, kHidden, kHidden, {0x31, 0xa5, 0x3c00});
        bench::t16_fill_input(input.p, input.bytes/2, seed);
        bench::t16_fill_weight(packed.weight, 34816, 5120, false, seed ^ 0x314159u);
        const std::size_t scratch_bytes =
            ops::detail::q4_linear_swiglu_sm75_transform_scratch_bytes();
        DeviceBuffer scratch(scratch_bytes);
        ops::detail::q4_linear_swiglu_sm75_transform_layout(
            packed.weight, QuantLayout::RowSplitSm75Fragment, scratch.p, scratch.bytes, stream);
        CUDA_CHECK(cudaStreamSynchronize(stream));
        const std::size_t workspace_bytes =
            ops::detail::q4_linear_swiglu_sm75_a8_workspace_bytes(kTokens);
        WorkspaceArena reference_workspace(std::max<std::size_t>(workspace_bytes, 256));
        WorkspaceArena candidate_workspace(std::max<std::size_t>(workspace_bytes, 256));
        Tensor x(input.p, DType::BF16, {kHidden, kTokens});
        Tensor reference_out(reference.p, DType::BF16, {kOutputRows, kTokens});
        Tensor candidate_out(candidate.p, DType::BF16, {kOutputRows, kTokens});
        const auto launch_reference = [&](cudaStream_t launch_stream) {
            for (std::int32_t offset = 0; offset < kTokens; offset += 8) {
                Tensor xs = x.slice(1, offset, 8);
                Tensor out = reference_out.slice(1, offset, 8);
                ops::detail::q4_linear_swiglu_sm75_a8_launch(
                    xs, packed.weight, out, reference_workspace, launch_stream);
            }
        };
        const auto launch_candidate = [&](cudaStream_t launch_stream) {
            ops::detail::q4_linear_swiglu_sm75_a8_launch(
                x, packed.weight, candidate_out, candidate_workspace, launch_stream);
        };
        reference.fill();
        candidate.fill();
        launch_reference(stream);
        launch_candidate(stream);
        CUDA_CHECK(cudaStreamSynchronize(stream));
        std::vector<std::uint16_t> host_reference(reference.bytes / sizeof(std::uint16_t));
        std::vector<std::uint16_t> host_candidate(candidate.bytes / sizeof(std::uint16_t));
        reference.copy_to_host(host_reference.data(), reference.bytes);
        candidate.copy_to_host(host_candidate.data(), candidate.bytes);
        std::size_t mismatches = 0;
        std::size_t first_mismatch = host_reference.size();
        for (std::size_t index = 0; index < host_reference.size(); ++index) {
            if (host_reference[index] != host_candidate[index]) {
                if (first_mismatch == host_reference.size()) { first_mismatch = index; }
                ++mismatches;
            }
        }
        const auto reference_timing = bench::measure_cold_launch(
            launch_reference, flush, stream, 5, 30);
        const auto candidate_timing = bench::measure_cold_launch(
            launch_candidate, flush, stream, 5, 30);
        std::printf(
            "RESULT q4_t16_weight_reuse exact=%d mismatches=%zu first=%zu "
            "reference_us=%.3f candidate_us=%.3f speedup=%.6f workspace=%zu\n",
            mismatches == 0 ? 1 : 0, mismatches, first_mismatch,
            reference_timing.median_us, candidate_timing.median_us,
            reference_timing.median_us / candidate_timing.median_us, workspace_bytes);
        CUDA_CHECK(cudaStreamDestroy(stream));
        return mismatches == 0 ? 0 : 2;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "q4_t16_weight_reuse_bench: %s\n", error.what());
        return 1;
    }
}
