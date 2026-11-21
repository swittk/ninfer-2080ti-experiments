#include "core/arena.h"
#include "core/device.h"
#include "ninfer_bench_common.h"
#include "ops/gdn_input_proj/q4_q5/q5_a8_sm75.h"
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
constexpr std::int32_t kHidden = 5120;
constexpr std::int32_t kQkRows = 4096;
constexpr std::int32_t kValueRows = 6144;
constexpr std::int32_t kParentRows = 12288;
constexpr std::int32_t kQkvRows = 10240;
constexpr std::int32_t kTokens = 16;
constexpr std::size_t kFlushBytes = 256ULL << 20;

struct MismatchCounts {
    std::size_t qk = 0;
    std::size_t value = 0;
    std::size_t z = 0;
};
}

int main(int argc, char** argv) {
    const std::uint32_t seed = argc > 1 ? static_cast<std::uint32_t>(std::strtoul(argv[1], nullptr, 10)) : 1234567u;
    if (seed == 0) return 2;
    try {
        cudaStream_t stream = nullptr;
        CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
        DeviceBuffer flush(kFlushBytes);
        DeviceBuffer input = bench::make_bf16(static_cast<std::size_t>(kHidden) * kTokens);
        DeviceBuffer reference_qkv(static_cast<std::size_t>(kQkvRows) * kTokens * sizeof(std::uint16_t));
        DeviceBuffer candidate_qkv(reference_qkv.bytes);
        DeviceBuffer reference_z(static_cast<std::size_t>(kValueRows) * kTokens * sizeof(std::uint16_t));
        DeviceBuffer candidate_z(reference_z.bytes);
        bench::PackedQuantizedWeight qk = bench::make_row_split_weight(
            QType::Q4G64_F16S, kQkRows, kHidden, kHidden, {0x31, 0xa5, 0x3c00});
        bench::PackedQuantizedWeight value_z = bench::make_row_split_weight(
            QType::Q5G64_F16S, kParentRows, kHidden, kHidden, {0x29, 0x9d, 0x3c00});
        bench::t16_fill_input(input.p, input.bytes/2, seed);
        bench::t16_fill_weight(qk.weight, 4096, 5120, false, seed ^ 0x141421u);
        bench::t16_fill_weight(value_z.weight, 12288, 5120, true, seed ^ 0x173205u);
        const std::size_t scratch_bytes = ops::detail::gdn_q5_sm75_transform_scratch_bytes();
        DeviceBuffer scratch(scratch_bytes);
        ops::detail::gdn_q5_sm75_transform_layout(
            value_z.weight, QuantLayout::RowSplitSm75Fragment,
            scratch.p, stream);
        CUDA_CHECK(cudaStreamSynchronize(stream));
        const std::size_t workspace_bytes = ops::detail::gdn_q5_sm75_a8_workspace_bytes(kTokens);
        WorkspaceArena reference_workspace(std::max<std::size_t>(workspace_bytes, 256));
        WorkspaceArena candidate_workspace(std::max<std::size_t>(workspace_bytes, 256));
        Tensor x(input.p, DType::BF16, {kHidden, kTokens});
        Tensor ref_qkv(reference_qkv.p, DType::BF16, {kQkvRows, kTokens});
        Tensor cand_qkv(candidate_qkv.p, DType::BF16, {kQkvRows, kTokens});
        Tensor ref_z(reference_z.p, DType::BF16, {kValueRows, kTokens});
        Tensor cand_z(candidate_z.p, DType::BF16, {kValueRows, kTokens});
        const auto launch_reference = [&](cudaStream_t launch_stream) {
            for (std::int32_t offset = 0; offset < kTokens; offset += 8) {
                Tensor xs = x.slice(1, offset, 8);
                Tensor qkv = ref_qkv.slice(1, offset, 8);
                Tensor z = ref_z.slice(1, offset, 8);
                ops::detail::gdn_q5_sm75_a8_launch(
                    xs, qk.weight, value_z.weight, qkv, z,
                    reference_workspace, launch_stream);
            }
        };
        const auto launch_candidate = [&](cudaStream_t launch_stream) {
            ops::detail::gdn_q5_sm75_a8_launch(
                x, qk.weight, value_z.weight, cand_qkv, cand_z,
                candidate_workspace, launch_stream);
        };
        reference_qkv.fill();
        candidate_qkv.fill();
        reference_z.fill();
        candidate_z.fill();
        launch_reference(stream);
        launch_candidate(stream);
        CUDA_CHECK(cudaStreamSynchronize(stream));
        std::vector<std::uint16_t> host_ref_qkv(reference_qkv.bytes / sizeof(std::uint16_t));
        std::vector<std::uint16_t> host_cand_qkv(candidate_qkv.bytes / sizeof(std::uint16_t));
        std::vector<std::uint16_t> host_ref_z(reference_z.bytes / sizeof(std::uint16_t));
        std::vector<std::uint16_t> host_cand_z(candidate_z.bytes / sizeof(std::uint16_t));
        reference_qkv.copy_to_host(host_ref_qkv.data(), reference_qkv.bytes);
        candidate_qkv.copy_to_host(host_cand_qkv.data(), candidate_qkv.bytes);
        reference_z.copy_to_host(host_ref_z.data(), reference_z.bytes);
        candidate_z.copy_to_host(host_cand_z.data(), candidate_z.bytes);
        MismatchCounts mismatches;
        for (std::int32_t token = 0; token < kTokens; ++token) {
            const std::size_t base = static_cast<std::size_t>(token) * kQkvRows;
            for (std::int32_t row = 0; row < kQkvRows; ++row) {
                if (host_ref_qkv[base + row] == host_cand_qkv[base + row]) { continue; }
                if (row < kQkRows) { ++mismatches.qk; }
                else { ++mismatches.value; }
            }
        }
        for (std::size_t index = 0; index < host_ref_z.size(); ++index) {
            mismatches.z += host_ref_z[index] != host_cand_z[index];
        }
        const auto reference_timing = bench::measure_cold_launch(
            launch_reference, flush, stream, 5, 30);
        const auto candidate_timing = bench::measure_cold_launch(
            launch_candidate, flush, stream, 5, 30);
        const std::size_t total = mismatches.qk + mismatches.value + mismatches.z;
        std::printf(
            "RESULT gdn_t16_weight_reuse exact=%d total=%zu qk=%zu value=%zu z=%zu "
            "reference_us=%.3f candidate_us=%.3f speedup=%.6f workspace=%zu\n",
            total == 0 ? 1 : 0, total, mismatches.qk, mismatches.value, mismatches.z,
            reference_timing.median_us, candidate_timing.median_us,
            reference_timing.median_us / candidate_timing.median_us, workspace_bytes);
        CUDA_CHECK(cudaStreamDestroy(stream));
        return total == 0 ? 0 : 2;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "gdn_t16_weight_reuse_bench: %s\n", error.what());
        return 1;
    }
}
