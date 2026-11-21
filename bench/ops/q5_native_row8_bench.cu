#include "core/arena.h"
#include "core/device.h"
#include "ninfer_bench_common.h"
#include "ops/linear_add/q5/q5_linear_add_a8_sm75.h"
#include "quantized_weight.cuh"
#include "t16_fixture.h"
#include <cstdlib>
#include <cuda_fp16.h>
#include <bit>
#include <cmath>
#include "../../tests/ops/op_check.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <exception>
#include <vector>
#include <string_view>

using namespace ninfer;

namespace {
constexpr std::int32_t kRows = 5120;
constexpr std::int32_t kHidden = 17408;

constexpr std::size_t kFlushBytes = 256ULL << 20;
}

int main(int argc, char** argv) {
    const std::uint32_t seed = argc > 1 ? static_cast<std::uint32_t>(std::strtoul(argv[1], nullptr, 10)) : 1234567u;
    if (seed == 0) return 2;
    const char* selected=std::getenv("NINFER_SERVE_Q5_BATCH");
    if(selected && std::string_view(selected)=="1") {
        std::fprintf(stderr,"Set NINFER_SERVE_Q5_BATCH=0 for an independent baseline comparison\n");
        return 2;
    }
    const int kTokens=argc>2?std::atoi(argv[2]):8;
    if(kTokens<1 || kTokens>8) return 2;
    try {
        cudaStream_t stream = nullptr;
        CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
        DeviceBuffer flush(kFlushBytes);
        DeviceBuffer input = bench::make_bf16(static_cast<std::size_t>(kHidden) * kTokens);
        DeviceBuffer initial = bench::make_bf16(static_cast<std::size_t>(kRows) * kTokens);
        DeviceBuffer reference(initial.bytes);
        DeviceBuffer candidate(initial.bytes);
        bench::PackedQuantizedWeight packed = bench::make_row_split_weight(
            QType::Q5G64_F16S, kRows, kHidden, kHidden, {0x31, 0xa5, 0x3c00});
        bench::t16_fill_input(input.p, input.bytes/2, seed);
        bench::t16_fill_input(initial.p, initial.bytes/2, seed ^ 0xa3b5u);
        bench::t16_fill_weight(packed.weight, 5120, 17408, true, seed ^ 0x271828u);
        std::vector<std::uint8_t> low(kRows*kHidden/2), high(kRows*kHidden/8);
        std::vector<std::uint16_t> weight_scales(kRows*kHidden/64), initial_bits(initial.bytes/2);
        CUDA_CHECK(cudaMemcpy(low.data(),packed.weight.qdata,low.size(),cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(high.data(),packed.weight.qhigh,high.size(),cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(weight_scales.data(),packed.weight.scales,weight_scales.size()*2,cudaMemcpyDeviceToHost));
        initial.copy_to_host(initial_bits.data(),initial.bytes);
        const std::size_t scratch_bytes =
            ops::detail::q5_linear_add_sm75_transform_scratch_bytes();
        DeviceBuffer scratch(scratch_bytes);
        ops::detail::q5_linear_add_sm75_transform_layout(
            packed.weight, QuantLayout::RowSplitSm75Fragment, scratch.p, scratch.bytes, stream);
        CUDA_CHECK(cudaStreamSynchronize(stream));
        const std::size_t workspace_bytes =
            ops::detail::q5_linear_add_sm75_a8_workspace_bytes(kTokens);
        WorkspaceArena reference_workspace(std::max<std::size_t>(workspace_bytes, 256));
        WorkspaceArena candidate_workspace(std::max<std::size_t>(workspace_bytes, 256));
        Tensor x(input.p, DType::BF16, {kHidden, kTokens});
        Tensor reference_out(reference.p, DType::BF16, {kRows, kTokens});
        Tensor candidate_out(candidate.p, DType::BF16, {kRows, kTokens});
        const auto reset_outputs = [&]() {
            CUDA_CHECK(cudaMemcpyAsync(reference.p, initial.p, initial.bytes,
                                       cudaMemcpyDeviceToDevice, stream));
            CUDA_CHECK(cudaMemcpyAsync(candidate.p, initial.p, initial.bytes,
                                       cudaMemcpyDeviceToDevice, stream));
        };
        const auto launch_reference = [&](cudaStream_t launch_stream) {
            CUDA_CHECK(cudaMemcpyAsync(reference.p, initial.p, initial.bytes, cudaMemcpyDeviceToDevice, launch_stream));
            for (std::int32_t offset = 0; offset < kTokens; offset += kTokens) {
                Tensor xs = x.slice(1, offset, kTokens);
                Tensor out = reference_out.slice(1, offset, kTokens);
                ops::detail::q5_linear_add_sm75_a8_launch(
                    xs, packed.weight, out, reference_workspace, launch_stream);
            }
        };
        const auto launch_candidate = [&](cudaStream_t launch_stream) {
            CUDA_CHECK(cudaMemcpyAsync(candidate.p, initial.p, initial.bytes, cudaMemcpyDeviceToDevice, launch_stream));
            ops::detail::q5_linear_add_sm75_a8_native_row8_launch(
                x, packed.weight, candidate_out, candidate_workspace, launch_stream);
        };
        reset_outputs();
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
        // Independently interpret canonical Q5 bytes and the actual quantized
        // input codes/scales; compare both schedules to an FP64 reduction oracle.
        std::vector<std::int8_t> aq(kHidden*kTokens);
        std::vector<std::uint16_t> as(kHidden/64*kTokens);
        CUDA_CHECK(cudaMemcpy(aq.data(),candidate_workspace.base(),aq.size(),cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(as.data(),static_cast<const char*>(candidate_workspace.base())+aq.size(),as.size()*2,cudaMemcpyDeviceToHost));
        const auto bf=[](std::uint16_t x){return std::bit_cast<float>(static_cast<std::uint32_t>(x)<<16);};
        std::vector<double> expected,old_values,new_values;
        constexpr int groups=kHidden/64;
        for(int t=0;t<kTokens;++t) for(int row : {0,1,1279,1280,2559,2560,5118,5119}) {
            double total=0;
            for(int g=0;g<groups;++g) {
                int dot=0;
                for(int j=0;j<64;++j) {
                    const auto index=static_cast<std::size_t>(row)*groups+g;
                    const int bits=((low[index*32+j/2]>>((j&1)*4))&15) |
                        (((high[index*8+j/8]>>(j&7))&1)<<4);
                    dot+=((bits^16)-16)*static_cast<int>(aq[t*kHidden+g*64+j]);
                }
                total+=static_cast<double>(dot)*__half2float(__ushort_as_half(weight_scales[row*groups+g]))*
                    __half2float(__ushort_as_half(as[t*groups+g]));
            }
            const auto index=t*kRows+row;
            expected.push_back(total+bf(initial_bits[index]));
            old_values.push_back(bf(host_reference[index]));new_values.push_back(bf(host_candidate[index]));
        }
        constexpr test::ReductionCriterion criterion{1.0/256.0,1.0/256.0,2.0/256.0};
        const auto old_stats=test::compute_reduction_stats(old_values.data(),expected.data(),expected.size());
        const auto new_stats=test::compute_reduction_stats(new_values.data(),expected.data(),expected.size());
        const bool oracle_pass=test::reduction_passes(old_stats,expected.size(),criterion) &&
            test::reduction_passes(new_stats,expected.size(),criterion);
        double error=0,norm=0;std::size_t bad=0;
        for(std::size_t i=0;i<host_reference.size();++i) {
            const double a=bf(host_reference[i]), b=bf(host_candidate[i]);
            error+=(a-b)*(a-b);norm+=a*a;bad+=!std::isfinite(b);
        }
        reset_outputs(); launch_candidate(stream); CUDA_CHECK(cudaStreamSynchronize(stream));
        std::vector<std::uint16_t> replay(host_candidate.size());candidate.copy_to_host(replay.data(),candidate.bytes);
        const bool deterministic=replay==host_candidate;
        std::printf("FP64 samples=%zu old_rel_l2=%.9g new_rel_l2=%.9g max_abs=%.9g gross_limit=%.9g oracle_pass=%d all_output_rel_l2=%.9g nonfinite=%zu deterministic=%d\n",
            expected.size(),old_stats.relative_l2,new_stats.relative_l2,new_stats.maximum_absolute_error,
            test::gross_error_limit(new_stats,criterion),oracle_pass,std::sqrt(error/std::max(norm,1e-30)),bad,deterministic);
        if(!oracle_pass || bad || !deterministic || std::sqrt(error/std::max(norm,1e-30))>1.0/256.0) return 2;
        const auto reference_timing = bench::measure_cold_launch(
            launch_reference, flush, stream, 5, 30);
        const auto candidate_timing = bench::measure_cold_launch(
            launch_candidate, flush, stream, 5, 30);
        std::printf(
            "RESULT q5_native_row8 exact=%d mismatches=%zu first=%zu "
            "reference_us=%.3f candidate_us=%.3f speedup=%.6f workspace=%zu\n",
            mismatches == 0 ? 1 : 0, mismatches, first_mismatch,
            reference_timing.median_us, candidate_timing.median_us,
            reference_timing.median_us / candidate_timing.median_us, workspace_bytes);
        CUDA_CHECK(cudaStreamDestroy(stream));
        return mismatches==0?0:3; // This native row mapping must remain bit-exact.
    } catch (const std::exception& error) {
        std::fprintf(stderr, "q5_native_row8_bench: %s\n", error.what());
        return 1;
    }
}
