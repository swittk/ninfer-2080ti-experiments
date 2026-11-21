#include "ops/linear/q4/q4_proposal_b2_sm75.h"

#include "core/device.h"
#include "ops/kernel/sampling_device.cuh"
#include "ops/linear/q4/q4_rowsplit_gemv.cuh"

#include <cub/block/block_radix_sort.cuh>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <mma.h>

#include <cstdint>
#include <stdexcept>

namespace ninfer::ops::detail {
namespace {

using b1 = nvcuda::wmma::experimental::precision::b1;

constexpr int kN            = 131072;
constexpr int kK            = 5120;
constexpr int kSourceGroups = 80;
constexpr int kB2GroupK     = 128;
constexpr int kB2Groups     = kK / kB2GroupK;
constexpr int kRows         = 16;
constexpr int kWarps        = 8;
constexpr int kThreads      = kWarps * 32;
constexpr int kCandidates   = 128;

constexpr int kProjectCtas    = kN / (kRows * kWarps);
constexpr int kCtasPerPartial = 8;
constexpr int kPartialBlocks  = kProjectCtas / kCtasPerPartial;
constexpr int kGroupPartials  = 16;
constexpr int kGroupBlocks    = kPartialBlocks / kGroupPartials;
constexpr int kSourceRowBytes = kSourceGroups * 32;

using GroupSort = cub::BlockRadixSort<unsigned long long, 256, 8>;
using FinalSort = cub::BlockRadixSort<unsigned long long, 256, 4>;

constexpr std::size_t kPackedBytes = static_cast<std::size_t>(kN) * kK / 4;
constexpr std::size_t kScaleBytes =
    static_cast<std::size_t>(kN) * kB2Groups * sizeof(__half);

struct State {
    const void* source = nullptr;
    uint4* packed = nullptr;
    std::uint32_t* scales = nullptr;
    uint2* activation_fragments = nullptr;
    __half* activation_scales = nullptr;
    unsigned long long* partial_keys = nullptr;
    unsigned long long* group_keys = nullptr;
    std::int32_t* candidate_ids = nullptr;
    std::uint8_t* exact_codes = nullptr;
    std::uint16_t* exact_scales = nullptr;
    __nv_bfloat16* exact_logits = nullptr;
};

State& state() {
    static State value;
    return value;
}

__device__ __forceinline__ int decode_q4(std::uint8_t packed, bool high) {
    const int nibble = high ? packed >> 4 : packed & 15;
    return (nibble ^ 8) - 8;
}

__device__ __forceinline__ void bmma_xor(int& c0, int& c1, unsigned a, unsigned b) {
    asm volatile(
        "mma.sync.aligned.m8n8k128.row.col.s32.b1.b1.s32.xor.popc "
        "{%0,%1}, {%2}, {%3}, {%0,%1};\n"
        : "+r"(c0), "+r"(c1)
        : "r"(a), "r"(b));
}

__global__ void transform_b2(const std::uint8_t* codes, const std::uint16_t* source_scales,
                             uint4* output, std::uint32_t* output_scales) {
    __shared__ __align__(16) std::uint32_t high_bits[2][32];
    __shared__ __align__(16) std::uint32_t low_bits[2][32];
    __shared__ std::uint32_t high_fragments[2][32];
    __shared__ std::uint32_t low_fragments[2][32];
    __shared__ std::uint16_t row_scales[2][8];

    const int warp = static_cast<int>(threadIdx.x) >> 5;
    const int lane = static_cast<int>(threadIdx.x) & 31;
    const int row_in_half = lane >> 2;
    const int lane_in_row = lane & 3;
    const int row_block = static_cast<int>(blockIdx.x);
    const int row = row_block * kRows + warp * 8 + row_in_half;

    for (int group = 0; group < kB2Groups; ++group) {
        const int source_group = group * 2 + (lane_in_row >> 1);
        const int half = lane_in_row & 1;
        const auto* input = codes + static_cast<std::int64_t>(row) * kSourceRowBytes +
                            source_group * 32 + half * 16;
        const float source_scale = __half2float(
            __ushort_as_half(source_scales[static_cast<std::int64_t>(row) * kSourceGroups +
                                            source_group]));

        float sum = 0.0f;
#pragma unroll
        for (int i = 0; i < 32; ++i) {
            const int value = decode_q4(input[i >> 1], (i & 1) != 0);
            sum += fabsf(static_cast<float>(value) * source_scale);
        }
        sum += __shfl_xor_sync(0xffffffffu, sum, 1, 4);
        sum += __shfl_xor_sync(0xffffffffu, sum, 2, 4);
        float alpha = sum * (1.0f / (128.0f * 1.5f));

#pragma unroll
        for (int iteration = 0; iteration < 2; ++iteration) {
            float numerator = 0.0f;
            float denominator = 0.0f;
#pragma unroll
            for (int i = 0; i < 32; ++i) {
                const int value = decode_q4(input[i >> 1], (i & 1) != 0);
                const float magnitude = fabsf(static_cast<float>(value) * source_scale);
                const float level = magnitude > 2.0f * alpha ? 3.0f : 1.0f;
                numerator += magnitude * level;
                denominator += level * level;
            }
            numerator += __shfl_xor_sync(0xffffffffu, numerator, 1, 4);
            numerator += __shfl_xor_sync(0xffffffffu, numerator, 2, 4);
            denominator += __shfl_xor_sync(0xffffffffu, denominator, 1, 4);
            denominator += __shfl_xor_sync(0xffffffffu, denominator, 2, 4);
            alpha = denominator > 0.0f ? numerator / denominator : 0.0f;
        }

        std::uint32_t high = 0;
        std::uint32_t low = 0;
#pragma unroll
        for (int i = 0; i < 32; ++i) {
            const int quantized = decode_q4(input[i >> 1], (i & 1) != 0);
            const float value = static_cast<float>(quantized) * source_scale;
            const bool negative = value < 0.0f;
            const bool strong = fabsf(value) > 2.0f * alpha;
            high |= static_cast<std::uint32_t>(negative) << i;
            low |= static_cast<std::uint32_t>(strong ? negative : !negative) << i;
        }

        high_bits[warp][lane] = high;
        low_bits[warp][lane] = low;
        if (lane_in_row == 0) {
            row_scales[warp][row_in_half] = __half_as_ushort(__float2half_rn(alpha));
        }
        __syncwarp();

        nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, 8, 8, 128, b1,
                               nvcuda::wmma::row_major>
            high_fragment;
        nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, 8, 8, 128, b1,
                               nvcuda::wmma::row_major>
            low_fragment;
        nvcuda::wmma::load_matrix_sync(high_fragment, high_bits[warp], 128);
        nvcuda::wmma::load_matrix_sync(low_fragment, low_bits[warp], 128);
        high_fragments[warp][lane] = high_fragment.x[0];
        low_fragments[warp][lane] = low_fragment.x[0];
        __syncthreads();

        if (warp == 0) {
            output[(static_cast<std::size_t>(row_block) * kB2Groups + group) * 32 + lane] =
                make_uint4(high_fragments[0][lane], low_fragments[0][lane],
                           high_fragments[1][lane], low_fragments[1][lane]);
            if (lane < 8) {
                output_scales[(static_cast<std::size_t>(row_block) * kB2Groups + group) * 8 +
                              lane] =
                    static_cast<std::uint32_t>(row_scales[0][lane]) |
                    (static_cast<std::uint32_t>(row_scales[1][lane]) << 16);
            }
        }
        __syncthreads();
    }
}

__global__ void prepare_b2(const __nv_bfloat16* hidden, uint2* output, __half* scales) {
    __shared__ __align__(16) std::uint32_t high_bits[8][32];
    __shared__ __align__(16) std::uint32_t low_bits[8][32];

    const int warp = static_cast<int>(threadIdx.x) >> 5;
    const int lane = static_cast<int>(threadIdx.x) & 31;
    const int group = static_cast<int>(blockIdx.x) * 8 + warp;
    if (group >= kB2Groups) { return; }

    high_bits[warp][lane] = 0;
    low_bits[warp][lane] = 0;
    if (lane < 4) {
        const int word = lane;
        const auto* input = hidden + group * kB2GroupK + word * 32;
        float sum = 0.0f;
#pragma unroll
        for (int i = 0; i < 32; ++i) { sum += fabsf(__bfloat162float(input[i])); }
        sum += __shfl_xor_sync(0x0000000fu, sum, 1, 4);
        sum += __shfl_xor_sync(0x0000000fu, sum, 2, 4);
        float alpha = sum * (1.0f / (128.0f * 1.5f));

#pragma unroll
        for (int iteration = 0; iteration < 2; ++iteration) {
            float numerator = 0.0f;
            float denominator = 0.0f;
#pragma unroll
            for (int i = 0; i < 32; ++i) {
                const float magnitude = fabsf(__bfloat162float(input[i]));
                const float level = magnitude > 2.0f * alpha ? 3.0f : 1.0f;
                numerator += magnitude * level;
                denominator += level * level;
            }
            numerator += __shfl_xor_sync(0x0000000fu, numerator, 1, 4);
            numerator += __shfl_xor_sync(0x0000000fu, numerator, 2, 4);
            denominator += __shfl_xor_sync(0x0000000fu, denominator, 1, 4);
            denominator += __shfl_xor_sync(0x0000000fu, denominator, 2, 4);
            alpha = denominator > 0.0f ? numerator / denominator : 0.0f;
        }

        std::uint32_t high = 0;
        std::uint32_t low = 0;
#pragma unroll
        for (int i = 0; i < 32; ++i) {
            const float value = __bfloat162float(input[i]);
            const bool negative = value < 0.0f;
            const bool strong = fabsf(value) > 2.0f * alpha;
            high |= static_cast<std::uint32_t>(negative) << i;
            low |= static_cast<std::uint32_t>(strong ? negative : !negative) << i;
        }
        high_bits[warp][word] = high;
        low_bits[warp][word] = low;
        if (word == 0) { scales[group] = __float2half_rn(alpha); }
    }
    __syncwarp();

    nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, 8, 8, 128, b1,
                           nvcuda::wmma::col_major>
        high_fragment;
    nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, 8, 8, 128, b1,
                           nvcuda::wmma::col_major>
        low_fragment;
    nvcuda::wmma::load_matrix_sync(high_fragment, high_bits[warp], 128);
    nvcuda::wmma::load_matrix_sync(low_fragment, low_bits[warp], 128);
    output[group * 32 + lane] = make_uint2(high_fragment.x[0], low_fragment.x[0]);
}

__global__ void project_local_top2(const uint4* weights, const std::uint32_t* weight_scales,
                                   const uint2* activation, const __half* activation_scales,
                                   unsigned long long* partial_keys) {
    __shared__ unsigned long long local[kWarps * 16];

    const int warp = static_cast<int>(threadIdx.x) >> 5;
    const int lane = static_cast<int>(threadIdx.x) & 31;
    const int cta = static_cast<int>(blockIdx.x);
    const int row_block = cta * kWarps + warp;
    const int row_base = row_block * kRows;
    const int row_in_half = lane >> 2;
    const int lane_in_row = lane & 3;

    float top = 0.0f;
    float bottom = 0.0f;
    uint4 packed = weights[(static_cast<std::int64_t>(row_block) * kB2Groups) * 32 + lane];
#pragma unroll 1
    for (int group = 0; group < kB2Groups; ++group) {
        uint4 next{};
        if (group + 1 < kB2Groups) {
            next = weights[(static_cast<std::int64_t>(row_block) * kB2Groups + group + 1) *
                           32 + lane];
        }
        const uint2 act = activation[group * 32 + lane];
        int top_q = 0;
        int bottom_q = 0;
        int a = 0;
        int ignored = 0;
        bmma_xor(a, ignored, packed.x, act.x);
        top_q += 4 * (128 - 2 * a);
        a = ignored = 0;
        bmma_xor(a, ignored, packed.x, act.y);
        top_q += 2 * (128 - 2 * a);
        a = ignored = 0;
        bmma_xor(a, ignored, packed.y, act.x);
        top_q += 2 * (128 - 2 * a);
        a = ignored = 0;
        bmma_xor(a, ignored, packed.y, act.y);
        top_q += 128 - 2 * a;

        a = ignored = 0;
        bmma_xor(a, ignored, packed.z, act.x);
        bottom_q += 4 * (128 - 2 * a);
        a = ignored = 0;
        bmma_xor(a, ignored, packed.z, act.y);
        bottom_q += 2 * (128 - 2 * a);
        a = ignored = 0;
        bmma_xor(a, ignored, packed.w, act.x);
        bottom_q += 2 * (128 - 2 * a);
        a = ignored = 0;
        bmma_xor(a, ignored, packed.w, act.y);
        bottom_q += 128 - 2 * a;

        std::uint32_t scale_pair = 0;
        if (lane_in_row == 0) {
            scale_pair = weight_scales[
                (static_cast<std::int64_t>(row_block) * kB2Groups + group) * 8 +
                row_in_half];
        }
        scale_pair = __shfl_sync(0xffffffffu, scale_pair, row_in_half * 4);
        const float top_scale =
            __half2float(__ushort_as_half(static_cast<std::uint16_t>(scale_pair)));
        const float bottom_scale =
            __half2float(__ushort_as_half(static_cast<std::uint16_t>(scale_pair >> 16)));
        const float activation_scale = __half2float(activation_scales[group]);
        top = __fmaf_rn(static_cast<float>(top_q), top_scale * activation_scale, top);
        bottom =
            __fmaf_rn(static_cast<float>(bottom_q), bottom_scale * activation_scale, bottom);
        packed = next;
    }

    if (lane_in_row == 0) {
        const int offset = warp * 16 + row_in_half;
        local[offset] = sampling_sort_key(
            __bfloat162float(__float2bfloat16_rn(top)), row_base + row_in_half);
        local[offset + 8] = sampling_sort_key(
            __bfloat162float(__float2bfloat16_rn(bottom)), row_base + row_in_half + 8);
    }
    __syncthreads();

    if (lane == 0) {
        const int offset = warp * 16;
        unsigned long long best0 = 0;
        unsigned long long best1 = 0;
#pragma unroll
        for (int i = 0; i < 16; ++i) {
            const unsigned long long value = local[offset + i];
            if (value > best0) {
                best1 = best0;
                best0 = value;
            } else if (value > best1) {
                best1 = value;
            }
        }
        const int partial = cta / kCtasPerPartial;
        const int slot = cta % kCtasPerPartial;
        const int destination = partial * kCandidates + slot * (2 * kWarps) + 2 * warp;
        partial_keys[destination] = best0;
        partial_keys[destination + 1] = best1;
    }
}

__global__ void topk_group(const unsigned long long* input, unsigned long long* output) {
    const int group = static_cast<int>(blockIdx.x);
    const int thread = static_cast<int>(threadIdx.x);
    const int first = group * kGroupPartials;
    unsigned long long keys[8];
#pragma unroll
    for (int item = 0; item < 8; ++item) {
        const int q = thread * 8 + item;
        const int partial = q / kCandidates;
        const int rank = q - partial * kCandidates;
        keys[item] = input[(first + partial) * kCandidates + rank];
    }
    __shared__ typename GroupSort::TempStorage storage;
    GroupSort(storage).SortDescending(keys);
#pragma unroll
    for (int item = 0; item < 8; ++item) {
        const int rank = thread * 8 + item;
        if (rank < kCandidates) { output[group * kCandidates + rank] = keys[item]; }
    }
}

__global__ void topk_final(const unsigned long long* input, std::int32_t* ids) {
    const int thread = static_cast<int>(threadIdx.x);
    unsigned long long keys[4];
#pragma unroll
    for (int item = 0; item < 4; ++item) {
        const int q = thread * 4 + item;
        keys[item] = q < kGroupBlocks * kCandidates ? input[q] : 0ull;
    }
    __shared__ typename FinalSort::TempStorage storage;
    FinalSort(storage).SortDescending(keys);
#pragma unroll
    for (int item = 0; item < 4; ++item) {
        const int rank = thread * 4 + item;
        if (rank < kCandidates) { ids[rank] = sampling_key_index(keys[item]); }
    }
}

__global__ void gather_exact(const std::int32_t* ids, const std::uint8_t* codes,
                             const std::uint16_t* scales, std::uint8_t* exact_codes,
                             std::uint16_t* exact_scales) {
    const int destination_row = static_cast<int>(blockIdx.x);
    const int source_row = ids[destination_row];
    for (int byte = static_cast<int>(threadIdx.x); byte < kSourceRowBytes;
         byte += blockDim.x) {
        exact_codes[static_cast<std::size_t>(destination_row) * kSourceRowBytes + byte] =
            codes[static_cast<std::size_t>(source_row) * kSourceRowBytes + byte];
    }
    for (int group = static_cast<int>(threadIdx.x); group < kSourceGroups;
         group += blockDim.x) {
        exact_scales[static_cast<std::size_t>(destination_row) * kSourceGroups + group] =
            scales[static_cast<std::size_t>(source_row) * kSourceGroups + group];
    }
}

__global__ void finish_exact(const __nv_bfloat16* logits, const std::int32_t* ids,
                             std::int32_t* token) {
    const int thread = static_cast<int>(threadIdx.x);
    __shared__ unsigned long long keys[kCandidates];
    if (thread < kCandidates) {
        keys[thread] = sampling_sort_key(__bfloat162float(logits[thread]), ids[thread]);
    }
    __syncthreads();
    if (thread == 0) {
        unsigned long long best = keys[0];
        for (int i = 1; i < kCandidates; ++i) {
            if (keys[i] > best) { best = keys[i]; }
        }
        token[0] = sampling_key_index(best);
    }
}

} // namespace

std::size_t q4_proposal_b2_reservation_bytes() noexcept {
    return kPackedBytes + kScaleBytes +
           static_cast<std::size_t>(kB2Groups) * 32 * sizeof(uint2) +
           static_cast<std::size_t>(kB2Groups) * sizeof(__half) +
           static_cast<std::size_t>(kPartialBlocks) * kCandidates *
               sizeof(unsigned long long) +
           static_cast<std::size_t>(kGroupBlocks) * kCandidates *
               sizeof(unsigned long long) +
           static_cast<std::size_t>(kCandidates) * sizeof(std::int32_t) +
           static_cast<std::size_t>(kCandidates) * kSourceRowBytes +
           static_cast<std::size_t>(kCandidates) * kSourceGroups * sizeof(std::uint16_t) +
           static_cast<std::size_t>(kCandidates) * sizeof(__nv_bfloat16);
}

void q4_proposal_b2_prepare(const Weight& weight, cudaStream_t stream) {
    if (weight.qtype != QType::Q4G64_F16S || weight.n != kN || weight.k != kK ||
        weight.qdata == nullptr || weight.scales == nullptr) {
        throw std::invalid_argument("SM75 B2 proposal preparation requires exact Q4 geometry");
    }
    auto& current = state();
    if (current.source == weight.qdata) { return; }

    CUDA_CHECK(cudaMalloc(&current.packed, kPackedBytes));
    CUDA_CHECK(cudaMalloc(&current.scales, kScaleBytes));
    CUDA_CHECK(cudaMalloc(
        &current.activation_fragments,
        static_cast<std::size_t>(kB2Groups) * 32 * sizeof(uint2)));
    CUDA_CHECK(cudaMalloc(
        &current.activation_scales, static_cast<std::size_t>(kB2Groups) * sizeof(__half)));
    CUDA_CHECK(cudaMalloc(
        &current.partial_keys,
        static_cast<std::size_t>(kPartialBlocks) * kCandidates * sizeof(unsigned long long)));
    CUDA_CHECK(cudaMalloc(
        &current.group_keys,
        static_cast<std::size_t>(kGroupBlocks) * kCandidates * sizeof(unsigned long long)));
    CUDA_CHECK(cudaMalloc(
        &current.candidate_ids, static_cast<std::size_t>(kCandidates) * sizeof(std::int32_t)));
    CUDA_CHECK(cudaMalloc(
        &current.exact_codes, static_cast<std::size_t>(kCandidates) * kSourceRowBytes));
    CUDA_CHECK(cudaMalloc(
        &current.exact_scales,
        static_cast<std::size_t>(kCandidates) * kSourceGroups * sizeof(std::uint16_t)));
    CUDA_CHECK(cudaMalloc(
        &current.exact_logits,
        static_cast<std::size_t>(kCandidates) * sizeof(__nv_bfloat16)));

    transform_b2<<<kN / kRows, 64, 0, stream>>>(
        static_cast<const std::uint8_t*>(weight.qdata),
        static_cast<const std::uint16_t*>(weight.scales), current.packed, current.scales);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(stream));
    current.source = weight.qdata;
}

bool q4_proposal_b2_ready(const Weight& weight) noexcept {
    return state().source == weight.qdata && state().packed != nullptr;
}

void q4_proposal_b2_launch(const Tensor& hidden, const Weight& weight, Tensor& proposal_tokens,
                           cudaStream_t stream) {
    auto& current = state();
    if (!q4_proposal_b2_ready(weight) || hidden.ne[0] != kK || hidden.ne[1] != 1 ||
        proposal_tokens.ne[0] != 1) {
        throw std::invalid_argument("SM75 B2 proposal launch requires prepared T1 geometry");
    }

    prepare_b2<<<(kB2Groups + 7) / 8, kThreads, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(hidden.data), current.activation_fragments,
        current.activation_scales);
    project_local_top2<<<kProjectCtas, kThreads, 0, stream>>>(
        current.packed, current.scales, current.activation_fragments, current.activation_scales,
        current.partial_keys);
    topk_group<<<kGroupBlocks, 256, 0, stream>>>(current.partial_keys, current.group_keys);
    topk_final<<<1, 256, 0, stream>>>(current.group_keys, current.candidate_ids);
    gather_exact<<<kCandidates, 256, 0, stream>>>(
        current.candidate_ids, static_cast<const std::uint8_t*>(weight.qdata),
        static_cast<const std::uint16_t*>(weight.scales), current.exact_codes,
        current.exact_scales);

    constexpr dim3 grid(
        (kCandidates + Q4GemvR4W1DirectSchedule::kRowsPerCta - 1) /
        Q4GemvR4W1DirectSchedule::kRowsPerCta);
    constexpr dim3 block(Q4GemvR4W1DirectSchedule::kThreads);
    q4_rowsplit_gemv_kernel<Q4GemvR4W1DirectSchedule><<<grid, block, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(hidden.data), current.exact_codes,
        reinterpret_cast<const std::uint8_t*>(current.exact_scales), current.exact_logits,
        nullptr, kCandidates, kK);
    finish_exact<<<1, kCandidates, 0, stream>>>(current.exact_logits, current.candidate_ids,
                                                static_cast<std::int32_t*>(proposal_tokens.data));
    CUDA_CHECK(cudaGetLastError());
}

} // namespace ninfer::ops::detail
