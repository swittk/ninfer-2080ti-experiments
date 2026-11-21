#pragma once

// Implements: include/ninfer/ops/argmax.h
// Match: contiguous BF16 [physical_rows,C], valid_rows <= physical_rows, with
// tuned routes qualified for the 248077-row and 131072-row vocabularies.
// Algorithm assumptions: one CTA reduces each blockDim.x-row tile and an atomic
// winner selects the exact value/lower-id maximum per column.

#include "ops/common/argmax_reduce.cuh"
#include <cuda_bf16.h>
#include <cstdint>
#include <climits>
#include <math_constants.h>

namespace ninfer::ops {

// Small-column execution uses 512 threads to reduce atomic contenders. Aggregate
// execution dispatches registered vocab profiles to a smaller block so the much
// larger 2-D grid exposes enough resident CTAs without oversized reductions.
inline constexpr int kArgmaxBlock          = 512;
inline constexpr int kArgmaxItemsPerThread = 1;

__launch_bounds__(kArgmaxBlock) __global__
    void argmax_kernel(const __nv_bfloat16* logits, std::int32_t* out, std::int32_t valid_rows,
                       std::int32_t physical_rows) {
    const std::int32_t t    = static_cast<std::int32_t>(blockIdx.x);
    const std::int64_t base = static_cast<std::int64_t>(t) * physical_rows;

    float best_value        = __bfloat162float(logits[base]);
    std::int32_t best_index = 0;
    for (std::int32_t v = static_cast<std::int32_t>(threadIdx.x); v < valid_rows; v += blockDim.x) {
        const float value = __bfloat162float(logits[base + v]);
        if (argmax_better(value, v, best_value, best_index)) {
            best_value = value;
            best_index = v;
        }
    }

    __shared__ float values[kArgmaxBlock];
    __shared__ std::int32_t indices[kArgmaxBlock];
    values[threadIdx.x]  = best_value;
    indices[threadIdx.x] = best_index;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            const float other_value        = values[threadIdx.x + stride];
            const std::int32_t other_index = indices[threadIdx.x + stride];
            if (argmax_better(other_value, other_index, values[threadIdx.x],
                              indices[threadIdx.x])) {
                values[threadIdx.x]  = other_value;
                indices[threadIdx.x] = other_index;
            }
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) { out[t] = indices[0]; }
}

__launch_bounds__(kArgmaxBlock) __global__
    void argmax_tiled_atomic_kernel(const __nv_bfloat16* logits, std::int32_t* out,
                                    std::int32_t valid_rows, std::int32_t physical_rows) {
    const std::int32_t t    = static_cast<std::int32_t>(blockIdx.y);
    const std::int64_t base = static_cast<std::int64_t>(t) * physical_rows;
    const std::int32_t tile_start =
        static_cast<std::int32_t>(blockIdx.x) * blockDim.x * kArgmaxItemsPerThread;

    // The row-zero NaN contract is independent of a previous graph winner.
    // All CTAs take this uniform branch, leaving exactly one writer. Without
    // this check a finite stale seed wins over NaN, or all-NaN tiles return
    // without writing any result at all.
    if (isnan(__bfloat162float(logits[base]))) {
        if (blockIdx.x == 0 && threadIdx.x == 0) out[t] = 0;
        return;
    }

    float best_value        = -CUDART_INF_F;
    std::int32_t best_index = INT32_MAX;
#pragma unroll
    for (int item = 0; item < kArgmaxItemsPerThread; ++item) {
        const std::int32_t v = tile_start + threadIdx.x + item * blockDim.x;
        if (v < valid_rows) {
            const float value = __bfloat162float(logits[base + v]);
            if (argmax_better(value, v, best_value, best_index)) {
                best_value = value;
                best_index = v;
            }
        }
    }

    argmax_block_reduce(best_value, best_index);
    if (threadIdx.x != 0 || best_index == INT32_MAX) { return; }

    // Multiple tile CTAs update the same winner. Start from the actual value atomically: the
    // output may contain a valid winner from a previous CUDA-Graph replay, and mixing a plain load
    // with concurrent CAS writes is a data race. Invalid stale values are normalized to row zero
    // before any logits access. From then on the CAS loop monotonically advances through the exact
    // (value, lower-id) argmax order, so no separate memset/reset node is required.
    int current = atomicAdd(reinterpret_cast<int*>(out + t), 0);
    while (true) {
        if (current < 0 || current >= valid_rows) {
            const int observed = atomicCAS(reinterpret_cast<int*>(out + t), current, 0);
            current            = observed == current ? 0 : observed;
            continue;
        }
        const float current_value = __bfloat162float(logits[base + current]);
        // The historical row-zero seed also defines behavior for NaN logits: row zero wins if it
        // is NaN, while NaNs at other rows never become winners. Normalize a stale NaN winner back
        // to row zero before comparing tile maxima so arbitrary preseeds preserve that contract.
        if (isnan(current_value) && current != 0) {
            const int observed = atomicCAS(reinterpret_cast<int*>(out + t), current, 0);
            current            = observed == current ? 0 : observed;
            continue;
        }
        if (!argmax_better(best_value, best_index, current_value, current)) { break; }
        const int observed = atomicCAS(reinterpret_cast<int*>(out + t), current, best_index);
        if (observed == current) { break; }
        current = observed;
    }
}

} // namespace ninfer::ops
