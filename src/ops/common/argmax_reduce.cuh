#pragma once

#include <cuda_runtime.h>
#include <cstdint>
#include <climits>
#include <math_constants.h>

namespace ninfer::ops {

__device__ __forceinline__ bool argmax_better(float value, std::int32_t index, float best_value,
                                              std::int32_t best_index) {
    return value > best_value || (value == best_value && index < best_index);
}

__device__ __forceinline__ void argmax_warp_reduce(float& value, std::int32_t& index) {
    constexpr unsigned int kMask = 0xffffffffu;
    for (int offset = 16; offset > 0; offset >>= 1) {
        const float other_value        = __shfl_down_sync(kMask, value, offset);
        const std::int32_t other_index = __shfl_down_sync(kMask, index, offset);
        if (argmax_better(other_value, other_index, value, index)) {
            value = other_value;
            index = other_index;
        }
    }
}

__device__ __forceinline__ void argmax_block_reduce(float& value, std::int32_t& index) {
    __shared__ float warp_values[16];
    __shared__ std::int32_t warp_indices[16];

    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    argmax_warp_reduce(value, index);
    if (lane == 0) {
        warp_values[warp]  = value;
        warp_indices[warp] = index;
    }
    __syncthreads();

    value = (lane < (blockDim.x >> 5)) ? warp_values[lane] : -CUDART_INF_F;
    index = (lane < (blockDim.x >> 5)) ? warp_indices[lane] : INT32_MAX;
    if (warp == 0) { argmax_warp_reduce(value, index); }
}


} // namespace ninfer::ops
