#pragma once

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstdint>

namespace ninfer::ops {

__device__ __forceinline__ std::uint16_t float_to_bf16_bits_rn(float value) {
    std::uint32_t bits = __float_as_uint(value);
    const std::uint32_t magnitude = bits & 0x7fffffffu;
    if (magnitude > 0x7f800000u) {
        // Preserve NaN sign/payload while ensuring the truncated BF16 remains NaN.
        return static_cast<std::uint16_t>((bits >> 16) | 0x0040u);
    }
    // Round to nearest, ties to even before discarding the low 16 FP32 bits.
    bits += 0x00007fffu + ((bits >> 16) & 1u);
    return static_cast<std::uint16_t>(bits >> 16);
}

__device__ __forceinline__ __nv_bfloat16 float_to_bf16_rn(float value) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    return __float2bfloat16_rn(value);
#else
    __nv_bfloat16_raw raw;
    raw.x = float_to_bf16_bits_rn(value);
    return __nv_bfloat16(raw);
#endif
}

__device__ __forceinline__ __nv_bfloat162 floats_to_bf162_rn(float lo, float hi) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    return __floats2bfloat162_rn(lo, hi);
#else
    __nv_bfloat162_raw raw;
    raw.x = static_cast<std::uint32_t>(float_to_bf16_bits_rn(lo)) |
            (static_cast<std::uint32_t>(float_to_bf16_bits_rn(hi)) << 16);
    return __nv_bfloat162(raw);
#endif
}

} // namespace ninfer::ops
