#pragma once

#include "ops/common/bf16.cuh"
#include "ops/common/memory.cuh"

#include <cuda_fp16.h>

namespace ninfer::ops {

__device__ __forceinline__ void ldmatrix_x2(unsigned& r0, unsigned& r1, unsigned addr) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0,%1}, [%2];\n"
                 : "=r"(r0), "=r"(r1)
                 : "r"(addr));
}

__device__ __forceinline__ void ldmatrix_x4(unsigned& r0, unsigned& r1, unsigned& r2, unsigned& r3,
                                            unsigned addr) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                 : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3)
                 : "r"(addr));
}

__device__ __forceinline__ void ldmatrix_x2_t(unsigned& r0, unsigned& r1, unsigned addr) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%0,%1}, [%2];\n"
                 : "=r"(r0), "=r"(r1)
                 : "r"(addr));
}

__device__ __forceinline__ void ldmatrix_x4_t(unsigned& r0, unsigned& r1, unsigned& r2,
                                              unsigned& r3, unsigned addr) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                 : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3)
                 : "r"(addr));
}

__device__ __forceinline__ void mma_f16_k8(float& c0, float& c1, float& c2, float& c3,
                                           unsigned a0, unsigned a1, unsigned b0) {
    asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32 "
                 "{%0,%1,%2,%3}, {%4,%5}, {%6}, {%0,%1,%2,%3};\n"
                 : "+f"(c0), "+f"(c1), "+f"(c2), "+f"(c3)
                 : "r"(a0), "r"(a1), "r"(b0));
}

__device__ __forceinline__ void mma_f16(float& c0, float& c1, float& c2, float& c3, unsigned a0,
                                        unsigned a1, unsigned a2, unsigned a3, unsigned b0,
                                        unsigned b1) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
                 "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                 : "+f"(c0), "+f"(c1), "+f"(c2), "+f"(c3)
                 : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
#else
    // Turing exposes the same m16n8 accumulator tile with K=8 FP16 MMA.
    // The Ampere K=16 fragments are two adjacent K=8 fragments, so preserve
    // the public wrapper shape by accumulating the two halves in order.
    mma_f16_k8(c0, c1, c2, c3, a0, a1, b0);
    mma_f16_k8(c0, c1, c2, c3, a2, a3, b1);
#endif
}

__device__ __forceinline__ unsigned bf16x2_to_f16x2(unsigned packed) {
    const float lo = __uint_as_float((packed & 0xffffu) << 16);
    const float hi = __uint_as_float(packed & 0xffff0000u);
    union PackedHalf2 {
        __half2 h;
        unsigned u;
    } out;
    out.h = __floats2half2_rn(lo, hi);
    return out.u;
}

__device__ __forceinline__ void mma_bf16(float& c0, float& c1, float& c2, float& c3, unsigned a0,
                                         unsigned a1, unsigned a2, unsigned a3, unsigned b0,
                                         unsigned b1) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
                 "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                 : "+f"(c0), "+f"(c1), "+f"(c2), "+f"(c3)
                 : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
#else
    // SM75 Tensor Cores do not accept BF16 operands. Convert each packed BF16
    // fragment to FP16 in registers and reuse the native Turing FP16 MMA path.
    mma_f16(c0, c1, c2, c3, bf16x2_to_f16x2(a0), bf16x2_to_f16x2(a1),
            bf16x2_to_f16x2(a2), bf16x2_to_f16x2(a3), bf16x2_to_f16x2(b0),
            bf16x2_to_f16x2(b1));
#endif
}

__device__ __forceinline__ void mma_s8(int& c0, int& c1, int& c2, int& c3, unsigned a0, unsigned a1,
                                       unsigned a2, unsigned a3, unsigned b0, unsigned b1) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 "
                 "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                 : "+r"(c0), "+r"(c1), "+r"(c2), "+r"(c3)
                 : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
#else
    auto mma_s8_k16 = [](int& d0, int& d1, unsigned a, unsigned b) {
        asm volatile("mma.sync.aligned.m8n8k16.row.col.s32.s8.s8.s32 "
                     "{%0,%1}, {%2}, {%3}, {%0,%1};\n"
                     : "+r"(d0), "+r"(d1)
                     : "r"(a), "r"(b));
    };
    // Decompose Ampere's logical m16n8k32 fragment into Turing's native
    // m8n8k16 integer MMA: two K halves for each 8-row accumulator half.
    mma_s8_k16(c0, c1, a0, b0);
    mma_s8_k16(c0, c1, a2, b1);
    mma_s8_k16(c2, c3, a1, b0);
    mma_s8_k16(c2, c3, a3, b1);
#endif
}

__device__ __forceinline__ unsigned f32x2_to_f16x2(float x, float y) {
    union PackedHalf2 {
        __half2 h;
        unsigned u;
    } out;
    out.h = __floats2half2_rn(x, y);
    return out.u;
}

__device__ __forceinline__ void mma_tf32_bits(float& c0, float& c1, float& c2, float& c3,
                                              unsigned a0, unsigned a1, unsigned a2, unsigned a3,
                                              unsigned b0, unsigned b1) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 "
                 "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                 : "+f"(c0), "+f"(c1), "+f"(c2), "+f"(c3)
                 : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
#else
    // SM75 has no native TF32 MMA, but TF32 and FP16 carry the same 10-bit
    // fraction precision. Repack the PTX TF32 fragment ownership into Turing's
    // native FP16 m16n8k8 layout using 4-lane subgroup shuffles, then convert
    // the selected scalars to FP16 and accumulate in FP32.
    constexpr unsigned mask = 0xffffffffU;
    const int lane_t = static_cast<int>(threadIdx.x) & 3;
    const int pair   = lane_t & 1;
    const int src0   = pair * 2;
    const int src1   = src0 + 1;
    const bool high  = lane_t >= 2;

    // Shuffle each source register independently. Selecting a0/a2 (or b0/b1)
    // before shuffling is wrong because source lanes make a different local
    // selection than the receiving lane.
    const unsigned a0_s0 = __shfl_sync(mask, a0, src0, 4);
    const unsigned a0_s1 = __shfl_sync(mask, a0, src1, 4);
    const unsigned a1_s0 = __shfl_sync(mask, a1, src0, 4);
    const unsigned a1_s1 = __shfl_sync(mask, a1, src1, 4);
    const unsigned a2_s0 = __shfl_sync(mask, a2, src0, 4);
    const unsigned a2_s1 = __shfl_sync(mask, a2, src1, 4);
    const unsigned a3_s0 = __shfl_sync(mask, a3, src0, 4);
    const unsigned a3_s1 = __shfl_sync(mask, a3, src1, 4);

    const unsigned b0_s0 = __shfl_sync(mask, b0, src0, 4);
    const unsigned b0_s1 = __shfl_sync(mask, b0, src1, 4);
    const unsigned b1_s0 = __shfl_sync(mask, b1, src0, 4);
    const unsigned b1_s1 = __shfl_sync(mask, b1, src1, 4);

    const float a_top0 = __uint_as_float(high ? a2_s0 : a0_s0);
    const float a_top1 = __uint_as_float(high ? a2_s1 : a0_s1);
    const float a_bot0 = __uint_as_float(high ? a3_s0 : a1_s0);
    const float a_bot1 = __uint_as_float(high ? a3_s1 : a1_s1);
    const float b_row0 = __uint_as_float(high ? b1_s0 : b0_s0);
    const float b_row1 = __uint_as_float(high ? b1_s1 : b0_s1);

    mma_f16_k8(c0, c1, c2, c3, f32x2_to_f16x2(a_top0, a_top1),
               f32x2_to_f16x2(a_bot0, a_bot1), f32x2_to_f16x2(b_row0, b_row1));
#endif
}

__device__ __forceinline__ void mma_tf32(float& c0, float& c1, float& c2, float& c3, float a0,
                                         float a1, float a2, float a3, float b0, float b1) {
    mma_tf32_bits(c0, c1, c2, c3, __float_as_uint(a0), __float_as_uint(a1), __float_as_uint(a2),
                  __float_as_uint(a3), __float_as_uint(b0), __float_as_uint(b1));
}

__device__ __forceinline__ void mma_nvfp4_e4m3(float& c0, float& c1, float& c2, float& c3,
                                               unsigned a0, unsigned a1, unsigned a2, unsigned a3,
                                               unsigned b0, unsigned b1, unsigned sfa,
                                               unsigned sfb) {
    constexpr unsigned short kScaleBlockId  = 0;
    constexpr unsigned short kScaleThreadId = 0;
    asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X."
                 "m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 "
                 "{%0,%1,%2,%3}, "
                 "{%4,%5,%6,%7}, "
                 "{%8,%9}, "
                 "{%0,%1,%2,%3}, "
                 "{%10}, "
                 "{%11,%12}, "
                 "{%13}, "
                 "{%14,%15};\n"
                 : "+f"(c0), "+f"(c1), "+f"(c2), "+f"(c3)
                 : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1), "r"(sfa),
                   "h"(kScaleBlockId), "h"(kScaleThreadId), "r"(sfb), "h"(kScaleBlockId),
                   "h"(kScaleThreadId));
}

} // namespace ninfer::ops
