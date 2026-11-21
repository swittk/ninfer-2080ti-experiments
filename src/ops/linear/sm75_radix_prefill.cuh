#pragma once
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cstdint>
namespace ninfer::ops::detail::resident_prefill {
// Expand the original signed G64 coefficients directly from the reversible
// resident radix permutation. No intermediate canonical matrix is materialized.
// Keep the native BF16-weight rounding before conversion to the FP16 BLAS operand.
template<bool Q5, int N, int K>
__global__ void expand_weights(const std::uint16_t* low,const std::uint32_t* high,
                               const std::uint16_t* scales,__half2* output,
                               std::int64_t pairs,int row_begin) {
    static_assert((!Q5 && N==4096 && K==5120) || (Q5 && N==5120 && K==6144));
    constexpr int groups=K/64;
    for (std::int64_t i=static_cast<std::int64_t>(blockIdx.x)*blockDim.x+threadIdx.x;
         i<pairs;i+=static_cast<std::int64_t>(gridDim.x)*blockDim.x) {
        const auto canonical_pair=i+static_cast<std::int64_t>(row_begin)*K/2;
        const int row=canonical_pair/(K/2),group=(canonical_pair%(K/2))/32,pair=canonical_pair%32;
        const int r=row%16,word=pair/2;
        const auto tile=static_cast<std::int64_t>(row/16)*groups+group;
        const int lane=4*(r%8)+word%4;
        const int part=4*(word/8)+2*(r/8)+(word/4)%2;
        const unsigned bits=(low[tile*256+lane*8+part]>>((pair&1)*8))&255U;
        int a=bits&15U,b=bits>>4;
        if constexpr (Q5) {
            const int byte=pair/4,hlane=4*(r%8)+2*(byte%2);
            const int shift=((byte/4)*2+r/8)*8+((byte%4)/2)*4;
            const auto at=tile*32+hlane;
            const unsigned h=((high[at]>>shift)&15U)|(((high[at+1]>>shift)&15U)<<4);
            const int bit=(pair%4)*2;
            a|=((h>>bit)&1U)<<4;b|=((h>>(bit+1))&1U)<<4;
            a=(a^16)-16;b=(b^16)-16;
        } else {
            a=(a^8)-8;b=(b^8)-8;
        }
        const float scale=__half2float(__ushort_as_half(scales[tile*16+r]));
        output[i]=__floats2half2_rn(__bfloat162float(__float2bfloat16_rn(a*scale)),
                                  __bfloat162float(__float2bfloat16_rn(b*scale)));
    }
}
} // namespace ninfer::ops::detail::resident_prefill
