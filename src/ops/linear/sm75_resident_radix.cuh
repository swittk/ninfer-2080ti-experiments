#pragma once
#include "core/device.h"
#include <cuda_fp16.h>
#include <cstdint>

namespace ninfer::ops::detail::resident_radix {
// Both representations have identical byte extents inside each 16-row block.
// Stage one such block in shared memory before writing to its original address.
template<bool ToPacked>
__global__ void permute_low(std::uint16_t* data, int groups) {
    extern __shared__ std::uint16_t tmp[];
    const int words = 16 * groups * 16;
    auto* plane = data + static_cast<std::int64_t>(blockIdx.x) * words;
    for (int i = threadIdx.x; i < words; i += blockDim.x) tmp[i] = plane[i];
    __syncthreads();
    for (int i = threadIdx.x; i < words; i += blockDim.x) {
        int from;
        if constexpr (ToPacked) {
            const int group=i/256, lane=(i%256)/8, part=i%8;
            const int row=lane/4+8*((part/2)%2);
            const int kword=8*(part/4)+4*(part%2)+lane%4;
            from=row*groups*16+group*16+kword;
        } else {
            const int row=i/(groups*16), group=(i%(groups*16))/16, kword=i%16;
            const int lane=4*(row%8)+kword%4;
            const int part=4*(kword/8)+2*(row/8)+(kword/4)%2;
            from=group*256+lane*8+part;
        }
        plane[i]=tmp[from];
    }
}

template<bool ToPacked>
__global__ void permute_high(std::uint8_t* data, int groups) {
    extern __shared__ std::uint8_t high_tmp[];
    const int bytes=16*groups*8;
    auto* plane=data+static_cast<std::int64_t>(blockIdx.x)*bytes;
    for(int i=threadIdx.x;i<bytes;i+=blockDim.x) high_tmp[i]=plane[i];
    __syncthreads();
    if constexpr(ToPacked) {
        auto* out=reinterpret_cast<std::uint32_t*>(plane);
        for(int i=threadIdx.x;i<groups*32;i+=blockDim.x) {
            const int group=i/32,lane=i%32,row=lane/4,l=lane%4;
            unsigned word=0;
#pragma unroll
            for(int half=0;half<2;++half) {
#pragma unroll
                for(int h=0;h<2;++h) {
                    const auto* source=high_tmp+(row+8*h)*groups*8+group*8+half*4;
                    const int shift=4*(l%2);
                    const unsigned lo=(source[l/2]>>shift)&15U;
                    const unsigned hi=(source[2+l/2]>>shift)&15U;
                    word|=(lo|(hi<<4))<<((half*2+h)*8);
                }
            }
            out[i]=word;
        }
    } else {
        const auto* packed=reinterpret_cast<const std::uint32_t*>(high_tmp);
        for(int i=threadIdx.x;i<bytes;i+=blockDim.x) {
            const int row=i/(groups*8),group=(i%(groups*8))/8,b=i%8;
            const int lane=4*(row%8)+2*(b%2);
            const int shift=((b/4)*2+row/8)*8+((b%4)/2)*4;
            plane[i]=static_cast<std::uint8_t>(((packed[group*32+lane]>>shift)&15U)|
                                             (((packed[group*32+lane+1]>>shift)&15U)<<4));
        }
    }
}

template<bool ToPacked>
__global__ void permute_scales(std::uint16_t* data,int groups) {
    extern __shared__ std::uint16_t tmp[];
    const int words=16*groups;
    auto* plane=data+static_cast<std::int64_t>(blockIdx.x)*words;
    for(int i=threadIdx.x;i<words;i+=blockDim.x)tmp[i]=plane[i];
    __syncthreads();
    for(int i=threadIdx.x;i<words;i+=blockDim.x) {
        const int from=ToPacked?(i%16)*groups+i/16:(i%groups)*16+i/groups;
        plane[i]=tmp[from];
    }
}

// No mutation of graph-owned resident weights: materialize one canonical view in
// caller-owned workspace, whose lifetime includes the consuming stream work.
__global__ void canonical_copy(const std::uint16_t* low,const std::uint32_t* high,
                               const std::uint16_t* scales,std::uint16_t* out_low,
                               std::uint8_t* out_high,std::uint16_t* out_scales,
                               int n,int groups) {
    const std::int64_t i=static_cast<std::int64_t>(blockIdx.x)*blockDim.x+threadIdx.x;
    if(i<static_cast<std::int64_t>(n)*groups*16) {
        const auto block=i/(16*groups*16);const int j=i%(16*groups*16);
        const int row=j/(groups*16),group=(j%(groups*16))/16,kword=j%16;
        const int lane=4*(row%8)+kword%4,part=4*(kword/8)+2*(row/8)+(kword/4)%2;
        out_low[i]=low[(block*groups+group)*256+lane*8+part];
    }
    if(high && i<static_cast<std::int64_t>(n)*groups*8) {
        const auto block=i/(16*groups*8);const int j=i%(16*groups*8);
        const int row=j/(groups*8),group=(j%(groups*8))/8,b=j%8;
        const int lane=4*(row%8)+2*(b%2),shift=((b/4)*2+row/8)*8+((b%4)/2)*4;
        const auto at=(block*groups+group)*32+lane;
        out_high[i]=static_cast<std::uint8_t>(((high[at]>>shift)&15U)|(((high[at+1]>>shift)&15U)<<4));
    }
    if(i<static_cast<std::int64_t>(n)*groups) {
        const auto block=i/(16*groups);const int j=i%(16*groups);
        out_scales[i]=scales[(block*groups+j%groups)*16+j/groups];
    }
}

// Only resident ragged dispatch uses these kernels. Existing full-width radix
// symbols and their arithmetic remain unchanged.
template<int Tokens>
__global__ void quantize_active(const __nv_bfloat16* x,signed char* q,float* scales,
                                int k,int valid_tokens) {
    const int group=(blockIdx.x*blockDim.x+threadIdx.x)/32,lane=threadIdx.x&31;
    const int groups=k/64;if(group>=Tokens*groups)return;
    const int t=group/groups,g=group%groups;
    const float a=t<valid_tokens?__bfloat162float(x[t*k+g*64+lane]):0.f;
    const float b=t<valid_tokens?__bfloat162float(x[t*k+g*64+lane+32]):0.f;
    float mx=fmaxf(fabsf(a),fabsf(b));
#pragma unroll
    for(int d=16;d;d>>=1)mx=fmaxf(mx,__shfl_xor_sync(0xffffffff,mx,d));
    int exponent=0;const float fraction=frexpf(mx,&exponent);
    const int shift=exponent-7+(fraction>127.f/128.f?1:0);
    const float scale=mx?ldexpf(1.f,shift):0.f,inv=mx?ldexpf(1.f,-shift):0.f;
    const int a0=max(-127,min(127,__float2int_rn(a*inv)));
    const int b0=max(-127,min(127,__float2int_rn(b*inv)));
    const int a1=max(-127,min(127,__float2int_rn(__fmaf_rn(-float(a0),scale,a)*inv*128.f)));
    const int b1=max(-127,min(127,__float2int_rn(__fmaf_rn(-float(b0),scale,b)*inv*128.f)));
    q[t*k+g*64+lane]=a0;q[t*k+g*64+lane+32]=b0;
    q[(t+Tokens)*k+g*64+lane]=a1;q[(t+Tokens)*k+g*64+lane+32]=b1;
    if(lane==0)scales[t*groups+g]=scale;
}

template<int GroupsOfFour,bool Residual>
__global__ void finish_active(const float* partial,const __nv_bfloat16* initial,
                             __nv_bfloat16* output,int n,int ld,int valid_tokens) {
    const int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i>=valid_tokens*n)return;
    const int block=i/(4*n),local=i%(4*n);
    float sum=0.f;
#pragma unroll
    for(int part=0;part<4;++part) {
        const auto offset=(static_cast<long long>(part)*GroupsOfFour+block)*8*n;
        sum+=partial[offset+local]+partial[offset+local+4*n];
    }
    const int destination=(i/n)*ld+i%n;
    if constexpr(Residual)sum+=__bfloat162float(initial[destination]);
    output[destination]=__float2bfloat16_rn(sum);
}
} // namespace ninfer::ops::detail::resident_radix
