#pragma once
#include "core/device.h"
#include "core/tensor.h"
#include "ops/common/mma.cuh"
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cstdint>
namespace ninfer::ops::detail::radix_projection {
__device__ __forceinline__ unsigned expand(unsigned p,unsigned high){
 const unsigned n=__byte_perm(p&0x0f0fu,(p>>4)&0x0f0fu,0x5140);
 const unsigned sign=((high&1u)<<4)|((high&2u)<<11)|((high&4u)<<18)|((high&8u)<<25);
 return n|(sign*15u);
}
__device__ __forceinline__ unsigned expand_q4(unsigned p){
 const unsigned n=__byte_perm(p&0x0f0fu,(p>>4)&0x0f0fu,0x5140);
 return n|((n&0x08080808u)*30u);
}
template<bool Q5>
__device__ __forceinline__ unsigned expand_typed(unsigned p,unsigned high){
 if constexpr(Q5)return expand(p,high);else return expand_q4(p);
}
template<bool Q5>
__global__ void repack(const unsigned char* src,const unsigned char* hs,const __half* ss,
                       uint4* dst,unsigned* dh,__half* ds,int n,int k){
 int lane=threadIdx.x&31;int unit=(blockIdx.x*blockDim.x+threadIdx.x)/32;
 int groups=k/64;if(unit>=n/16*groups)return;
 int rb=unit/groups,g=unit%groups,r=rb*16+(lane>>2),l=lane&3;
 // Four 16-value segments, two rows, mapping required by m16n8k32.
 uint4 p{};unsigned high=0;
#pragma unroll
 for(int h=0;h<2;++h){
#pragma unroll
  for(int half=0;half<2;++half){
   unsigned pack=0,hp=0;
#pragma unroll
   for(int q=0;q<8;++q){int kk=g*64+half*32+(q>=4?16:0)+4*l+(q&3);int rr=r+8*h;
    unsigned v=(src[(static_cast<long long>(rr)*k+kk)/2]>>((kk&1)*4))&15;
    unsigned sign=Q5?((hs[(static_cast<long long>(rr)*k+kk)/8]>>(kk&7))&1):(v>>3);
    pack|=v<<(4*q);hp|=sign<<q;
   }
   int ix=half*2+h;reinterpret_cast<unsigned*>(&p)[ix]=pack;high|=hp<<(ix*8);
  }
 }
 dst[unit*32+lane]=p;if constexpr(Q5)dh[unit*32+lane]=high;
 if(l==0){ds[unit*16+(lane>>2)]=ss[(r)*groups+g];ds[unit*16+8+(lane>>2)]=ss[(r+8)*groups+g];}
}
__global__ void quantize(const __nv_bfloat16* x,signed char* q,float* scales,int k){
 int group=(blockIdx.x*blockDim.x+threadIdx.x)/32,lane=threadIdx.x&31;
 int groups=k/64;if(group>=4*groups)return;
 int t=group/groups,g=group%groups;
 float a=__bfloat162float(x[t*k+g*64+lane]),b=__bfloat162float(x[t*k+g*64+lane+32]);
 float mx=fmaxf(fabsf(a),fabsf(b));
#pragma unroll
 for(int d=16;d;d>>=1)mx=fmaxf(mx,__shfl_xor_sync(0xffffffff,mx,d));
 // Power-of-two radix preserves BF16 mantissa bits rather than rounding
 // every value to an arbitrary scale. High and low limbs share exact exponents.
 int exponent=0;float fraction=frexpf(mx,&exponent);
 int shift=exponent-7+(fraction>127.f/128.f?1:0);
 float s=mx?ldexpf(1.f,shift):0.f,inv=mx?ldexpf(1.f,-shift):0.f;
 int a0=max(-127,min(127,__float2int_rn(a*inv))),b0=max(-127,min(127,__float2int_rn(b*inv)));
 int a1=max(-127,min(127,__float2int_rn(__fmaf_rn(-float(a0),s,a)*inv*128.f)));
 int b1=max(-127,min(127,__float2int_rn(__fmaf_rn(-float(b0),s,b)*inv*128.f)));
 q[t*k+g*64+lane]=a0;q[t*k+g*64+lane+32]=b0;
 q[(t+4)*k+g*64+lane]=a1;q[(t+4)*k+g*64+lane+32]=b1;
 if(lane==0)scales[t*groups+g]=s;
}
template<int Parts,bool Q5=true>
__global__ void matmul(const uint4* w,const unsigned* hs,const __half* scales,
                       const signed char* q,const float* as,float* tmp,int n,int k){
 int lane=threadIdx.x&31,warp=threadIdx.x>>5,rb=blockIdx.x*4+warp;
 if(rb>=n/16)return;
 int gid=lane>>2,l=lane&3,groups=k/64,part=blockIdx.y;
 int first=groups*part/Parts,last=groups*(part+1)/Parts;
 float f0=0,f1=0,f2=0,f3=0;
 for(int g=first;g<last;++g){
  auto at=(static_cast<long long>(rb)*groups+g)*32+lane;uint4 p=w[at];unsigned h=0;if constexpr(Q5)h=hs[at];
  const auto* a=reinterpret_cast<const unsigned*>(q+gid*k+g*64);
  int c0=0,c1=0,c2=0,c3=0;
  ops::mma_s8(c0,c1,c2,c3,expand_typed<Q5>(p.x&65535u,h),expand_typed<Q5>(p.y&65535u,h>>8),
      expand_typed<Q5>(p.x>>16,h>>4),expand_typed<Q5>(p.y>>16,h>>12),a[l],a[4+l]);
  ops::mma_s8(c0,c1,c2,c3,expand_typed<Q5>(p.z&65535u,h>>16),expand_typed<Q5>(p.w&65535u,h>>24),
      expand_typed<Q5>(p.z>>16,h>>20),expand_typed<Q5>(p.w>>16,h>>28),a[8+l],a[12+l]);
  float sw0=__half2float(scales[(rb*groups+g)*16+gid]);float sw1=__half2float(scales[(rb*groups+g)*16+8+gid]);
  int ta=l*2,tb=ta+1;
  float sa=as[(ta&3)*groups+g]*(ta>=4?1.f/128:1.f),sb=as[(tb&3)*groups+g]*(tb>=4?1.f/128:1.f);
  f0=__fmaf_rn(float(c0),sw0*sa,f0);f1=__fmaf_rn(float(c1),sw0*sb,f1);
  f2=__fmaf_rn(float(c2),sw1*sa,f2);f3=__fmaf_rn(float(c3),sw1*sb,f3);
 }
 int ta=l*2,tb=ta+1,r=rb*16+gid;auto offset=static_cast<long long>(part)*8*n;
 tmp[offset+ta*n+r]=f0;tmp[offset+tb*n+r]=f1;
 tmp[offset+ta*n+r+8]=f2;tmp[offset+tb*n+r+8]=f3;
}
template<int Parts,bool Residual>
__global__ void finish(const float* tmp,const __nv_bfloat16* initial,__nv_bfloat16* output,int n,int ld){
 int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=4*n)return;
 float sum=0;
#pragma unroll
 for(int p=0;p<Parts;++p){auto o=static_cast<long long>(p)*8*n;sum+=tmp[o+i]+tmp[o+i+4*n];}
 int dest=(i/n)*ld+i%n;
 if constexpr(Residual)sum+=__bfloat162float(initial[dest]);output[dest]=__float2bfloat16_rn(sum);
}
}
