#include "ops/linear_swiglu/q4/q4_linear_swiglu_a8_sm75.h"
#include "core/device.h"
#include "ops/common/mma.cuh"
#include "ops/kernel/rmsnorm.cuh"
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <algorithm>
#include <cstdint>
#include <stdexcept>
namespace ninfer::ops::detail::tp2_q4_fast {
namespace {
constexpr int kGateUpRows=17664,kM=8832,kK=5120,kGroupK=64,kGroups=80,kOutputGroups=kM/kGroupK;
constexpr int kWarps=16,kThreads=kWarps*32,kOutputRowsPerCta=kWarps*8;
constexpr int kRowsPerFragmentBlock=8,kLanes=32,kRowBytes=kGroups*kLanes;
constexpr int kTransformBlockBytes=kRowsPerFragmentBlock*kRowBytes;
constexpr int kRowsPerHalf=kM,kRowBlocksPerHalf=kRowsPerHalf/kRowsPerFragmentBlock;
constexpr std::size_t kHalfCodeBytes=(std::size_t)kRowsPerHalf*kRowBytes;
constexpr std::size_t kTotalCodeBytes=2*kHalfCodeBytes;
constexpr std::size_t kTotalScaleBytes=(std::size_t)2*kRowsPerHalf*kGroups*sizeof(std::uint16_t);
__device__ __forceinline__ std::uint32_t expand_q4_pair(std::uint16_t p){
 const std::uint32_t x=(std::uint32_t)p;
 const std::uint32_t n=__byte_perm(x&0x0f0fu,(x>>4)&0x0f0fu,0x5140);
 return n|((n&0x08080808u)*30u);
}
__launch_bounds__(512) __global__ void rmsnorm_offset_quantize_a8_t4(
 const __nv_bfloat162* x,const __nv_bfloat162* weight,std::int8_t* q,__half* scales,float eps,int d){
 constexpr int Block=512,Pairs=2560,Groups=80,PairsPerThread=5;
 int t=(int)blockIdx.x,tid=(int)threadIdx.x,lane=tid&31,warp=tid>>5;
 std::int64_t base=(std::int64_t)t*Pairs;
 __nv_bfloat162 values[PairsPerThread];float sum=0.f;
#pragma unroll
 for(int k=0;k<PairsPerThread;++k){int pair=tid+k*Block;values[k]=x[base+pair];float2 v=__bfloat1622float2(values[k]);sum+=v.x*v.x+v.y*v.y;}
 __shared__ float warp_sums[Block/32];__shared__ float inv_shared;
 float total=block_reduce_sum<Block>(sum,warp_sums);
 if(tid==0)inv_shared=rsqrtf(total/static_cast<float>(d)+eps);
 __syncthreads();float inv=inv_shared;
#pragma unroll
 for(int k=0;k<PairsPerThread;++k){
  int pair=tid+k*Block,group=warp+k*(Block/32);float2 xv=__bfloat1622float2(values[k]),wv=__bfloat1622float2(weight[pair]);
  __nv_bfloat162 rounded=__floats2bfloat162_rn(
      rmsnorm_epilogue<RmsEpilogue::Offset>(xv.x,inv,wv.x,0.f),
      rmsnorm_epilogue<RmsEpilogue::Offset>(xv.y,inv,wv.y,0.f));
  float2 y=__bfloat1622float2(rounded);
  float m=fmaxf(fabsf(y.x),fabsf(y.y));
#pragma unroll
  for(int off=16;off;off>>=1)m=fmaxf(m,__shfl_xor_sync(0xffffffffu,m,off));
  float s=m?m/127.f:0.f;if(lane==0)scales[t*Groups+group]=__float2half_rn(s);s=__shfl_sync(0xffffffffu,s,0);float iv=s?1.f/s:0.f;
  int a=__float2int_rn(y.x*iv),b=__float2int_rn(y.y*iv);a=max(-127,min(127,a));b=max(-127,min(127,b));
  std::int64_t d=(std::int64_t)t*5120+pair*2;q[d]=(std::int8_t)a;q[d+1]=(std::int8_t)b;
 }
}


template<int Tokens>
__global__ void quantize_a8(const __nv_bfloat16* x,std::int8_t* q,__half* scales){
 int warp=(int)threadIdx.x>>5,lane=(int)threadIdx.x&31,unit=(int)blockIdx.x*8+warp;
 if(unit>=Tokens*kGroups)return;
 int t=unit/kGroups,g=unit-t*kGroups,d0=g*kGroupK+lane,d1=d0+32;
 float x0=__bfloat162float(x[(std::int64_t)t*kK+d0]);
 float x1=__bfloat162float(x[(std::int64_t)t*kK+d1]);
 float m=fmaxf(fabsf(x0),fabsf(x1));
#pragma unroll
 for(int off=16;off;off>>=1)m=fmaxf(m,__shfl_xor_sync(0xffffffffu,m,off));
 float s=m>0?m/127.0f:0.0f,inv=s>0?1.0f/s:0.0f;
 int a=__float2int_rn(x0*inv),b=__float2int_rn(x1*inv);
 a=max(-127,min(127,a));b=max(-127,min(127,b));
 q[(std::int64_t)t*kK+d0]=(std::int8_t)a;q[(std::int64_t)t*kK+d1]=(std::int8_t)b;
 if(lane==0)scales[t*kGroups+g]=__float2half_rn(s);
}
template<int Tokens>
__global__ void q4_a8_swiglu(const std::uint8_t* frag,const std::uint8_t* scale_bytes,
                                const std::int8_t* aq,const __half* as,__nv_bfloat16* out){
 int warp=(int)threadIdx.x>>5,lane=(int)threadIdx.x&31;
 int rb=(int)blockIdx.x*kWarps+warp,rowbase=rb*8,gid=lane>>2,lid=lane&3;
 extern __shared__ __align__(16) std::int8_t s_aq[];
 const std::int8_t* act_q=aq;
 if constexpr(Tokens==4){
  for(int v=(int)threadIdx.x;v<(Tokens*kK)/16;v+=kThreads)reinterpret_cast<uint4*>(s_aq)[v]=reinterpret_cast<const uint4*>(aq)[v];
  __syncthreads();
  act_q=s_aq;
 }
 float g0=0,g1=0,u0=0,u1=0;
 const std::int64_t fi0=((std::int64_t)rb*kGroups)*32+lane;
 uint4 p=*reinterpret_cast<const uint4*>(frag+fi0*16);
#pragma unroll 1
 for(int group=0;group<kGroups;++group){
  uint4 p_next{};
  if(group+1<kGroups){
   const std::int64_t fin=((std::int64_t)rb*kGroups+group+1)*32+lane;
   p_next=*reinterpret_cast<const uint4*>(frag+fin*16);
  }
  unsigned a0=expand_q4_pair((std::uint16_t)p.x);
  unsigned a1=expand_q4_pair((std::uint16_t)p.z);
  unsigned a2=expand_q4_pair((std::uint16_t)(p.x>>16));
  unsigned a3=expand_q4_pair((std::uint16_t)(p.z>>16));
  int c0=0,c1=0,c2=0,c3=0;int tok=lane>>2,q=lane&3;unsigned b0=0,b1=0;
  if(tok<Tokens){const auto* ab=reinterpret_cast<const std::uint32_t*>(act_q+(std::int64_t)tok*kK+group*kGroupK);b0=ab[q];b1=ab[4+q];}
  mma_s8(c0,c1,c2,c3,a0,a1,a2,a3,b0,b1);
  a0=expand_q4_pair((std::uint16_t)p.y);a1=expand_q4_pair((std::uint16_t)p.w);
  a2=expand_q4_pair((std::uint16_t)(p.y>>16));a3=expand_q4_pair((std::uint16_t)(p.w>>16));
  if(tok<Tokens){const auto* ab=reinterpret_cast<const std::uint32_t*>(act_q+(std::int64_t)tok*kK+group*kGroupK+32);b0=ab[q];b1=ab[4+q];}else b0=b1=0;
  mma_s8(c0,c1,c2,c3,a0,a1,a2,a3,b0,b1);
  int ta=2*lid,tb=ta+1,row=rowbase+gid;
  const auto* ss=reinterpret_cast<const std::uint32_t*>(scale_bytes);
  std::uint32_t sw=0;
  if(lid==0){
   const std::int64_t si=((std::int64_t)rb*kGroups+group)*kRowsPerFragmentBlock+gid;
   sw=ss[si];
  }
  sw=__shfl_sync(0xffffffffu,sw,gid*4);
  float wg=__half2float(__ushort_as_half((std::uint16_t)sw));
  float wu=__half2float(__ushort_as_half((std::uint16_t)(sw>>16)));
  float sa=ta<Tokens?__half2float(as[ta*kGroups+group]):0.0f;
  float sb=tb<Tokens?__half2float(as[tb*kGroups+group]):0.0f;
  g0=__fmaf_rn((float)c0,wg*sa,g0);g1=__fmaf_rn((float)c1,wg*sb,g1);
  u0=__fmaf_rn((float)c2,wu*sa,u0);u1=__fmaf_rn((float)c3,wu*sb,u1);
  p=p_next;
 }
 int ta=2*lid,tb=ta+1,row=rowbase+gid;
 if(ta<Tokens)out[(std::int64_t)ta*kM+row]=__float2bfloat16_rn((g0/(1.0f+expf(-g0)))*u0);
 if(tb<Tokens)out[(std::int64_t)tb*kM+row]=__float2bfloat16_rn((g1/(1.0f+expf(-g1)))*u1);
}
__global__ void q4_a8_swiglu_t4_interleaved_chunk40(
    const std::uint8_t* frag,const std::uint8_t* scale_bytes,
    const std::int8_t* aq,const __half* as,__nv_bfloat16* out){
 constexpr int Tokens=4,ChunkGroups=40;
 int warp=(int)threadIdx.x>>5,lane=(int)threadIdx.x&31;
 int rb=(int)blockIdx.x*kWarps+warp,rowbase=rb*8,gid=lane>>2,lid=lane&3;
 extern __shared__ __align__(16) std::int8_t s_aq[];
 float g0=0,g1=0,u0=0,u1=0;
#pragma unroll 1
 for(int base=0;base<kGroups;base+=ChunkGroups){
  constexpr int count=ChunkGroups;
  constexpr int vectors_per_token=count*kGroupK/16;
  for(int v=(int)threadIdx.x;v<Tokens*vectors_per_token;v+=kThreads){
   int t=v/vectors_per_token,lv=v-t*vectors_per_token;
   reinterpret_cast<uint4*>(s_aq)[v]=reinterpret_cast<const uint4*>(
       aq+(std::int64_t)t*kK+base*kGroupK)[lv];
  }
  __syncthreads();
  const std::int64_t fi0=((std::int64_t)rb*kGroups+base)*32+lane;
  uint4 p=*reinterpret_cast<const uint4*>(frag+fi0*16);
#pragma unroll 1
  for(int group=base;group<base+count;++group){
   uint4 pn{};if(group+1<base+count){const std::int64_t fin=((std::int64_t)rb*kGroups+group+1)*32+lane;pn=*reinterpret_cast<const uint4*>(frag+fin*16);}
   unsigned a0=expand_q4_pair((std::uint16_t)p.x),a1=expand_q4_pair((std::uint16_t)p.z);
   unsigned a2=expand_q4_pair((std::uint16_t)(p.x>>16)),a3=expand_q4_pair((std::uint16_t)(p.z>>16));
   int c0=0,c1=0,c2=0,c3=0;int tok=lane>>2,q=lane&3;const std::uint32_t*ab=nullptr;unsigned b0=0,b1=0;
   if(tok<Tokens){ab=reinterpret_cast<const std::uint32_t*>(s_aq+((std::int64_t)tok*count+(group-base))*kGroupK);b0=ab[q];b1=ab[4+q];}
   mma_s8(c0,c1,c2,c3,a0,a1,a2,a3,b0,b1);
   a0=expand_q4_pair((std::uint16_t)p.y);a1=expand_q4_pair((std::uint16_t)p.w);
   a2=expand_q4_pair((std::uint16_t)(p.y>>16));a3=expand_q4_pair((std::uint16_t)(p.w>>16));
   if(tok<Tokens){b0=ab[8+q];b1=ab[12+q];}else b0=b1=0;
   mma_s8(c0,c1,c2,c3,a0,a1,a2,a3,b0,b1);
   int ta=2*lid,tb=ta+1,row=rowbase+gid;const auto*ss=reinterpret_cast<const std::uint32_t*>(scale_bytes);std::uint32_t sw=0;
   if(lid==0)sw=ss[((std::int64_t)rb*kGroups+group)*8+gid];sw=__shfl_sync(0xffffffffu,sw,gid*4);
   float wg=__half2float(__ushort_as_half((std::uint16_t)sw)),wu=__half2float(__ushort_as_half((std::uint16_t)(sw>>16)));
   float sa=ta<Tokens?__half2float(as[ta*kGroups+group]):0.f,sb=tb<Tokens?__half2float(as[tb*kGroups+group]):0.f;
   g0=__fmaf_rn((float)c0,wg*sa,g0);g1=__fmaf_rn((float)c1,wg*sb,g1);u0=__fmaf_rn((float)c2,wu*sa,u0);u1=__fmaf_rn((float)c3,wu*sb,u1);p=pn;
  }
  __syncthreads();
 }
 int ta=2*lid,tb=ta+1,row=rowbase+gid;
 if(ta<Tokens)out[(std::int64_t)ta*kM+row]=__float2bfloat16_rn((g0/(1.f+expf(-g0)))*u0);
 if(tb<Tokens)out[(std::int64_t)tb*kM+row]=__float2bfloat16_rn((g1/(1.f+expf(-g1)))*u1);
}

__global__ void q4_a8_swiglu_t4_interleaved_chunk40_to_a8(
    const std::uint8_t* frag,const std::uint8_t* scale_bytes,
    const std::int8_t* aq,const __half* as,std::int8_t* out_q,__half* out_scales){
 constexpr int Tokens=4,ChunkGroups=40;
 int warp=(int)threadIdx.x>>5,lane=(int)threadIdx.x&31;
 int rb=(int)blockIdx.x*kWarps+warp,rowbase=rb*8,gid=lane>>2,lid=lane&3;
 extern __shared__ __align__(16) std::int8_t s_aq[];
 float g0=0,g1=0,u0=0,u1=0;
#pragma unroll 1
 for(int base=0;base<kGroups;base+=ChunkGroups){
  constexpr int count=ChunkGroups;
  constexpr int vectors_per_token=count*kGroupK/16;
  for(int v=(int)threadIdx.x;v<Tokens*vectors_per_token;v+=kThreads){
   int t=v/vectors_per_token,lv=v-t*vectors_per_token;
   reinterpret_cast<uint4*>(s_aq)[v]=reinterpret_cast<const uint4*>(
       aq+(std::int64_t)t*kK+base*kGroupK)[lv];
  }
  __syncthreads();
  const std::int64_t fi0=((std::int64_t)rb*kGroups+base)*32+lane;
  uint4 p=*reinterpret_cast<const uint4*>(frag+fi0*16);
#pragma unroll 1
  for(int group=base;group<base+count;++group){
   uint4 pn{};if(group+1<base+count){const std::int64_t fin=((std::int64_t)rb*kGroups+group+1)*32+lane;pn=*reinterpret_cast<const uint4*>(frag+fin*16);}
   unsigned a0=expand_q4_pair((std::uint16_t)p.x),a1=expand_q4_pair((std::uint16_t)p.z);
   unsigned a2=expand_q4_pair((std::uint16_t)(p.x>>16)),a3=expand_q4_pair((std::uint16_t)(p.z>>16));
   int c0=0,c1=0,c2=0,c3=0;int tok=lane>>2,q=lane&3;const std::uint32_t*ab=nullptr;unsigned b0=0,b1=0;
   if(tok<Tokens){ab=reinterpret_cast<const std::uint32_t*>(s_aq+((std::int64_t)tok*count+(group-base))*kGroupK);b0=ab[q];b1=ab[4+q];}
   mma_s8(c0,c1,c2,c3,a0,a1,a2,a3,b0,b1);
   a0=expand_q4_pair((std::uint16_t)p.y);a1=expand_q4_pair((std::uint16_t)p.w);
   a2=expand_q4_pair((std::uint16_t)(p.y>>16));a3=expand_q4_pair((std::uint16_t)(p.w>>16));
   if(tok<Tokens){b0=ab[8+q];b1=ab[12+q];}else b0=b1=0;
   mma_s8(c0,c1,c2,c3,a0,a1,a2,a3,b0,b1);
   int ta=2*lid,tb=ta+1,row=rowbase+gid;const auto*ss=reinterpret_cast<const std::uint32_t*>(scale_bytes);std::uint32_t sw=0;
   if(lid==0)sw=ss[((std::int64_t)rb*kGroups+group)*8+gid];sw=__shfl_sync(0xffffffffu,sw,gid*4);
   float wg=__half2float(__ushort_as_half((std::uint16_t)sw)),wu=__half2float(__ushort_as_half((std::uint16_t)(sw>>16)));
   float sa=ta<Tokens?__half2float(as[ta*kGroups+group]):0.f,sb=tb<Tokens?__half2float(as[tb*kGroups+group]):0.f;
   g0=__fmaf_rn((float)c0,wg*sa,g0);g1=__fmaf_rn((float)c1,wg*sb,g1);u0=__fmaf_rn((float)c2,wu*sa,u0);u1=__fmaf_rn((float)c3,wu*sb,u1);p=pn;
  }
  __syncthreads();
 }
 int ta=2*lid,tb=ta+1,row_local=warp*8+gid;
 constexpr int OutputGroupsPerCta=kOutputRowsPerCta/kGroupK;
 const int output_group=warp/8;
 float v0=0.f,v1=0.f;
 if(ta<Tokens){__nv_bfloat16 b=__float2bfloat16_rn((g0/(1.f+expf(-g0)))*u0);v0=__bfloat162float(b);}
 if(tb<Tokens){__nv_bfloat16 b=__float2bfloat16_rn((g1/(1.f+expf(-g1)))*u1);v1=__bfloat162float(b);}
 float m0=fabsf(v0),m1=fabsf(v1);
#pragma unroll
 for(int off=4;off<=16;off<<=1){
  m0=fmaxf(m0,__shfl_xor_sync(0xffffffffu,m0,off));
  m1=fmaxf(m1,__shfl_xor_sync(0xffffffffu,m1,off));
 }
 auto* warp_max=reinterpret_cast<float*>(s_aq);
 if(gid==0){
  if(ta<Tokens)warp_max[warp*Tokens+ta]=m0;
  if(tb<Tokens)warp_max[warp*Tokens+tb]=m1;
 }
 __syncthreads();
 auto* qscale=warp_max+kWarps*Tokens;
 if(threadIdx.x<Tokens*OutputGroupsPerCta){
  int unit=(int)threadIdx.x,t=unit/OutputGroupsPerCta,og=unit-t*OutputGroupsPerCta;
  int first_warp=og*8;float m=warp_max[first_warp*Tokens+t];
#pragma unroll
  for(int w=1;w<8;++w)m=fmaxf(m,warp_max[(first_warp+w)*Tokens+t]);
  float s=m?m/127.f:0.f;qscale[t*OutputGroupsPerCta+og]=s;
  out_scales[(std::int64_t)t*kOutputGroups+(std::int64_t)blockIdx.x*OutputGroupsPerCta+og]=__float2half_rn(s);
 }
 __syncthreads();
 if(ta<Tokens){float s=qscale[ta*OutputGroupsPerCta+output_group],inv=s?1.f/s:0.f;int qv=__float2int_rn(v0*inv);qv=max(-127,min(127,qv));out_q[(std::int64_t)ta*kM+(std::int64_t)blockIdx.x*kOutputRowsPerCta+row_local]=(std::int8_t)qv;}
 if(tb<Tokens){float s=qscale[tb*OutputGroupsPerCta+output_group],inv=s?1.f/s:0.f;int qv=__float2int_rn(v1*inv);qv=max(-127,min(127,qv));out_q[(std::int64_t)tb*kM+(std::int64_t)blockIdx.x*kOutputRowsPerCta+row_local]=(std::int8_t)qv;}
}

__global__ void transform_to_interleaved(const std::uint8_t* src,std::uint8_t* dst,
                                         const std::uint16_t* scales,std::uint32_t* scale_dst){
 int rb=(int)blockIdx.x;
 for(int item=(int)threadIdx.x;item<kGroups*kLanes;item+=blockDim.x){
  int group=item/kLanes,lane=item-group*kLanes,row=lane>>2,q=lane&3;
  const auto* gate=src+((std::int64_t)(rb*8+row)*kGroups+group)*32;
  const auto* up=src+kHalfCodeBytes+((std::int64_t)(rb*8+row)*kGroups+group)*32;
  auto* out=reinterpret_cast<std::uint32_t*>(dst+((std::int64_t)rb*kGroups*32+item)*16);
#pragma unroll
  for(int half=0;half<2;++half){
   int bo=half*16+q*2;
   out[half]=((std::uint32_t)*reinterpret_cast<const std::uint16_t*>(gate+bo))|
             ((std::uint32_t)*reinterpret_cast<const std::uint16_t*>(gate+bo+8)<<16);
   out[half+2]=((std::uint32_t)*reinterpret_cast<const std::uint16_t*>(up+bo))|
               ((std::uint32_t)*reinterpret_cast<const std::uint16_t*>(up+bo+8)<<16);
  }
 }
 for(int item=(int)threadIdx.x;item<kGroups*8;item+=blockDim.x){
  int group=item/8,row=item-group*8,global_row=rb*8+row;
  std::uint16_t gate=scales[(std::int64_t)global_row*kGroups+group];
  std::uint16_t up=scales[(std::int64_t)(global_row+kM)*kGroups+group];
  scale_dst[(std::int64_t)rb*kGroups*8+item]=(std::uint32_t)gate|((std::uint32_t)up<<16);
 }
}

__global__ void transform_from_interleaved(const std::uint8_t* src,std::uint8_t* dst,
                                           const std::uint32_t* scales,std::uint16_t* scale_dst){
 int rb=(int)blockIdx.x;
 for(int item=(int)threadIdx.x;item<kGroups*kLanes;item+=blockDim.x){
  int group=item/kLanes,lane=item-group*kLanes,row=lane>>2,q=lane&3;
  const auto* in=reinterpret_cast<const std::uint32_t*>(src+((std::int64_t)rb*kGroups*32+item)*16);
  auto* gate=dst+((std::int64_t)(rb*8+row)*kGroups+group)*32;
  auto* up=dst+kHalfCodeBytes+((std::int64_t)(rb*8+row)*kGroups+group)*32;
#pragma unroll
  for(int half=0;half<2;++half){
   int bo=half*16+q*2;
   std::uint32_t gv=in[half],uv=in[half+2];
   *reinterpret_cast<std::uint16_t*>(gate+bo)=(std::uint16_t)gv;
   *reinterpret_cast<std::uint16_t*>(gate+bo+8)=(std::uint16_t)(gv>>16);
   *reinterpret_cast<std::uint16_t*>(up+bo)=(std::uint16_t)uv;
   *reinterpret_cast<std::uint16_t*>(up+bo+8)=(std::uint16_t)(uv>>16);
  }
 }
 for(int item=(int)threadIdx.x;item<kGroups*8;item+=blockDim.x){
  int group=item/8,row=item-group*8,global_row=rb*8+row;
  std::uint32_t pair=scales[(std::int64_t)rb*kGroups*8+item];
  scale_dst[(std::int64_t)global_row*kGroups+group]=(std::uint16_t)pair;
  scale_dst[(std::int64_t)(global_row+kM)*kGroups+group]=(std::uint16_t)(pair>>16);
 }
}

}
extern "C" std::size_t tp2_fast_q4_code_bytes(){return kTotalCodeBytes;}
extern "C" std::size_t tp2_fast_q4_scale_bytes(){return kTotalScaleBytes;}
extern "C" std::size_t tp2_fast_q4_scratch_bytes(){return kTotalCodeBytes+kTotalScaleBytes;}
extern "C" int tp2_fast_q4_rows(){return kM;}
extern "C" int tp2_fast_q4_output_groups(){return kOutputGroups;}
extern "C" void tp2_fast_q4_transform(std::uint8_t*c,std::uint8_t*s,void*x,cudaStream_t st){auto*sc=(std::uint8_t*)x;auto*ss=sc+kTotalCodeBytes;transform_to_interleaved<<<kRowBlocksPerHalf,256,0,st>>>(c,sc,(const std::uint16_t*)s,(std::uint32_t*)ss);CUDA_CHECK(cudaGetLastError());CUDA_CHECK(cudaMemcpyAsync(c,sc,kTotalCodeBytes,cudaMemcpyDeviceToDevice,st));CUDA_CHECK(cudaMemcpyAsync(s,ss,kTotalScaleBytes,cudaMemcpyDeviceToDevice,st));}
extern "C" void tp2_fast_q4_run(const std::uint8_t*c,const std::uint8_t*s,const std::int8_t*iq,const __half*is,std::int8_t*oq,__half*os,cudaStream_t st){constexpr std::size_t sh=(std::size_t)4*40*kGroupK;q4_a8_swiglu_t4_interleaved_chunk40_to_a8<<<kM/kOutputRowsPerCta,kThreads,sh,st>>>(c,s,iq,is,oq,os);CUDA_CHECK(cudaGetLastError());}
}
