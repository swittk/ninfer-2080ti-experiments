#include "ops/linear_add/w8/w8_mtp_down_w4a8_sm75.h"
#include "core/device.h"
#include "ops/common/mma.cuh"
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <stdexcept>
namespace ninfer::ops::detail { namespace {
constexpr int N=5120,K=17408,GK=32,G=K/GK,ROWS=16,RB=N/ROWS,WARPS=8,THREADS=256,RPC=WARPS*ROWS;
constexpr std::size_t PACKED=(std::size_t)RB*G*32*sizeof(uint2),SCALE=(std::size_t)RB*G*8*sizeof(std::uint32_t),AQ=(std::size_t)4*K,AS=(std::size_t)4*G*sizeof(__half);
struct State{const void*source=nullptr;uint2*packed=nullptr;std::uint32_t*scales=nullptr;std::int8_t*aq=nullptr;__half*as=nullptr;};State&state(){static State s;return s;}
int mode(){static int v=[](){const char*e=std::getenv("NINFER_SM75_MTP_DOWN_W4_MODE");int x=e?std::atoi(e):0;return x<0?0:(x>3?3:x);}();return v;}
__device__ __forceinline__ std::uint32_t expand4(std::uint16_t p){std::uint32_t x=p,n=__byte_perm(x&0x0f0fu,(x>>4)&0x0f0fu,0x5140);return n|((n&0x08080808u)*30u);}
__global__ void transform(const std::int8_t*codes,const std::uint16_t*src_sc,uint2*out,std::uint32_t*out_sc){
 __shared__ __align__(16) std::uint8_t canon[2][8][16];__shared__ std::uint16_t qscale[2][8];__shared__ std::uint32_t frag[2][32];int warp=(int)threadIdx.x>>5,lane=(int)threadIdx.x&31,gid=lane>>2,lid=lane&3,rb=(int)blockIdx.x,row=rb*ROWS+warp*8+gid;
 for(int g=0;g<G;++g){const auto*p=codes+(std::int64_t)row*K+g*GK+lid*8;int m=0;
#pragma unroll
  for(int i=0;i<8;++i)m=max(m,abs((int)p[i]));m=max(m,__shfl_xor_sync(__activemask(),m,1,4));m=max(m,__shfl_xor_sync(__activemask(),m,2,4));float a=m?(float)m/7.f:0.f;
#pragma unroll
  for(int it=0;it<2;++it){float num=0.f,den=0.f;
#pragma unroll
   for(int i=0;i<8;++i){int q=a?__float2int_rn((float)p[i]/a):0;q=max(-7,min(7,q));num+=(float)p[i]*(float)q;den+=(float)(q*q);}num+=__shfl_xor_sync(__activemask(),num,1,4);num+=__shfl_xor_sync(__activemask(),num,2,4);den+=__shfl_xor_sync(__activemask(),den,1,4);den+=__shfl_xor_sync(__activemask(),den,2,4);a=den>0.f?num/den:0.f;}
  if(lid==0){float ss=__half2float(__ushort_as_half(src_sc[(std::int64_t)row*G+g]));qscale[warp][gid]=__half_as_ushort(__float2half_rn(ss*a));}
#pragma unroll
  for(int i=0;i<4;++i){int q0=a?__float2int_rn((float)p[2*i]/a):0,q1=a?__float2int_rn((float)p[2*i+1]/a):0;q0=max(-7,min(7,q0));q1=max(-7,min(7,q1));canon[warp][gid][lid*4+i]=(std::uint8_t)((q0&15)|((q1&15)<<4));}
  __syncwarp();frag[warp][lane]=(std::uint32_t)*reinterpret_cast<std::uint16_t*>(&canon[warp][gid][lid*2])|((std::uint32_t)*reinterpret_cast<std::uint16_t*>(&canon[warp][gid][lid*2+8])<<16);__syncthreads();if(warp==0){out[((std::int64_t)rb*G+g)*32+lane]=make_uint2(frag[0][lane],frag[1][lane]);if(lane<8)out_sc[((std::int64_t)rb*G+g)*8+lane]=(std::uint32_t)qscale[0][lane]|((std::uint32_t)qscale[1][lane]<<16);}__syncthreads();}
}
template<int T> __global__ void quant(const __nv_bfloat16*x,std::int8_t*q,__half*s){int warp=(int)threadIdx.x>>5,lane=(int)threadIdx.x&31,u=(int)blockIdx.x*8+warp;if(u>=T*G)return;int t=u/G,g=u-t*G,d=g*GK+lane;float v=__bfloat162float(x[(std::int64_t)t*K+d]),m=fabsf(v);for(int o=16;o;o>>=1)m=fmaxf(m,__shfl_xor_sync(__activemask(),m,o));float sc=m?m/127.f:0.f,iv=sc?1.f/sc:0.f;int a=__float2int_rn(v*iv);a=max(-127,min(127,a));q[(std::int64_t)t*K+d]=(std::int8_t)a;if(lane==0)s[t*G+g]=__float2half_rn(sc);}
template<int T> __launch_bounds__(THREADS) __global__ void add(const uint2*w,const std::uint32_t*ws,const std::int8_t*aq,const __half*as,__nv_bfloat16*res){int warp=(int)threadIdx.x>>5,lane=(int)threadIdx.x&31,rb=(int)blockIdx.x*WARPS+warp,rowbase=rb*ROWS,gid=lane>>2,lid=lane&3;float f0=0.f,f1=0.f,f2=0.f,f3=0.f;uint2 p=w[((std::int64_t)rb*G)*32+lane];
#pragma unroll 1
 for(int g=0;g<G;++g){uint2 pn{};if(g+1<G)pn=w[((std::int64_t)rb*G+g+1)*32+lane];int c0=0,c1=0,c2=0,c3=0,t=lane>>2,q=lane&3;unsigned b0=0,b1=0;if(t<T){const auto*ab=reinterpret_cast<const std::uint32_t*>(aq+(std::int64_t)t*K+g*GK);b0=ab[q];b1=ab[4+q];}mma_s8(c0,c1,c2,c3,expand4((std::uint16_t)p.x),expand4((std::uint16_t)p.y),expand4((std::uint16_t)(p.x>>16)),expand4((std::uint16_t)(p.y>>16)),b0,b1);std::uint32_t sp=0;if(lid==0)sp=ws[((std::int64_t)rb*G+g)*8+gid];sp=__shfl_sync(__activemask(),sp,gid*4);float w0=__half2float(__ushort_as_half((std::uint16_t)sp)),w1=__half2float(__ushort_as_half((std::uint16_t)(sp>>16)));int t0=2*lid,t1=t0+1;float s0=t0<T?__half2float(as[t0*G+g]):0.f,s1=t1<T?__half2float(as[t1*G+g]):0.f;f0=__fmaf_rn((float)c0,w0*s0,f0);f1=__fmaf_rn((float)c1,w0*s1,f1);f2=__fmaf_rn((float)c2,w1*s0,f2);f3=__fmaf_rn((float)c3,w1*s1,f3);p=pn;}
 int t0=2*lid,t1=t0+1,r0=rowbase+gid,r1=r0+8;if(t0<T){auto i0=(std::int64_t)t0*N+r0,i1=(std::int64_t)t0*N+r1;res[i0]=__float2bfloat16_rn(__bfloat162float(res[i0])+f0);res[i1]=__float2bfloat16_rn(__bfloat162float(res[i1])+f2);}if(t1<T){auto i0=(std::int64_t)t1*N+r0,i1=(std::int64_t)t1*N+r1;res[i0]=__float2bfloat16_rn(__bfloat162float(res[i0])+f1);res[i1]=__float2bfloat16_rn(__bfloat162float(res[i1])+f3);}}
template<int T> void launch(const Tensor&x,Tensor&res,cudaStream_t stream){auto&s=state();quant<T><<<(T*G+7)/8,256,0,stream>>>((const __nv_bfloat16*)x.data,s.aq,s.as);add<T><<<N/RPC,THREADS,0,stream>>>(s.packed,s.scales,s.aq,s.as,(__nv_bfloat16*)res.data);CUDA_CHECK(cudaGetLastError());}
}
std::size_t w8_mtp_down_w4a8_sm75_reservation_bytes() noexcept{return PACKED+SCALE+AQ+AS;}
void w8_mtp_down_w4a8_sm75_prepare(const Weight&w,cudaStream_t stream){if(w.qtype!=QType::W8G32_F16S||w.n!=N||w.k!=K||!w.qdata||!w.scales)throw std::invalid_argument("SM75 MTP W4/A8 down preparation requires exact W8 geometry");if(mode()==0)return;auto&s=state();if(s.source==w.qdata)return;CUDA_CHECK(cudaMalloc(&s.packed,PACKED));CUDA_CHECK(cudaMalloc(&s.scales,SCALE));CUDA_CHECK(cudaMalloc(&s.aq,AQ));CUDA_CHECK(cudaMalloc(&s.as,AS));transform<<<RB,64,0,stream>>>((const std::int8_t*)w.qdata,(const std::uint16_t*)w.scales,s.packed,s.scales);CUDA_CHECK(cudaGetLastError());CUDA_CHECK(cudaStreamSynchronize(stream));s.source=w.qdata;}
bool w8_mtp_down_w4a8_sm75_use(const Weight&w,int tokens) noexcept{int bit=tokens==1?1:(tokens==4?2:0);return bit&&(mode()&bit)&&state().source==w.qdata&&state().packed;}
void w8_mtp_down_w4a8_sm75_launch(const Tensor&x,const Weight&w,Tensor&res,cudaStream_t stream){if(!w8_mtp_down_w4a8_sm75_use(w,x.ne[1])||x.dtype!=DType::BF16||x.ne[0]!=K||res.dtype!=DType::BF16||res.ne[0]!=N||res.ne[1]!=x.ne[1])throw std::invalid_argument("SM75 MTP W4/A8 down launch requires prepared T1/T4 geometry");if(x.ne[1]==1)launch<1>(x,res,stream);else launch<4>(x,res,stream);}
} // namespace ninfer::ops::detail
