#include "ops/linear_swiglu/q4/q4_linear_swiglu_b2_sm75.h"
#include "core/device.h"
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <mma.h>
#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <stdexcept>

namespace ninfer::ops::detail {
namespace {
using b1 = nvcuda::wmma::experimental::precision::b1;
constexpr int kM=17408,kK=5120,kExactGroups=80,kB2GroupK=128,kB2Groups=40;
constexpr int kRowsPerBlock=8,kRowBlocks=kM/kRowsPerBlock,kWarps=16,kThreads=512;
constexpr int kRowsPerCta=kWarps*kRowsPerBlock,kOutputGroups=kM/64;
constexpr int kFirstSelectedLayer=28,kSelectedLayers=8;
constexpr std::size_t kPackedBytes=(std::size_t)kRowBlocks*kB2Groups*32*sizeof(uint4);
constexpr std::size_t kScaleBytes=(std::size_t)kRowBlocks*kB2Groups*8*sizeof(std::uint32_t);
constexpr std::size_t kActivationBytes=(std::size_t)kB2Groups*32*sizeof(uint2);
constexpr std::size_t kActivationScaleBytes=(std::size_t)4*kB2Groups*sizeof(__half);
struct Entry{const void*source=nullptr;uint4*packed=nullptr;std::uint32_t*scales=nullptr;};
struct State{Entry entries[kSelectedLayers];int count=0;uint2*activation=nullptr;__half*activation_scales=nullptr;};
State& state(){static State s;return s;}
Entry* find_entry(const void*source) noexcept{auto&s=state();for(int i=0;i<s.count;++i)if(s.entries[i].source==source)return&s.entries[i];return nullptr;}
__device__ __forceinline__ int q4_nibble(std::uint32_t word,int index){int n=(int)((word>>(4*index))&15u);return(n^8)-8;}
__device__ __forceinline__ int q4_value(uint4 a,uint4 b,bool up,int index){int seg=index>>3,within=index&7;std::uint32_t word=0;if(!up){word=seg==0?a.x:seg==1?a.y:seg==2?b.x:b.y;}else{word=seg==0?a.z:seg==1?a.w:seg==2?b.z:b.w;}return q4_nibble(word,within);}
__device__ __forceinline__ void encode_b2(uint4 first,uint4 second,bool up,float scale0,float scale1,std::uint32_t&high,std::uint32_t&low,float&alpha){
 float sum=0.f;
#pragma unroll
 for(int i=0;i<32;++i){float sc=i<16?scale0:scale1;sum+=fabsf((float)q4_value(first,second,up,i)*sc);}
 sum+=__shfl_xor_sync(0xffffffffu,sum,1,4);sum+=__shfl_xor_sync(0xffffffffu,sum,2,4);alpha=sum*(1.f/(128.f*1.5f));
#pragma unroll
 for(int it=0;it<2;++it){float num=0.f,den=0.f;
#pragma unroll
  for(int i=0;i<32;++i){float sc=i<16?scale0:scale1,mag=fabsf((float)q4_value(first,second,up,i)*sc),level=mag>2.f*alpha?3.f:1.f;num+=mag*level;den+=level*level;}
  num+=__shfl_xor_sync(0xffffffffu,num,1,4);num+=__shfl_xor_sync(0xffffffffu,num,2,4);den+=__shfl_xor_sync(0xffffffffu,den,1,4);den+=__shfl_xor_sync(0xffffffffu,den,2,4);alpha=den>0.f?num/den:0.f;}
 high=0;low=0;
#pragma unroll
 for(int i=0;i<32;++i){float sc=i<16?scale0:scale1,v=(float)q4_value(first,second,up,i)*sc;bool neg=v<0.f,strong=fabsf(v)>2.f*alpha;high|=(std::uint32_t)neg<<i;low|=(std::uint32_t)(strong?neg:!neg)<<i;}
}

__device__ __forceinline__ void encode_b2_canonical(const std::uint8_t* input,float scale,std::uint32_t&high,std::uint32_t&low,float&alpha){
 float sum=0.f;
#pragma unroll
 for(int i=0;i<32;++i){int n=(i&1)?input[i>>1]>>4:input[i>>1]&15;n=(n^8)-8;sum+=fabsf((float)n*scale);}sum+=__shfl_xor_sync(0xffffffffu,sum,1,4);sum+=__shfl_xor_sync(0xffffffffu,sum,2,4);alpha=sum*(1.f/(128.f*1.5f));
#pragma unroll
 for(int it=0;it<2;++it){float num=0.f,den=0.f;
#pragma unroll
  for(int i=0;i<32;++i){int n=(i&1)?input[i>>1]>>4:input[i>>1]&15;n=(n^8)-8;float mag=fabsf((float)n*scale),level=mag>2.f*alpha?3.f:1.f;num+=mag*level;den+=level*level;}num+=__shfl_xor_sync(0xffffffffu,num,1,4);num+=__shfl_xor_sync(0xffffffffu,num,2,4);den+=__shfl_xor_sync(0xffffffffu,den,1,4);den+=__shfl_xor_sync(0xffffffffu,den,2,4);alpha=den>0.f?num/den:0.f;}
 high=0;low=0;
#pragma unroll
 for(int i=0;i<32;++i){int n=(i&1)?input[i>>1]>>4:input[i>>1]&15;n=(n^8)-8;float v=(float)n*scale;bool neg=v<0.f,strong=fabsf(v)>2.f*alpha;high|=(std::uint32_t)neg<<i;low|=(std::uint32_t)(strong?neg:!neg)<<i;}
}
__device__ __forceinline__ void bmma_xor(int&c0,int&c1,unsigned a,unsigned b){asm volatile("mma.sync.aligned.m8n8k128.row.col.s32.b1.b1.s32.xor.popc {%0,%1}, {%2}, {%3}, {%0,%1};\n":"+r"(c0),"+r"(c1):"r"(a),"r"(b));}

__global__ void transform_b2(const std::uint8_t*codes,const std::uint16_t*source_scales,uint4*out,std::uint32_t*out_scales){
 __shared__ __align__(16) std::uint32_t gh[32],gl[32],uh[32],ul[32],ghf[32],glf[32],uhf[32],ulf[32];__shared__ std::uint16_t gs[8],us[8];int lane=(int)threadIdx.x&31,row=lane>>2,lid=lane&3,rb=(int)blockIdx.x;constexpr int RowBytes=kExactGroups*32;
 for(int g=0;g<kB2Groups;++g){int gr=rb*8+row,sg=g*2+(lid>>1);const auto*gate=codes+(std::int64_t)gr*RowBytes+g*64+lid*16;const auto*up=codes+(std::int64_t)(gr+kM)*RowBytes+g*64+lid*16;float gscale=__half2float(__ushort_as_half(source_scales[(std::int64_t)gr*kExactGroups+sg])),uscale=__half2float(__ushort_as_half(source_scales[(std::int64_t)(gr+kM)*kExactGroups+sg]));std::uint32_t a,b,c,d;float ag,au;encode_b2_canonical(gate,gscale,a,b,ag);encode_b2_canonical(up,uscale,c,d,au);gh[lane]=a;gl[lane]=b;uh[lane]=c;ul[lane]=d;if(lid==0){gs[row]=__half_as_ushort(__float2half_rn(ag));us[row]=__half_as_ushort(__float2half_rn(au));}__syncwarp();nvcuda::wmma::fragment<nvcuda::wmma::matrix_a,8,8,128,b1,nvcuda::wmma::row_major>f0,f1,f2,f3;nvcuda::wmma::load_matrix_sync(f0,gh,128);nvcuda::wmma::load_matrix_sync(f1,gl,128);nvcuda::wmma::load_matrix_sync(f2,uh,128);nvcuda::wmma::load_matrix_sync(f3,ul,128);ghf[lane]=f0.x[0];glf[lane]=f1.x[0];uhf[lane]=f2.x[0];ulf[lane]=f3.x[0];__syncwarp();out[((std::int64_t)rb*kB2Groups+g)*32+lane]=make_uint4(ghf[lane],glf[lane],uhf[lane],ulf[lane]);if(lane<8)out_scales[((std::int64_t)rb*kB2Groups+g)*8+lane]=(std::uint32_t)gs[lane]|((std::uint32_t)us[lane]<<16);__syncwarp();}
}
__global__ void prepare_b2(const __nv_bfloat16*hidden,uint2*out,__half*scales){
 __shared__ __align__(16) std::uint32_t hi[8][32],lo[8][32];int warp=(int)threadIdx.x>>5,lane=(int)threadIdx.x&31,g=(int)blockIdx.x*8+warp;if(g>=kB2Groups)return;hi[warp][lane]=0;lo[warp][lane]=0;
 if(lane<16){int t=lane>>2,w=lane&3;const auto*p=hidden+(std::int64_t)t*kK+g*kB2GroupK+w*32;float sum=0.f;
#pragma unroll
  for(int i=0;i<32;++i)sum+=fabsf(__bfloat162float(p[i]));sum+=__shfl_xor_sync(0x0000ffffu,sum,1,4);sum+=__shfl_xor_sync(0x0000ffffu,sum,2,4);float alpha=sum*(1.f/(128.f*1.5f));
#pragma unroll
  for(int it=0;it<2;++it){float num=0.f,den=0.f;
#pragma unroll
   for(int i=0;i<32;++i){float mag=fabsf(__bfloat162float(p[i])),level=mag>2.f*alpha?3.f:1.f;num+=mag*level;den+=level*level;}num+=__shfl_xor_sync(0x0000ffffu,num,1,4);num+=__shfl_xor_sync(0x0000ffffu,num,2,4);den+=__shfl_xor_sync(0x0000ffffu,den,1,4);den+=__shfl_xor_sync(0x0000ffffu,den,2,4);alpha=den>0.f?num/den:0.f;}
  std::uint32_t h=0,l=0;
#pragma unroll
  for(int i=0;i<32;++i){float v=__bfloat162float(p[i]);bool neg=v<0.f,strong=fabsf(v)>2.f*alpha;h|=(std::uint32_t)neg<<i;l|=(std::uint32_t)(strong?neg:!neg)<<i;}hi[warp][t*4+w]=h;lo[warp][t*4+w]=l;if(w==0)scales[t*kB2Groups+g]=__float2half_rn(alpha);}
 __syncwarp();nvcuda::wmma::fragment<nvcuda::wmma::matrix_b,8,8,128,b1,nvcuda::wmma::col_major>fh,fl;nvcuda::wmma::load_matrix_sync(fh,hi[warp],128);nvcuda::wmma::load_matrix_sync(fl,lo[warp],128);out[g*32+lane]=make_uint2(fh.x[0],fl.x[0]);
}

__launch_bounds__(kThreads) __global__ void swiglu_b2_to_a8(const uint4*weights,const std::uint32_t*weight_scales,const uint2*activation,const __half*activation_scales,std::int8_t*output_q,__half*output_scales){
 __shared__ float warp_max[kWarps][4];__shared__ float qscale[4][2];int warp=(int)threadIdx.x>>5,lane=(int)threadIdx.x&31,rb=(int)blockIdx.x*kWarps+warp,rowbase=rb*8,gid=lane>>2,lid=lane&3;float fg0=0.f,fg1=0.f,fu0=0.f,fu1=0.f;uint4 p=weights[((std::int64_t)rb*kB2Groups)*32+lane];
#pragma unroll 1
 for(int g=0;g<kB2Groups;++g){uint4 pn{};if(g+1<kB2Groups)pn=weights[((std::int64_t)rb*kB2Groups+g+1)*32+lane];uint2 act=activation[g*32+lane];int g0=0,g1=0,u0=0,u1=0,a=0,b=0;
  bmma_xor(a,b,p.x,act.x);g0+=4*(128-2*a);g1+=4*(128-2*b);a=b=0;bmma_xor(a,b,p.x,act.y);g0+=2*(128-2*a);g1+=2*(128-2*b);a=b=0;bmma_xor(a,b,p.y,act.x);g0+=2*(128-2*a);g1+=2*(128-2*b);a=b=0;bmma_xor(a,b,p.y,act.y);g0+=128-2*a;g1+=128-2*b;
  a=b=0;bmma_xor(a,b,p.z,act.x);u0+=4*(128-2*a);u1+=4*(128-2*b);a=b=0;bmma_xor(a,b,p.z,act.y);u0+=2*(128-2*a);u1+=2*(128-2*b);a=b=0;bmma_xor(a,b,p.w,act.x);u0+=2*(128-2*a);u1+=2*(128-2*b);a=b=0;bmma_xor(a,b,p.w,act.y);u0+=128-2*a;u1+=128-2*b;
  std::uint32_t sp=0;if(lid==0)sp=weight_scales[((std::int64_t)rb*kB2Groups+g)*8+gid];sp=__shfl_sync(0xffffffffu,sp,gid*4);float gs=__half2float(__ushort_as_half((std::uint16_t)sp)),us=__half2float(__ushort_as_half((std::uint16_t)(sp>>16)));int t0=2*lid,t1=t0+1;float as0=t0<4?__half2float(activation_scales[t0*kB2Groups+g]):0.f,as1=t1<4?__half2float(activation_scales[t1*kB2Groups+g]):0.f;fg0=__fmaf_rn((float)g0,gs*as0,fg0);fg1=__fmaf_rn((float)g1,gs*as1,fg1);fu0=__fmaf_rn((float)u0,us*as0,fu0);fu1=__fmaf_rn((float)u1,us*as1,fu1);p=pn;}
 int t0=2*lid,t1=t0+1;float v0=0.f,v1=0.f;if(t0<4){__nv_bfloat16 b=__float2bfloat16_rn((fg0/(1.f+expf(-fg0)))*fu0);v0=__bfloat162float(b);}if(t1<4){__nv_bfloat16 b=__float2bfloat16_rn((fg1/(1.f+expf(-fg1)))*fu1);v1=__bfloat162float(b);}float m0=fabsf(v0),m1=fabsf(v1);
#pragma unroll
 for(int off=4;off<=16;off<<=1){m0=fmaxf(m0,__shfl_xor_sync(0xffffffffu,m0,off));m1=fmaxf(m1,__shfl_xor_sync(0xffffffffu,m1,off));}if(gid==0){if(t0<4)warp_max[warp][t0]=m0;if(t1<4)warp_max[warp][t1]=m1;}__syncthreads();
 if(threadIdx.x<8){int t=(int)threadIdx.x>>1,og=(int)threadIdx.x&1,first=og*8;float m=warp_max[first][t];
#pragma unroll
  for(int w=1;w<8;++w)m=fmaxf(m,warp_max[first+w][t]);float s=m>0.f?m/127.f:0.f;qscale[t][og]=s;output_scales[(std::int64_t)t*kOutputGroups+(std::int64_t)blockIdx.x*2+og]=__float2half_rn(s);}__syncthreads();
 int local=warp*8+gid,og=warp>>3;std::int64_t row=(std::int64_t)blockIdx.x*kRowsPerCta+local;if(t0<4){float s=qscale[t0][og],inv=s>0.f?1.f/s:0.f;int q=__float2int_rn(v0*inv);q=max(-127,min(127,q));output_q[(std::int64_t)t0*kM+row]=(std::int8_t)q;}if(t1<4){float s=qscale[t1][og],inv=s>0.f?1.f/s:0.f;int q=__float2int_rn(v1*inv);q=max(-127,min(127,q));output_q[(std::int64_t)t1*kM+row]=(std::int8_t)q;}
}
}

bool q4_linear_swiglu_b2_sm75_select_layer(int layer) noexcept{
 static const int first=[](){const char*e=std::getenv("NINFER_SM75_MLP_B2_FIRST");return e?std::atoi(e):0;}();
 static const int count=[](){const char*e=std::getenv("NINFER_SM75_MLP_B2_COUNT");int v=e?std::atoi(e):0;return v<0?0:(v>kSelectedLayers?kSelectedLayers:v);}();
 return layer>=first&&layer<first+count;
}
std::size_t q4_linear_swiglu_b2_sm75_reservation_bytes() noexcept{return(std::size_t)kSelectedLayers*(kPackedBytes+kScaleBytes)+kActivationBytes+kActivationScaleBytes;}
void q4_linear_swiglu_b2_sm75_prepare(const Weight&w,cudaStream_t stream){
 if(w.qtype!=QType::Q4G64_F16S||w.n!=2*kM||w.k!=kK||!w.qdata||!w.scales)throw std::invalid_argument("SM75 B2 SwiGLU preparation requires exact Q4 geometry");if(find_entry(w.qdata))return;auto&s=state();if(s.count>=kSelectedLayers)throw std::runtime_error("SM75 B2 SwiGLU entry capacity exceeded");if(!s.activation){CUDA_CHECK(cudaMalloc(&s.activation,kActivationBytes));CUDA_CHECK(cudaMalloc(&s.activation_scales,kActivationScaleBytes));}Entry&e=s.entries[s.count++];e.source=w.qdata;CUDA_CHECK(cudaMalloc(&e.packed,kPackedBytes));CUDA_CHECK(cudaMalloc(&e.scales,kScaleBytes));transform_b2<<<kRowBlocks,32,0,stream>>>(static_cast<const std::uint8_t*>(w.qdata),static_cast<const std::uint16_t*>(w.scales),e.packed,e.scales);CUDA_CHECK(cudaGetLastError());CUDA_CHECK(cudaStreamSynchronize(stream));
}
bool q4_linear_swiglu_b2_sm75_ready(const Weight&w) noexcept{return find_entry(w.qdata)!=nullptr;}
void q4_linear_swiglu_b2_sm75_launch(const Tensor&x,const Weight&w,Tensor&oq,Tensor&os,cudaStream_t stream){Entry*e=find_entry(w.qdata);if(!e||x.dtype!=DType::BF16||x.ne[0]!=kK||x.ne[1]!=4||oq.dtype!=DType::I8||oq.ne[0]!=kM||oq.ne[1]!=4||os.dtype!=DType::FP16||os.ne[0]!=kOutputGroups||os.ne[1]!=4)throw std::invalid_argument("SM75 B2 SwiGLU launch requires prepared T4 geometry");auto&s=state();prepare_b2<<<(kB2Groups+7)/8,256,0,stream>>>(static_cast<const __nv_bfloat16*>(x.data),s.activation,s.activation_scales);swiglu_b2_to_a8<<<kM/kRowsPerCta,kThreads,0,stream>>>(e->packed,e->scales,s.activation,s.activation_scales,static_cast<std::int8_t*>(oq.data),static_cast<__half*>(os.data));CUDA_CHECK(cudaGetLastError());}
} // namespace ninfer::ops::detail
