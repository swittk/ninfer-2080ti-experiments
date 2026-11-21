#include "ops/linear/nu3_mlp_sm75.h"
#include "core/device.h"
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <math_constants.h>
#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <fstream>
#include <vector>
#include <string>
#include <filesystem>
namespace ninfer::ops::detail { namespace {
constexpr int kM=17408,kK=5120,kGroups=40,kGroupK=64,kWarps=16,kThreads=512,kOutputRowsPerCta=128,kOutputGroups=kM/64;
constexpr int DownN=5120,DownK=17408,DownGroups=136;
__device__ __forceinline__ int nu3_level(unsigned index){
 return static_cast<std::int8_t>(__byte_perm(0x00fefcf9u,0x07040301u,index));
}
__device__ __forceinline__ unsigned unpack3(unsigned low,unsigned high){
 low=(low|(low<<4))&0x0f0fu;low=(low|(low<<2))&0x3333u;
 high=(high|(high<<6))&0x0303u;high=(high|(high<<3))&0x1111u;
 return __byte_perm(0x00fefcf9u,0x07040301u,low|(high<<2));
}
__device__ __forceinline__ void mma16(int& x,int& y,unsigned a,unsigned b){
 asm volatile("mma.sync.aligned.m8n8k16.row.col.s32.s8.s8.s32 {%0,%1}, {%2}, {%3}, {%0,%1};\n"
              : "+r"(x),"+r"(y):"r"(a),"r"(b));
}
template<int Tokens>
__global__ void quantize_a8(const __nv_bfloat16* x,std::int8_t* q,__half* scales){
 int warp=(int)threadIdx.x>>5,lane=(int)threadIdx.x&31,unit=(int)blockIdx.x*8+warp;
 if(unit>=Tokens*(kK/64))return;
 int t=unit/(kK/64),g=unit-t*(kK/64),d0=g*kGroupK+lane,d1=d0+32;
 float x0=__bfloat162float(x[(std::int64_t)t*kK+d0]);
 float x1=__bfloat162float(x[(std::int64_t)t*kK+d1]);
 float m=fmaxf(fabsf(x0),fabsf(x1));
#pragma unroll
 for(int off=16;off;off>>=1)m=fmaxf(m,__shfl_xor_sync(0xffffffffu,m,off));
 float s=m>0?m/127.0f:0.0f,inv=s>0?1.0f/s:0.0f;
 int a=__float2int_rn(x0*inv),b=__float2int_rn(x1*inv);
 a=max(-127,min(127,a));b=max(-127,min(127,b));
 q[(std::int64_t)t*kK+d0]=(std::int8_t)a;q[(std::int64_t)t*kK+d1]=(std::int8_t)b;
 if(lane==0)scales[t*(kK/64)+g]=__float2half_rn(s);
}
__global__ void gate_w3g128(const uint4* low,const uint2* high,const unsigned* scales,
 const std::int8_t* aq,const __half* as,std::int8_t* out_q,__half* out_scales,float* raw){
 constexpr int Tokens=4,ChunkGroups=20;
 const int warp=threadIdx.x>>5,lane=threadIdx.x&31,gid=lane>>2,lid=lane&3;
 const int rb=blockIdx.x*kWarps+warp;
 extern __shared__ __align__(16) std::int8_t s_aq[];
 float g0=0,g1=0,u0=0,u1=0;
 for(int first=0;first<kGroups;first+=ChunkGroups){
  for(int v=threadIdx.x;v<4*ChunkGroups*8;v+=kThreads){
   const int t=v/(ChunkGroups*8),local=v%(ChunkGroups*8);
   reinterpret_cast<uint4*>(s_aq)[v]=reinterpret_cast<const uint4*>(aq+t*kK+first*128)[local];
  }
  __syncthreads();
  auto i=(static_cast<long long>(rb)*kGroups+first)*32+lane;uint4 p=low[i];uint2 h=high[i];
  for(int group=first;group<first+ChunkGroups;++group){
   uint4 pn{};uint2 hn{};
   if(group+1<first+ChunkGroups){const auto n=(static_cast<long long>(rb)*kGroups+group+1)*32+lane;pn=low[n];hn=high[n];}
   unsigned s=0;if(lid==0)s=scales[(static_cast<long long>(rb)*kGroups+group)*8+gid];s=__shfl_sync(0xffffffffu,s,gid*4);
   const float wg=__half2float(__ushort_as_half(s&65535)),wu=__half2float(__ushort_as_half(s>>16));
   const int ta=lid*2,tb=ta+1;
#pragma unroll
   for(int half=0;half<2;++half){
    int c0=0,c1=0,c2=0,c3=0;
#pragma unroll
    for(int j=0;j<4;++j){
     const int step=half*4+j;
     const unsigned gl=((step<4?p.x:p.y)>>(8*(step%4)))&255;
     const unsigned ul=((step<4?p.z:p.w)>>(8*(step%4)))&255;
     const unsigned wg0=unpack3(gl,(h.x>>(step*4))&15),wu0=unpack3(ul,(h.y>>(step*4))&15);
     const unsigned a=gid<4?reinterpret_cast<const unsigned*>(s_aq+(gid*ChunkGroups+group-first)*128)[step*4+lid]:0;
     mma16(c0,c1,wg0,a);mma16(c2,c3,wu0,a);
    }
    const float sa=ta<4?__half2float(as[ta*(kK/64)+group*2+half]):0,sb=tb<4?__half2float(as[tb*(kK/64)+group*2+half]):0;
    g0=__fmaf_rn(float(c0),wg*sa,g0);g1=__fmaf_rn(float(c1),wg*sb,g1);
    u0=__fmaf_rn(float(c2),wu*sa,u0);u1=__fmaf_rn(float(c3),wu*sb,u1);
   }
   p=pn;h=hn;
  }
  __syncthreads();
 }
 int ta=2*lid,tb=ta+1,row_local=warp*8+gid;
 constexpr int OutputGroupsPerCta=kOutputRowsPerCta/kGroupK;
 const int output_group=warp/8;
 float v0=0.f,v1=0.f;
 if(ta<Tokens){__nv_bfloat16 b=__float2bfloat16_rn((g0/(1.f+expf(-g0)))*u0);v0=__bfloat162float(b);}
 if(tb<Tokens){__nv_bfloat16 b=__float2bfloat16_rn((g1/(1.f+expf(-g1)))*u1);v1=__bfloat162float(b);}
 if(raw){if(ta<4)raw[ta*kM+blockIdx.x*128+row_local]=v0;if(tb<4)raw[tb*kM+blockIdx.x*128+row_local]=v1;}
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
__launch_bounds__(128,4) __global__ void down_w3g128(const uint4* low,const uint2* high,const __half* scales,
 const std::int8_t* aq,const __half* as,__nv_bfloat16* residual){
 const int part=threadIdx.x>>5,lane=threadIdx.x&31,gid=lane>>2,lid=lane&3,rb=blockIdx.x;
 const int ta=2*lid,tb=ta+1,half=part&1;
 const auto* low_words=reinterpret_cast<const unsigned*>(low);
 const auto* high_words=reinterpret_cast<const std::uint16_t*>(high);
 float f0=0,f1=0,f2=0,f3=0;
 auto first=(static_cast<long long>(rb)*DownGroups+part/2)*32+lane;
 uint2 l=make_uint2(low_words[first*4+half],low_words[first*4+2+half]);
 unsigned h=unsigned(high_words[first*4+half])|(unsigned(high_words[first*4+2+half])<<16);
 for(int group64=part;group64<DownK/64;group64+=4){
  uint2 next{};unsigned next_h=0;
  if(group64+4<DownK/64){
   const auto ix=(static_cast<long long>(rb)*DownGroups+(group64+4)/2)*32+lane;
   next=make_uint2(low_words[ix*4+half],low_words[ix*4+2+half]);
   next_h=unsigned(high_words[ix*4+half])|(unsigned(high_words[ix*4+2+half])<<16);
  }
  int c0=0,c1=0,c2=0,c3=0;
#pragma unroll
  for(int j=0;j<4;++j){
   const unsigned a0=unpack3((l.x>>(8*j))&255,(h>>(4*j))&15);
   const unsigned a1=unpack3((l.y>>(8*j))&255,(h>>(16+4*j))&15);
   const unsigned x=gid<4?reinterpret_cast<const unsigned*>(aq+gid*DownK+group64*64)[j*4+lid]:0;
   mma16(c0,c1,a0,x);mma16(c2,c3,a1,x);
  }
  float w0=0,w1=0;
  if(lid==0){const auto at=(static_cast<long long>(rb)*DownGroups+group64/2)*16+gid;w0=__half2float(scales[at]);w1=__half2float(scales[at+8]);}
  w0=__shfl_sync(0xffffffffu,w0,gid*4);w1=__shfl_sync(0xffffffffu,w1,gid*4);
  const float sa=ta<4?__half2float(as[ta*(DownK/64)+group64]):0,sb=tb<4?__half2float(as[tb*(DownK/64)+group64]):0;
  f0=__fmaf_rn(float(c0),w0*sa,f0);f1=__fmaf_rn(float(c1),w0*sb,f1);
  f2=__fmaf_rn(float(c2),w1*sa,f2);f3=__fmaf_rn(float(c3),w1*sb,f3);
  l=next;h=next_h;
 }
 __shared__ float partial[4][4][16];
 if(ta<4){partial[part][ta][gid]=f0;partial[part][ta][gid+8]=f2;}
 if(tb<4){partial[part][tb][gid]=f1;partial[part][tb][gid+8]=f3;}
 __syncthreads();
 if(part==0){
  if(ta<4){const float a=((partial[0][ta][gid]+partial[1][ta][gid])+partial[2][ta][gid])+partial[3][ta][gid];const float b=((partial[0][ta][gid+8]+partial[1][ta][gid+8])+partial[2][ta][gid+8])+partial[3][ta][gid+8];const int row=rb*16+gid;residual[ta*DownN+row]=__float2bfloat16_rn(__bfloat162float(residual[ta*DownN+row])+a);residual[ta*DownN+row+8]=__float2bfloat16_rn(__bfloat162float(residual[ta*DownN+row+8])+b);}
  if(tb<4){const float a=((partial[0][tb][gid]+partial[1][tb][gid])+partial[2][tb][gid])+partial[3][tb][gid];const float b=((partial[0][tb][gid+8]+partial[1][tb][gid+8])+partial[2][tb][gid+8])+partial[3][tb][gid+8];const int row=rb*16+gid;residual[tb*DownN+row]=__float2bfloat16_rn(__bfloat162float(residual[tb*DownN+row])+a);residual[tb*DownN+row+8]=__float2bfloat16_rn(__bfloat162float(residual[tb*DownN+row+8])+b);}
 }
}
__device__ __forceinline__ int nearest_nu3(float value){
 const int book[8]={-7,-4,-2,0,1,3,4,7};
 int best=book[0];float error=fabsf(value-best);
#pragma unroll
 for(int i=1;i<8;++i){float e=fabsf(value-book[i]);if(e<error){error=e;best=book[i];}}
 return best;
}
template<bool Q5,bool Nonuniform=false>
__global__ void recode(std::uint8_t* data,std::uint8_t* high,__half* scales,int groups){
 const int lane=threadIdx.x&31,group=(blockIdx.x*blockDim.x+threadIdx.x)/32;if(group>=groups)return;
 const auto at=static_cast<long long>(group)*64;
 const unsigned a=data[at+lane],b=data[at+32+lane];
 unsigned ha=0,hb=0;
 if constexpr(Q5){ha=high[static_cast<long long>(group)*16+lane/4];hb=high[static_cast<long long>(group)*16+8+lane/4];}
 int qi[4]={int(a&15),int(a>>4),int(b&15),int(b>>4)};
 if constexpr(Q5){qi[0]-=((ha>>((lane%4)*2))&1)*16;qi[1]-=((ha>>((lane%4)*2+1))&1)*16;qi[2]-=((hb>>((lane%4)*2))&1)*16;qi[3]-=((hb>>((lane%4)*2+1))&1)*16;}
 else{for(int j=0;j<4;++j)if(qi[j]>=8)qi[j]-=16;}
 const float s0=__half2float(scales[group*2]),s1=__half2float(scales[group*2+1]);
 float value[4]={qi[0]*s0,qi[1]*s0,qi[2]*s1,qi[3]*s1};float bound=0;
 for(int j=0;j<4;++j)bound=fmaxf(bound,Nonuniform ? fabsf(value[j])/7.f : fmaxf(value[j]/3.f,-value[j]/4.f));
 for(int off=16;off;off>>=1)bound=fmaxf(bound,__shfl_xor_sync(0xffffffffu,bound,off));
 float best=0,error_best=CUDART_INF_F;
 // A fixed per-group weight reconstruction objective, with no activation data,
 // model fitting, learned parameters, or checkpoint-specific constants.
 for(int candidate=0;candidate<9;++candidate){
  const float scale=__half2float(__float2half_rn(bound*(1.f-.05f*candidate)));
  const float inv=scale?1.f/scale:0.f;float error=0;
  for(int j=0;j<4;++j){const int q=Nonuniform ? nearest_nu3(value[j]*inv) : max(-4,min(3,__float2int_rn(value[j]*inv)));const float delta=value[j]-q*scale;error=__fmaf_rn(delta,delta,error);}
  for(int off=16;off;off>>=1)error+=__shfl_xor_sync(0xffffffffu,error,off);
  if(error<error_best){error_best=error;best=scale;}
 }
 const float inv=best?1.f/best:0.f;
 for(int j=0;j<4;++j)qi[j]=Nonuniform ? nearest_nu3(value[j]*inv) : max(-4,min(3,__float2int_rn(value[j]*inv)));
 unsigned sign0=0,sign1=0;
 if constexpr(Q5){
  unsigned sa=(qi[0]<0?1u:0u)|(qi[1]<0?2u:0u),sb=(qi[2]<0?1u:0u)|(qi[3]<0?2u:0u);
  for(int j=0;j<4;++j){sign0|=__shfl_sync(0xffffffffu,sa,(lane/4)*4+j)<<(2*j);sign1|=__shfl_sync(0xffffffffu,sb,(lane/4)*4+j)<<(2*j);}
 }
 __syncwarp();
 data[at+lane]=(qi[0]&15)|((qi[1]&15)<<4);data[at+32+lane]=(qi[2]&15)|((qi[3]&15)<<4);
 if constexpr(Q5){if(lane%4==0){high[static_cast<long long>(group)*16+lane/4]=sign0;high[static_cast<long long>(group)*16+8+lane/4]=sign1;}}
 if(lane==0){scales[group*2]=__float2half_rn(best);scales[group*2+1]=__float2half_rn(best);}
}
namespace nu3_layout {
__device__ __forceinline__ unsigned index_of(int value){
 switch(value){case -7:return 0;case -4:return 1;case -2:return 2;case 0:return 3;
 case 1:return 4;case 3:return 5;case 4:return 6;case 7:return 7;default:return 255;}
}
__device__ __forceinline__ int value_of(unsigned index){
 return static_cast<std::int8_t>(__byte_perm(0x00fefcf9u,0x07040301u,index));
}
template<bool Gate>
__device__ __forceinline__ int canonical(const std::uint8_t* data,const std::uint8_t* high,std::uint64_t at){
 int value=(data[at/2]>>(4*(at%2)))&15;
 if constexpr(Gate){if(value>=8)value-=16;}
 else value-=((high[at/8]>>(at%8))&1)*16;
 return value;
}
template<bool Gate>
__global__ void pack_codes(const std::uint8_t* source,const std::uint8_t* source_high,
 uint4* low,uint2* high,int k,std::size_t cells,unsigned* errors){
 const auto at=static_cast<std::size_t>(blockIdx.x)*blockDim.x+threadIdx.x;if(at>=cells)return;
 const int groups=k/128,lane=at%32,gid=lane/4,lid=lane%4,g=(at/32)%groups,rb=at/(32*groups);
 const int r0=rb*(Gate?8:16)+gid,r1=Gate?r0+17408:r0+8;
 uint4 l{};uint2 h{};
 for(int step=0;step<8;++step){unsigned c0=0,c1=0,sign0=0,sign1=0;
  for(int j=0;j<4;++j){
   const int column=g*128+step*16+lid*4+j;
   unsigned a=index_of(canonical<Gate>(source,source_high,static_cast<std::uint64_t>(r0)*k+column));
   unsigned b=index_of(canonical<Gate>(source,source_high,static_cast<std::uint64_t>(r1)*k+column));
   if(a==255||b==255)atomicAdd(errors,1u);
   c0|=(a&3)<<(2*j);c1|=(b&3)<<(2*j);sign0|=((a>>2)&1)<<j;sign1|=((b>>2)&1)<<j;
  }
  if(step<4){l.x|=c0<<(step*8);l.z|=c1<<(step*8);}else{l.y|=c0<<((step-4)*8);l.w|=c1<<((step-4)*8);}
  h.x|=sign0<<(step*4);h.y|=sign1<<(step*4);
 }
 low[at]=l;high[at]=h;
}
template<bool Gate>
__global__ void pack_scales(const std::uint16_t* source,std::uint16_t* out,int n,int k){
 const auto i=static_cast<std::size_t>(blockIdx.x)*blockDim.x+threadIdx.x;
 const int groups=k/128;if(i>=static_cast<std::size_t>(n)*groups)return;
 if constexpr(Gate){
  const int which=i%2,row=(i/2)%8,g=(i/16)%groups,rb=i/(16*groups);
  const int source_row=rb*8+row+which*17408;
  out[i]=source[(static_cast<std::size_t>(source_row)*groups+g)*2];
 }else{
  const int row=i%16,g=(i/16)%groups,rb=i/(16*groups);
  out[i]=source[(static_cast<std::size_t>(rb*16+row)*groups+g)*2];
 }
}
template<bool Gate>
__device__ __forceinline__ int read_packed(const uint4* low,const uint2* high,int row,int column,int k){
 const int r=Gate?row%17408:row,rb=r/(Gate?8:16),gid=r%8;
 const int group=column/128,step=(column%128)/16,lid=(column%16)/4,j=column%4;
 const auto at=(static_cast<std::size_t>(rb)*(k/128)+group)*32+gid*4+lid;
 const uint4 p=low[at];const uint2 h=high[at];const bool second=Gate?row>=17408:r%16>=8;
 const unsigned bytes=second?(step<4?p.z:p.w):(step<4?p.x:p.y),bits=second?h.y:h.x;
 return value_of(((bytes>>(8*(step%4)+j*2))&3)|(((bits>>(step*4+j))&1)<<2));
}
template<bool Gate>
__global__ void unpack_codes(const uint4* low,const uint2* high,std::uint32_t* out,std::uint8_t* signs,int n,int k){
 const auto i=static_cast<std::size_t>(blockIdx.x)*blockDim.x+threadIdx.x;if(i>=static_cast<std::size_t>(n)*k/8)return;
 const int row=i/(k/8),col=(i%(k/8))*8;unsigned packed=0,bits=0;
 for(int j=0;j<8;++j){const int value=read_packed<Gate>(low,high,row,col+j,k);packed|=(value&15)<<(4*j);bits|=(value<0?1u:0u)<<j;}
 out[i]=packed;if constexpr(!Gate)signs[i]=bits;
}
template<bool Gate>
__global__ void unpack_scales(const std::uint16_t* source,std::uint16_t* out,int n,int k){
 const auto i=static_cast<std::size_t>(blockIdx.x)*blockDim.x+threadIdx.x;const int groups=k/128;
 if(i>=static_cast<std::size_t>(n)*groups*2)return;
 const int row=i/(groups*2),g=(i%(groups*2))/2;
 if constexpr(Gate){const int r=row%17408;out[i]=source[((static_cast<std::size_t>(r/8)*groups+g)*8+r%8)*2+(row>=17408)];}
 else out[i]=source[(static_cast<std::size_t>(row/16)*groups+g)*16+row%16];
}
inline std::size_t cells(const ninfer::Weight& w){return static_cast<std::size_t>(w.n/16)*(w.k/128)*32;}
inline std::size_t low_bytes(const ninfer::Weight& w){return cells(w)*16;}
inline void transform(ninfer::Weight& w,void* scratch,std::size_t capacity,bool packed,cudaStream_t stream){
 const bool gate=w.n==34816&&w.k==5120&&w.qtype==ninfer::QType::Q4G64_F16S;
 const bool down=w.n==5120&&w.k==17408&&w.qtype==ninfer::QType::Q5G64_F16S;
 if(!gate&&!down)throw std::invalid_argument("unsupported NU3 source geometry");
 const auto count=cells(w),compressed=count*24,data_bytes=static_cast<std::size_t>(w.n)*w.k/2,sign_bytes=down?data_bytes/4:0;
 const auto scale_bytes=static_cast<std::size_t>(w.n)*w.k/64*2;
 if(capacity<data_bytes+sign_bytes||capacity<compressed+4||capacity<scale_bytes)throw std::invalid_argument("NU3 transform scratch too small");
 auto* data=const_cast<std::uint8_t*>(static_cast<const std::uint8_t*>(w.qdata));
 auto* signs=const_cast<std::uint8_t*>(static_cast<const std::uint8_t*>(w.qhigh));
 auto* scales=const_cast<std::uint16_t*>(static_cast<const std::uint16_t*>(w.scales));
 if(packed){
  auto* error=reinterpret_cast<unsigned*>(static_cast<std::uint8_t*>(scratch)+compressed);
  CUDA_CHECK(cudaMemsetAsync(error,0,4,stream));
  if(gate)pack_codes<true><<<(count+255)/256,256,0,stream>>>(data,signs,static_cast<uint4*>(scratch),reinterpret_cast<uint2*>(static_cast<std::uint8_t*>(scratch)+count*16),w.k,count,error);
  else pack_codes<false><<<(count+255)/256,256,0,stream>>>(data,signs,static_cast<uint4*>(scratch),reinterpret_cast<uint2*>(static_cast<std::uint8_t*>(scratch)+count*16),w.k,count,error);
  CUDA_CHECK(cudaGetLastError());unsigned bad=0;CUDA_CHECK(cudaMemcpyAsync(&bad,error,4,cudaMemcpyDeviceToHost,stream));CUDA_CHECK(cudaStreamSynchronize(stream));
  if(bad)throw std::invalid_argument("NU3 packing requires valid converted source codes");
  CUDA_CHECK(cudaMemcpyAsync(data,scratch,compressed,cudaMemcpyDeviceToDevice,stream));
  const auto units=static_cast<std::size_t>(w.n)*(w.k/128);
  if(gate)pack_scales<true><<<(units+255)/256,256,0,stream>>>(scales,static_cast<std::uint16_t*>(scratch),w.n,w.k);
  else pack_scales<false><<<(units+255)/256,256,0,stream>>>(scales,static_cast<std::uint16_t*>(scratch),w.n,w.k);
  CUDA_CHECK(cudaGetLastError());CUDA_CHECK(cudaMemcpyAsync(scales,scratch,scale_bytes/2,cudaMemcpyDeviceToDevice,stream));
 }else{
  const auto units=static_cast<std::size_t>(w.n)*w.k/8;
  const auto* low=reinterpret_cast<const uint4*>(data);const auto* high=reinterpret_cast<const uint2*>(data+count*16);
  if(gate)unpack_codes<true><<<(units+255)/256,256,0,stream>>>(low,high,static_cast<std::uint32_t*>(scratch),static_cast<std::uint8_t*>(scratch)+data_bytes,w.n,w.k);
  else unpack_codes<false><<<(units+255)/256,256,0,stream>>>(low,high,static_cast<std::uint32_t*>(scratch),static_cast<std::uint8_t*>(scratch)+data_bytes,w.n,w.k);
  CUDA_CHECK(cudaGetLastError());CUDA_CHECK(cudaMemcpyAsync(data,scratch,data_bytes,cudaMemcpyDeviceToDevice,stream));
  if(down)CUDA_CHECK(cudaMemcpyAsync(signs,static_cast<std::uint8_t*>(scratch)+data_bytes,sign_bytes,cudaMemcpyDeviceToDevice,stream));
  if(gate)unpack_scales<true><<<(scale_bytes/2+255)/256,256,0,stream>>>(scales,static_cast<std::uint16_t*>(scratch),w.n,w.k);
  else unpack_scales<false><<<(scale_bytes/2+255)/256,256,0,stream>>>(scales,static_cast<std::uint16_t*>(scratch),w.n,w.k);
  CUDA_CHECK(cudaGetLastError());CUDA_CHECK(cudaMemcpyAsync(scales,scratch,scale_bytes,cudaMemcpyDeviceToDevice,stream));
 }
}
} // namespace nu3_layout


} // anonymous namespace
namespace {
template<bool Q5>
__global__ void recode_factory_bf16(const __nv_bfloat16* source,std::uint8_t* data,
 std::uint8_t* high,__half* scales,std::uint64_t base_group,std::uint32_t groups,unsigned* invalid){
 const int lane=threadIdx.x&31;
 const auto local=(blockIdx.x*blockDim.x+threadIdx.x)/32;
 if(local>=groups)return;
 const auto group=base_group+local;
 const auto input=static_cast<std::uint64_t>(local)*128;
 float value[4]={__bfloat162float(source[input+lane*2]),
   __bfloat162float(source[input+lane*2+1]),
   __bfloat162float(source[input+64+lane*2]),
   __bfloat162float(source[input+65+lane*2])};
 bool bad=false;float bound=0;
 for(int j=0;j<4;++j){bad=bad||!isfinite(value[j]);bound=fmaxf(bound,fabsf(value[j])/7.f);}
 if(__any_sync(0xffffffffu,bad)){if(lane==0)atomicAdd(invalid,1u);return;}
 for(int off=16;off;off>>=1)bound=fmaxf(bound,__shfl_xor_sync(0xffffffffu,bound,off));
 float best=0,error_best=CUDART_INF_F;
 for(int candidate=0;candidate<9;++candidate){
  const float scale=__half2float(__float2half_rn(bound*(1.f-.05f*candidate)));
  const float inv=scale?1.f/scale:0.f;float error=0;
  for(int j=0;j<4;++j){const int q=nearest_nu3(value[j]*inv);const float delta=value[j]-q*scale;error=__fmaf_rn(delta,delta,error);}
  for(int off=16;off;off>>=1)error+=__shfl_xor_sync(0xffffffffu,error,off);
  if(error<error_best){error_best=error;best=scale;}
 }
 const float inv=best?1.f/best:0.f;
 int q[4];for(int j=0;j<4;++j)q[j]=nearest_nu3(value[j]*inv);
 const auto at=group*64;
 data[at+lane]=(q[0]&15)|((q[1]&15)<<4);
 data[at+32+lane]=(q[2]&15)|((q[3]&15)<<4);
 if constexpr(Q5){
  const unsigned a=(q[0]<0?1u:0u)|(q[1]<0?2u:0u),b=(q[2]<0?1u:0u)|(q[3]<0?2u:0u);
  unsigned lo=0,hi=0;
  for(int j=0;j<4;++j){lo|=__shfl_sync(0xffffffffu,a,(lane/4)*4+j)<<(2*j);hi|=__shfl_sync(0xffffffffu,b,(lane/4)*4+j)<<(2*j);}
  if(lane%4==0){high[group*16+lane/4]=lo;high[group*16+8+lane/4]=hi;}
 }
 if(lane==0){scales[group*2]=__float2half_rn(best);scales[group*2+1]=__float2half_rn(best);}
}
struct FactoryBuffer {
 void* p=nullptr;
 explicit FactoryBuffer(std::size_t bytes){CUDA_CHECK(cudaMalloc(&p,bytes));}
 ~FactoryBuffer(){if(p)cudaFree(p);}
 FactoryBuffer(const FactoryBuffer&)=delete;
 FactoryBuffer& operator=(const FactoryBuffer&)=delete;
};
} // namespace
void nu3_mlp_prepare_bf16(Weight& weight,const char* first,const char* second,cudaStream_t stream){
 const bool gate=weight.n==34816&&weight.k==5120&&weight.qtype==QType::Q4G64_F16S;
 const bool down=weight.n==5120&&weight.k==17408&&weight.qtype==QType::Q5G64_F16S;
 if((!gate&&!down)||weight.layout!=QuantLayout::RowSplit||!weight.qdata||!weight.scales||
    (down&&!weight.qhigh)||!first||(gate&&!second)||(!gate&&second))
  throw std::invalid_argument("factory NU3 preparation requires complete supported canonical sources");
 const std::uint64_t elements=static_cast<std::uint64_t>(weight.n)*weight.k;
 const std::uint64_t file_elements=gate?elements/2:elements;
 for(const char* path:{first,second})if(path){
  if(!std::filesystem::is_regular_file(path)||std::filesystem::file_size(path)!=file_elements*2)
   throw std::invalid_argument(std::string("factory BF16 file size mismatch: ")+path);
 }
 constexpr std::uint64_t chunk_elements=8ULL*1024*1024;
 FactoryBuffer temporary(chunk_elements*2),error_device(sizeof(unsigned));
 CUDA_CHECK(cudaMemsetAsync(error_device.p,0,sizeof(unsigned),stream));
 std::vector<std::uint16_t> host(chunk_elements);
 std::uint64_t base=0;
 for(const char* path:{first,second})if(path){
  std::ifstream input(path,std::ios::binary);if(!input)throw std::runtime_error("cannot read authenticated BF16 source");
  for(std::uint64_t done=0;done<file_elements;){
   const auto count=std::min(chunk_elements,file_elements-done);
   input.read(reinterpret_cast<char*>(host.data()),static_cast<std::streamsize>(count*2));
   if(!input)throw std::runtime_error("factory BF16 source truncated during preparation");
   CUDA_CHECK(cudaMemcpyAsync(temporary.p,host.data(),count*2,cudaMemcpyHostToDevice,stream));
   const auto groups=static_cast<std::uint32_t>(count/128);
   auto* data=const_cast<std::uint8_t*>(static_cast<const std::uint8_t*>(weight.qdata));
   auto* high=const_cast<std::uint8_t*>(static_cast<const std::uint8_t*>(weight.qhigh));
   auto* scales=const_cast<__half*>(static_cast<const __half*>(weight.scales));
   if(gate)recode_factory_bf16<false><<<(groups+7)/8,256,0,stream>>>(static_cast<const __nv_bfloat16*>(temporary.p),data,high,scales,(base+done)/128,groups,static_cast<unsigned*>(error_device.p));
   else recode_factory_bf16<true><<<(groups+7)/8,256,0,stream>>>(static_cast<const __nv_bfloat16*>(temporary.p),data,high,scales,(base+done)/128,groups,static_cast<unsigned*>(error_device.p));
   CUDA_CHECK(cudaGetLastError());CUDA_CHECK(cudaStreamSynchronize(stream));done+=count;
  }
  base+=file_elements;
 }
 unsigned invalid=0;CUDA_CHECK(cudaMemcpy(&invalid,error_device.p,sizeof(invalid),cudaMemcpyDeviceToHost));
 if(invalid)throw std::invalid_argument("factory source has nonfinite values");
 if(base!=elements)throw std::logic_error("factory NU3 preparation did not cover the full matrix");
}

void nu3_mlp_prepare(Weight& weight,cudaStream_t stream){
 const bool gate=weight.n==34816&&weight.k==5120&&weight.qtype==QType::Q4G64_F16S;
 const bool down=weight.n==5120&&weight.k==17408&&weight.qtype==QType::Q5G64_F16S;
 if((!gate&&!down)||weight.layout!=QuantLayout::RowSplit||!weight.qdata||!weight.scales||(down&&!weight.qhigh))
  throw std::invalid_argument("NU3 MLP requires canonical supported source weights");
 const int groups=weight.n*(weight.k/128);
 auto* data=const_cast<std::uint8_t*>(static_cast<const std::uint8_t*>(weight.qdata));
 auto* high=const_cast<std::uint8_t*>(static_cast<const std::uint8_t*>(weight.qhigh));
 auto* scales=const_cast<__half*>(static_cast<const __half*>(weight.scales));
 if(gate)recode<false,true><<<(groups+7)/8,256,0,stream>>>(data,high,scales,groups);
 else recode<true,true><<<(groups+7)/8,256,0,stream>>>(data,high,scales,groups);
 CUDA_CHECK(cudaGetLastError());CUDA_CHECK(cudaStreamSynchronize(stream));
}
void nu3_mlp_transform(Weight& weight,QuantLayout target,void* scratch,std::size_t bytes,cudaStream_t stream){
 if(weight.layout==target)return;
 if(!((weight.layout==QuantLayout::RowSplit&&target==QuantLayout::RowSplitSm75Nu3)||
      (weight.layout==QuantLayout::RowSplitSm75Nu3&&target==QuantLayout::RowSplit)))
  throw std::invalid_argument("NU3 layout transitions require canonical or packed source");
 nu3_layout::transform(weight,scratch,bytes,target==QuantLayout::RowSplitSm75Nu3,stream);
 weight.layout=target;
}
void nu3_mlp_launch(const Tensor& input,const Weight& gate,const Weight& down,Tensor& residual,
 WorkspaceArena& workspace,cudaStream_t stream){
 if(input.dtype!=DType::BF16||residual.dtype!=DType::BF16||input.ne[0]!=5120||input.ne[1]!=4||
    residual.ne[0]!=5120||residual.ne[1]!=4||!input.is_contiguous()||!residual.is_contiguous()||
    gate.layout!=QuantLayout::RowSplitSm75Nu3||down.layout!=QuantLayout::RowSplitSm75Nu3||
    gate.n!=34816||gate.k!=5120||gate.qtype!=QType::Q4G64_F16S||
    down.n!=5120||down.k!=17408||down.qtype!=QType::Q5G64_F16S)
  throw std::invalid_argument("NU3 MLP requires native contiguous C1 D3 four-column verification");
 auto scope=workspace.scope();
 Tensor iq=workspace.alloc(DType::I8,{5120,4}),is=workspace.alloc(DType::FP16,{80,4});
 Tensor aq=workspace.alloc(DType::I8,{17408,4}),as=workspace.alloc(DType::FP16,{272,4});
 quantize_a8<4><<<(4*(kK/64)+7)/8,256,0,stream>>>((const __nv_bfloat16*)input.data,(std::int8_t*)iq.data,(__half*)is.data);
 gate_w3g128<<<kM/128,512,10240,stream>>>((const uint4*)gate.qdata,
  (const uint2*)((const std::uint8_t*)gate.qdata+nu3_layout::low_bytes(gate)),(const unsigned*)gate.scales,
  (const std::int8_t*)iq.data,(const __half*)is.data,(std::int8_t*)aq.data,(__half*)as.data,nullptr);
 down_w3g128<<<DownN/16,128,0,stream>>>((const uint4*)down.qdata,
  (const uint2*)((const std::uint8_t*)down.qdata+nu3_layout::low_bytes(down)),(const __half*)down.scales,
  (const std::int8_t*)aq.data,(const __half*)as.data,(__nv_bfloat16*)residual.data);
 CUDA_CHECK(cudaGetLastError());
}
} // namespace ninfer::ops::detail
