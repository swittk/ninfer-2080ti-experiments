#pragma once
#include "core/device.h"
#include "core/arena.h"
#include "ops/common/mma.cuh"
#include "../headscale_gqa.cuh"
#include <cub/device/device_segmented_radix_sort.cuh>
namespace ninfer::ops::value_residual {
using G=Gqa27Geometry;
constexpr unsigned Mask=0xffffffffu;
constexpr int Rank=32, MaxCorrections=4096, QueryRows=96, SampleLimit=16384;
constexpr int KeysPerPart=512, MaxMasks=8192, CorrectionBlock=32, CorrectionParts=129;
constexpr float Log2E=1.4426950408889634074f;
__global__ void quantize_queries(const __nv_bfloat16* input,std::int8_t* output,float* scales){
 int lane=threadIdx.x&31,row=(blockIdx.x*blockDim.x+threadIdx.x)/32;if(row>=96)return;
 float values[8],maximum=0;
#pragma unroll
 for(int j=0;j<8;++j){values[j]=__bfloat162float(input[row*256+lane+j*32]);maximum=fmaxf(maximum,fabsf(values[j]));}
 maximum=warp_max(maximum,Mask);float s=maximum>0?maximum/127.f:0,inv=s>0?1.f/s:0;
#pragma unroll
 for(int j=0;j<8;++j)output[row*256+lane+j*32]=gqa_kv_quant_code(values[j],inv);
 if(lane==0)scales[row]=s;
}
__device__ __forceinline__ int query_index(int row,int head){return (row/6)*24+head*6+row%6;}

__device__ __forceinline__ unsigned half_pair(float a,float b){
 union {__half2 h;unsigned u;} p;
 p.h=__halves2half2(__float2half_rn(a),__float2half_rn(b));return p.u;
}
__device__ __forceinline__ float coord_at(const unsigned* packed,int blocks,int head,int key,int feature){
 const int block=key/32,within=key%32,n=feature/8,nt=within/8;
 const int lane=(feature%8)*4+(within%8)/2;
 const auto offset=(((static_cast<std::size_t>(head)*blocks+block)*4+n)*4+nt)*32+lane;
 const unsigned bits=packed[offset];
 return __half2float(__ushort_as_half((within&1)?bits>>16:bits&65535));
}
// Input is logical [head,key,rank]. Output stores exactly the native m16n8k8 B
// fragments; no coefficient value or rank changes during this permutation.
__global__ void pack_coordinates(const __half* linear,unsigned* packed,int keys,int blocks){
 const auto i=static_cast<std::size_t>(blockIdx.x)*blockDim.x+threadIdx.x;
 const auto words=static_cast<std::size_t>(4)*blocks*512;if(i>=words)return;
 int lane=i%32,nt=(i/32)%4,n=(i/128)%4,block=(i/512)%blocks,head=i/(512*blocks);
 int feature=n*8+lane/4,k0=block*32+nt*8+(lane%4)*2;
 __half a=__float2half_rn(0),b=a;
 if(k0<keys)a=linear[(static_cast<std::size_t>(head)*keys+k0)*32+feature];
 if(k0+1<keys)b=linear[(static_cast<std::size_t>(head)*keys+k0+1)*32+feature];
 packed[i]=static_cast<unsigned>(__half_as_ushort(a))|(static_cast<unsigned>(__half_as_ushort(b))<<16);
}
// Exact current-query QK over the deterministic stratified prefix sample.
__global__ void sample_qk(const std::int8_t* q,const float* qs,const std::int8_t* key,
 const __half* ks,const int* table,int old_end,int samples,float* scores){
 const int warp=threadIdx.x>>5,lane=threadIdx.x&31,gid=lane>>2,lid=lane&3;
 const int tiles=(samples+31)/32,work=blockIdx.x*8+warp;
 if(work>=4*tiles*2)return;
 const int head=work/(tiles*2),sample_tile=(work/2)%tiles,rowbase=(work%2)*16;
 const int r0=rowbase+gid,r1=r0+8,qa=query_index(r0,head),qb=query_index(r1,head);
 int physical[4],actual_key[4];
#pragma unroll
 for(int nt=0;nt<4;++nt){int sample=sample_tile*32+nt*8+gid;
  actual_key[nt]=sample<samples?static_cast<int>(static_cast<long long>(sample)*old_end/samples):0;
  physical[nt]=sample<samples?table[actual_key[nt]/64]:-1;
 }
 int dot[4][4]{};
#pragma unroll
 for(int kk=0;kk<8;++kk){
  unsigned a0=r0<24?__ldg(reinterpret_cast<const unsigned*>(q+qa*256+kk*32+lid*4)):0;
  unsigned a1=r1<24?__ldg(reinterpret_cast<const unsigned*>(q+qb*256+kk*32+lid*4)):0;
  unsigned a2=r0<24?__ldg(reinterpret_cast<const unsigned*>(q+qa*256+kk*32+16+lid*4)):0;
  unsigned a3=r1<24?__ldg(reinterpret_cast<const unsigned*>(q+qb*256+kk*32+16+lid*4)):0;
#pragma unroll
  for(int nt=0;nt<4;++nt){unsigned b0=0,b1=0;
   if(physical[nt]>=0){auto at=gqa_kv_quant_code_index<G>(physical[nt],head,kk*32+lid*4,actual_key[nt]%64);
    b0=__ldg(reinterpret_cast<const unsigned*>(key+at));b1=__ldg(reinterpret_cast<const unsigned*>(key+at+16));}
   mma_s8(dot[nt][0],dot[nt][1],dot[nt][2],dot[nt][3],a0,a1,a2,a3,b0,b1);
  }
 }
 const float q0=r0<24?qs[qa]:0,q1=r1<24?qs[qb]:0;
#pragma unroll
 for(int nt=0;nt<4;++nt){
  int sa=sample_tile*32+nt*8+2*lid,sb=sa+1;
  float ka=0,kb=0;
  if(sa<samples){int ix=static_cast<int>(static_cast<long long>(sa)*old_end/samples),p=table[ix/64];if(p>=0)ka=__half2float(ks[gqa_kv_quant_scale_index<G>(p,head,0,ix%64)]);}
  if(sb<samples){int ix=static_cast<int>(static_cast<long long>(sb)*old_end/samples),p=table[ix/64];if(p>=0)kb=__half2float(ks[gqa_kv_quant_scale_index<G>(p,head,0,ix%64)]);}
  if(r0<24){if(sa<samples)scores[(head*24+r0)*samples+sa]=q0*ka*float(dot[nt][0])*.0625f;if(sb<samples)scores[(head*24+r0)*samples+sb]=q0*kb*float(dot[nt][1])*.0625f;}
  if(r1<24){if(sa<samples)scores[(head*24+r1)*samples+sa]=q1*ka*float(dot[nt][2])*.0625f;if(sb<samples)scores[(head*24+r1)*samples+sb]=q1*kb*float(dot[nt][3])*.0625f;}
 }
}
__global__ void choose_threshold(const float* sorted,int samples,int order,float* threshold){
 int q=blockIdx.x*blockDim.x+threadIdx.x;if(q<96)threshold[q]=sorted[q*samples+order-1];
}
// QK, all-key normalization and rank32 value contraction share one key scan.
// The heavy mask is exact for this invocation's sampled threshold.
__global__ void allkey_coordinates(const std::int8_t* q,const float* qs,const int* positions,
 const std::int8_t* keys,const __half* ks,const int* table,const unsigned* coords,
 const float* threshold,int count,int old_end,int blocks,int parts,
 unsigned* masks,float* output,float4* state){
 int warp=threadIdx.x>>5,lane=threadIdx.x&31,gid=lane>>2,lid=lane&3;
 int work=blockIdx.x*4+warp;if(work>=4*parts*2)return;
 int tile=work%2,part=(work/2)%parts,head=work/(parts*2),r0=tile*16+gid,r1=r0+8;
 int qa=query_index(r0,head),qb=query_index(r1,head),pos0=r0<24?positions[r0/6]:-1,pos1=r1<24?positions[r1/6]:-1;
 float qs0=r0<24?qs[qa]:0,qs1=r1<24?qs[qb]:0;
 float cut0=r0<24?threshold[head*24+r0]:CUDART_INF_F,cut1=r1<24?threshold[head*24+r1]:CUDART_INF_F;
 unsigned qr[8][4];
#pragma unroll
 for(int kk=0;kk<8;++kk){
  qr[kk][0]=r0<24?__ldg(reinterpret_cast<const unsigned*>(q+qa*256+kk*32+lid*4)):0;
  qr[kk][1]=r1<24?__ldg(reinterpret_cast<const unsigned*>(q+qb*256+kk*32+lid*4)):0;
  qr[kk][2]=r0<24?__ldg(reinterpret_cast<const unsigned*>(q+qa*256+kk*32+16+lid*4)):0;
  qr[kk][3]=r1<24?__ldg(reinterpret_cast<const unsigned*>(q+qb*256+kk*32+16+lid*4)):0;
 }
 float accumulator[4][4]{};float m0=-CUDART_INF_F,m1=-CUDART_INF_F,l0=0,l1=0,old0=0,old1=0;
 int begin=part*KeysPerPart,end=min(begin+KeysPerPart,count);
 for(int first=begin;first<end;first+=32){
  int block=first/32,physical=table[first/64],pageoff=first%64;int dot[4][4]{};
#pragma unroll
  for(int kk=0;kk<8;++kk){
#pragma unroll
   for(int nt=0;nt<4;++nt){unsigned b0=0,b1=0;int key=first+nt*8+gid;
    if(physical>=0&&key<count){auto at=gqa_kv_quant_code_index<G>(physical,head,kk*32+lid*4,pageoff+nt*8+gid);b0=__ldg(reinterpret_cast<const unsigned*>(keys+at));b1=__ldg(reinterpret_cast<const unsigned*>(keys+at+16));}
    mma_s8(dot[nt][0],dot[nt][1],dot[nt][2],dot[nt][3],qr[kk][0],qr[kk][1],qr[kk][2],qr[kk][3],b0,b1);
   }
  }
  float score[4][4],bm0=-CUDART_INF_F,bm1=-CUDART_INF_F;unsigned mask0=0,mask1=0;
#pragma unroll
  for(int nt=0;nt<4;++nt){
   int a=first+nt*8+2*lid,b=a+1;float sa=0,sb=0;
   if(physical>=0){if(a<count)sa=__half2float(ks[gqa_kv_quant_scale_index<G>(physical,head,0,a%64)]);if(b<count)sb=__half2float(ks[gqa_kv_quant_scale_index<G>(physical,head,0,b%64)]);}
   score[nt][0]=physical>=0&&r0<24&&a<count&&a<=pos0?qs0*sa*float(dot[nt][0])*.0625f:-CUDART_INF_F;
   score[nt][1]=physical>=0&&r0<24&&b<count&&b<=pos0?qs0*sb*float(dot[nt][1])*.0625f:-CUDART_INF_F;
   score[nt][2]=physical>=0&&r1<24&&a<count&&a<=pos1?qs1*sa*float(dot[nt][2])*.0625f:-CUDART_INF_F;
   score[nt][3]=physical>=0&&r1<24&&b<count&&b<=pos1?qs1*sb*float(dot[nt][3])*.0625f:-CUDART_INF_F;
   bm0=fmaxf(bm0,fmaxf(score[nt][0],score[nt][1]));bm1=fmaxf(bm1,fmaxf(score[nt][2],score[nt][3]));
   if(a<old_end&&score[nt][0]>=cut0)mask0|=1U<<(nt*8+2*lid);
   if(b<old_end&&score[nt][1]>=cut0)mask0|=1U<<(nt*8+2*lid+1);
   if(a<old_end&&score[nt][2]>=cut1)mask1|=1U<<(nt*8+2*lid);
   if(b<old_end&&score[nt][3]>=cut1)mask1|=1U<<(nt*8+2*lid+1);
  }
  mask0|=__shfl_xor_sync(Mask,mask0,1);mask0|=__shfl_xor_sync(Mask,mask0,2);
  mask1|=__shfl_xor_sync(Mask,mask1,1);mask1|=__shfl_xor_sync(Mask,mask1,2);
  if(lid==0){if(r0<24)masks[(head*24+r0)*blocks+block]=mask0;if(r1<24)masks[(head*24+r1)*blocks+block]=mask1;}
  bm0=warp_max<4>(bm0,Mask);bm1=warp_max<4>(bm1,Mask);
  float nm0=fmaxf(m0,bm0),nm1=fmaxf(m1,bm1);
  float alpha0=m0>-CUDART_INF_F?exp2_approx((m0-nm0)*Log2E):0,alpha1=m1>-CUDART_INF_F?exp2_approx((m1-nm1)*Log2E):0;
  float beta0=bm0>-CUDART_INF_F?exp2_approx((bm0-nm0)*Log2E):0,beta1=bm1>-CUDART_INF_F?exp2_approx((bm1-nm1)*Log2E):0;
  unsigned pwords[4][2];float block0=0,block1=0;
#pragma unroll
  for(int nt=0;nt<4;++nt){
   float a=score[nt][0]>-CUDART_INF_F?exp2_approx((score[nt][0]-bm0)*Log2E):0,b=score[nt][1]>-CUDART_INF_F?exp2_approx((score[nt][1]-bm0)*Log2E):0;
   float c=score[nt][2]>-CUDART_INF_F?exp2_approx((score[nt][2]-bm1)*Log2E):0,d=score[nt][3]>-CUDART_INF_F?exp2_approx((score[nt][3]-bm1)*Log2E):0;
   block0+=a+b;block1+=c+d;pwords[nt][0]=half_pair(a,b);pwords[nt][1]=half_pair(c,d);
  }
  block0=warp_sum<4>(block0,Mask);block1=warp_sum<4>(block1,Mask);
  l0=l0*alpha0+block0*beta0;l1=l1*alpha1+block1*beta1;
  old0=old0*alpha0+(first<old_end?block0*beta0:0);old1=old1*alpha1+(first<old_end?block1*beta1:0);m0=nm0;m1=nm1;
#pragma unroll
  for(int n=0;n<4;++n){float a=0,b=0,c=0,d=0;
   if(first<old_end){
#pragma unroll
    for(int nt=0;nt<4;++nt){auto at=(((static_cast<std::size_t>(head)*blocks+block)*4+n)*4+nt)*32+lane;
     unsigned value=__ldg(coords+at);mma_f16_k8(a,b,c,d,pwords[nt][0],pwords[nt][1],value);}
   }
   accumulator[n][0]=__fmaf_rn(a,beta0,accumulator[n][0]*alpha0);accumulator[n][1]=__fmaf_rn(b,beta0,accumulator[n][1]*alpha0);
   accumulator[n][2]=__fmaf_rn(c,beta1,accumulator[n][2]*alpha1);accumulator[n][3]=__fmaf_rn(d,beta1,accumulator[n][3]*alpha1);
  }
 }
 if(lid==0){if(r0<24)state[(head*24+r0)*parts+part]=make_float4(m0,l0,old0,0);if(r1<24)state[(head*24+r1)*parts+part]=make_float4(m1,l1,old1,0);}
#pragma unroll
 for(int n=0;n<4;++n){int d=n*8+2*lid;
  if(r0<24){auto at=((head*24+r0)*parts+part)*32+d;output[at]=accumulator[n][0];output[at+1]=accumulator[n][1];}
  if(r1<24){auto at=((head*24+r1)*parts+part)*32+d;output[at]=accumulator[n][2];output[at+1]=accumulator[n][3];}
 }
}
__global__ void compact_masks(const unsigned* masks,int blocks,int old_end,int* indices,int* counts,int* fallback_heads){
 int q=blockIdx.x,t=threadIdx.x,lane=t&31,warp=t>>5;
 unsigned local[32];int count=0;
#pragma unroll
 for(int j=0;j<32;++j){int b=t*32+j;local[j]=b<blocks?masks[q*blocks+b]:0;count+=__popc(local[j]);}
 int inclusive=count;
#pragma unroll
 for(int off=1;off<32;off<<=1){int x=__shfl_up_sync(Mask,inclusive,off);if(lane>=off)inclusive+=x;}
 __shared__ int warp_sum[8],total;
 if(lane==31)warp_sum[warp]=inclusive;
 __syncthreads();
 int prefix=inclusive-count;for(int i=0;i<warp;++i)prefix+=warp_sum[i];
 if(t==255){total=prefix+count;counts[q]=total;if(total>MaxCorrections||total<min(2048,old_end))atomicOr(fallback_heads+q/24,1);}
 __syncthreads();
 if(total>MaxCorrections||total<min(2048,old_end))return;
#pragma unroll
 for(int j=0;j<32;++j){unsigned bits=local[j];while(bits){int bit=__ffs(bits)-1;indices[q*MaxCorrections+prefix++]=(t*32+j)*32+bit;bits&=bits-1;}}
}
__global__ void combine_coordinates(const float* partial,const float4* state,int parts,float* coordinates,float4* global){
 int q=blockIdx.x,t=threadIdx.x;__shared__ float maximum;
 float mx=-CUDART_INF_F;for(int s=t;s<parts;s+=256)mx=fmaxf(mx,state[q*parts+s].x);
 __shared__ float warps[8];mx=warp_max(mx,Mask);if((t&31)==0)warps[t>>5]=mx;__syncthreads();
 if(t==0){mx=-CUDART_INF_F;for(int w=0;w<8;++w)mx=fmaxf(mx,warps[w]);maximum=mx;}__syncthreads();
 if(t<32){float sum=0,den=0,old=0;
  for(int s=0;s<parts;++s){float4 v=state[q*parts+s];float f=v.x>-CUDART_INF_F?exp2_approx((v.x-maximum)*Log2E):0;
   sum=__fmaf_rn(partial[(q*parts+s)*32+t],f,sum);den=__fmaf_rn(v.y,f,den);old=__fmaf_rn(v.z,f,old);}
  coordinates[q*32+t]=den>0?sum/den:0;if(t==0)global[q]=make_float4(maximum,den,den>0?old/den:0,0);
 }
}
// Fixed-size selected chunks plus a canonical tail chunk. Accumulate V and its
// rank coordinates separately, reconstructing the residual only once per query.
__global__ void corrections(const std::int8_t* query,const float* qs,const int* positions,
 const std::int8_t* keys,const __half* ks,const std::int8_t* values,const __half* vs,
 const int* table,const unsigned* coords,const int* indices,const int* counts,const int* fallback_heads,
 const float4* global,int key_count,int old_end,int blocks,float* result){
 int q=blockIdx.y,chunk=blockIdx.x,t=threadIdx.x,lane=t&31,warp=t>>5,head=q/24,row=q%24;
 const bool tail=chunk==128;float* out=result+(q*CorrectionParts+chunk)*289;
 if(fallback_heads[head]){out[t]=0;if(t<33)out[256+t]=0;return;}
 int count=counts[q],offset=chunk*32,qactual=query_index(row,head),qpos=positions[row/6];
 __shared__ int selected[32];__shared__ float p[32];
 if(t<32){int k=tail?old_end+t:(offset+t<count?indices[q*MaxCorrections+offset+t]:-1);selected[t]=k<key_count?k:-1;}
 __syncthreads();float4 state=global[q];
 for(int j=warp;j<32;j+=8){int k=selected[j],dot=0;
  if(k>=0&&k<=qpos){int ph=table[k/64];
#pragma unroll
   for(int d=lane;d<256;d+=32){dot+=int(query[qactual*256+d])*int(keys[gqa_kv_quant_code_index<G>(ph,head,d,k%64)]);}
  }
  for(int off=16;off;off>>=1)dot+=__shfl_down_sync(Mask,dot,off);
  if(lane==0){float probability=0;if(k>=0&&k<=qpos&&state.y>0){int ph=table[k/64];float s=__half2float(ks[gqa_kv_quant_scale_index<G>(ph,head,0,k%64)]);float score=qs[qactual]*s*float(dot)*.0625f;probability=exp2_approx((score-state.x)*Log2E)/state.y;}p[j]=probability;}
 }
 __syncthreads();
 float total=0,coordinate=0,mass=0;
 for(int j=0;j<32;++j){int k=selected[j];float probability=p[j];if(k>=0&&probability!=0){int ph=table[k/64];
  float scale=__half2float(vs[gqa_kv_quant_scale_index<G>(ph,head,t/64,k%64)]);
  float v=__half2float(__float2half_rn(float(values[gqa_kv_quant_code_index<G>(ph,head,t,k%64)])*scale));
  total=__fmaf_rn(probability,v,total);
  if(t<32&&!tail)coordinate=__fmaf_rn(probability,coord_at(coords,blocks,head,k,t),coordinate);
 }
 if(!tail)mass+=probability;
 }
 out[t]=total;if(t<32)out[256+t]=coordinate;if(t==32)out[288]=mass;
}
__global__ void finish(const float* coordinates,const float4* global,const float* corrections,
 const __half* basis,const __half* mean,const int* fallback_heads,__nv_bfloat16* output){
 int q=blockIdx.x,d=threadIdx.x,head=q/24,row=q%24;if(fallback_heads[head])return;
 __shared__ float remaining[32],mass;
 float value=0;for(int p=0;p<CorrectionParts;++p)value+=corrections[(q*CorrectionParts+p)*289+d];
 if(d<32){float c=0;for(int p=0;p<CorrectionParts;++p)c+=corrections[(q*CorrectionParts+p)*289+256+d];remaining[d]=coordinates[q*32+d]-c;}
 if(d==32){float m=0;for(int p=0;p<CorrectionParts;++p)m+=corrections[(q*CorrectionParts+p)*289+288];mass=global[q].z-m;}
 __syncthreads();
 value=__fmaf_rn(mass,__half2float(mean[head*256+d]),value);
 for(int r=0;r<32;++r)value=__fmaf_rn(remaining[r],__half2float(basis[(head*256+d)*32+r]),value);
 output[query_index(row,head)*256+d]=__float2bfloat16_rn(value);
}
}
