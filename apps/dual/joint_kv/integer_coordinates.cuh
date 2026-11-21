#pragma once
#include "value_residual.cuh"
namespace ninfer::ops::integer_joint {
using namespace ninfer::ops::value_residual;
__device__ __forceinline__ float unpack_half(unsigned bits){return __half2float(__ushort_as_half(bits&65535U));}
__global__ void quantize_keys(const unsigned* source,std::int8_t* dest,__half* scales,int blocks,int rows){
 const int lane=threadIdx.x&31;
 const int row=(blockIdx.x*blockDim.x+threadIdx.x)/32;
 if(row>=4*rows)return;
 const int head=row/rows,key=row%rows,block=key/32,nt=(key%32)/8,kr=key%8;
 float v[2],maximum=0;
#pragma unroll
 for(int j=0;j<2;++j){int d=lane+j*32;
  const auto at=(((std::size_t(head)*blocks+block)*8+d/8)*4+nt)*32+kr*4+(d%8)/2;
  v[j]=unpack_half(__ldg(source+at)>>(16*(d&1)));maximum=fmaxf(maximum,fabsf(v[j]));
 }
 maximum=warp_max(maximum,Mask);const __half hs=__float2half_rn(maximum/127.f);
 const float scale=__half2float(hs),inverse=scale>0?1.f/scale:0;
 if(lane==0)scales[row]=hs;
#pragma unroll
 for(int j=0;j<2;++j){int d=lane+j*32;
  const auto at=(((std::size_t(head)*blocks+block)*4+d/16)*4+nt)*128+(kr*4+(d%16)/4)*4+d%4;
  dest[at]=static_cast<std::int8_t>(max(-127,min(127,__float2int_rn(v[j]*inverse))));
 }
}
__global__ void quantize_values(const unsigned* source,std::int8_t* dest,__half* scales,int blocks){
 const int lane=threadIdx.x&31;const int unit=(blockIdx.x*blockDim.x+threadIdx.x)/32;
 if(unit>=4*blocks*32)return;
 const int d=unit%32,block=(unit/32)%blocks,head=unit/(32*blocks);
 const auto at=(((std::size_t(head)*blocks+block)*4+d/8)*4+lane/8)*32+(d%8)*4+(lane%8)/2;
 const float value=unpack_half(__ldg(source+at)>>(16*(lane&1)));
 const float maximum=warp_max(fabsf(value),Mask);const __half hs=__float2half_rn(maximum/127.f);
 const float scale=__half2float(hs),inverse=scale>0?1.f/scale:0;
 if(lane==0)scales[unit]=hs;
 const auto out=(((std::size_t(head)*blocks+block)*4+d/8)*2+lane/16)*128+((d%8)*4+(lane%16)/4)*4+lane%4;
 dest[out]=static_cast<std::int8_t>(max(-127,min(127,__float2int_rn(value*inverse))));
}
__global__ void quantize_projected_queries(const __half* input,std::int8_t* output,float* scales){
 const int lane=threadIdx.x&31,row=(blockIdx.x*blockDim.x+threadIdx.x)/32;if(row>=96)return;
 const float a=__half2float(input[row*64+lane]),b=__half2float(input[row*64+lane+32]);
 const float maximum=warp_max(fmaxf(fabsf(a),fabsf(b)),Mask),scale=maximum/127.f,inverse=scale>0?1.f/scale:0;
 output[row*64+lane]=static_cast<std::int8_t>(max(-127,min(127,__float2int_rn(a*inverse))));
 output[row*64+lane+32]=static_cast<std::int8_t>(max(-127,min(127,__float2int_rn(b*inverse))));
 if(lane==0)scales[row]=scale;
}
__global__ void sample_scores(const std::int8_t* q,const float* qscale,const float* shift,
 const unsigned* packet,const __half* keyscale,int samples,float* scores){
 const int warp=threadIdx.x>>5,lane=threadIdx.x&31,gid=lane>>2,lid=lane&3,tiles=(samples+31)/32;
 const int work=blockIdx.x*8+warp;if(work>=4*tiles*2)return;
 const int head=work/(tiles*2),tile=(work/2)%tiles,base=(work%2)*16,r0=base+gid,r1=r0+8;
 const int qa=query_index(r0,head),qb=query_index(r1,head);int dot[4][4]{};
#pragma unroll
 for(int kk=0;kk<2;++kk){
  unsigned qr[4]={r0<24?__ldg(reinterpret_cast<const unsigned*>(q+qa*64+kk*32+lid*4)):0,
    r1<24?__ldg(reinterpret_cast<const unsigned*>(q+qb*64+kk*32+lid*4)):0,
    r0<24?__ldg(reinterpret_cast<const unsigned*>(q+qa*64+kk*32+16+lid*4)):0,
    r1<24?__ldg(reinterpret_cast<const unsigned*>(q+qb*64+kk*32+16+lid*4)):0};
#pragma unroll
  for(int nt=0;nt<4;++nt){
   const auto at=((std::size_t(head)*tiles+tile)*4+kk*2)*4*32+nt*32+lane;
   const unsigned b0=__ldg(packet+at),b1=__ldg(packet+at+4*32);
   mma_s8(dot[nt][0],dot[nt][1],dot[nt][2],dot[nt][3],qr[0],qr[1],qr[2],qr[3],b0,b1);
  }
 }
 const float s0=r0<24?shift[qa]:0,s1=r1<24?shift[qb]:0,q0=r0<24?qscale[qa]:0,q1=r1<24?qscale[qb]:0;
#pragma unroll
 for(int nt=0;nt<4;++nt){const int a=tile*32+nt*8+2*lid,b=a+1;
  const float ka=a<samples?__half2float(keyscale[head*samples+a]):0,kb=b<samples?__half2float(keyscale[head*samples+b]):0;
  if(r0<24){if(a<samples)scores[(head*24+r0)*samples+a]=float(dot[nt][0])*(q0*ka)*.0625f+s0;if(b<samples)scores[(head*24+r0)*samples+b]=float(dot[nt][1])*(q0*kb)*.0625f+s0;}
  if(r1<24){if(a<samples)scores[(head*24+r1)*samples+a]=float(dot[nt][2])*(q1*ka)*.0625f+s1;if(b<samples)scores[(head*24+r1)*samples+b]=float(dot[nt][3])*(q1*kb)*.0625f+s1;}
 }
}
__device__ __forceinline__ unsigned probability_pair(float a,float b){
 const unsigned x=static_cast<unsigned>(max(0,min(127,__float2int_rn(a*127.f))));
 const unsigned y=static_cast<unsigned>(max(0,min(127,__float2int_rn(b*127.f))));return x|(y<<8);
}
// Twenty-four real query rows are three native eight-row warp tiles.
// Keep each row's reduction order; no padded query rows or extra storage.
__device__ __forceinline__ void mma_rows8(int& a,int& b,unsigned q0,unsigned q1,unsigned k0,unsigned k1){
 asm volatile("mma.sync.aligned.m8n8k16.row.col.s32.s8.s8.s32 {%0,%1}, {%2}, {%3}, {%0,%1};\n" : "+r"(a),"+r"(b):"r"(q0),"r"(k0));
 asm volatile("mma.sync.aligned.m8n8k16.row.col.s32.s8.s8.s32 {%0,%1}, {%2}, {%3}, {%0,%1};\n" : "+r"(a),"+r"(b):"r"(q1),"r"(k1));
}
template<bool Indexed, bool DirectMetadata=false>
__global__ void allkey(const std::int8_t* q,const float* qscale,const float* qs,const int* positions,
 const unsigned* keys,const __half* linear_keys,const __half* linear_values,const unsigned* coords,
 const float* threshold,int count,int old_end,int blocks,int parts,
 unsigned* masks,float* output,float4* state,int capacity,const int* indices,const int* union_counts,const int* failed,int range_begin,int range_end,int first_part,int storage_blocks,int storage_begin,const __half* key_scales,const __half* value_scales,unsigned* head_unions=nullptr){
 static_assert(!Indexed);
 const int warp=threadIdx.x>>5,lane=threadIdx.x&31,gid=lane>>2,lid=lane&3;
 const int head=blockIdx.x/parts,part=blockIdx.x%parts,row=warp*8+gid;
 const int packed_blocks=storage_blocks>0?storage_blocks:blocks;
 const int count_here=min(count,positions[3]+1),qa=query_index(row,head),pos=positions[row/6];
 const float shift=qs[qa],cut=threshold[head*24+row],scale=qscale[qa];
 unsigned qr[2][2];
#pragma unroll
 for(int kk=0;kk<2;++kk){qr[kk][0]=__ldg(reinterpret_cast<const unsigned*>(q+qa*64+kk*32+lid*4));qr[kk][1]=__ldg(reinterpret_cast<const unsigned*>(q+qa*64+kk*32+16+lid*4));}
 __shared__ unsigned union_masks[2][3];
 int selected_count=0;float accumulator[4][2]{};float maximum=-CUDART_INF_F,denominator=0,old_mass=0;
 const int begin=(part+first_part)*KeysPerPart,end=min(begin+KeysPerPart,min(count_here,range_end));
 for(int first=begin;first<end;first+=32){
  const int block=first/32;int dots[4][2]{};
#pragma unroll
  for(int kk=0;kk<2;++kk){
#pragma unroll
   for(int n=0;n<4;++n){unsigned b0=0,b1=0;
    if(first>=range_begin&&first<old_end){auto at=(((std::size_t(head)*packed_blocks+block-storage_begin)*4+kk*2)*4+n)*32+lane;b0=__ldg(keys+at);b1=__ldg(keys+at+4*32);}
    mma_rows8(dots[n][0],dots[n][1],qr[kk][0],qr[kk][1],b0,b1);
   }
  }
  float score[4][2];unsigned selected=0;
#pragma unroll
  for(int n=0;n<4;++n){const int a=first+n*8+2*lid,b=a+1;float ka=0,kb=0;
   if(first>=range_begin&&first<old_end){auto at=(std::size_t(head)*packed_blocks+block-storage_begin)*32+n*8+2*lid;ka=__half2float(key_scales[at]);kb=__half2float(key_scales[at+1]);}
   score[n][0]=a>=range_begin&&a<count_here&&a<=pos?float(dots[n][0])*(scale*ka)*.0625f+shift:-CUDART_INF_F;
   score[n][1]=b>=range_begin&&b<count_here&&b<=pos?float(dots[n][1])*(scale*kb)*.0625f+shift:-CUDART_INF_F;
   if(a<old_end&&score[n][0]>=cut)selected|=1U<<(n*8+2*lid);
   if(b<old_end&&score[n][1]>=cut)selected|=1U<<(n*8+2*lid+1);
  }
  selected|=__shfl_xor_sync(Mask,selected,1);selected|=__shfl_xor_sync(Mask,selected,2);
  if(lid==0){if constexpr(DirectMetadata)selected_count+=__popc(selected);else masks[(head*24+row)*blocks+block]=selected;}
  unsigned whole=selected;whole|=__shfl_xor_sync(Mask,whole,4);whole|=__shfl_xor_sync(Mask,whole,8);whole|=__shfl_xor_sync(Mask,whole,16);
  const int slot=(first/32)&1;if(lane==0)union_masks[slot][warp]=whole;
  __syncthreads();
  unsigned excluded=union_masks[slot][0]|union_masks[slot][1]|union_masks[slot][2];
  if constexpr(DirectMetadata)if(threadIdx.x==0)head_unions[head*blocks+block]=excluded;
  // The head union must retain the selected mask even for tail words: native
  // selection counts use it. Excluding the entire tail is only a softmax rule.
  if(first>=old_end)excluded=0xffffffffU;
  float block_max=-CUDART_INF_F;
#pragma unroll
  for(int n=0;n<4;++n){const int c=n*8+2*lid;
   if(excluded&(1U<<c))score[n][0]=-CUDART_INF_F;
   if(excluded&(1U<<(c+1)))score[n][1]=-CUDART_INF_F;
   block_max=fmaxf(block_max,fmaxf(score[n][0],score[n][1]));
  }
  block_max=warp_max<4>(block_max,Mask);const float next=fmaxf(maximum,block_max);
  const float alpha=maximum>-CUDART_INF_F?exp2_approx((maximum-next)*Log2E):0;
  const float beta=block_max>-CUDART_INF_F?exp2_approx((block_max-next)*Log2E):0;
  unsigned probabilities[4];float block_mass=0;
#pragma unroll
  for(int n=0;n<4;++n){float a=score[n][0]>-CUDART_INF_F?exp2_approx((score[n][0]-block_max)*Log2E):0,b=score[n][1]>-CUDART_INF_F?exp2_approx((score[n][1]-block_max)*Log2E):0;block_mass+=a+b;probabilities[n]=probability_pair(a,b);}
  // Every tail score was just excluded. Therefore its block_mass and beta are
  // both zero: old-prefix mass follows exactly the denominator recurrence.
  // DirectMetadata emits that identical value without a duplicate live float.
  block_mass=warp_sum<4>(block_mass,Mask);denominator=denominator*alpha+block_mass*beta;if constexpr(!DirectMetadata)old_mass=old_mass*alpha+(first<old_end?block_mass*beta:0);maximum=next;
  unsigned assembled[4];
#pragma unroll
  for(int n=0;n<4;++n){const int owner=gid*4+(lid&1)*2;assembled[n]=__shfl_sync(Mask,probabilities[n],owner)|(__shfl_sync(Mask,probabilities[n],owner+1)<<16);}
  const unsigned p0=assembled[lid/2],p1=assembled[2+lid/2];
#pragma unroll
  for(int n=0;n<4;++n){unsigned b0=0,b1=0;float s0=0,s1=0;
   if(first>=range_begin&&first<old_end){auto at=((std::size_t(head)*packed_blocks+block-storage_begin)*4+n)*2*32+lane;b0=__ldg(coords+at);b1=__ldg(coords+at+32);auto si=(std::size_t(head)*packed_blocks+block-storage_begin)*32+n*8+2*lid;s0=__half2float(value_scales[si]);s1=__half2float(value_scales[si+1]);}
   int a=0,b=0;mma_rows8(a,b,p0,p1,b0,b1);
   const float u=float(a)*(s0*(1.f/127.f)),v=float(b)*(s1*(1.f/127.f));
   accumulator[n][0]=__fmaf_rn(u,beta,accumulator[n][0]*alpha);accumulator[n][1]=__fmaf_rn(v,beta,accumulator[n][1]*alpha);
  }
 }
 if(lid==0){if constexpr(DirectMetadata)masks[(head*24+row)*parts+part]=static_cast<unsigned>(selected_count);}
 if(lid==0)state[(head*24+row)*parts+part]=make_float4(maximum,denominator,DirectMetadata?denominator:old_mass,0);
#pragma unroll
 for(int n=0;n<4;++n){const int d=n*8+2*lid;auto at=((head*24+row)*parts+part)*32+d;output[at]=accumulator[n][0];output[at+1]=accumulator[n][1];}
}
__global__ void reduce_selected_counts(const unsigned* partial,int parts,int* counts,int* failed){
 const int q=blockIdx.x,t=threadIdx.x,lane=t&31,warp=t>>5;int count=0;
 for(int i=t;i<parts;i+=256)count+=partial[q*parts+i];
 for(int off=16;off;off>>=1)count+=__shfl_down_sync(0xffffffffu,count,off);
 __shared__ int totals[8];if(lane==0)totals[warp]=count;__syncthreads();
 if(t==0){int sum=0;for(int i=0;i<8;++i)sum+=totals[i];counts[q]=sum;if(sum>4096)atomicOr(failed+q/24,1);}
}

}
