// Frozen pre-optimization reference. Do not share the candidate implementation.
#pragma once
#include "../../apps/dual/joint_kv/value_residual.cuh"
namespace ninfer::ops::integer_joint_reference {
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
template<bool Indexed>
__global__ void allkey(const std::int8_t* q,const float* qscale,const float* qs,const int* positions,
 const unsigned* keys,const __half* linear_keys,const __half* linear_values,const unsigned* coords,
 const float* threshold,int count,int old_end,int blocks,int parts,
 unsigned* masks,float* output,float4* state,int capacity,const int* indices,const int* union_counts,const int* failed,int range_begin,int range_end,int first_part,int storage_blocks,int storage_begin,const __half* key_scales,const __half* value_scales){
 const int packed_blocks=storage_blocks>0?storage_blocks:blocks;
 int warp=threadIdx.x>>5,lane=threadIdx.x&31,gid=lane>>2,lid=lane&3;
 int work=blockIdx.x*4+warp;if(work>=4*parts*2)return;
 int tile=work%2,part=(work/2)%parts,head=work/(parts*2),r0=tile*16+gid,r1=r0+8;
 static_assert(!Indexed,"fused union is the all-key nonselected pass");
 // Two adjacent warps own all24 real query rows for one head/partition.
 // Alternating tiny mask slots prevent a fast warp from overwriting a mask
 // before its partner reads it, requiring only one64-thread barrier per tile.
 __shared__ unsigned group_masks[2][2][2];
 const int group=warp/2;
 const int* qpos=positions+(Indexed?head*4:0);
 const int count_here=Indexed?(failed[head]?0:qpos[3]+1):min(count,qpos[3]+1);
 int qa=query_index(r0,head),qb=query_index(r1,head),pos0=r0<24?qpos[r0/6]:-1,pos1=r1<24?qpos[r1/6]:-1;
 auto source_key=[&](int slot){if constexpr(Indexed)return slot<union_counts[head]?indices[head*65536+slot]:old_end+slot-union_counts[head];else return slot;};
 float qs0=r0<24?qs[qa]:0,qs1=r1<24?qs[qb]:0;
 float cut0=!Indexed&&r0<24?threshold[head*24+r0]:CUDART_INF_F,cut1=!Indexed&&r1<24?threshold[head*24+r1]:CUDART_INF_F;
 unsigned qr[2][4];
#pragma unroll
 for(int kk=0;kk<2;++kk){
  qr[kk][0]=r0<24?__ldg(reinterpret_cast<const unsigned*>(q+qa*64+kk*32+lid*4)):0;
  qr[kk][1]=r1<24?__ldg(reinterpret_cast<const unsigned*>(q+qb*64+kk*32+lid*4)):0;
  qr[kk][2]=r0<24?__ldg(reinterpret_cast<const unsigned*>(q+qa*64+kk*32+16+lid*4)):0;
  qr[kk][3]=r1<24?__ldg(reinterpret_cast<const unsigned*>(q+qb*64+kk*32+16+lid*4)):0;
 }
 const float query_scale0=r0<24?qscale[qa]:0,query_scale1=r1<24?qscale[qb]:0;
 float accumulator[4][4]{};float m0=-CUDART_INF_F,m1=-CUDART_INF_F,l0=0,l1=0,old0=0,old1=0;
 int begin=(part+first_part)*KeysPerPart,end=min(begin+KeysPerPart,min(count_here,range_end));
 for(int first=begin;first<end;first+=32){
  int block=first/32;int dot[4][4]{};
#pragma unroll
  for(int kk=0;kk<2;++kk){
#pragma unroll
   for(int nt=0;nt<4;++nt){unsigned b0=0,b1=0;
    if(first>=range_begin&&first<old_end){
     const auto at=(((std::size_t(head)*packed_blocks+block-storage_begin)*4+kk*2)*4+nt)*32+lane;
     b0=__ldg(keys+at);b1=__ldg(keys+at+4*32);
    }
    mma_s8(dot[nt][0],dot[nt][1],dot[nt][2],dot[nt][3],qr[kk][0],qr[kk][1],qr[kk][2],qr[kk][3],b0,b1);
   }
  }
  float score[4][4],bm0=-CUDART_INF_F,bm1=-CUDART_INF_F;unsigned mask0=0,mask1=0;
#pragma unroll
  for(int nt=0;nt<4;++nt){
   int a=first+nt*8+2*lid,b=a+1;
   float ka=0,kb=0;
   if(first>=range_begin&&first<old_end){const auto at=(std::size_t(head)*packed_blocks+block-storage_begin)*32+nt*8+2*lid;ka=__half2float(key_scales[at]);kb=__half2float(key_scales[at+1]);}

   score[nt][0]=r0<24&&a>=range_begin&&a<count_here&&a<=pos0?float(dot[nt][0])*(query_scale0*ka)*.0625f+qs0:-CUDART_INF_F;
   score[nt][1]=r0<24&&b>=range_begin&&b<count_here&&b<=pos0?float(dot[nt][1])*(query_scale0*kb)*.0625f+qs0:-CUDART_INF_F;
   score[nt][2]=r1<24&&a>=range_begin&&a<count_here&&a<=pos1?float(dot[nt][2])*(query_scale1*ka)*.0625f+qs1:-CUDART_INF_F;
   score[nt][3]=r1<24&&b>=range_begin&&b<count_here&&b<=pos1?float(dot[nt][3])*(query_scale1*kb)*.0625f+qs1:-CUDART_INF_F;
   bm0=fmaxf(bm0,fmaxf(score[nt][0],score[nt][1]));bm1=fmaxf(bm1,fmaxf(score[nt][2],score[nt][3]));
   if(a<old_end&&score[nt][0]>=cut0)mask0|=1U<<(nt*8+2*lid);
   if(b<old_end&&score[nt][1]>=cut0)mask0|=1U<<(nt*8+2*lid+1);
   if(a<old_end&&score[nt][2]>=cut1)mask1|=1U<<(nt*8+2*lid);
   if(b<old_end&&score[nt][3]>=cut1)mask1|=1U<<(nt*8+2*lid+1);
  }
  mask0|=__shfl_xor_sync(Mask,mask0,1);mask0|=__shfl_xor_sync(Mask,mask0,2);
  mask1|=__shfl_xor_sync(Mask,mask1,1);mask1|=__shfl_xor_sync(Mask,mask1,2);
  if constexpr(!Indexed){  if(lid==0){if(r0<24)masks[(head*24+r0)*blocks+block]=mask0;if(r1<24)masks[(head*24+r1)*blocks+block]=mask1;}}
  unsigned union_local=mask0|mask1;
  union_local|=__shfl_xor_sync(Mask,union_local,4);union_local|=__shfl_xor_sync(Mask,union_local,8);union_local|=__shfl_xor_sync(Mask,union_local,16);
  const int slot=(first/32)&1;
  if(lane==0)group_masks[group][slot][tile]=union_local;
  asm volatile("barrier.sync %0, 64;"::"r"(group+1):"memory");
  unsigned excluded=group_masks[group][slot][0]|group_masks[group][slot][1];
  if(first>=old_end)excluded=0xffffffffU;
  bm0=-CUDART_INF_F;bm1=-CUDART_INF_F;
#pragma unroll
  for(int nt=0;nt<4;++nt){int c=nt*8+2*lid;
   if(excluded&(1U<<c)){score[nt][0]=-CUDART_INF_F;score[nt][2]=-CUDART_INF_F;}
   if(excluded&(1U<<(c+1))){score[nt][1]=-CUDART_INF_F;score[nt][3]=-CUDART_INF_F;}
   bm0=fmaxf(bm0,fmaxf(score[nt][0],score[nt][1]));bm1=fmaxf(bm1,fmaxf(score[nt][2],score[nt][3]));
  }
  bm0=warp_max<4>(bm0,Mask);bm1=warp_max<4>(bm1,Mask);
  float nm0=fmaxf(m0,bm0),nm1=fmaxf(m1,bm1);
  float alpha0=m0>-CUDART_INF_F?exp2_approx((m0-nm0)*Log2E):0,alpha1=m1>-CUDART_INF_F?exp2_approx((m1-nm1)*Log2E):0;
  float beta0=bm0>-CUDART_INF_F?exp2_approx((bm0-nm0)*Log2E):0,beta1=bm1>-CUDART_INF_F?exp2_approx((bm1-nm1)*Log2E):0;
  unsigned pwords[4][2];float block0=0,block1=0;
#pragma unroll
  for(int nt=0;nt<4;++nt){
   float a=score[nt][0]>-CUDART_INF_F?exp2_approx((score[nt][0]-bm0)*Log2E):0,b=score[nt][1]>-CUDART_INF_F?exp2_approx((score[nt][1]-bm0)*Log2E):0;
   float c=score[nt][2]>-CUDART_INF_F?exp2_approx((score[nt][2]-bm1)*Log2E):0,d=score[nt][3]>-CUDART_INF_F?exp2_approx((score[nt][3]-bm1)*Log2E):0;
   block0+=a+b;block1+=c+d;pwords[nt][0]=probability_pair(a,b);pwords[nt][1]=probability_pair(c,d);
  }
  block0=warp_sum<4>(block0,Mask);block1=warp_sum<4>(block1,Mask);
  l0=l0*alpha0+block0*beta0;l1=l1*alpha1+block1*beta1;
  old0=old0*alpha0+(first<old_end?block0*beta0:0);old1=old1*alpha1+(first<old_end?block1*beta1:0);m0=nm0;m1=nm1;
  unsigned assembled[4][2];
#pragma unroll
  for(int nt=0;nt<4;++nt){
#pragma unroll
   for(int row=0;row<2;++row){const int owner=gid*4+(lid&1)*2;
    assembled[nt][row]=__shfl_sync(Mask,pwords[nt][row],owner)|(__shfl_sync(Mask,pwords[nt][row],owner+1)<<16);
   }
  }
  const unsigned p0=assembled[lid/2][0],p1=assembled[lid/2][1],p2=assembled[2+lid/2][0],p3=assembled[2+lid/2][1];
#pragma unroll
  for(int n=0;n<4;++n){unsigned b0=0,b1=0;float s0=0,s1=0;
   if(first>=range_begin&&first<old_end){
    const auto at=((std::size_t(head)*packed_blocks+block-storage_begin)*4+n)*2*32+lane;
    b0=__ldg(coords+at);b1=__ldg(coords+at+32);
    const auto si=(std::size_t(head)*packed_blocks+block-storage_begin)*32+n*8+2*lid;
    s0=__half2float(value_scales[si]);s1=__half2float(value_scales[si+1]);
   }
   int a=0,b=0,c=0,d=0;mma_s8(a,b,c,d,p0,p1,p2,p3,b0,b1);
   const float a0=float(a)*(s0*(1.f/127.f)),a1=float(b)*(s1*(1.f/127.f)),a2=float(c)*(s0*(1.f/127.f)),a3=float(d)*(s1*(1.f/127.f));
   accumulator[n][0]=__fmaf_rn(a0,beta0,accumulator[n][0]*alpha0);accumulator[n][1]=__fmaf_rn(a1,beta0,accumulator[n][1]*alpha0);
   accumulator[n][2]=__fmaf_rn(a2,beta1,accumulator[n][2]*alpha1);accumulator[n][3]=__fmaf_rn(a3,beta1,accumulator[n][3]*alpha1);
  }

 }
 if(lid==0){if(r0<24)state[(head*24+r0)*parts+part]=make_float4(m0,l0,old0,0);if(r1<24)state[(head*24+r1)*parts+part]=make_float4(m1,l1,old1,0);}
#pragma unroll
 for(int n=0;n<4;++n){int d=n*8+2*lid;
  if(r0<24){auto at=((head*24+r0)*parts+part)*32+d;output[at]=accumulator[n][0];output[at+1]=accumulator[n][1];}
  if(r1<24){auto at=((head*24+r1)*parts+part)*32+d;output[at]=accumulator[n][2];output[at+1]=accumulator[n][3];}
 }
}

}
