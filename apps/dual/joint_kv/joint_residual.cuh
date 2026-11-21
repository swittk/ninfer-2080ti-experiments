#pragma once
#include "value_residual.cuh"
namespace ninfer::ops::joint_residual {
using namespace ninfer::ops::value_residual;
constexpr int KeyRank=64;
__global__ void project_query(const std::int8_t* q,const float* qs,const __half* key_basis,const __half* key_mean,__half* projected,float* shift){
 const int row=blockIdx.x,t=threadIdx.x,slice=t/64,d=t%64,head=(row%24)/6;
 float dot=0,mu=0;
 for(int j=0;j<64;++j){const int k=slice*64+j;float x=float(q[row*256+k]);dot=__fmaf_rn(x,__half2float(key_basis[(head*256+k)*KeyRank+d]),dot);if(d==0)mu=__fmaf_rn(x,__half2float(key_mean[head*256+k]),mu);}
 __shared__ float partial[4][64],means[4];partial[slice][d]=dot;if(d==0)means[slice]=mu;__syncthreads();
 if(slice==0){dot=(partial[0][d]+partial[1][d])+(partial[2][d]+partial[3][d]);projected[row*KeyRank+d]=__float2half_rn(dot*qs[row]);
  if(d==0)shift[row]=((means[0]+means[1])+(means[2]+means[3]))*qs[row]*.0625f;
 }
}
__global__ void pack_key_coordinates(const __half* input,unsigned* output,int capacity,int keys,int slots,int old_end,bool sample){
 const auto index=static_cast<std::size_t>(blockIdx.x)*blockDim.x+threadIdx.x;const int tiles=(slots+31)/32;
 if(index>=std::size_t(4)*tiles*8*4*32)return;
 int lane=index%32,nt=(index/32)%4,kk=(index/128)%8,tile=(index/1024)%tiles,head=index/(1024*tiles);
 int at=tile*32+nt*8+lane/4;unsigned result=0;
 if(at<slots){int source=sample?static_cast<int>(static_cast<long long>(at)*old_end/slots):at;
  if(source<keys)result=*reinterpret_cast<const unsigned*>(input+(static_cast<std::size_t>(head)*capacity+source)*KeyRank+kk*8+2*(lane%4));}
 output[index]=result;
}
__global__ void sample_scores(const __half* q,const float* shift,const unsigned* packet,int samples,float* scores){
 int warp=threadIdx.x>>5,lane=threadIdx.x&31,gid=lane>>2,lid=lane&3,tiles=(samples+31)/32;
 int work=blockIdx.x*8+warp;if(work>=4*tiles*2)return;
 int head=work/(tiles*2),tile=(work/2)%tiles,base=(work%2)*16,r0=base+gid,r1=r0+8;
 int qa=query_index(r0,head),qb=query_index(r1,head);float dots[4][4]{};
#pragma unroll
 for(int kk=0;kk<8;++kk){unsigned a=r0<24?__ldg(reinterpret_cast<const unsigned*>(q+qa*64+kk*8+2*lid)):0;
  unsigned b=r1<24?__ldg(reinterpret_cast<const unsigned*>(q+qb*64+kk*8+2*lid)):0;
#pragma unroll
  for(int nt=0;nt<4;++nt){unsigned key=__ldg(packet+(((static_cast<std::size_t>(head)*tiles+tile)*8+kk)*4+nt)*32+lane);
   mma_f16_k8(dots[nt][0],dots[nt][1],dots[nt][2],dots[nt][3],a,b,key);}
 }
 float s0=r0<24?shift[qa]:0,s1=r1<24?shift[qb]:0;
#pragma unroll
 for(int nt=0;nt<4;++nt){int a=tile*32+nt*8+2*lid,b=a+1;
  if(r0<24){if(a<samples)scores[(head*24+r0)*samples+a]=dots[nt][0]*.0625f+s0;if(b<samples)scores[(head*24+r0)*samples+b]=dots[nt][1]*.0625f+s0;}
  if(r1<24){if(a<samples)scores[(head*24+r1)*samples+a]=dots[nt][2]*.0625f+s1;if(b<samples)scores[(head*24+r1)*samples+b]=dots[nt][3]*.0625f+s1;}
 }
}
template<bool Indexed>
__global__ void allkey(const __half* q,const float* qs,const int* positions,
 const unsigned* keys,const __half* linear_keys,const __half* linear_values,const unsigned* coords,
 const float* threshold,int count,int old_end,int blocks,int parts,
 unsigned* masks,float* output,float4* state,int capacity,const int* indices,const int* union_counts,const int* failed){
 int warp=threadIdx.x>>5,lane=threadIdx.x&31,gid=lane>>2,lid=lane&3;
 int work=blockIdx.x*4+warp;if(work>=4*parts*2)return;
 int tile=work%2,part=(work/2)%parts,head=work/(parts*2),r0=tile*16+gid,r1=r0+8;
 const int* qpos=positions+(Indexed?head*4:0);
 const int count_here=Indexed?(failed[head]?0:qpos[3]+1):count;
 int qa=query_index(r0,head),qb=query_index(r1,head),pos0=r0<24?qpos[r0/6]:-1,pos1=r1<24?qpos[r1/6]:-1;
 auto source_key=[&](int slot){if constexpr(Indexed)return slot<union_counts[head]?indices[head*65536+slot]:old_end+slot-union_counts[head];else return slot;};
 float qs0=r0<24?qs[qa]:0,qs1=r1<24?qs[qb]:0;
 float cut0=!Indexed&&r0<24?threshold[head*24+r0]:CUDART_INF_F,cut1=!Indexed&&r1<24?threshold[head*24+r1]:CUDART_INF_F;
 unsigned qr[8][2];
#pragma unroll
 for(int kk=0;kk<8;++kk){
  qr[kk][0]=r0<24?__ldg(reinterpret_cast<const unsigned*>(q+qa*64+kk*8+2*lid)):0;
  qr[kk][1]=r1<24?__ldg(reinterpret_cast<const unsigned*>(q+qb*64+kk*8+2*lid)):0;
 }
 float accumulator[4][4]{};float m0=-CUDART_INF_F,m1=-CUDART_INF_F,l0=0,l1=0,old0=0,old1=0;
 int begin=part*KeysPerPart,end=min(begin+KeysPerPart,count_here);
 for(int first=begin;first<end;first+=32){
  int block=first/32;float dot[4][4]{};
#pragma unroll
  for(int kk=0;kk<8;++kk){
#pragma unroll
   for(int nt=0;nt<4;++nt){unsigned operand=0;int slot=first+nt*8+gid;
    if constexpr(Indexed){if(slot<count_here){int source=source_key(slot);
      const auto at=(((static_cast<std::size_t>(head)*blocks+source/32)*8+kk)*4+(source%32)/8)*32+(source%8)*4+lid;
      operand=__ldg(keys+at);}}
    else operand=__ldg(keys+(((static_cast<std::size_t>(head)*blocks+block)*8+kk)*4+nt)*32+lane);
    mma_f16_k8(dot[nt][0],dot[nt][1],dot[nt][2],dot[nt][3],qr[kk][0],qr[kk][1],operand);
   }
  }
  float score[4][4],bm0=-CUDART_INF_F,bm1=-CUDART_INF_F;unsigned mask0=0,mask1=0;
#pragma unroll
  for(int nt=0;nt<4;++nt){
   int a=first+nt*8+2*lid,b=a+1;
   score[nt][0]=r0<24&&a<count_here&&a<=pos0?dot[nt][0]*.0625f+qs0:-CUDART_INF_F;
   score[nt][1]=r0<24&&b<count_here&&b<=pos0?dot[nt][1]*.0625f+qs0:-CUDART_INF_F;
   score[nt][2]=r1<24&&a<count_here&&a<=pos1?dot[nt][2]*.0625f+qs1:-CUDART_INF_F;
   score[nt][3]=r1<24&&b<count_here&&b<=pos1?dot[nt][3]*.0625f+qs1:-CUDART_INF_F;
   bm0=fmaxf(bm0,fmaxf(score[nt][0],score[nt][1]));bm1=fmaxf(bm1,fmaxf(score[nt][2],score[nt][3]));
   if(a<old_end&&score[nt][0]>=cut0)mask0|=1U<<(nt*8+2*lid);
   if(b<old_end&&score[nt][1]>=cut0)mask0|=1U<<(nt*8+2*lid+1);
   if(a<old_end&&score[nt][2]>=cut1)mask1|=1U<<(nt*8+2*lid);
   if(b<old_end&&score[nt][3]>=cut1)mask1|=1U<<(nt*8+2*lid+1);
  }
  mask0|=__shfl_xor_sync(Mask,mask0,1);mask0|=__shfl_xor_sync(Mask,mask0,2);
  mask1|=__shfl_xor_sync(Mask,mask1,1);mask1|=__shfl_xor_sync(Mask,mask1,2);
  if constexpr(!Indexed){  if(lid==0){if(r0<24)masks[(head*24+r0)*blocks+block]=mask0;if(r1<24)masks[(head*24+r1)*blocks+block]=mask1;}}
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
#pragma unroll
   for(int nt=0;nt<4;++nt){unsigned value=0;
    if constexpr(Indexed){int k0=first+nt*8+2*lid,k1=k0+1;unsigned a0=0,a1=0;
      if(k0<count_here){int source=source_key(k0);const auto at=(((static_cast<std::size_t>(head)*blocks+source/32)*4+n)*4+(source%32)/8)*32+gid*4+(source%8)/2;a0=(__ldg(coords+at)>>(16*(source&1)))&65535U;}
      if(k1<count_here){int source=source_key(k1);const auto at=(((static_cast<std::size_t>(head)*blocks+source/32)*4+n)*4+(source%32)/8)*32+gid*4+(source%8)/2;a1=(__ldg(coords+at)>>(16*(source&1)))&65535U;}
      value=a0|(a1<<16);
    }else value=__ldg(coords+(((static_cast<std::size_t>(head)*blocks+block)*4+n)*4+nt)*32+lane);
    mma_f16_k8(a,b,c,d,pwords[nt][0],pwords[nt][1],value);
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

}
