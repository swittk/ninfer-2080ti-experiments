#pragma once
#include "value_residual.cuh"

namespace ninfer::ops::value_residual {
__global__ void combine_coordinates_parallel(const float* partial,const float4* state,int parts,float* coordinates,float4* global){
 int q=blockIdx.x,t=threadIdx.x,lane=t&31,warp=t>>5;
 __shared__ float maxima[8],vector[8][32],denominator[8],oldmass[8],maximum;
 float local=-CUDART_INF_F;for(int s=t;s<parts;s+=256)local=fmaxf(local,state[q*parts+s].x);
 local=warp_max(local,Mask);if(lane==0)maxima[warp]=local;__syncthreads();
 if(t==0){local=-CUDART_INF_F;for(int w=0;w<8;++w)local=fmaxf(local,maxima[w]);maximum=local;}__syncthreads();
 float value=0,den=0,old=0;
 for(int s=warp;s<parts;s+=8){float m=0,l=0,o=0;
  if(lane==0){float4 st=state[q*parts+s];m=st.x;l=st.y;o=st.z;}
  m=__shfl_sync(Mask,m,0);l=__shfl_sync(Mask,l,0);o=__shfl_sync(Mask,o,0);
  float factor=m>-CUDART_INF_F?exp2_approx((m-maximum)*Log2E):0;
  value=__fmaf_rn(partial[(q*parts+s)*32+lane],factor,value);
  den=__fmaf_rn(l,factor,den);old=__fmaf_rn(o,factor,old);
 }
 vector[warp][lane]=value;if(lane==0){denominator[warp]=den;oldmass[warp]=old;}__syncthreads();
 if(warp==0){value=0;den=0;old=0;for(int w=0;w<8;++w){value+=vector[w][lane];den+=denominator[w];old+=oldmass[w];}
  coordinates[q*32+lane]=den>0?value/den:0;if(lane==0)global[q]=make_float4(maximum,den,den>0?old/den:0,0);
 }
}
}
