#pragma once
#include "joint_residual.cuh"
namespace ninfer::ops::joint_residual {
__global__ void normalize_exact_subset(const __nv_bfloat16* partial,const float* maxima,const float* denominator,
 int splits,const int* failed,float* output,float4* global){
 int q=blockIdx.x,d=threadIdx.x,head=q/24,row=q%24,qa=query_index(row,head);
 if(failed[head]){output[q*256+d]=0;if(d==0)global[q]=make_float4(-CUDART_INF_F,0,0,0);return;}
 __shared__ float ms[128],ls[128],factors[128],maximum,den;
 if(d<splits){auto idx=gqa_partial_stat_index<G>(qa%24,qa/24,d,4);ms[d]=maxima[idx];ls[d]=denominator[idx];}__syncthreads();
 if(d==0){float maximum2=-CUDART_INF_F;for(int s=0;s<splits;++s)maximum2=fmaxf(maximum2,ms[s]);maximum=maximum2;
  float sum=0;for(int s=0;s<splits;++s){float f=ms[s]>-CUDART_INF_F?exp2_approx((ms[s]-maximum2)*Log2E):0;factors[s]=f;sum=__fmaf_rn(ls[s],f,sum);}den=sum;global[q]=make_float4(maximum2,sum,0,0);
 }__syncthreads();
 float value=0;for(int s=0;s<splits;++s)value=__fmaf_rn(__bfloat162float(partial[gqa_partial_acc_index<G>(qa%24,d,qa/24,s,4)]),factors[s],value);
 output[q*256+d]=den>0?value/den:0;
}
__global__ void finish_joint(const float* all_coordinates,const float4* all_state,const float* subset_coordinates,
 const float4* subset_state,const float* exact_values,const float4* exact_state,const __half* basis,const __half* mean,
 int* failed,__nv_bfloat16* output){
 int q=blockIdx.x,d=threadIdx.x,head=q/24,row=q%24;
 __shared__ int skip; if(d==0)skip=failed[head];__syncthreads();if(skip)return;
 const float4 a=all_state[q],s=subset_state[q],e=exact_state[q];
 float maximum=fmaxf(a.x,e.x),wa=a.y*exp2_approx((a.x-maximum)*Log2E),ws=s.y*exp2_approx((s.x-maximum)*Log2E),we=e.y*exp2_approx((e.x-maximum)*Log2E);
 float residual=wa-ws,den=fmaxf(0.f,residual)+we;
 if(!(den>0)||!isfinite(den)||residual<-.00001f*fmaxf(wa,1.f)){if(d==0)atomicOr(failed+head,1);return;}
 __shared__ float coordinate[32];
 if(d<32)coordinate[d]=(wa*all_coordinates[q*32+d]-ws*subset_coordinates[q*32+d])/den;
 __syncthreads();float v=exact_values[q*256+d]*(we/den)+__half2float(mean[head*256+d])*(fmaxf(0.f,residual)/den);
 for(int r=0;r<32;++r)v=__fmaf_rn(coordinate[r],__half2float(basis[(head*256+d)*32+r]),v);
 if(!isfinite(v)){atomicOr(failed+head,1);return;}
 output[query_index(row,head)*256+d]=__float2bfloat16_rn(v);
}
}
