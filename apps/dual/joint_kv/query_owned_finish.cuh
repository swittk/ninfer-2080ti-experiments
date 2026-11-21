#pragma once
#include "runtime_kernels.cuh"
namespace ninfer::apps::dual::joint_runtime {
// One-time source-derived permutation. Staging is supplied by the existing
// preparation workspace, so persistent allocation and graph addresses do not move.
__global__ void transpose_value_basis_for_queries(const std::uint16_t* source,
                                                  std::uint16_t* destination){
 const int i=blockIdx.x*blockDim.x+threadIdx.x;
 if(i<4*256*32){int head=i/(256*32),rank=(i/256)%32,d=i%256;
  destination[i]=source[(head*256+d)*32+rank];}
}

// Same reduction trees and nonlinear boundaries as combine_coordinates_parallel,
// normalize_exact_subset, then finish_local_state. No intermediate global writes.
__global__ void query_owned_finish(const float* partial,const float4* state,int parts,
 const __nv_bfloat16* exact_partial,const float* exact_maxima,const float* exact_denominator,
 int splits,const __half* basis_by_coordinate,const __half* mean,int* failed,float* out,float* out_m,float* out_l){
 using namespace ninfer::ops::value_residual;
 const int q=blockIdx.x,t=threadIdx.x,lane=t&31,warp=t>>5,head=q/24;
 const int qa=query_index(q%24,head);
 __shared__ int skip;
 if(t==0)skip=failed[head];__syncthreads();if(skip)return;
 __shared__ float max8[8],vectors[8][32],dens[8],maximum,coordinate[32],all_max,all_den;
 __shared__ float ms[128],ls[128],factors[128],exact_max,exact_den;
 float m=-CUDART_INF_F;
 for(int s=t;s<parts;s+=256)m=fmaxf(m,state[q*parts+s].x);
 m=warp_max(m,Mask);if(lane==0)max8[warp]=m;
 if(t<splits){auto at=gqa_partial_stat_index<G>(qa%24,qa/24,t,4);ms[t]=exact_maxima[at];ls[t]=exact_denominator[at];}
 __syncthreads();
 if(t==0){
  m=-CUDART_INF_F;for(int w=0;w<8;++w)m=fmaxf(m,max8[w]);maximum=m;
  float m2=-CUDART_INF_F;for(int s=0;s<splits;++s)m2=fmaxf(m2,ms[s]);exact_max=m2;

 }
 __syncthreads();
 // The factors are independent. Compute them beside coordinate accumulation,
 // but retain the exact original sequential denominator FMA below.
 if(t<splits)factors[t]=ms[t]>-CUDART_INF_F?exp2_approx((ms[t]-exact_max)*Log2E):0;
 float value=0,den=0;
 for(int s=warp;s<parts;s+=8){float local_m=0,local_l=0;
  if(lane==0){const float4 v=state[q*parts+s];local_m=v.x;local_l=v.y;}
  local_m=__shfl_sync(Mask,local_m,0);local_l=__shfl_sync(Mask,local_l,0);
  float factor=local_m>-CUDART_INF_F?exp2_approx((local_m-maximum)*Log2E):0;
  value=__fmaf_rn(partial[(q*parts+s)*32+lane],factor,value);
  den=__fmaf_rn(local_l,factor,den);
 }
 vectors[warp][lane]=value;if(lane==0)dens[warp]=den;__syncthreads();
 if(warp==0){value=0;den=0;for(int w=0;w<8;++w){value+=vectors[w][lane];den+=dens[w];}
  coordinate[lane]=den>0?value/den:0;if(lane==0){all_max=maximum;all_den=den;}
 }
 if(t==0){float sum=0;for(int s=0;s<splits;++s)sum=__fmaf_rn(ls[s],factors[s],sum);exact_den=sum;}
 __syncthreads();
 const float combined_m=fmaxf(all_max,exact_max);
 if(combined_m==-CUDART_INF_F){out[qa*256+t]=0;if(t==0){out_m[qa]=-CUDART_INF_F;out_l[qa]=0;}return;}
 const float wr=all_max>-CUDART_INF_F?all_den*exp2_approx((all_max-combined_m)*Log2E):0;
 const float we=exact_max>-CUDART_INF_F?exact_den*exp2_approx((exact_max-combined_m)*Log2E):0;
 const float combined_den=wr+we;
 if(!(combined_den>0)||!isfinite(combined_den)){if(t==0)atomicOr(failed+head,1);return;}
 float exact_value=0;
 for(int s=0;s<splits;++s)exact_value=__fmaf_rn(__bfloat162float(exact_partial[gqa_partial_acc_index<G>(qa%24,t,qa/24,s,4)]),factors[s],exact_value);
 exact_value=exact_den>0?exact_value/exact_den:0;
 float result=__half2float(mean[head*256+t]);
 for(int k=0;k<32;++k)result=__fmaf_rn(coordinate[k],__half2float(basis_by_coordinate[(head*32+k)*256+t]),result);
 // Match the existing kernel's multiply-then-FMA choice explicitly. Fusion
 // otherwise lets the compiler choose the opposite (one-bit-different) contraction.
 result=__fmaf_rn(wr/combined_den,result,__fmul_rn(we/combined_den,exact_value));
 if(!isfinite(result)){atomicOr(failed+head,1);return;}out[qa*256+t]=result;
 if(t==0){out_m[qa]=combined_m+logf(combined_den);out_l[qa]=1;}
}
}
