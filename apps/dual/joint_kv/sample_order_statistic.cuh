#pragma once
#include <cuda_runtime.h>
#include <cstdint>
namespace ninfer::ops::value_residual {
// Exact order statistic, not an approximate histogram threshold. Only the
// containing radix bin survives each pass; complete sorting is unnecessary.
__global__ void sample_order_statistic(const float* samples,int width,int requested,float* output){
 constexpr unsigned Full=0xffffffffu;
 const int row=blockIdx.x,t=threadIdx.x,lane=t&31,warp=t>>5;
 unsigned values[64];unsigned lower=Full,upper=0;
#pragma unroll
 for(int j=0;j<64;++j){int at=t+j*256;unsigned value=0;
  if(at<width){unsigned raw=__float_as_uint(samples[row*width+at]);value=(raw&0x80000000u)?~raw:(raw^0x80000000u);lower=min(lower,value);upper=max(upper,value);}
  values[j]=value;
 }
 for(int off=16;off;off>>=1){lower=min(lower,__shfl_xor_sync(Full,lower,off));upper=max(upper,__shfl_xor_sync(Full,upper,off));}
 __shared__ unsigned lo[8],hi[8],hist[256],prefix,active_mask,next_prefix;
 __shared__ int shift,order,next_order,warp_totals[8];
 if(lane==0){lo[warp]=lower;hi[warp]=upper;}__syncthreads();
 if(t==0){lower=Full;upper=0;for(int w=0;w<8;++w){lower=min(lower,lo[w]);upper=max(upper,hi[w]);}
  unsigned diff=lower^upper;shift=diff?((31-__clz(diff))/8)*8:-1;order=requested;
  active_mask=shift<0?Full:shift==24?0u:(Full<<(shift+8));prefix=upper&active_mask;
 }
 __syncthreads();
 if(shift<0){if(t==0)output[row]=__uint_as_float((prefix&0x80000000u)?prefix^0x80000000u:~prefix);return;}
 for(int pass=0;pass<4;++pass){
  hist[t]=0;__syncthreads();
  unsigned match_mask=active_mask,match=prefix;int bit=shift;
#pragma unroll
  for(int j=0;j<64;++j){unsigned value=values[j];bool eligible=t+j*256<width&&(value&match_mask)==match;
   unsigned active=__ballot_sync(Full,eligible);
   if(eligible){unsigned bin=(value>>bit)&255u;unsigned peers=__match_any_sync(active,bin);
    if(lane==__ffs(peers)-1)atomicAdd(hist+bin,__popc(peers));}
  }
  __syncthreads();
  const int count=static_cast<int>(hist[255-t]);int sum=count;
  for(int off=1;off<32;off<<=1){int other=__shfl_up_sync(Full,sum,off);if(lane>=off)sum+=other;}
  if(lane==31)warp_totals[warp]=sum;__syncthreads();
  int preceding=sum-count;for(int w=0;w<warp;++w)preceding+=warp_totals[w];
  if(order>preceding&&order<=preceding+count){next_prefix=prefix|(static_cast<unsigned>(255-t)<<bit);next_order=order-preceding;}
  __syncthreads();
  if(t==0){prefix=next_prefix;active_mask|=255u<<bit;order=next_order;shift=bit-8;}
  __syncthreads();
  if(shift<0){if(t==0)output[row]=__uint_as_float((prefix&0x80000000u)?prefix^0x80000000u:~prefix);return;}
 }
}
}
