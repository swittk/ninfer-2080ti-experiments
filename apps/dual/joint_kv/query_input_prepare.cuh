#pragma once
#include "value_residual.cuh"
namespace ninfer::ops::joint_residual {
// Exact composition of quantize_queries, project_query, then
// quantize_projected_queries; the intermediate I8 query has CTA lifetime only.
__global__ void prepare_integer_query_roles(const __nv_bfloat16* input,const __half* basis,
 const __half* mean,__half* projected,float* shift,std::int8_t* output,float* output_scale,
 unsigned* union_mask,int union_words,int* metadata){
 constexpr int Rank=64;
 const int row=blockIdx.x,t=threadIdx.x,lane=t&31,slice=t/64,d=t%64,head=(row%24)/6;
 // The next kernel is the first consumer. Disjoint stores can be owned by this
 // producer grid rather than separate memset graph nodes, with identical zeros.
 const int global=blockIdx.x*blockDim.x+t,stride=gridDim.x*blockDim.x;
 for(int i=global;i<union_words;i+=stride)union_mask[i]=0;
 if(global<104)metadata[global]=0;
 __shared__ std::int8_t codes[256];
 __shared__ float query_scale,partial[4][64],means[4];
 __shared__ __half projection[64];
 if(t>=256){
  float x[8],maximum=0;
#pragma unroll
  for(int j=0;j<8;++j){x[j]=__bfloat162float(input[row*256+lane+j*32]);maximum=fmaxf(maximum,fabsf(x[j]));}
  maximum=warp_max(maximum,0xffffffffU);
  const float scale=maximum>0?maximum/127.f:0,inverse=scale>0?1.f/scale:0;
#pragma unroll
  for(int j=0;j<8;++j)codes[lane+j*32]=gqa_kv_quant_code(x[j],inverse);
  if(lane==0)query_scale=scale;
 }
 __syncthreads();
 float dot=0;
 // Eight projection warps do not serialize the independent mean projection.
 // One producer warp owns quantization and the four ordered mean dot products.
 if(t<256){
  for(int j=0;j<64;++j){const int k=slice*64+j;
   dot=__fmaf_rn(float(codes[k]),__half2float(basis[(head*256+k)*Rank+d]),dot);
  }
  partial[slice][d]=dot;
 }else if(lane<4){
  float mu=0;for(int j=0;j<64;++j){const int k=lane*64+j;mu=__fmaf_rn(float(codes[k]),__half2float(mean[head*256+k]),mu);}
  means[lane]=mu;
 }
 __syncthreads();
 if(slice==0){
  dot=(partial[0][d]+partial[1][d])+(partial[2][d]+partial[3][d]);
  const __half value=__float2half_rn(dot*query_scale);
  projection[d]=value;projected[row*64+d]=value;
  if(d==0)shift[row]=((means[0]+means[1])+(means[2]+means[3]))*query_scale*.0625f;
 }
 __syncthreads();
 if(t>=256){
  const float a=__half2float(projection[lane]),b=__half2float(projection[lane+32]);
  const float maximum=warp_max(fmaxf(fabsf(a),fabsf(b)),0xffffffffU);
  const float scale=maximum/127.f,inverse=scale>0?1.f/scale:0;
  output[row*64+lane]=static_cast<std::int8_t>(max(-127,min(127,__float2int_rn(a*inverse))));
  output[row*64+lane+32]=static_cast<std::int8_t>(max(-127,min(127,__float2int_rn(b*inverse))));
  if(lane==0)output_scale[row]=scale;
 }
}
}
