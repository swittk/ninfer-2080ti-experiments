#pragma once
#include "partition_scan.cuh"
#include "sample_order_statistic.cuh"
#include "ordered_union.cuh"
#include "parallel_coordinate_reduce.cuh"
#include "indexed_subset_attention.cuh"
#include "joint_combine.cuh"
namespace ninfer::apps::dual::joint_runtime {
using namespace ninfer::ops;
constexpr int MaxTail=4096,UnionLimit=ordered_union::kUnionLimit,SubsetRows=UnionLimit+MaxTail,SubsetPages=SubsetRows/64;
constexpr int LocalSubsetSplits=64;
__global__ void decode_plane(const std::int8_t* input,const __half* scales,const int* table,int head,int begin,int count,bool value,float* matrix){
 const auto i=static_cast<std::size_t>(blockIdx.x)*blockDim.x+threadIdx.x;if(i>=std::size_t(count)*256)return;
 int logical=begin+i/256,d=i%256,ph=table[logical/64];
 float x=0;
 if(ph>=0){const auto ix=gqa_kv_quant_code_index<Gqa27Geometry>(ph,head,d,logical%64);const auto si=gqa_kv_quant_scale_index<Gqa27Geometry>(ph,head,d/64,logical%64);
  x=float(input[ix])*__half2float(scales[si]);if(value)x=__half2float(__float2half_rn(x));}
 matrix[i]=x;
}
__global__ void moment_blocks(const float* matrix,int rows,double* output){
 int d=threadIdx.x,begin=blockIdx.x*256,end=min(begin+256,rows);double sum=0;
 for(int i=begin;i<end;++i)sum+=double(matrix[i*256+d]);output[blockIdx.x*256+d]=sum;
}
__global__ void finish_moments(const double* partial,int blocks,double* output){int d=threadIdx.x;double sum=0;for(int b=0;b<blocks;++b)sum+=partial[b*256+d];output[d]=sum;}
__global__ void center_matrix(float* matrix,const __half* mean,int rows){
 const auto i=static_cast<std::size_t>(blockIdx.x)*blockDim.x+threadIdx.x;if(i<std::size_t(rows)*256)matrix[i]-=__half2float(mean[i%256]);
}
__global__ void basis_to_float(const __half* basis,int count,float* output){int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<count)output[i]=__half2float(basis[i]);}
// cuSolver returns column-major double eigenvectors. Select the fixed largest
// eigenmodes, canonicalize each sign and convert to row-major FP16 once.
__global__ void select_basis(const double* vectors,int rank,__half* half_basis){
 int feature=blockIdx.x,d=threadIdx.x;if(feature>=rank)return;
 const double* v=vectors+(256-rank+feature)*256;__shared__ double sign;
 if(d==0){int best=0;for(int i=1;i<256;++i)if(fabs(v[i])>fabs(v[best]))best=i;sign=v[best]<0?-1:1;}
 __syncthreads();half_basis[d*rank+feature]=__float2half_rn(float(v[d]*sign));
}
template<int Rank,bool Key>
__global__ void pack_projected(const float* matrix,int begin,int rows,int head,int global_blocks,unsigned* output,int storage_begin=0){
 const auto i=static_cast<std::size_t>(blockIdx.x)*blockDim.x+threadIdx.x;
 constexpr int Words=Rank*16;
 if(i>=std::size_t(rows/32)*Words)return;
 int lane=i%32,nt=(i/32)%4,n=(i/128)%(Rank/8),block=i/Words;
 unsigned word=0;
 if constexpr(Key){int row=block*32+nt*8+lane/4,feature=n*8+2*(lane%4);
  word=unsigned(__half_as_ushort(__float2half_rn(matrix[row*Rank+feature])))|(unsigned(__half_as_ushort(__float2half_rn(matrix[row*Rank+feature+1])))<<16);
 }else{int row=block*32+nt*8+2*(lane%4),feature=n*8+lane/4;
  word=unsigned(__half_as_ushort(__float2half_rn(matrix[row*Rank+feature])))|(unsigned(__half_as_ushort(__float2half_rn(matrix[(row+1)*Rank+feature])))<<16);
 }
 const auto at=((static_cast<std::size_t>(head)*global_blocks+begin/32-storage_begin+block)*(Rank/8)+n)*128+nt*32+lane;output[at]=word;
}
__global__ void sample_from_partition(const unsigned* keys,const int* table,int blocks,int old_end,int samples,unsigned* out,int storage_begin=0){
 const auto i=static_cast<std::size_t>(blockIdx.x)*blockDim.x+threadIdx.x;int tiles=(samples+31)/32;
 if(i>=std::size_t(4)*tiles*1024)return;
 int lane=i%32,nt=(i/32)%4,kk=(i/128)%8,tile=(i/1024)%tiles,head=i/(1024*tiles),sample=tile*32+nt*8+lane/4;
 unsigned word=0;
 if(sample<samples){int source=static_cast<int>(static_cast<long long>(sample)*old_end/samples);if(table[source/64]>=0){auto at=(((static_cast<std::size_t>(head)*blocks+source/32-storage_begin)*8+kk)*4+(source%32)/8)*32+(source%8)*4+lane%4;word=keys[at];}}
 out[i]=word;
}
__global__ void count_local_masks(const unsigned* masks,int blocks,int* counts,int* failed){
 int q=blockIdx.x,t=threadIdx.x,warp=t>>5,lane=t&31;int count=0;for(int b=t;b<blocks;b+=256)count+=__popc(masks[q*blocks+b]);
 for(int off=16;off;off>>=1)count+=__shfl_down_sync(0xffffffffu,count,off);__shared__ int sums[8];if(lane==0)sums[warp]=count;__syncthreads();
 if(t==0){int total=0;for(int w=0;w<8;++w)total+=sums[w];counts[q]=total;if(total>4096)atomicOr(failed+q/24,1);}
}
__global__ void union_masks_local(const unsigned* input,int blocks,unsigned* output){int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<4*blocks){int h=i/blocks,b=i%blocks;unsigned mask=0;for(int q=0;q<24;++q)mask|=input[(h*24+q)*blocks+b];output[i]=mask;}}
__global__ void subset_positions(const int* positions,const int* counts,int tail_begin,int range_end,int* output){int i=threadIdx.x;if(i<16)output[i]=counts[i/4]+max(0,min(positions[i%4]+1,range_end)-tail_begin)-1;}
// Store a normalized FP32 value vector and log partition. The existing
// two-device merge can consume these as (acc=normalized,m=logZ,l=1).
__global__ void finish_local_state(const float* coords,const float4* approx,const float* exact_value,const float4* exact,
 const __half* basis,const __half* mean,int* failed,float* out,float* out_m,float* out_l){
 int q=blockIdx.x,d=threadIdx.x,h=q/24,row=q%24,qa=value_residual::query_index(row,h);__shared__ int skip;
 if(d==0)skip=failed[h];__syncthreads();if(skip)return;
 float4 r=approx[q],e=exact[q];float m=fmaxf(r.x,e.x);
 if(m==-CUDART_INF_F){out[qa*256+d]=0;if(d==0){out_m[qa]=-CUDART_INF_F;out_l[qa]=0;}return;}
 float wr=r.x>-CUDART_INF_F?r.y*exp2_approx((r.x-m)*value_residual::Log2E):0;
 float we=e.x>-CUDART_INF_F?e.y*exp2_approx((e.x-m)*value_residual::Log2E):0;
 float den=wr+we;if(!(den>0)||!isfinite(den)){if(d==0)atomicOr(failed+h,1);return;}
 float v=__half2float(mean[h*256+d]);for(int k=0;k<32;++k)v=__fmaf_rn(coords[q*32+k],__half2float(basis[(h*256+d)*32+k]),v);
 v=(wr/den)*v+(we/den)*exact_value[q*256+d];
 if(!isfinite(v)){atomicOr(failed+h,1);return;}out[qa*256+d]=v;
 if(d==0){out_m[qa]=m+logf(den);out_l[qa]=1;}
}
}
