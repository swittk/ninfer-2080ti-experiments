#pragma once
#include <cuda_runtime.h>
#include <cstddef>
#include <cub/block/block_scan.cuh>
namespace ninfer::ops::ordered_union {
inline constexpr int kUnionLimit = 65536;
inline constexpr int kMaximumWords = 8192;
inline constexpr int kHeads = 4;
inline constexpr int kWordsPerTile = 256;
inline constexpr int tiles_for(int words) { return (words + kWordsPerTile - 1) / kWordsPerTile; }
inline constexpr std::size_t workspace_words(int words) { return std::size_t(kHeads) * (words + tiles_for(words)); }
// There are at most 8192 32-key words per head. Separate tiles expose the
// parallel work instead of assigning every word and output index to one CTA.
__global__ void tile_prefix(const unsigned* masks,int blocks,int* word_prefix,
                            int* tile_counts,const int* failed){
 const int h=blockIdx.y,t=threadIdx.x,tiles=(blocks+255)/256;
 if(failed[h])return;
 const int b=blockIdx.x*256+t;
 const int count=b<blocks?__popc(masks[h*blocks+b]):0;
 using Scan=cub::BlockScan<int,256>;
 __shared__ typename Scan::TempStorage scratch;
 int prefix,total;Scan(scratch).ExclusiveSum(count,prefix,total);
 if(b<blocks)word_prefix[h*blocks+b]=prefix;
 if(t==0)tile_counts[h*tiles+blockIdx.x]=total;
}
__global__ void head_prefix(int blocks,int* tile_counts,int* counts,int* failed){
 const int h=blockIdx.x,t=threadIdx.x,tiles=(blocks+255)/256;
 if(failed[h]){if(t==0)counts[h]=0;return;}
 const int count=t<tiles?tile_counts[h*tiles+t]:0;int sum=count;
 for(int off=1;off<32;off<<=1){int x=__shfl_up_sync(0xffffffffu,sum,off);if(t>=off)sum+=x;}
 if(t<tiles)tile_counts[h*tiles+t]=sum-count;
 if(t==31){counts[h]=sum;if(sum>kUnionLimit)failed[h]=1;}
}
__global__ void scatter(const unsigned* masks,int blocks,const int* word_prefix,
                         const int* tile_prefix,int* indices,const int* failed){
 const int h=blockIdx.y,b=blockIdx.x*256+threadIdx.x,tiles=(blocks+255)/256;
 if(failed[h]||b>=blocks)return;
 int offset=word_prefix[h*blocks+b]+tile_prefix[h*tiles+blockIdx.x];
 unsigned word=masks[h*blocks+b];
 while(word){int bit=__ffs(word)-1;indices[h*kUnionLimit+offset++]=b*32+bit;word&=word-1;}
}
}
