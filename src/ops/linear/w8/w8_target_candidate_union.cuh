#pragma once
#include "ops/kernel/sampling_device.cuh"
#include <cuda_bf16.h>
#include <cstdint>
namespace ninfer::ops::detail::target_union {
// Fill existing padded exact-head rows. Invalid or unused tail proposals never
// become candidates and never produce an out-of-range weight read.
static __global__ void prepare(const std::int32_t* drafts,const std::int32_t* extent,
                               std::int32_t* rows,int vocab) {
    const int i=threadIdx.x;if(i>=16)return;
    const int count=max(0,min(3,extent[0]));
    int id=count?drafts[min(i,count-1)]:0;
    rows[i]=(id>=0 && id<vocab)?id:0;
}
static __global__ void merge(const __nv_bfloat16* exact,const std::int32_t* drafts,
    const std::int32_t* extent,__nv_bfloat16* metadata,std::int32_t* tokens,int vocab) {
    const int col=threadIdx.x;if(col>=4)return;
    const int count=max(0,min(3,extent[0]));if(count==0)return;
    auto* ids=reinterpret_cast<std::int32_t*>(metadata+static_cast<std::int64_t>(col)*vocab);
    auto* scores=metadata+static_cast<std::int64_t>(col)*vocab+256;
    unsigned long long best=0,worst=~0ULL;int weakest=0;
    __nv_bfloat16 best_score{};
    for(int i=0;i<128;++i){
        const auto key=sampling_sort_key(__bfloat162float(scores[i]),ids[i]);
        if(key>best){best=key;best_score=scores[i];}
        if(key<worst){worst=key;weakest=i;}
    }
    const auto original=best;
    for(int i=0;i<count;++i){
        const auto id=drafts[i];if(id<0 || id>=vocab)continue;
        const auto score=exact[col*16+i];
        const auto key=sampling_sort_key(__bfloat162float(score),id);
        if(key>best){best=key;best_score=score;}
    }
    // Retain the current indexed-metadata layout for diagnostics and consumers.
    // Only the union's winner is necessary for strict greedy acceptance.
    if(best!=original){ids[weakest]=sampling_key_index(best);scores[weakest]=best_score;}
    tokens[col]=sampling_key_index(best);
}
} // namespace ninfer::ops::detail::target_union
