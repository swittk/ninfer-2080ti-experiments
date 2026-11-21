#pragma once
#include "attention_packet.h"
#include <cuda_runtime.h>
namespace ninfer::apps::dual::attention_packet {
// Copy the original bit patterns into one DMA source. No arithmetic conversion
// or token reordering occurs. Only position alignment padding is initialized.
__global__ void pack_input(const uint4* q,const int* positions,const uint4* k,const uint4* v,
                           uint4* packet,int tokens,bool include_kv) {
    const int i=blockIdx.x*blockDim.x+threadIdx.x;
    const int q_words=tokens*QueryHeads*HeadDim*2/16;
    const int pos_words=(tokens+3)/4;
    const int kv_words=tokens*KvHeads*HeadDim*2/16;
    const int total=q_words+pos_words+(include_kv?2*kv_words:0);
    if(i>=total)return;
    if(i<q_words){packet[i]=q[i];return;}
    const int after_q=i-q_words;
    if(after_q<pos_words){
        const int p=after_q*4;
        packet[i]=make_uint4(p<tokens?static_cast<unsigned>(positions[p]):0U,
            p+1<tokens?static_cast<unsigned>(positions[p+1]):0U,
            p+2<tokens?static_cast<unsigned>(positions[p+2]):0U,
            p+3<tokens?static_cast<unsigned>(positions[p+3]):0U);
        return;
    }
    const int at=after_q-pos_words;
    packet[i]=at<kv_words?k[at]:v[at-kv_words];
}
}
