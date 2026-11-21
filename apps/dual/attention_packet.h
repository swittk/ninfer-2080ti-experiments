#pragma once
#include <cstddef>
#include <stdexcept>
namespace ninfer::apps::dual::attention_packet {
inline constexpr int HeadDim=256, QueryHeads=24, KvHeads=4, MaximumTokens=6;
struct InputLayout {
    std::size_t query_bytes, position_offset, position_bytes, key_offset, value_offset, kv_bytes, bytes;
};
constexpr InputLayout input_layout(int tokens,bool include_kv) {
    if(tokens<1||tokens>MaximumTokens)throw std::invalid_argument("attention packet requires 1..6 columns");
    const std::size_t q=std::size_t(tokens)*QueryHeads*HeadDim*2;
    const std::size_t positions=std::size_t(tokens)*4;
    const std::size_t key=q+((positions+15)/16)*16;
    const std::size_t kv=std::size_t(tokens)*KvHeads*HeadDim*2;
    return {q,q,positions,key,key+kv,kv,include_kv?key+2*kv:key};
}
struct StateLayout {std::size_t numerator_bytes,maximum_offset,normalizer_offset,stat_bytes,bytes;};
constexpr StateLayout state_layout(int tokens) {
    if(tokens<1||tokens>MaximumTokens)throw std::invalid_argument("attention state packet requires 1..6 columns");
    const auto numerator=std::size_t(tokens)*QueryHeads*HeadDim*4;
    const auto stat=std::size_t(tokens)*QueryHeads*4;
    return {numerator,numerator,numerator+stat,stat,numerator+2*stat};
}
} // namespace ninfer::apps::dual::attention_packet
