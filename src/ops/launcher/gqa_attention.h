#pragma once

// ninfer::ops::detail - private launch prototypes for gqa_attention policies.

#include "core/paged_kv_cache.h"
#include "core/tensor.h"
#include "ninfer/ops/gqa_attention.h"

#include <cuda_runtime.h>

#include <cstdint>

namespace ninfer::ops::detail {

enum class GqaAttentionRoute { SmallT, ChunkedSmallT, Prompt };

struct GqaSmallTInvocation {
    const Tensor* valid_columns = nullptr;
    const Tensor* table_rows    = nullptr;
    std::int32_t full_width     = 0;
    std::int32_t column_begin   = 0;
    std::int32_t width          = 0;
    std::int32_t batch_size     = 1;
};

std::int32_t gqa_attention_split_capacity(std::int32_t q_heads, std::int32_t tokens,
                                          DType cache_dtype, GqaExecutionEnvelope envelope);

bool gqa_attention_uses_small_t(std::int32_t tokens);

GqaAttentionRoute gqa_attention_resolve_route(std::int32_t q_heads, std::int32_t width,
                                              std::int32_t batch_size,
                                              GqaExecutionEnvelope envelope);

const char* gqa_attention_route_name(GqaAttentionRoute route);

void gqa_attention_small_t_launch(const Tensor& q, const Tensor& k, const Tensor& v,
                                  const Tensor& positions, const Tensor& valid_columns,
                                  const Tensor& table_rows, float scale,
                                  PagedKVBatchLayerView cache, GqaExecutionEnvelope envelope,
                                  std::int32_t column_begin, std::int32_t width,
                                  Tensor& partial_acc, Tensor& partial_m, Tensor& partial_l,
                                  Tensor& out, cudaStream_t stream);

void gqa_attention_cached_small_t_launch(const Tensor& q, const Tensor& positions, float scale,
                                         const PagedKVLayerView& cache,
                                         GqaExecutionEnvelope envelope, Tensor& partial_acc,
                                         Tensor& partial_m, Tensor& partial_l, Tensor& out,
                                         cudaStream_t stream);

// Exact context-parallel primitives for BF16 and INT8 paged KV. They expose one contiguous
// global split range plus its online-softmax state for deterministic ordered merge.
void gqa_attention_cached_small_t_partial_range_launch(
    const Tensor& q, const Tensor& positions, float scale, const PagedKVLayerView& cache,
    GqaExecutionEnvelope envelope, std::int32_t split_begin, std::int32_t split_count,
    Tensor& partial_acc, Tensor& partial_m, Tensor& partial_l, cudaStream_t stream,
    std::int32_t split_subdivide = 1,
    std::int32_t partition_capacity = 0);

void gqa_attention_small_t_reduce_range_state_launch(
    std::int32_t q_heads, std::int32_t tokens, std::int32_t total_split_count,
    std::int32_t split_begin, std::int32_t split_count, const Tensor& partial_acc,
    const Tensor& partial_m, const Tensor& partial_l, Tensor& state_acc, Tensor& state_m,
    Tensor& state_l, cudaStream_t stream);

void gqa_attention_small_t_merge_two_states_launch(
    std::int32_t q_heads, std::int32_t tokens, const Tensor& state_acc0,
    const Tensor& state_m0, const Tensor& state_l0, const Tensor& state_acc1,
    const Tensor& state_m1, const Tensor& state_l1, Tensor& out, cudaStream_t stream);

void gqa_attention_prompt_launch(const Tensor& q, const Tensor& k, const Tensor& v,
                                 const Tensor& positions, const Tensor& valid_columns,
                                 const Tensor& table_rows, float scale, PagedKVBatchLayerView cache,
                                 Tensor& out, cudaStream_t stream);

// Build a request-local page-aligned sink/recent alias without modifying any
// original KV page or its page table. Query RoPE coordinates remain global.
// Temporary proposal-only retrieval oracle; never supplies a target attention view.
void gqa_proposal_retrieved_view_launch(const Tensor& query,const Tensor& positions,
    const PagedKVBatchLayerView& cache,const Tensor& source_row,std::uint32_t window,
    Tensor& scores,Tensor& normalizers,Tensor& local_positions,Tensor& local_tables,
    Tensor& local_row,cudaStream_t stream);

void gqa_target_guided_proposal_view_launch(const Tensor& guide,const Tensor& guide_position,
    const Tensor& positions,const PagedKVBatchLayerView& cache,const Tensor& source_row,
    std::uint32_t window,Tensor& scores,Tensor& normalizers,Tensor& local_positions,
    Tensor& local_tables,Tensor& local_row,cudaStream_t stream,const Tensor* licensed_columns=nullptr);

void gqa_recent_view_launch(const Tensor& positions,const Tensor& source_tables,
    const Tensor& source_row,std::uint32_t recent_tokens,std::uint32_t sink_tokens,
    Tensor& local_positions,Tensor& local_tables,Tensor& local_row,cudaStream_t stream);

void gqa_kv_append_launch(const Tensor& k, const Tensor& v, const Tensor& positions,
                          PagedKVLayerView cache, cudaStream_t stream);

void gqa_attention_prompt_attention_launch(const Tensor& q, const Tensor& positions, float scale,
                                           const PagedKVLayerView& cache, Tensor& out,
                                           cudaStream_t stream);

} // namespace ninfer::ops::detail
