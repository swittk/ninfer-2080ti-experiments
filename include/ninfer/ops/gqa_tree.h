#pragma once

#include "core/tensor.h"

#include <cuda_runtime_api.h>

namespace ninfer::ops {

/**
 * Compute exact online-softmax state for a small parent-indexed verifier tree tail.
 *
 * q is contiguous BF16 [256,24,N], k/v are BF16 [256,4,N], and parent_ids is device I32 [N].
 * Every node must be topologically ordered: parent is -1 or in [0,node). Each query node attends
 * only to its root-to-self ancestry. Outputs are unnormalized FP32 acc [256,24,N], m/l [24,N],
 * suitable for exact ordered merge with a committed-prefix attention state. N must be in [1,32].
 * The inputs and committed-prefix state are read-only.
 */
void gqa_attention_tree_tail_state(const Tensor& q, const Tensor& k, const Tensor& v,
                                   const Tensor& parent_ids, float scale, Tensor& state_acc,
                                   Tensor& state_m, Tensor& state_l, cudaStream_t stream);

/**
 * Merge a committed-prefix online-softmax state with one ancestor-only BF16 tree-tail state.
 *
 * Prefix and tail acc are contiguous FP32 [256,24,N], m/l are FP32 [24,N], and out is BF16
 * [256,24,N]. The ordered merge is mathematically exact modulo the established FP32/BF16
 * numerical contract and moves no prefix KV history.
 */
void gqa_attention_tree_merge_prefix(const Tensor& prefix_acc, const Tensor& prefix_m,
                                     const Tensor& prefix_l, const Tensor& tail_acc,
                                     const Tensor& tail_m, const Tensor& tail_l, Tensor& out,
                                     cudaStream_t stream);

} // namespace ninfer::ops
