#pragma once

#include "core/gdn_replay_records.h"
#include "core/linear_attention_state.h"
#include "core/tensor.h"

#include <cuda_runtime_api.h>

#include <cstdint>
#include <span>

namespace ninfer::ops {

/**
 * Parent-indexed convolution oracle for hybrid GDN tree verification.
 *
 * `projected` is contiguous BF16 [C,N], `conv_weight` is BF16 [C,4],
 * `committed_history` is BF16 [C,3], and `parent_ids` is device I32 [N]. Parents must be in
 * topological order: every entry is -1 (the committed root) or in [0,node). `node_history` is
 * caller-owned BF16 [C,3,N]. Query/key/value are contiguous BF16 [Q,N], [K,N], [V,N] and their
 * row counts must sum to C. The committed history is read-only.
 *
 * This dense per-node history is the correctness-oracle representation. Production tree
 * verification is expected to retain compact raw transition records and replay only the selected
 * root-to-leaf path.
 */
void gdn_projected_conv_tree(const Tensor& projected, const Tensor& conv_weight,
                             const Tensor& committed_history, const Tensor& parent_ids,
                             Tensor& node_history, Tensor& query, Tensor& key, Tensor& value,
                             cudaStream_t stream);

/**
 * Parent-indexed normalized Gated DeltaNet recurrence.
 *
 * q/k are contiguous BF16 [128,Hq,N], v/out are BF16 [128,Hv,N], g/beta are FP32 [Hv,N],
 * committed_state is read-only FP32 [128,128,Hv], parent_ids is device I32 [N], and node_states
 * is caller-owned FP32 [128,128,Hv,N]. Parents follow the same topological contract as the
 * convolution oracle. Hq must be 16 and Hv must be 32 or 48.
 *
 * Every node reloads either committed_state or its parent's dense oracle state, applies one exact
 * normalized GDN transition, writes its output and persists its resulting state. The committed
 * state is never modified.
 */
void gated_delta_net_tree_snapshot(const Tensor& q, const Tensor& k, const Tensor& v,
                                   const Tensor& g, const Tensor& beta, float scale,
                                   const Tensor& committed_state, const Tensor& parent_ids,
                                   Tensor& node_states, Tensor& out, cudaStream_t stream);

/**
 * Gather arbitrary flattened tree nodes into one compact ReplaySSM record row.
 *
 * `path_indices` is device I32 [L]. Sources are flattened tree transition tensors. Destinations are
 * one record row: conv BF16 [C,L], key BF16 [128,Hq,L], value BF16 [128,Hv,L], gate FP32
 * [2,Hv,L]. An identity index vector stores compact factors for all nodes; a root-to-leaf vector
 * stores only a selected path.
 */
void gdn_tree_gather_path_records(const Tensor& projected, const Tensor& key, const Tensor& value,
                                  const Tensor& g, const Tensor& beta,
                                  const Tensor& path_indices, Tensor& conv_record,
                                  Tensor& key_record, Tensor& value_record, Tensor& gate_record,
                                  cudaStream_t stream);

/**
 * Fold one selected root-to-leaf path from all-node compact tree records.
 *
 * `records` stores one transition factor per flattened tree node in topological order. The caller
 * supplies selected node indices in root-to-leaf order after target verification. `record_row`
 * selects the physical record row and `linear_state_slot` selects the committed state slot. Node
 * indices must be unique and in [0, records.spec.width). The record planes remain read-only; the
 * op replays only selected factors, updates recurrent state once, and commits convolution history
 * as tail_3(old_history || selected conv inputs).
 */
void gdn_tree_fold_selected_path(const GdnReplayRecords& records,
                                 LinearAttentionStateAllLayersView states,
                                 std::int32_t record_row,
                                 std::int32_t linear_state_slot,
                                 std::span<const std::int32_t> selected_nodes,
                                 cudaStream_t stream);

} // namespace ninfer::ops
