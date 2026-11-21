#include "ninfer/ops/gdn_tree.h"

#include "core/device.h"
#include "ops/gdn_input_proj/gdn_conv.cuh"
#include "ops/linear_attention/gated_delta_net/launch.h"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>

namespace ninfer::ops {
namespace {

constexpr std::int32_t kStateDim = 128;
constexpr std::int32_t kMaximumTreeNodes = 32;

void require_tensor(const Tensor& tensor, DType dtype, const char* label) {
    if (tensor.data == nullptr || tensor.dtype != dtype || !tensor.is_contiguous()) {
        throw std::invalid_argument(std::string("gdn tree: invalid ") + label);
    }
}

void require_tail_ones(const Tensor& tensor, int first, const char* label) {
    for (int dim = first; dim < 4; ++dim) {
        if (tensor.ne[dim] != 1) {
            throw std::invalid_argument(std::string("gdn tree: invalid ") + label + " rank");
        }
    }
}

std::int32_t require_parent_ids(const Tensor& parent_ids) {
    require_tensor(parent_ids, DType::I32, "parent_ids");
    require_tail_ones(parent_ids, 1, "parent_ids");
    const std::int32_t nodes = parent_ids.ne[0];
    if (nodes <= 0 || nodes > kMaximumTreeNodes) {
        throw std::invalid_argument("gdn tree: node count must be in [1,32]");
    }
    return nodes;
}

void require_nonoverlap(const Tensor& lhs, const Tensor& rhs, const char* label) {
    const auto lhs_begin = reinterpret_cast<std::uintptr_t>(lhs.data);
    const auto rhs_begin = reinterpret_cast<std::uintptr_t>(rhs.data);
    const auto lhs_end = lhs_begin + lhs.bytes();
    const auto rhs_end = rhs_begin + rhs.bytes();
    if (lhs_begin < rhs_end && rhs_begin < lhs_end) {
        throw std::invalid_argument(std::string("gdn tree: overlapping ") + label);
    }
}

__global__ void projected_conv_tree_kernel(
    const __nv_bfloat16* __restrict__ projected,
    const __nv_bfloat16* __restrict__ conv_weight,
    const __nv_bfloat16* __restrict__ committed_history,
    const std::int32_t* __restrict__ parent_ids,
    __nv_bfloat16* __restrict__ node_history,
    __nv_bfloat16* __restrict__ query,
    __nv_bfloat16* __restrict__ key,
    __nv_bfloat16* __restrict__ value,
    std::int32_t channels, std::int32_t query_rows, std::int32_t key_rows,
    std::int32_t value_rows, std::int32_t nodes) {
    const std::int32_t row =
        static_cast<std::int32_t>(blockIdx.x * blockDim.x + threadIdx.x);
    if (row >= channels) { return; }

    const float w0 = __bfloat162float(conv_weight[row]);
    const float w1 = __bfloat162float(conv_weight[channels + row]);
    const float w2 = __bfloat162float(conv_weight[2LL * channels + row]);
    const float w3 = __bfloat162float(conv_weight[3LL * channels + row]);

    for (std::int32_t node = 0; node < nodes; ++node) {
        const std::int32_t parent = parent_ids[node];
        const __nv_bfloat16* history = parent < 0
            ? committed_history
            : node_history + static_cast<std::int64_t>(parent) * 3LL * channels;
        const float s0 = __bfloat162float(history[row]);
        const float s1 = __bfloat162float(history[channels + row]);
        const float s2 = __bfloat162float(history[2LL * channels + row]);
        const float p = __bfloat162float(
            projected[static_cast<std::int64_t>(node) * channels + row]);

        float conv = fmaf(w0, s0, 0.0F);
        conv = fmaf(w1, s1, conv);
        conv = fmaf(w2, s2, conv);
        conv = fmaf(w3, p, conv);
        const __nv_bfloat16 output = __float2bfloat16_rn(silu(conv));
        if (row < query_rows) {
            query[static_cast<std::int64_t>(node) * query_rows + row] = output;
        } else if (row < query_rows + key_rows) {
            key[static_cast<std::int64_t>(node) * key_rows + row - query_rows] = output;
        } else {
            value[static_cast<std::int64_t>(node) * value_rows + row - query_rows - key_rows] =
                output;
        }

        __nv_bfloat16* destination =
            node_history + static_cast<std::int64_t>(node) * 3LL * channels;
        destination[row] = __float2bfloat16_rn(s1);
        destination[channels + row] = __float2bfloat16_rn(s2);
        destination[2LL * channels + row] = __float2bfloat16_rn(p);
    }
}

template <class T>
__global__ void gather_vectors_kernel(const T* __restrict__ source,
                                      const std::int32_t* __restrict__ path,
                                      T* __restrict__ destination,
                                      std::int32_t elements_per_node,
                                      std::int32_t path_length) {
    const std::int64_t total =
        static_cast<std::int64_t>(elements_per_node) * path_length;
    const std::int64_t index =
        static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= total) { return; }
    const std::int32_t path_column = static_cast<std::int32_t>(index / elements_per_node);
    const std::int32_t inner = static_cast<std::int32_t>(index % elements_per_node);
    const std::int32_t source_node = path[path_column];
    destination[index] =
        source[static_cast<std::int64_t>(source_node) * elements_per_node + inner];
}

__global__ void gather_gate_kernel(const float* __restrict__ g,
                                   const float* __restrict__ beta,
                                   const std::int32_t* __restrict__ path,
                                   float* __restrict__ gate_record,
                                   std::int32_t value_heads,
                                   std::int32_t path_length) {
    const std::int64_t total = static_cast<std::int64_t>(value_heads) * path_length;
    const std::int64_t index =
        static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= total) { return; }
    const std::int32_t path_column = static_cast<std::int32_t>(index / value_heads);
    const std::int32_t head = static_cast<std::int32_t>(index % value_heads);
    const std::int32_t source_node = path[path_column];
    const std::int64_t source_index =
        static_cast<std::int64_t>(source_node) * value_heads + head;
    const std::int64_t destination_index = index * 2;
    gate_record[destination_index] = g[source_index];
    gate_record[destination_index + 1] = beta[source_index];
}

} // namespace


void gdn_projected_conv_tree(const Tensor& projected, const Tensor& conv_weight,
                             const Tensor& committed_history, const Tensor& parent_ids,
                             Tensor& node_history, Tensor& query, Tensor& key, Tensor& value,
                             cudaStream_t stream) {
    require_tensor(projected, DType::BF16, "projected");
    require_tensor(conv_weight, DType::BF16, "conv_weight");
    require_tensor(committed_history, DType::BF16, "committed_history");
    require_tensor(node_history, DType::BF16, "node_history");
    require_tensor(query, DType::BF16, "query");
    require_tensor(key, DType::BF16, "key");
    require_tensor(value, DType::BF16, "value");
    const std::int32_t nodes = require_parent_ids(parent_ids);
    const std::int32_t channels = projected.ne[0];
    const std::int32_t query_rows = query.ne[0];
    const std::int32_t key_rows = key.ne[0];
    const std::int32_t value_rows = value.ne[0];
    if (channels <= 0 || channels % 128 != 0 || projected.ne[1] != nodes ||
        projected.ne[2] != 1 || projected.ne[3] != 1 ||
        conv_weight.ne[0] != channels || conv_weight.ne[1] != 4 ||
        conv_weight.ne[2] != 1 || conv_weight.ne[3] != 1 ||
        committed_history.ne[0] != channels || committed_history.ne[1] != 3 ||
        committed_history.ne[2] != 1 || committed_history.ne[3] != 1 ||
        node_history.ne[0] != channels || node_history.ne[1] != 3 ||
        node_history.ne[2] != nodes || node_history.ne[3] != 1 ||
        query_rows <= 0 || key_rows <= 0 || value_rows <= 0 ||
        query_rows + key_rows + value_rows != channels ||
        query.ne[1] != nodes || key.ne[1] != nodes || value.ne[1] != nodes ||
        query.ne[2] != 1 || query.ne[3] != 1 || key.ne[2] != 1 || key.ne[3] != 1 ||
        value.ne[2] != 1 || value.ne[3] != 1) {
        throw std::invalid_argument("gdn tree: projected convolution shape mismatch");
    }
    require_nonoverlap(committed_history, node_history, "conv history");
    require_nonoverlap(projected, node_history, "projected/node history");
    constexpr std::int32_t threads = 256;
    projected_conv_tree_kernel<<<static_cast<unsigned>((channels + threads - 1) / threads),
                                 threads, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(projected.data),
        static_cast<const __nv_bfloat16*>(conv_weight.data),
        static_cast<const __nv_bfloat16*>(committed_history.data),
        static_cast<const std::int32_t*>(parent_ids.data),
        static_cast<__nv_bfloat16*>(node_history.data),
        static_cast<__nv_bfloat16*>(query.data),
        static_cast<__nv_bfloat16*>(key.data),
        static_cast<__nv_bfloat16*>(value.data), channels, query_rows, key_rows, value_rows,
        nodes);
    CUDA_CHECK(cudaGetLastError());
}

void gated_delta_net_tree_snapshot(const Tensor& q, const Tensor& k, const Tensor& v,
                                   const Tensor& g, const Tensor& beta, float scale,
                                   const Tensor& committed_state, const Tensor& parent_ids,
                                   Tensor& node_states, Tensor& out, cudaStream_t stream) {
    require_tensor(q, DType::BF16, "q");
    require_tensor(k, DType::BF16, "k");
    require_tensor(v, DType::BF16, "v");
    require_tensor(g, DType::FP32, "g");
    require_tensor(beta, DType::FP32, "beta");
    require_tensor(committed_state, DType::FP32, "committed_state");
    require_tensor(node_states, DType::FP32, "node_states");
    require_tensor(out, DType::BF16, "out");
    const std::int32_t nodes = require_parent_ids(parent_ids);
    const std::int32_t qk_heads = q.ne[1];
    const std::int32_t value_heads = v.ne[1];
    if (scale <= 0.0F || q.ne[0] != kStateDim || k.ne[0] != kStateDim ||
        v.ne[0] != kStateDim || out.ne[0] != kStateDim || qk_heads != 16 ||
        k.ne[1] != qk_heads || (value_heads != 32 && value_heads != 48) ||
        out.ne[1] != value_heads || value_heads % qk_heads != 0 ||
        q.ne[2] != nodes || k.ne[2] != nodes || v.ne[2] != nodes ||
        out.ne[2] != nodes || q.ne[3] != 1 || k.ne[3] != 1 || v.ne[3] != 1 ||
        out.ne[3] != 1 || g.ne[0] != value_heads || beta.ne[0] != value_heads ||
        g.ne[1] != nodes || beta.ne[1] != nodes || g.ne[2] != 1 || beta.ne[2] != 1 ||
        g.ne[3] != 1 || beta.ne[3] != 1 ||
        committed_state.ne[0] != kStateDim || committed_state.ne[1] != kStateDim ||
        committed_state.ne[2] != value_heads || committed_state.ne[3] != 1 ||
        node_states.ne[0] != kStateDim || node_states.ne[1] != kStateDim ||
        node_states.ne[2] != value_heads || node_states.ne[3] != nodes) {
        throw std::invalid_argument("gdn tree: recurrent shape mismatch");
    }
    require_nonoverlap(committed_state, node_states, "committed/node recurrent state");
    detail::gated_delta_net::launch_recurrent_tree_snapshot(
        q, k, v, g, beta, scale, committed_state, parent_ids, node_states, out, stream);
}

void gdn_tree_gather_path_records(const Tensor& projected, const Tensor& key, const Tensor& value,
                                  const Tensor& g, const Tensor& beta,
                                  const Tensor& path_indices, Tensor& conv_record,
                                  Tensor& key_record, Tensor& value_record, Tensor& gate_record,
                                  cudaStream_t stream) {
    require_tensor(projected, DType::BF16, "projected");
    require_tensor(key, DType::BF16, "key");
    require_tensor(value, DType::BF16, "value");
    require_tensor(g, DType::FP32, "g");
    require_tensor(beta, DType::FP32, "beta");
    require_tensor(path_indices, DType::I32, "path_indices");
    require_tensor(conv_record, DType::BF16, "conv_record");
    require_tensor(key_record, DType::BF16, "key_record");
    require_tensor(value_record, DType::BF16, "value_record");
    require_tensor(gate_record, DType::FP32, "gate_record");
    require_tail_ones(path_indices, 1, "path_indices");
    const std::int32_t nodes = projected.ne[1];
    const std::int32_t path_length = path_indices.ne[0];
    const std::int32_t channels = projected.ne[0];
    const std::int32_t qk_elements = key.ne[0] * key.ne[1];
    const std::int32_t value_elements = value.ne[0] * value.ne[1];
    const std::int32_t value_heads = value.ne[1];
    if (nodes <= 0 || nodes > kMaximumTreeNodes || path_length <= 0 || path_length > nodes ||
        projected.ne[2] != 1 || projected.ne[3] != 1 ||
        key.ne[2] != nodes || key.ne[3] != 1 || value.ne[2] != nodes || value.ne[3] != 1 ||
        key.ne[0] != kStateDim || value.ne[0] != kStateDim || key.ne[1] != 16 ||
        (value_heads != 32 && value_heads != 48) ||
        g.ne[0] != value_heads || beta.ne[0] != value_heads ||
        g.ne[1] != nodes || beta.ne[1] != nodes || g.ne[2] != 1 || beta.ne[2] != 1 ||
        conv_record.ne[0] != channels || conv_record.ne[1] != path_length ||
        conv_record.ne[2] != 1 || conv_record.ne[3] != 1 ||
        key_record.ne[0] != kStateDim || key_record.ne[1] != 16 ||
        key_record.ne[2] != path_length || key_record.ne[3] != 1 ||
        value_record.ne[0] != kStateDim || value_record.ne[1] != value_heads ||
        value_record.ne[2] != path_length || value_record.ne[3] != 1 ||
        gate_record.ne[0] != 2 || gate_record.ne[1] != value_heads ||
        gate_record.ne[2] != path_length || gate_record.ne[3] != 1) {
        throw std::invalid_argument("gdn tree: selected-path record shape mismatch");
    }
    constexpr std::int32_t threads = 256;
    auto launch_bf16 = [&](const Tensor& source, Tensor& destination,
                           std::int32_t elements_per_node) {
        const std::int64_t total =
            static_cast<std::int64_t>(elements_per_node) * path_length;
        gather_vectors_kernel<<<static_cast<unsigned>((total + threads - 1) / threads),
                                threads, 0, stream>>>(
            static_cast<const __nv_bfloat16*>(source.data),
            static_cast<const std::int32_t*>(path_indices.data),
            static_cast<__nv_bfloat16*>(destination.data), elements_per_node, path_length);
        CUDA_CHECK(cudaGetLastError());
    };
    launch_bf16(projected, conv_record, channels);
    launch_bf16(key, key_record, qk_elements);
    launch_bf16(value, value_record, value_elements);
    const std::int64_t gate_total = static_cast<std::int64_t>(value_heads) * path_length;
    gather_gate_kernel<<<static_cast<unsigned>((gate_total + threads - 1) / threads),
                         threads, 0, stream>>>(
        static_cast<const float*>(g.data), static_cast<const float*>(beta.data),
        static_cast<const std::int32_t*>(path_indices.data),
        static_cast<float*>(gate_record.data), value_heads, path_length);
    CUDA_CHECK(cudaGetLastError());
}


void gdn_tree_fold_selected_path(const GdnReplayRecords& records,
                                 LinearAttentionStateAllLayersView states,
                                 std::int32_t record_row,
                                 std::int32_t linear_state_slot,
                                 std::span<const std::int32_t> selected_nodes,
                                 cudaStream_t stream) {
    const GdnReplayRecordSpec& record = records.spec;
    const LinearAttentionStatePoolSpec& state = states.spec;
    const bool geometry_48 = record.layers > 0 && record.layers <= 48 &&
                             record.qk_heads == 16 && record.value_heads == 48 &&
                             record.conv_channels == 10240;
    const bool geometry_30 = record.layers > 0 && record.layers <= 30 &&
                             record.qk_heads == 16 && record.value_heads == 32 &&
                             record.conv_channels == 8192;
    if ((!geometry_48 && !geometry_30) || record.record_capacity <= 0 ||
        record.width <= 0 || record.width > kMaximumTreeNodes ||
        record.key_dim != kStateDim || record.value_dim != kStateDim) {
        throw std::invalid_argument("gdn tree: unsupported selected-fold record geometry");
    }
    if (record_row < 0 || record_row >= record.record_capacity ||
        linear_state_slot < 0 || linear_state_slot >= state.slot_count ||
        selected_nodes.empty() || selected_nodes.size() >
            static_cast<std::size_t>(detail::gated_delta_net::kMaximumTreeFoldPath)) {
        throw std::invalid_argument("gdn tree: selected-fold metadata is out of range");
    }
    for (std::size_t index = 0; index < selected_nodes.size(); ++index) {
        const std::int32_t node = selected_nodes[index];
        if (node < 0 || node >= record.width) {
            throw std::invalid_argument("gdn tree: selected node is out of range");
        }
        if (std::find(selected_nodes.begin(), selected_nodes.begin() +
                      static_cast<std::ptrdiff_t>(index), node) !=
            selected_nodes.begin() + static_cast<std::ptrdiff_t>(index)) {
            throw std::invalid_argument("gdn tree: selected nodes must be unique");
        }
    }
    if (state.layers != static_cast<std::uint32_t>(record.layers) ||
        state.conv_channels != record.conv_channels || state.conv_width != 3 ||
        state.value_heads != record.value_heads || state.value_head_dim != kStateDim ||
        state.key_head_dim != kStateDim || state.conv_dtype != DType::BF16) {
        throw std::invalid_argument("gdn tree: selected-fold state geometry mismatch");
    }
    require_tensor(records.conv, DType::BF16, "tree conv records");
    require_tensor(records.key, DType::BF16, "tree key records");
    require_tensor(records.value, DType::BF16, "tree value records");
    require_tensor(records.gate, DType::FP32, "tree gate records");
    require_tensor(states.conv_layer0, DType::BF16, "tree convolution state");
    require_tensor(states.recurrent_layer0, DType::FP32, "tree recurrent state");
    const std::int32_t outer = record.layers * record.record_capacity;
    if (records.conv.ne[0] != record.conv_channels || records.conv.ne[1] != record.width ||
        records.conv.ne[2] != outer || records.key.ne[0] != kStateDim ||
        records.key.ne[1] != record.qk_heads || records.key.ne[2] != record.width ||
        records.key.ne[3] != outer || records.value.ne[0] != kStateDim ||
        records.value.ne[1] != record.value_heads || records.value.ne[2] != record.width ||
        records.value.ne[3] != outer || records.gate.ne[0] != 2 ||
        records.gate.ne[1] != record.value_heads || records.gate.ne[2] != record.width ||
        records.gate.ne[3] != outer) {
        throw std::invalid_argument("gdn tree: selected-fold record plane mismatch");
    }
    const bool single_layer = record.layers == 1;
    if ((single_layer && (states.conv_layer_stride_bytes != 0 ||
                          states.recurrent_layer_stride_bytes != 0)) ||
        (!single_layer && (states.conv_layer_stride_bytes <= 0 ||
                           states.recurrent_layer_stride_bytes <= 0))) {
        throw std::invalid_argument("gdn tree: selected-fold state stride mismatch");
    }
    detail::gated_delta_net::GdnTreeFoldKernelPath packed{};
    packed.record_row = record_row;
    packed.linear_state_slot = linear_state_slot;
    packed.length = static_cast<std::int32_t>(selected_nodes.size());
    std::copy(selected_nodes.begin(), selected_nodes.end(), packed.node);
    detail::gated_delta_net::launch_tree_replay_fold_selected(
        records, states, packed, stream);
}

} // namespace ninfer::ops
