#include "ninfer/ops/gdn_tree.h"

#include "core/device.h"
#include "core/gdn_replay_records.h"
#include "core/layout.h"
#include "core/linear_attention_state.h"
#include "ninfer/ops/gated_delta_net.h"
#include "ninfer/ops/gdn_replay.h"
#include "ops/gdn_input_proj/gdn_projected_conv.h"
#include "ops/op_tester.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <span>
#include <string>
#include <vector>

using namespace ninfer;
using namespace ninfer::test;

namespace {

constexpr std::int32_t kStateDim = 128;
constexpr std::int32_t kQkHeads = 16;
constexpr std::int32_t kValueHeads = 48;
constexpr std::int32_t kQueryRows = kStateDim * kQkHeads;
constexpr std::int32_t kKeyRows = kQueryRows;
constexpr std::int32_t kValueRows = kStateDim * kValueHeads;
constexpr std::int32_t kChannels = kQueryRows + kKeyRows + kValueRows;
constexpr std::int32_t kNodes = 6;
constexpr std::int32_t kDepth = 3;
constexpr float kScale = 1.0F / std::sqrt(static_cast<float>(kStateDim));

std::vector<std::uint16_t> make_bf16(std::size_t count, std::uint32_t seed,
                                     float lo = -0.08F, float hi = 0.08F) {
    std::vector<float> values(count);
    fill_uniform(values, seed, lo, hi);
    round_to_bf16(values);
    std::vector<std::uint16_t> bits(count);
    for (std::size_t index = 0; index < count; ++index) {
        bits[index] = f32_to_bf16(values[index]);
    }
    return bits;
}

template <class T>
DeviceBuffer upload(std::span<const T> values) {
    DeviceBuffer buffer(std::max<std::size_t>(sizeof(T), values.size_bytes()));
    if (!values.empty()) { buffer.copy_from_host(values.data(), values.size_bytes()); }
    return buffer;
}

template <class T>
DeviceBuffer upload(const std::vector<T>& values) {
    return upload<T>(std::span<const T>(values));
}

template <class T, std::size_t N>
DeviceBuffer upload(const std::array<T, N>& values) {
    return upload<T>(std::span<const T>(values));
}

template <class T>
std::vector<T> download(const Tensor& tensor) {
    std::vector<T> values(static_cast<std::size_t>(tensor.numel()));
    cuda_check(cudaMemcpy(values.data(), tensor.data, tensor.bytes(), cudaMemcpyDeviceToHost),
               "download tree tensor");
    return values;
}

template <class T>
int compare_slice(const std::string& label, std::span<const T> lhs, std::span<const T> rhs) {
    if (lhs.size() != rhs.size()) {
        std::cerr << label << ": size mismatch\n";
        return 1;
    }
    for (std::size_t index = 0; index < lhs.size(); ++index) {
        if (lhs[index] != rhs[index]) {
            std::cerr << label << ": mismatch at " << index << "\n";
            return 1;
        }
    }
    return 0;
}

template <class T>
int compare_slice(const std::string& label, const std::vector<T>& lhs,
                  std::span<const T> rhs) {
    return compare_slice<T>(label, std::span<const T>(lhs), rhs);
}

template <class T>
int compare_slice(const std::string& label, std::span<const T> lhs,
                  const std::vector<T>& rhs) {
    return compare_slice<T>(label, lhs, std::span<const T>(rhs));
}

template <class T>
int compare_slice(const std::string& label, const std::vector<T>& lhs,
                  const std::vector<T>& rhs) {
    return compare_slice<T>(label, std::span<const T>(lhs), std::span<const T>(rhs));
}

template <class T>
std::span<const T> column(std::span<const T> values, std::int32_t rows, std::int32_t index) {
    return values.subspan(static_cast<std::size_t>(rows) * index, rows);
}

std::vector<std::uint16_t> gather_bf16_columns(const std::vector<std::uint16_t>& source,
                                               std::int32_t rows,
                                               std::span<const std::int32_t> indices) {
    std::vector<std::uint16_t> out(static_cast<std::size_t>(rows) * indices.size());
    for (std::size_t column_index = 0; column_index < indices.size(); ++column_index) {
        const auto source_begin =
            source.begin() + static_cast<std::ptrdiff_t>(rows) * indices[column_index];
        std::copy_n(source_begin, rows,
                    out.begin() + static_cast<std::ptrdiff_t>(rows) * column_index);
    }
    return out;
}

std::vector<float> gather_fp32_columns(const std::vector<float>& source, std::int32_t rows,
                                       std::span<const std::int32_t> indices) {
    std::vector<float> out(static_cast<std::size_t>(rows) * indices.size());
    for (std::size_t column_index = 0; column_index < indices.size(); ++column_index) {
        const auto source_begin =
            source.begin() + static_cast<std::ptrdiff_t>(rows) * indices[column_index];
        std::copy_n(source_begin, rows,
                    out.begin() + static_cast<std::ptrdiff_t>(rows) * column_index);
    }
    return out;
}

struct TreeRun {
    DeviceBuffer projected;
    DeviceBuffer conv_weight;
    DeviceBuffer committed_conv;
    DeviceBuffer parents;
    DeviceBuffer node_history;
    DeviceBuffer query;
    DeviceBuffer key;
    DeviceBuffer value;
    DeviceBuffer g;
    DeviceBuffer beta;
    DeviceBuffer committed_recurrent;
    DeviceBuffer node_states;
    DeviceBuffer output;

    Tensor projected_tensor;
    Tensor conv_weight_tensor;
    Tensor committed_conv_tensor;
    Tensor parent_tensor;
    Tensor node_history_tensor;
    Tensor query_tensor;
    Tensor key_tensor;
    Tensor value_tensor;
    Tensor g_tensor;
    Tensor beta_tensor;
    Tensor committed_recurrent_tensor;
    Tensor node_states_tensor;
    Tensor output_tensor;

    TreeRun(const std::vector<std::uint16_t>& projected_bits,
            const std::vector<std::uint16_t>& conv_weight_bits,
            const std::vector<std::uint16_t>& committed_conv_bits,
            const std::array<std::int32_t, kNodes>& parent_values,
            const std::vector<float>& g_values, const std::vector<float>& beta_values,
            const std::vector<float>& committed_recurrent_values)
        : projected(upload(projected_bits)),
          conv_weight(upload(conv_weight_bits)),
          committed_conv(upload(committed_conv_bits)),
          parents(upload(parent_values)),
          node_history(static_cast<std::size_t>(kChannels) * 3 * kNodes * sizeof(std::uint16_t)),
          query(static_cast<std::size_t>(kQueryRows) * kNodes * sizeof(std::uint16_t)),
          key(static_cast<std::size_t>(kKeyRows) * kNodes * sizeof(std::uint16_t)),
          value(static_cast<std::size_t>(kValueRows) * kNodes * sizeof(std::uint16_t)),
          g(upload(g_values)),
          beta(upload(beta_values)),
          committed_recurrent(upload(committed_recurrent_values)),
          node_states(static_cast<std::size_t>(kStateDim) * kStateDim * kValueHeads * kNodes *
                      sizeof(float)),
          output(static_cast<std::size_t>(kValueRows) * kNodes * sizeof(std::uint16_t)),
          projected_tensor(projected.p, DType::BF16, {kChannels, kNodes}),
          conv_weight_tensor(conv_weight.p, DType::BF16, {kChannels, 4}),
          committed_conv_tensor(committed_conv.p, DType::BF16, {kChannels, 3}),
          parent_tensor(parents.p, DType::I32, {kNodes}),
          node_history_tensor(node_history.p, DType::BF16, {kChannels, 3, kNodes}),
          query_tensor(query.p, DType::BF16, {kQueryRows, kNodes}),
          key_tensor(key.p, DType::BF16, {kKeyRows, kNodes}),
          value_tensor(value.p, DType::BF16, {kValueRows, kNodes}),
          g_tensor(g.p, DType::FP32, {kValueHeads, kNodes}),
          beta_tensor(beta.p, DType::FP32, {kValueHeads, kNodes}),
          committed_recurrent_tensor(committed_recurrent.p, DType::FP32,
                                     {kStateDim, kStateDim, kValueHeads}),
          node_states_tensor(node_states.p, DType::FP32,
                             {kStateDim, kStateDim, kValueHeads, kNodes}),
          output_tensor(output.p, DType::BF16, {kStateDim, kValueHeads, kNodes}) {
        node_history.fill(0xff);
        query.fill(0xff);
        key.fill(0xff);
        value.fill(0xff);
        node_states.fill(0xff);
        output.fill(0xff);
        ops::gdn_projected_conv_tree(projected_tensor, conv_weight_tensor,
                                     committed_conv_tensor, parent_tensor,
                                     node_history_tensor, query_tensor, key_tensor,
                                     value_tensor, nullptr);
        ops::gated_delta_net_tree_snapshot(
            query_tensor.view({kStateDim, kQkHeads, kNodes, 1}),
            key_tensor.view({kStateDim, kQkHeads, kNodes, 1}),
            value_tensor.view({kStateDim, kValueHeads, kNodes, 1}),
            g_tensor, beta_tensor, kScale, committed_recurrent_tensor, parent_tensor,
            node_states_tensor, output_tensor, nullptr);
        cuda_synchronize();
    }
};

struct CanonicalBranch {
    DeviceBuffer projected;
    DeviceBuffer conv_states;
    DeviceBuffer initial;
    DeviceBuffer bases;
    DeviceBuffer query;
    DeviceBuffer key;
    DeviceBuffer value;
    DeviceBuffer g;
    DeviceBuffer beta;
    DeviceBuffer recurrent_states;
    DeviceBuffer output;

    Tensor conv_states_tensor;
    Tensor recurrent_states_tensor;
    Tensor query_tensor;
    Tensor key_tensor;
    Tensor value_tensor;
    Tensor output_tensor;

    CanonicalBranch(const std::vector<std::uint16_t>& tree_projected,
                    const std::vector<std::uint16_t>& conv_weight_bits,
                    const std::vector<std::uint16_t>& committed_conv_bits,
                    const std::vector<float>& tree_g, const std::vector<float>& tree_beta,
                    const std::vector<float>& committed_recurrent_values,
                    std::span<const std::int32_t> path)
        : projected(upload(gather_bf16_columns(tree_projected, kChannels, path))),
          conv_states(static_cast<std::size_t>(kChannels) * 3 * (kDepth + 1) *
                      sizeof(std::uint16_t)),
          initial(upload(std::array<std::int32_t, 1>{0})),
          bases(upload(std::array<std::int32_t, 1>{1})),
          query(static_cast<std::size_t>(kQueryRows) * kDepth * sizeof(std::uint16_t)),
          key(static_cast<std::size_t>(kKeyRows) * kDepth * sizeof(std::uint16_t)),
          value(static_cast<std::size_t>(kValueRows) * kDepth * sizeof(std::uint16_t)),
          g(upload(gather_fp32_columns(tree_g, kValueHeads, path))),
          beta(upload(gather_fp32_columns(tree_beta, kValueHeads, path))),
          recurrent_states(static_cast<std::size_t>(kStateDim) * kStateDim * kValueHeads *
                           (kDepth + 1) * sizeof(float)),
          output(static_cast<std::size_t>(kValueRows) * kDepth * sizeof(std::uint16_t)),
          conv_states_tensor(conv_states.p, DType::BF16,
                             {kChannels, 3, kDepth + 1}),
          recurrent_states_tensor(recurrent_states.p, DType::FP32,
                                  {kStateDim, kStateDim, kValueHeads, kDepth + 1}),
          query_tensor(query.p, DType::BF16, {kQueryRows, kDepth}),
          key_tensor(key.p, DType::BF16, {kKeyRows, kDepth}),
          value_tensor(value.p, DType::BF16, {kValueRows, kDepth}),
          output_tensor(output.p, DType::BF16, {kStateDim, kValueHeads, kDepth}) {
        conv_states.fill(0);
        recurrent_states.fill(0);
        cuda_check(cudaMemcpy(conv_states_tensor.slice(2, 0, 1).data,
                              committed_conv_bits.data(),
                              committed_conv_bits.size() * sizeof(std::uint16_t),
                              cudaMemcpyHostToDevice),
                   "upload canonical root convolution state");
        cuda_check(cudaMemcpy(recurrent_states_tensor.slice(3, 0, 1).data,
                              committed_recurrent_values.data(),
                              committed_recurrent_values.size() * sizeof(float),
                              cudaMemcpyHostToDevice),
                   "upload canonical root recurrent state");
        Tensor projected_tensor(projected.p, DType::BF16, {kChannels, kDepth});
        Tensor weight_tensor(const_cast<std::uint16_t*>(conv_weight_bits.data()), DType::BF16,
                             {kChannels, 4});
        DeviceBuffer device_weight = upload(conv_weight_bits);
        weight_tensor.data = device_weight.p;
        Tensor initial_tensor(initial.p, DType::I32, {1});
        Tensor base_tensor(bases.p, DType::I32, {1});
        ops::detail::gdn_projected_conv_snapshot_launch(
            projected_tensor, weight_tensor, conv_states_tensor, Tensor{}, initial_tensor,
            base_tensor, query_tensor, key_tensor, value_tensor, nullptr);
        Tensor g_tensor(g.p, DType::FP32, {kValueHeads, kDepth});
        Tensor beta_tensor(beta.p, DType::FP32, {kValueHeads, kDepth});
        ops::gated_delta_net_snapshot(
            query_tensor.view({kStateDim, kQkHeads, kDepth, 1}),
            key_tensor.view({kStateDim, kQkHeads, kDepth, 1}),
            value_tensor.view({kStateDim, kValueHeads, kDepth, 1}),
            g_tensor, beta_tensor, kScale, true, recurrent_states_tensor, Tensor{},
            initial_tensor, base_tensor, output_tensor, nullptr);
        cuda_synchronize();
    }
};

int verify_branch(const TreeRun& tree, const CanonicalBranch& branch,
                  std::span<const std::int32_t> path, const std::string& label) {
    const auto tree_q = download<std::uint16_t>(tree.query_tensor);
    const auto tree_k = download<std::uint16_t>(tree.key_tensor);
    const auto tree_v = download<std::uint16_t>(tree.value_tensor);
    const auto tree_history = download<std::uint16_t>(tree.node_history_tensor);
    const auto tree_out = download<std::uint16_t>(tree.output_tensor);
    const auto tree_state = download<float>(tree.node_states_tensor);
    const auto branch_q = download<std::uint16_t>(branch.query_tensor);
    const auto branch_k = download<std::uint16_t>(branch.key_tensor);
    const auto branch_v = download<std::uint16_t>(branch.value_tensor);
    const auto branch_history = download<std::uint16_t>(branch.conv_states_tensor);
    const auto branch_out = download<std::uint16_t>(branch.output_tensor);
    const auto branch_state = download<float>(branch.recurrent_states_tensor);

    int failures = 0;
    const std::size_t history_stride = static_cast<std::size_t>(kChannels) * 3;
    const std::size_t state_stride =
        static_cast<std::size_t>(kStateDim) * kStateDim * kValueHeads;
    for (std::int32_t depth = 0; depth < kDepth; ++depth) {
        const std::int32_t node = path[static_cast<std::size_t>(depth)];
        failures += compare_slice(label + " query depth=" + std::to_string(depth),
                                  column<std::uint16_t>(tree_q, kQueryRows, node),
                                  column<std::uint16_t>(branch_q, kQueryRows, depth));
        failures += compare_slice(label + " key depth=" + std::to_string(depth),
                                  column<std::uint16_t>(tree_k, kKeyRows, node),
                                  column<std::uint16_t>(branch_k, kKeyRows, depth));
        failures += compare_slice(label + " value depth=" + std::to_string(depth),
                                  column<std::uint16_t>(tree_v, kValueRows, node),
                                  column<std::uint16_t>(branch_v, kValueRows, depth));
        failures += compare_slice(
            label + " conv history depth=" + std::to_string(depth),
            std::span<const std::uint16_t>(tree_history).subspan(
                static_cast<std::size_t>(node) * history_stride, history_stride),
            std::span<const std::uint16_t>(branch_history).subspan(
                static_cast<std::size_t>(depth + 1) * history_stride, history_stride));
        failures += compare_slice(label + " output depth=" + std::to_string(depth),
                                  column<std::uint16_t>(tree_out, kValueRows, node),
                                  column<std::uint16_t>(branch_out, kValueRows, depth));
        failures += compare_slice(
            label + " recurrent state depth=" + std::to_string(depth),
            std::span<const float>(tree_state).subspan(
                static_cast<std::size_t>(node) * state_stride, state_stride),
            std::span<const float>(branch_state).subspan(
                static_cast<std::size_t>(depth + 1) * state_stride, state_stride));
    }
    return failures;
}

int verify_selected_path_fold(const TreeRun& tree,
                              const std::vector<std::uint16_t>& projected_bits,
                              const std::vector<float>& g_values,
                              const std::vector<float>& beta_values,
                              const std::vector<std::uint16_t>& committed_conv_bits,
                              const std::vector<float>& committed_recurrent_values,
                              std::span<const std::int32_t> path) {
    LayoutBuilder record_builder;
    const GdnReplayRecordLayout record_layout = plan_gdn_replay_records(
        record_builder, {.layers = 1,
                         .record_capacity = 1,
                         .width = kNodes,
                         .conv_channels = kChannels,
                         .qk_heads = kQkHeads,
                         .value_heads = kValueHeads,
                         .key_dim = kStateDim,
                         .value_dim = kStateDim});
    DeviceBuffer record_storage(record_builder.finish(256));
    record_storage.fill(0xff);
    GdnReplayRecords records({record_storage.p, record_storage.bytes}, record_layout);
    GdnReplayRecordLayer record = records.layer(0, 1);
    constexpr std::array<std::int32_t, kNodes> all_nodes{0, 1, 2, 3, 4, 5};
    DeviceBuffer all_nodes_buffer = upload(all_nodes);
    Tensor all_nodes_tensor(all_nodes_buffer.p, DType::I32, {kNodes});
    ops::gdn_tree_gather_path_records(
        tree.projected_tensor,
        tree.key_tensor.view({kStateDim, kQkHeads, kNodes, 1}),
        tree.value_tensor.view({kStateDim, kValueHeads, kNodes, 1}),
        tree.g_tensor, tree.beta_tensor, all_nodes_tensor, record.conv, record.key, record.value,
        record.gate, nullptr);

    LayoutBuilder state_builder;
    const LinearAttentionStatePoolLayout state_layout = plan_linear_attention_state_pool(
        state_builder, {.layers = 1,
                        .conv_channels = kChannels,
                        .conv_width = 3,
                        .value_heads = kValueHeads,
                        .value_head_dim = kStateDim,
                        .key_head_dim = kStateDim,
                        .slot_count = 2,
                        .conv_dtype = DType::BF16});
    DeviceBuffer state_storage(state_builder.finish(256));
    state_storage.fill(0);
    LinearAttentionStatePool state_pool({state_storage.p, state_storage.bytes}, state_layout);
    Tensor committed_conv = state_pool.conv_slot(0, 0);
    Tensor committed_recurrent = state_pool.recurrent_slot(0, 0);
    cuda_check(cudaMemcpy(committed_conv.data, committed_conv_bits.data(),
                          committed_conv_bits.size() * sizeof(std::uint16_t),
                          cudaMemcpyHostToDevice),
               "upload fold root convolution state");
    cuda_check(cudaMemcpy(committed_recurrent.data, committed_recurrent_values.data(),
                          committed_recurrent_values.size() * sizeof(float),
                          cudaMemcpyHostToDevice),
               "upload fold root recurrent state");
    ops::gdn_tree_fold_selected_path(records, state_pool.all_layers_view(), 0, 0, path,
                                      nullptr);
    cuda_synchronize();

    const std::int32_t selected = path.back();
    const auto tree_history = download<std::uint16_t>(tree.node_history_tensor);
    const auto tree_state = download<float>(tree.node_states_tensor);
    const auto folded_history = download<std::uint16_t>(committed_conv);
    const auto folded_state = download<float>(committed_recurrent);
    const std::size_t history_stride = static_cast<std::size_t>(kChannels) * 3;
    const std::size_t state_stride =
        static_cast<std::size_t>(kStateDim) * kStateDim * kValueHeads;
    int failures = 0;
    failures += compare_slice(
        "selected-path folded convolution state", folded_history,
        std::span<const std::uint16_t>(tree_history).subspan(
            static_cast<std::size_t>(selected) * history_stride, history_stride));
    failures += compare_slice(
        "selected-path folded recurrent state", folded_state,
        std::span<const float>(tree_state).subspan(
            static_cast<std::size_t>(selected) * state_stride, state_stride));

    // One ordinary continuation from the folded state must equal a one-node tree rooted at the
    // selected dense oracle state.
    const std::vector<std::uint16_t> next_projected = make_bf16(kChannels, 4101U);
    const std::vector<float> next_g = [] {
        std::vector<float> values(kValueHeads);
        fill_uniform(values, 4103U, -1.1F, -0.03F);
        return values;
    }();
    const std::vector<float> next_beta = [] {
        std::vector<float> values(kValueHeads);
        fill_uniform(values, 4105U, 0.03F, 0.95F);
        return values;
    }();
    DeviceBuffer next_projected_device = upload(next_projected);
    DeviceBuffer next_g_device = upload(next_g);
    DeviceBuffer next_beta_device = upload(next_beta);
    DeviceBuffer next_parent_device = upload(std::array<std::int32_t, 1>{-1});
    DeviceBuffer next_history_device(static_cast<std::size_t>(kChannels) * 3 *
                                     sizeof(std::uint16_t));
    DeviceBuffer next_q_device(static_cast<std::size_t>(kQueryRows) * sizeof(std::uint16_t));
    DeviceBuffer next_k_device(static_cast<std::size_t>(kKeyRows) * sizeof(std::uint16_t));
    DeviceBuffer next_v_device(static_cast<std::size_t>(kValueRows) * sizeof(std::uint16_t));
    DeviceBuffer next_state_device(state_stride * sizeof(float));
    DeviceBuffer next_out_device(static_cast<std::size_t>(kValueRows) * sizeof(std::uint16_t));
    Tensor next_projected_tensor(next_projected_device.p, DType::BF16, {kChannels, 1});
    Tensor conv_weight_tensor = tree.conv_weight_tensor;
    Tensor next_parent_tensor(next_parent_device.p, DType::I32, {1});
    Tensor next_history_tensor(next_history_device.p, DType::BF16, {kChannels, 3, 1});
    Tensor next_q_tensor(next_q_device.p, DType::BF16, {kQueryRows, 1});
    Tensor next_k_tensor(next_k_device.p, DType::BF16, {kKeyRows, 1});
    Tensor next_v_tensor(next_v_device.p, DType::BF16, {kValueRows, 1});
    Tensor next_g_tensor(next_g_device.p, DType::FP32, {kValueHeads, 1});
    Tensor next_beta_tensor(next_beta_device.p, DType::FP32, {kValueHeads, 1});
    Tensor next_state_tensor(next_state_device.p, DType::FP32,
                             {kStateDim, kStateDim, kValueHeads, 1});
    Tensor next_out_tensor(next_out_device.p, DType::BF16,
                           {kStateDim, kValueHeads, 1});
    ops::gdn_projected_conv_tree(next_projected_tensor, conv_weight_tensor,
                                 committed_conv, next_parent_tensor, next_history_tensor,
                                 next_q_tensor, next_k_tensor, next_v_tensor, nullptr);
    ops::gated_delta_net_tree_snapshot(
        next_q_tensor.view({kStateDim, kQkHeads, 1, 1}),
        next_k_tensor.view({kStateDim, kQkHeads, 1, 1}),
        next_v_tensor.view({kStateDim, kValueHeads, 1, 1}),
        next_g_tensor, next_beta_tensor, kScale, committed_recurrent,
        next_parent_tensor, next_state_tensor, next_out_tensor, nullptr);

    DeviceBuffer canonical_projected_device = upload(next_projected);
    DeviceBuffer canonical_q_device(next_q_device.bytes);
    DeviceBuffer canonical_k_device(next_k_device.bytes);
    DeviceBuffer canonical_v_device(next_v_device.bytes);
    Tensor canonical_projected(canonical_projected_device.p, DType::BF16, {kChannels, 1});
    Tensor canonical_q(canonical_q_device.p, DType::BF16, {kQueryRows, 1});
    Tensor canonical_k(canonical_k_device.p, DType::BF16, {kKeyRows, 1});
    Tensor canonical_v(canonical_v_device.p, DType::BF16, {kValueRows, 1});
    DeviceBuffer initial = upload(std::array<std::int32_t, 1>{0});
    DeviceBuffer base = upload(std::array<std::int32_t, 1>{1});
    Tensor initial_tensor(initial.p, DType::I32, {1});
    Tensor base_tensor(base.p, DType::I32, {1});
    ops::detail::gdn_projected_conv_snapshot_launch(
        canonical_projected, conv_weight_tensor, state_pool.conv[0], Tensor{},
        initial_tensor, base_tensor, canonical_q, canonical_k, canonical_v, nullptr);
    DeviceBuffer canonical_out_device(next_out_device.bytes);
    Tensor canonical_out(canonical_out_device.p, DType::BF16,
                         {kStateDim, kValueHeads, 1});
    ops::gated_delta_net_snapshot(
        canonical_q.view({kStateDim, kQkHeads, 1, 1}),
        canonical_k.view({kStateDim, kQkHeads, 1, 1}),
        canonical_v.view({kStateDim, kValueHeads, 1, 1}),
        next_g_tensor, next_beta_tensor, kScale, true, state_pool.recurrent[0],
        Tensor{}, initial_tensor, base_tensor, canonical_out, nullptr);
    cuda_synchronize();
    failures += compare_slice("post-tree ordinary continuation output",
                              download<std::uint16_t>(next_out_tensor),
                              download<std::uint16_t>(canonical_out));
    failures += compare_slice("post-tree ordinary continuation conv state",
                              download<std::uint16_t>(next_history_tensor),
                              download<std::uint16_t>(state_pool.conv_slot(0, 1)));
    failures += compare_slice("post-tree ordinary continuation recurrent state",
                              download<float>(next_state_tensor),
                              download<float>(state_pool.recurrent_slot(0, 1)));
    return failures;
}

int run_tree_oracle() {
    constexpr std::array<std::int32_t, kNodes> parents{-1, 0, 1, -1, 3, 4};
    constexpr std::array<std::int32_t, kDepth> path_a{0, 1, 2};
    constexpr std::array<std::int32_t, kDepth> path_b{3, 4, 5};
    const std::vector<std::uint16_t> projected_bits =
        make_bf16(static_cast<std::size_t>(kChannels) * kNodes, 4001U);
    const std::vector<std::uint16_t> conv_weight_bits =
        make_bf16(static_cast<std::size_t>(kChannels) * 4, 4003U, -0.04F, 0.04F);
    const std::vector<std::uint16_t> committed_conv_bits =
        make_bf16(static_cast<std::size_t>(kChannels) * 3, 4005U);
    std::vector<float> g_values(static_cast<std::size_t>(kValueHeads) * kNodes);
    std::vector<float> beta_values(g_values.size());
    fill_uniform(g_values, 4007U, -1.1F, -0.03F);
    fill_uniform(beta_values, 4009U, 0.03F, 0.95F);
    std::vector<float> committed_recurrent_values(
        static_cast<std::size_t>(kStateDim) * kStateDim * kValueHeads);
    fill_uniform(committed_recurrent_values, 4011U, -0.025F, 0.025F);

    TreeRun tree(projected_bits, conv_weight_bits, committed_conv_bits, parents,
                 g_values, beta_values, committed_recurrent_values);
    CanonicalBranch branch_a(projected_bits, conv_weight_bits, committed_conv_bits,
                             g_values, beta_values, committed_recurrent_values, path_a);
    CanonicalBranch branch_b(projected_bits, conv_weight_bits, committed_conv_bits,
                             g_values, beta_values, committed_recurrent_values, path_b);
    int failures = 0;
    failures += verify_branch(tree, branch_a, path_a, "branch A");
    failures += verify_branch(tree, branch_b, path_b, "branch B");
    failures += verify_selected_path_fold(tree, projected_bits, g_values, beta_values,
                                          committed_conv_bits, committed_recurrent_values,
                                          path_b);
    failures += compare_slice("committed convolution input remained read-only",
                              download<std::uint16_t>(tree.committed_conv_tensor),
                              committed_conv_bits);
    failures += compare_slice("committed recurrent input remained read-only",
                              download<float>(tree.committed_recurrent_tensor),
                              committed_recurrent_values);
    return failures;
}

} // namespace

int main() {
    if (cuda_unavailable()) {
        std::cout << "SKIP: no usable CUDA device\n";
        return 77;
    }
    const int failures = run_tree_oracle();
    std::cout << (failures == 0 ? "OK" : "FAIL")
              << " gdn_tree_replay parent-indexed oracle\n";
    return failures == 0 ? 0 : 1;
}
