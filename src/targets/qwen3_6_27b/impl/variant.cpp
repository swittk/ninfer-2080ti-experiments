#include "ops/linear/nu3_mlp_sm75.h"
#include "targets/qwen3_6_27b/impl/variant.h"
#if defined(NINFER_CUDA_SM75)
#include "targets/qwen3_6_27b/impl/sm75_prefill_dense.h"
#endif

#include "ninfer/ops/attn_input_proj.h"
#include "ninfer/ops/gdn_gating_proj.h"
#include "ninfer/ops/gdn_input_proj.h"
#include "ninfer/ops/linear.h"
#include "ninfer/ops/linear_add.h"
#include "ninfer/ops/linear_pair.h"
#include "ninfer/ops/linear_swiglu.h"
#include "ninfer/ops/mtp_pack.h"
#include "ops/linear_add/q5/q5_linear_add_a8_sm75.h"
#include "ops/linear/q5/q5_sm75_canonical.h"
#include "ops/linear_swiglu/q4/q4_linear_swiglu_a8_sm75.h"
#include "ops/marlin/mtp_q4_marlin.h"
#include "ops/linear/sm75_radix_projection.h"
#include "ops/gdn_input_proj/q4_q5/q5_a8_sm75.h"
#include "ops/gdn_input_proj/gdn_projected_conv.h"
#include "ninfer/ops/residual_add.h"
#include "ninfer/ops/silu_mul.h"

#include <algorithm>
#include <cstdlib>
#include <stdexcept>
#include <filesystem>
#include <string>
#include <cstdio>

#define NINFER_QWEN36_SM75_DENSE_A8 1
#define NINFER_QWEN36_VARIANT    ::ninfer::targets::qwen3_6_27b::detail::Variant
#define NINFER_QWEN36_RUNTIME_NS qwen3_6_27b_runtime
#include "targets/qwen3_6/impl/runtime/instantiate.h"

namespace ninfer::targets::qwen3_6_27b::detail {
std::unique_ptr<qwen3_6::detail::PrefillDenseOverride>
Variant::make_prefill_mlp(WorkspaceArena& workspace, cudaStream_t stream) {
#if defined(NINFER_CUDA_SM75)
    if (Sm75PrefillDense::enabled()) { return std::make_unique<Sm75PrefillDense>(workspace, stream); }
#else
    (void)workspace; (void)stream;
#endif
    return {};
}

bool Variant::sm75_resident_prefill_enabled() noexcept {
#if defined(NINFER_CUDA_SM75)
    static const bool enabled = [] {
        const char* a8 = std::getenv("NINFER_SERVE_C4_A8");
        const char* resident = std::getenv("NINFER_SERVE_RESIDENT_PREFILL");
        return a8 && a8[0] == '1' && a8[1] == '\0' &&
               resident && resident[0] == '1' && resident[1] == '\0';
    }();
    return enabled;
#else
    return false;
#endif
}
namespace {

std::vector<GraphExecutionProfile>
graph_profiles_through(std::uint32_t max_frontier,
                       const std::vector<std::uint32_t>& preferred_ends) {
    std::vector<GraphExecutionProfile> out;
    std::uint32_t begin = 0;
    for (const std::uint32_t preferred_end : preferred_ends) {
        if (begin > max_frontier) { break; }
        const std::uint32_t end = std::min(preferred_end, max_frontier);
        out.push_back({begin, end});
        if (end == max_frontier) { return out; }
        begin = end + 1;
    }
    if (begin <= max_frontier) { out.push_back({begin, max_frontier}); }
    return out;
}

void validate_token_interval(std::int32_t first, std::int32_t last) {
    if (first <= 0 || last < first) {
        throw std::invalid_argument("invalid target leaf token interval");
    }
}

#if defined(NINFER_SM86) || defined(NINFER_CUDA_SM75)
constexpr ops::LinearPolicy kNvfp4TextPolicy = ops::LinearPolicy::A16Only;
#else
constexpr ops::LinearPolicy kNvfp4TextPolicy = ops::LinearPolicy::AllowA4;
#endif

ops::LinearPolicy text_policy(const Weight& weight) {
    return weight.qtype == QType::NVFP4 ? kNvfp4TextPolicy : ops::LinearPolicy::A16Only;
}

Weight canonical_radix_view(const Weight& weight, WorkspaceArena& workspace, cudaStream_t stream) {
#if defined(NINFER_CUDA_SM75)
    return ops::detail::sm75_radix_canonical_copy(weight,workspace,stream);
#else
    (void)workspace;(void)stream;return weight;
#endif
}

constexpr std::size_t kMinimumLeafWorkspaceBytes = 1;

std::size_t gdn_record_workspace_bytes(const Tensor& hidden,
                                       const Variant::GdnProjectionWeights& weights) {
    const std::int32_t batch = hidden.ne[2];
    const std::int32_t width = hidden.ne[1];
    if (std::holds_alternative<SplitGdnInputProjectionPayload>(weights.input_projection)) {
        return std::max(kMinimumLeafWorkspaceBytes,
                        ops::gdn_input_proj_conv_record_workspace_capacity_bytes(
                            TextConfig::key_dim, TextConfig::key_dim, TextConfig::value_dim, batch,
                            width, width));
    }
    return std::max(
        kMinimumLeafWorkspaceBytes,
        ops::gdn_input_proj_conv_record_workspace_capacity_bytes(
            QType::NVFP4, 16384, TextConfig::hidden, kNvfp4TextPolicy, batch, width, width));
}

} // namespace

std::vector<GraphExecutionProfile> Variant::ordinary_graph_profiles(std::uint32_t capacity) {
    // E+1 is the one-token visible window. Early ranges limit empty producer CTAs; later ranges
    // follow measured split-policy transitions until the producer grid reaches its fixed cap.
    return graph_profiles_through(capacity - 1, {127, 511, 2047, 4095, 8197, 16389, 32767});
}

std::vector<GraphExecutionProfile> Variant::mtp_graph_profiles(std::uint32_t capacity,
                                                               std::uint32_t draft_window) {
    if (draft_window == 0 || capacity == 0) { return {}; }
    // Bound the final AR window E+2K at split-policy transitions until the grid reaches its cap.
    std::vector<std::uint32_t> ends;
    const auto add_shifted = [&](std::uint32_t visible_end, std::uint32_t offset) {
        if (visible_end >= offset) { ends.push_back(visible_end - offset); }
    };
    for (const std::uint32_t visible_end : {128U, 512U, 2048U, 4096U, 8198U, 16390U, 32768U}) {
        add_shifted(visible_end, 2 * draft_window);
    }
    // Target verify and MTP batch both have T=K+1 and W=E+K+1. Preserve one concrete INT8
    // implementation per range at the T=4/5/6 launch boundaries.
    if (draft_window == 3) {
        add_shifted(1029, draft_window + 1);
    } else if (draft_window == 4) {
        for (const std::uint32_t visible_end : {128U, 512U, 1029U}) {
            add_shifted(visible_end, draft_window + 1);
        }
    } else if (draft_window == 5) {
        for (const std::uint32_t visible_end : {128U, 160U, 2054U, 8198U}) {
            add_shifted(visible_end, draft_window + 1);
        }
    }
    std::sort(ends.begin(), ends.end());
    ends.erase(std::unique(ends.begin(), ends.end()), ends.end());
    return graph_profiles_through(capacity - 1, ends);
}

std::vector<GraphExecutionProfile> Variant::dflash_graph_profiles(std::uint32_t, std::uint32_t,
                                                                  std::uint32_t) {
    return {};
}

void Variant::attention_projection(const Tensor& hidden,
                                   const FullAttentionProjectionWeights& weights, Tensor& query,
                                   Tensor& gate, Tensor& key, Tensor& value, qwen3_6::TextPhase,
                                   WorkspaceArena& workspace, cudaStream_t stream,
                                   qwen3_6::detail::PrefillDenseOverride* prefill) {
    if (const auto* split = std::get_if<SplitAttentionProjectionPayload>(&weights)) {
        if (prefill && prefill->attention_input(hidden, split->query_key, split->gate_value,
                                               query, gate, key, value, stream)) return;
        ops::attn_input_proj(hidden, split->query_key, split->gate_value, query, gate, key, value,
                             stream);
        return;
    }
    const Weight& fused = std::get<FusedAttentionProjectionPayload>(weights).query_key_gate_value;
    ops::attn_input_proj(hidden, fused, query, gate, key, value, text_policy(fused), workspace,
                         stream);
}

void Variant::attention_output_projection(const Tensor& attention, const Weight& weight,
                                          Tensor& residual, qwen3_6::TextPhase phase,
                                          WorkspaceArena& workspace, cudaStream_t stream) {
#if defined(NINFER_CUDA_SM75)
    if (phase == qwen3_6::TextPhase::Verify &&
        ops::detail::sm75_radix_projection_launch(attention, weight, residual, true, stream)) { return; }
#endif
    auto canonical_scope=workspace.scope();
    const Weight canonical=canonical_radix_view(weight,workspace,stream);
    ops::linear_add(attention, canonical, residual, text_policy(canonical), workspace, stream);
}

void Variant::mtp_attention_projection(const Tensor& hidden,
                                       const MtpAttentionProjectionWeights& weights, Tensor& query,
                                       Tensor& gate, Tensor& key, Tensor& value,
                                       WorkspaceArena& workspace, cudaStream_t stream) {
    auto scope     = workspace.scope();
    const int cols = hidden.ne[1];
    Tensor packed  = workspace.alloc(DType::BF16, {TextConfig::mtp_attention_input_rows, cols});
    ops::linear(hidden, weights.packed, packed, stream);
    Tensor query_heads = query.view({TextConfig::head_dim, TextConfig::query_heads, cols});
    Tensor key_heads   = key.view({TextConfig::head_dim, TextConfig::kv_heads, cols});
    Tensor gate_heads  = gate.view({TextConfig::head_dim, TextConfig::query_heads, cols});
    Tensor value_heads = value.view({TextConfig::head_dim, TextConfig::kv_heads, cols});
    ops::mtp_split_attn_in(packed, query_heads, key_heads, gate_heads, value_heads, stream);
}

void Variant::mtp_kv_projection(const Tensor& hidden, const MtpAttentionProjectionWeights& weights,
                                Tensor& key, Tensor& value, WorkspaceArena&, cudaStream_t stream) {
    if (weights.key.qtype == QType::W8G32_F16S && weights.value.qtype == QType::W8G32_F16S) {
        ops::linear_pair(hidden, weights.key, weights.value, key, value, stream);
        return;
    }
    ops::linear(hidden, weights.key, key, stream);
    ops::linear(hidden, weights.value, value, stream);
}

void Variant::mtp_q_gate_projection(const Tensor& hidden,
                                    const MtpAttentionProjectionWeights& weights, Tensor& query,
                                    Tensor& gate, WorkspaceArena&, cudaStream_t stream,
                                    bool retained_packed_prefix) {
    if (retained_packed_prefix) {
        ops::detail::mtp_q4_marlin_retained_prefix_launch(
            hidden, weights.packed, weights.query, query, stream);
    } else {
        ops::linear(hidden, weights.query, query, stream);
    }
    ops::linear(hidden, weights.output_gate, gate, stream);
}

void Variant::gdn_input_projection(const Tensor& hidden, const GdnProjectionWeights& weights,
                                   Tensor& qkv, Tensor& output_gate, qwen3_6::TextPhase,
                                   WorkspaceArena& workspace, cudaStream_t stream,
                                   qwen3_6::detail::PrefillDenseOverride* prefill) {
    Tensor output_gate_flat =
        output_gate.view({TextConfig::value_dim, static_cast<int>(hidden.ne[1])});
    if (const auto* split =
            std::get_if<SplitGdnInputProjectionPayload>(&weights.input_projection)) {
        // The prefill executor can consume the resident representation directly.
        // Materialize canonical planes only for the independent native fallback.
        if (prefill && prefill->gdn_input(hidden, split->query_key, split->value_z,
                                           qkv, output_gate_flat, stream)) { return; }
        auto qk_scope=workspace.scope();
        const Weight canonical_qk=canonical_radix_view(split->query_key,workspace,stream);
#if defined(NINFER_CUDA_SM75)
        if (split->value_z.layout == QuantLayout::RowSplitSm75Fragment) {
            auto scope = workspace.scope();
            const Weight canonical = ops::detail::q5_sm75_canonical_copy(
                split->value_z, workspace, stream);
            ops::gdn_input_proj(hidden, canonical_qk, canonical, qkv, output_gate_flat, stream);
            return;
        }
#endif
        ops::gdn_input_proj(hidden, canonical_qk, split->value_z, qkv, output_gate_flat,
                            stream);
        return;
    }
    const Weight& fused =
        std::get<FusedGdnInputProjectionPayload>(weights.input_projection).query_key_value_z;
    ops::gdn_input_proj(hidden, fused, qkv, output_gate_flat, text_policy(fused), workspace,
                        stream);
}

void Variant::gdn_input_projection_snapshot(
    const Tensor& hidden, const GdnProjectionWeights& weights, const Tensor& conv_weight,
    Tensor& conv_states, const Tensor& valid_columns, const Tensor& initial_slot,
    const Tensor& snapshot_base_slot, Tensor& query, Tensor& key, Tensor& value,
    Tensor& output_gate, qwen3_6::TextPhase phase, WorkspaceArena& workspace, cudaStream_t stream) {
    Tensor output_gate_view = output_gate.view({TextConfig::value_dim, hidden.ne[1], hidden.ne[2]});
    if (const auto* split =
            std::get_if<SplitGdnInputProjectionPayload>(&weights.input_projection)) {
#if defined(NINFER_CUDA_SM75)
        const int columns = hidden.ne[1] * hidden.ne[2];
        if (phase == qwen3_6::TextPhase::Verify && columns >= 1 && columns <= 8 &&
            split->value_z.layout == QuantLayout::RowSplitSm75Fragment) {
            auto scope = workspace.scope();
            Tensor projected = workspace.alloc(DType::BF16, {TextConfig::convolution_dim, columns});
            Tensor hidden_flat = hidden.view({TextConfig::hidden, columns});
            Tensor z_flat = output_gate_view.view({TextConfig::value_dim, columns});
            ops::detail::gdn_q5_sm75_a8_launch(hidden_flat, split->query_key, split->value_z,
                                               projected, z_flat, workspace, stream);
            ops::detail::gdn_projected_conv_snapshot_launch(
                projected, conv_weight, conv_states, valid_columns, initial_slot,
                snapshot_base_slot, query, key, value, stream);
            return;
        }
#endif
        auto canonical_scope=workspace.scope();
        const Weight canonical_qk=canonical_radix_view(split->query_key,workspace,stream);
        ops::gdn_input_proj_conv_snapshot(hidden, canonical_qk, split->value_z, conv_weight,
                                          conv_states, valid_columns, initial_slot,
                                          snapshot_base_slot, query, key, value, output_gate_view,
                                          workspace, stream);
        return;
    }
    const Weight& fused =
        std::get<FusedGdnInputProjectionPayload>(weights.input_projection).query_key_value_z;
    ops::gdn_input_proj_conv_snapshot(hidden, fused, conv_weight, conv_states, valid_columns,
                                      initial_slot, snapshot_base_slot, query, key, value,
                                      output_gate_view, text_policy(fused), workspace, stream);
}

void Variant::gdn_input_projection_record(const Tensor& hidden, const GdnProjectionWeights& weights,
                                          const Tensor& conv_weight, const Tensor& conv_states,
                                          const Tensor& valid_columns, const Tensor& initial_slots,
                                          Tensor& conv_record, Tensor& query, Tensor& key,
                                          Tensor& value, Tensor& output_gate, qwen3_6::TextPhase phase,
                                          WorkspaceArena& workspace, cudaStream_t stream) {
    Tensor output_gate_view = output_gate.view({TextConfig::value_dim, hidden.ne[1], hidden.ne[2]});
    if (const auto* split =
            std::get_if<SplitGdnInputProjectionPayload>(&weights.input_projection)) {
#if defined(NINFER_CUDA_SM75)
        const int columns = hidden.ne[1] * hidden.ne[2];
        if (phase == qwen3_6::TextPhase::Verify && columns >= 1 && columns <= 16 &&
            split->value_z.layout == QuantLayout::RowSplitSm75Fragment) {
            Tensor hidden_flat = hidden.view({TextConfig::hidden, columns});
            Tensor record_flat = conv_record.view({TextConfig::convolution_dim, columns});
            Tensor z_flat = output_gate_view.view({TextConfig::value_dim, columns});
            // T16 preserves two-T8 arithmetic while sharing loaded weight fragments.
            // Ragged groups keep the qualified smaller-column path.
            if (columns == 16) {
                ops::detail::gdn_q5_sm75_a8_launch(hidden_flat, split->query_key, split->value_z,
                                                   record_flat, z_flat, workspace, stream);
            } else for (int offset = 0; offset < columns; offset += 8) {
                auto slice_scope = workspace.scope();
                const int count = std::min(8, columns - offset);
                Tensor hidden_slice = hidden_flat.slice(1, offset, count);
                Tensor record_slice = record_flat.slice(1, offset, count);
                Tensor gate_slice = z_flat.slice(1, offset, count);
                ops::detail::gdn_q5_sm75_a8_launch(hidden_slice, split->query_key, split->value_z,
                                                   record_slice, gate_slice, workspace, stream);
            }
            ops::detail::gdn_projected_conv_record_launch(
                conv_record, conv_weight, conv_states, valid_columns, initial_slots,
                query, key, value, stream);
            return;
        }
#endif
        auto workspace_scope     = workspace.scope();
        const DeviceSpan storage = workspace.alloc_bytes(gdn_record_workspace_bytes(hidden, weights));
        WorkspaceArena leaf_workspace(storage);
        auto canonical_scope=workspace.scope();
        const Weight canonical_qk=canonical_radix_view(split->query_key,workspace,stream);
        ops::gdn_input_proj_conv_record(hidden, canonical_qk, split->value_z, conv_weight,
                                        conv_states, valid_columns, initial_slots, conv_record,
                                        query, key, value, output_gate_view, leaf_workspace,
                                        stream);
        return;
    }
    auto workspace_scope     = workspace.scope();
    const DeviceSpan storage = workspace.alloc_bytes(gdn_record_workspace_bytes(hidden, weights));
    WorkspaceArena leaf_workspace(storage);
    const Weight& fused =
        std::get<FusedGdnInputProjectionPayload>(weights.input_projection).query_key_value_z;
    ops::gdn_input_proj_conv_record(hidden, fused, conv_weight, conv_states, valid_columns,
                                    initial_slots, conv_record, query, key, value, output_gate_view,
                                    text_policy(fused), leaf_workspace, stream);
}

void Variant::gdn_output_projection(const Tensor& hidden, const Weight& weight, Tensor& residual,
                                    qwen3_6::TextPhase phase, WorkspaceArena& workspace,
                                    cudaStream_t stream) {
#if defined(NINFER_CUDA_SM75)
    if (phase == qwen3_6::TextPhase::Verify &&
        ops::detail::sm75_radix_projection_launch(hidden, weight, residual, true, stream)) { return; }
#endif
    auto canonical_scope=workspace.scope();
    const Weight canonical=canonical_radix_view(weight,workspace,stream);
    ops::linear_add(hidden, canonical, residual, text_policy(canonical), workspace, stream);
}

void Variant::gdn_norm_control_projection(const Tensor& residual, const Tensor& norm_weight,
                                          float eps, const GdnProjectionWeights& weights,
                                          Tensor& hidden, Tensor& g, Tensor& beta,
                                          WorkspaceArena& workspace, cudaStream_t stream) {
    ops::gdn_norm_gating_proj(residual, norm_weight, eps, weights.a_projection,
                              weights.b_projection, weights.a_log, weights.dt_bias, workspace,
                              hidden, g, beta, stream);
}

bool Variant::try_post_mixer_rmsnorm_sm75(
    Tensor& residual,const Tensor& norm_weight,float eps,const PostMixerWeights& weights,
    qwen3_6::TextPhase phase,WorkspaceArena& workspace,cudaStream_t stream) {
#if defined(NINFER_CUDA_SM75)
    if (phase == qwen3_6::TextPhase::Verify && residual.ne[0] == TextConfig::hidden &&
        residual.ne[1] == 4 && weights.gate_up.qtype == QType::Q4G64_F16S &&
        weights.gate_up.layout == QuantLayout::RowSplitSm75Fragment &&
        weights.down.qtype == QType::Q5G64_F16S &&
        weights.down.layout == QuantLayout::RowSplitSm75Fragment) {
        auto scope = workspace.scope();
        Tensor activation_q = workspace.alloc(DType::I8, {TextConfig::intermediate, 4});
        Tensor activation_s = workspace.alloc(DType::FP16, {TextConfig::intermediate / 64, 4});
        ops::detail::q4_linear_swiglu_sm75_rmsnorm_a8_to_a8_launch(
            residual, norm_weight, eps, weights.gate_up, activation_q, activation_s, workspace,
            stream);
        ops::detail::q5_linear_add_sm75_a8_prequant_launch(
            activation_q, activation_s, weights.down, residual, stream);
        return true;
    }
#else
    (void)residual; (void)norm_weight; (void)eps; (void)weights;
    (void)phase; (void)workspace; (void)stream;
#endif
    return false;
}

void Variant::post_mixer(const Tensor& hidden, const PostMixerWeights& weights, Tensor& residual,
                         qwen3_6::TextPhase phase, WorkspaceArena& workspace, cudaStream_t stream) {
    auto scope = workspace.scope();
    if(weights.gate_up.layout==QuantLayout::RowSplitSm75Nu3 || weights.down.layout==QuantLayout::RowSplitSm75Nu3) {
        if(phase!=qwen3_6::TextPhase::Verify)
            throw std::logic_error("NU3 packed target MLP must be restored before prefill");
        ops::detail::nu3_mlp_launch(hidden,weights.gate_up,weights.down,residual,workspace,stream);
        return;
    }
#if defined(NINFER_CUDA_SM75)
    if (phase == qwen3_6::TextPhase::Prefill &&
        (weights.gate_up.layout == QuantLayout::RowSplitSm75Fragment ||
         weights.down.layout == QuantLayout::RowSplitSm75Fragment)) {
        Tensor activation = workspace.alloc(DType::BF16, {TextConfig::intermediate, hidden.ne[1]});
        {
            auto weight_scope = workspace.scope();
            const Weight canonical = ops::detail::q4_linear_swiglu_sm75_canonical_copy(
                weights.gate_up, workspace, stream);
            ops::linear_swiglu(hidden, canonical, activation, ops::LinearPolicy::A16Only,
                                workspace, stream);
        }
        {
            auto weight_scope = workspace.scope();
            const Weight canonical = ops::detail::q5_sm75_canonical_copy(weights.down, workspace, stream);
            ops::linear_add(activation, canonical, residual, ops::LinearPolicy::A16Only,
                            workspace, stream);
        }
        return;
    }
    if (phase == qwen3_6::TextPhase::Verify && hidden.ne[1] == 16 &&
        weights.gate_up.layout == QuantLayout::RowSplitSm75Fragment &&
        weights.down.layout == QuantLayout::RowSplitSm75Fragment) {
        Tensor activation = workspace.alloc(DType::BF16, {TextConfig::intermediate, 16});
        ops::detail::q4_linear_swiglu_sm75_a8_launch(hidden, weights.gate_up, activation,
                                                      workspace, stream);
        ops::detail::q5_linear_add_sm75_a8_launch(activation, weights.down, residual,
                                                  workspace, stream);
        return;
    }
    if (phase == qwen3_6::TextPhase::Verify && hidden.ne[1] > 8 && hidden.ne[1] < 16 &&
        weights.gate_up.layout == QuantLayout::RowSplitSm75Fragment &&
        weights.down.layout == QuantLayout::RowSplitSm75Fragment) {
        // Keep qualified T8 slices for ragged T9..T15 groups.
        // No recurrent state, KV positions or request membership is altered.
        for (int offset = 0; offset < hidden.ne[1]; offset += 8) {
            auto slice_scope = workspace.scope();
            const int count = std::min(8, hidden.ne[1] - offset);
            Tensor input_slice = hidden.slice(1, offset, count);
            Tensor residual_slice = residual.slice(1, offset, count);
            post_mixer(input_slice, weights, residual_slice, phase, workspace, stream);
        }
        return;
    }
#endif
#if defined(NINFER_CUDA_SM75)
    if (phase == qwen3_6::TextPhase::Verify && hidden.ne[1] == 4 &&
        weights.gate_up.qtype == QType::Q4G64_F16S &&
        weights.gate_up.layout == QuantLayout::RowSplitSm75Fragment &&
        weights.down.qtype == QType::Q5G64_F16S &&
        weights.down.layout == QuantLayout::RowSplitSm75Fragment) {
        Tensor activation_q = workspace.alloc(DType::I8, {TextConfig::intermediate, 4});
        Tensor activation_s = workspace.alloc(DType::FP16, {TextConfig::intermediate / 64, 4});
        ops::detail::q4_linear_swiglu_sm75_a8_to_a8_launch(
            hidden, weights.gate_up, activation_q, activation_s, workspace, stream);
        ops::detail::q5_linear_add_sm75_a8_prequant_launch(
            activation_q, activation_s, weights.down, residual, stream);
        return;
    }
#endif
    Tensor activation = workspace.alloc(DType::BF16, {TextConfig::intermediate, hidden.ne[1]});
    ops::LinearPolicy gate_policy = text_policy(weights.gate_up);
#if defined(NINFER_CUDA_SM75)
    if (phase == qwen3_6::TextPhase::Verify &&
        weights.gate_up.qtype == QType::Q4G64_F16S &&
        weights.gate_up.layout == QuantLayout::RowSplitSm75Fragment) {
        gate_policy = ops::LinearPolicy::AllowA8;
    }
#endif
    ops::linear_swiglu(hidden, weights.gate_up, activation, gate_policy, workspace, stream);
#if defined(NINFER_CUDA_SM75)
    if (phase == qwen3_6::TextPhase::Verify && weights.down.qtype == QType::Q5G64_F16S &&
        weights.down.layout == QuantLayout::RowSplitSm75Fragment && hidden.ne[1] <= 8) {
        ops::detail::q5_linear_add_sm75_a8_launch(activation, weights.down, residual, workspace,
                                                  stream);
    } else
#endif
    {
        ops::linear_add(activation, weights.down, residual, text_policy(weights.down), workspace,
                        stream);
    }
}

void Variant::mtp_post_mixer(const Tensor& hidden, const MtpPostMixerWeights& weights,
                             Tensor& residual, WorkspaceArena& workspace, cudaStream_t stream) {
    auto scope     = workspace.scope();
    const int cols = hidden.ne[1];
    Tensor activation = workspace.alloc(DType::BF16, {TextConfig::intermediate, cols});
    const auto launch_prepared_marlin = [&](const Tensor& input, const Weight& weight,
                                             Tensor& output, const char* label) {
        // Ragged C1..C4 verification/alignment produces T4/T8/T12/T16. Never
        // launch an unqualified native wide kernel or share its prepared scratch.
        // Ordered <=T8 slices preserve per-column ownership on this stream.
        for (std::int32_t offset = 0; offset < input.ne[1]; offset += 8) {
            const auto width = static_cast<std::int32_t>(std::min<std::int64_t>(8, input.ne[1] - offset));
            Tensor input_slice = input.slice(1, offset, width);
            Tensor output_slice = output.slice(1, offset, width);
            if (!ops::detail::mtp_q4_marlin_launch(input_slice, weight, output_slice, stream)) {
                throw std::logic_error(std::string("prepared MTP ") + label +
                                       " Marlin slice rejected its qualified shape");
            }
        }
    };
    // Keep high-M prefill on its existing dense route. This composition is
    // for small/ragged decode batches, not a T8 loop over a long prompt.
    const bool small_batch = cols <= 32;
    if (small_batch && ops::detail::mtp_q4_marlin_ready(weights.gate_up)) {
        Tensor gate_up = workspace.alloc(DType::BF16, {TextConfig::mtp_mlp_gate_up_rows, cols});
        launch_prepared_marlin(hidden, weights.gate_up, gate_up, "gate/up");
        ops::silu_mul(gate_up.slice(0, 0, TextConfig::intermediate),
                      gate_up.slice(0, TextConfig::intermediate, TextConfig::intermediate),
                      activation, stream);
    } else if (weights.gate_up.qtype == QType::Q4G64_F16S) {
        ops::linear_swiglu(hidden, weights.gate_up, activation, workspace, stream);
    } else {
        Tensor gate_up = workspace.alloc(DType::BF16, {TextConfig::mtp_mlp_gate_up_rows, cols});
        ops::linear(hidden, weights.gate_up, gate_up, stream);
        ops::silu_mul(gate_up.slice(0, 0, TextConfig::intermediate),
                      gate_up.slice(0, TextConfig::intermediate, TextConfig::intermediate),
                      activation, stream);
    }
    Tensor delta = workspace.alloc(DType::BF16, {TextConfig::hidden, cols});
    if (small_batch && ops::detail::mtp_q4_marlin_ready(weights.down)) {
        launch_prepared_marlin(activation, weights.down, delta, "down");
    } else {
        ops::linear(activation, weights.down, delta, stream);
    }
    ops::residual_add(delta, residual, stream);
}

void Variant::sm75_prepare_nu3_mlp(ModelView& model,cudaStream_t stream) {
    if(const char* directory=std::getenv("NINFER_NU3_FACTORY_BF16_DIR")) {
        const std::filesystem::path root(directory);
        // Private evaluation only: runner authenticates the pinned factory source
        // and all tensor hashes. Fail before conversion if any input is missing.
        for(int layer=0;layer<TextConfig::layers;++layer) {
            char number[16];std::snprintf(number,sizeof(number),"layer%02d_",layer);
            for(const char* kind:{"gate_proj","up_proj","down_proj"}) {
                const auto path=root/(std::string(number)+kind+".bf16");
                constexpr std::uint64_t bytes=std::uint64_t(TextConfig::hidden)*TextConfig::intermediate*2;
                if(!std::filesystem::is_regular_file(path)||std::filesystem::file_size(path)!=bytes)
                    throw std::invalid_argument("factory-source NU3 requires every complete MLP tensor");
            }
        }
        auto prepare=[&](auto& mixer,int layer){
            char number[16];std::snprintf(number,sizeof(number),"layer%02d_",layer);
            const auto g=(root/(std::string(number)+"gate_proj.bf16")).string();
            const auto u=(root/(std::string(number)+"up_proj.bf16")).string();
            const auto d=(root/(std::string(number)+"down_proj.bf16")).string();
            ops::detail::nu3_mlp_prepare_bf16(mixer.gate_up,g.c_str(),u.c_str(),stream);
            ops::detail::nu3_mlp_prepare_bf16(mixer.down,d.c_str(),nullptr,stream);
        };
        for(int layer=0;layer<TextConfig::layers;++layer) {
            if(TextConfig::is_full_attention(layer))prepare(model.full_layers.at(TextConfig::full_attention_index(layer)).post_mixer,layer);
            else prepare(model.gdn_layers.at(TextConfig::gdn_index(layer)).post_mixer,layer);
        }
        std::fprintf(stderr,"NU3_FACTORY_BF16_PREPARED target_matrices=%d fixed_codebook source_only temporary_bytes=16777220\n",TextConfig::layers*2);
        return;
    }

    for(auto& layer:model.full_layers){ops::detail::nu3_mlp_prepare(layer.post_mixer.gate_up,stream);ops::detail::nu3_mlp_prepare(layer.post_mixer.down,stream);}
    for(auto& layer:model.gdn_layers){ops::detail::nu3_mlp_prepare(layer.post_mixer.gate_up,stream);ops::detail::nu3_mlp_prepare(layer.post_mixer.down,stream);}
    std::fprintf(stderr,"NU3_MLP_PREPARED target_matrices=%zu source_device_only acceptance_unchanged\n",2*(model.full_layers.size()+model.gdn_layers.size()));
}



void Variant::sm75_transform_gdn_value_z(ModelView& model, QuantLayout target, void* scratch,
                                         cudaStream_t stream) {
#if defined(NINFER_CUDA_SM75)
    for (auto& layer : model.gdn_layers) {
        if (auto* split = std::get_if<SplitGdnInputProjectionPayload>(&layer.projection.input_projection)) {
            if (split->value_z.qtype == QType::Q5G64_F16S &&
                split->value_z.qdata != nullptr) {
                ops::detail::gdn_q5_sm75_transform_layout(split->value_z, target, scratch, stream);
            }
        }
    }
#else
    (void)model; (void)target; (void)scratch; (void)stream;
#endif
}

std::size_t Variant::mtp_attention_projection_workspace_capacity_bytes(std::int32_t first,
                                                                       std::int32_t last) {
    validate_token_interval(first, last);
    WorkspaceLayoutBuilder layout;
    (void)layout.alloc(DType::BF16, {TextConfig::mtp_attention_input_rows, last});
    return layout.peak_bytes(1);
}

std::size_t Variant::mtp_kv_projection_workspace_capacity_bytes(std::int32_t first,
                                                                std::int32_t last) {
    validate_token_interval(first, last);
    return 0;
}

std::size_t Variant::mtp_q_gate_projection_workspace_capacity_bytes(std::int32_t first,
                                                                    std::int32_t last) {
    validate_token_interval(first, last);
    return 0;
}

std::size_t Variant::attention_projection_workspace_capacity_bytes(WeightsProfile weights_profile,
                                                                   qwen3_6::TextPhase phase,
                                                                   std::int32_t first,
                                                                   std::int32_t last) {
    validate_token_interval(first, last);
    switch (weights_profile) {
    case WeightsProfile::GroupwiseInt:
    case WeightsProfile::GroupwiseIntW8Endpoints:
#if defined(NINFER_CUDA_SM75)
        if (phase == qwen3_6::TextPhase::Prefill && Sm75PrefillDense::enabled() &&
            first <= 4096 && last >= 64) {
            return Sm75PrefillDense::projection_workspace_bytes(7168, 5120, std::min(last, 4096));
        }
#endif
        return 0;
    case WeightsProfile::Nvfp4:
        return ops::attn_input_proj_workspace_capacity_bytes(
            QType::NVFP4, 14336, TextConfig::hidden, kNvfp4TextPolicy, first, last);
    }
    throw std::logic_error("invalid 27B weights profile");
}

std::size_t Variant::attention_output_projection_workspace_capacity_bytes(
    WeightsProfile weights_profile, qwen3_6::TextPhase phase, std::int32_t first, std::int32_t last) {
    validate_token_interval(first, last);
    switch (weights_profile) {
    case WeightsProfile::GroupwiseInt:
    case WeightsProfile::GroupwiseIntW8Endpoints:
#if defined(NINFER_CUDA_SM75)
        if (phase == qwen3_6::TextPhase::Prefill && Sm75PrefillDense::enabled() &&
            first <= 4096 && last >= 64) {
            return Sm75PrefillDense::projection_workspace_bytes(5120, 6144, std::min(last, 4096));
        }
#endif
        return ops::linear_add_workspace_capacity_bytes(QType::Q5G64_F16S, TextConfig::hidden,
                                                        TextConfig::query_size,
                                                        ops::LinearPolicy::A16Only, first, last);
    case WeightsProfile::Nvfp4:
        return ops::linear_add_workspace_capacity_bytes(QType::NVFP4, TextConfig::hidden,
                                                        TextConfig::query_size, kNvfp4TextPolicy,
                                                        first, last);
    }
    throw std::logic_error("invalid 27B weights profile");
}

std::size_t Variant::gdn_input_projection_workspace_capacity_bytes(WeightsProfile weights_profile,
                                                                   qwen3_6::TextPhase phase,
                                                                   std::int32_t first,
                                                                   std::int32_t last) {
    validate_token_interval(first, last);
    switch (weights_profile) {
    case WeightsProfile::GroupwiseInt:
    case WeightsProfile::GroupwiseIntW8Endpoints:
#if defined(NINFER_CUDA_SM75)
        if (phase == qwen3_6::TextPhase::Prefill && Sm75PrefillDense::enabled() &&
            first <= 4096 && last >= 64) {
            return Sm75PrefillDense::projection_workspace_bytes(12288, 5120, std::min(last, 4096));
        }
        if (phase == qwen3_6::TextPhase::Prefill && sm75_resident_prefill_enabled()) {
            return ops::detail::q5_sm75_canonical_workspace_bytes(12288, TextConfig::hidden);
        }
#endif
        return 0;
    case WeightsProfile::Nvfp4:
        return ops::gdn_input_proj_workspace_capacity_bytes(QType::NVFP4, 16384, TextConfig::hidden,
                                                            kNvfp4TextPolicy, first, last);
    }
    throw std::logic_error("invalid 27B weights profile");
}

std::size_t Variant::gdn_input_projection_snapshot_workspace_capacity_bytes(
    WeightsProfile weights_profile, qwen3_6::TextPhase phase, std::int32_t batch_size, std::int32_t first,
    std::int32_t last) {
    validate_token_interval(first, last);
    switch (weights_profile) {
    case WeightsProfile::GroupwiseInt:
    case WeightsProfile::GroupwiseIntW8Endpoints: {
        const std::size_t base = ops::gdn_input_proj_conv_snapshot_workspace_capacity_bytes(
            TextConfig::key_dim, TextConfig::key_dim, TextConfig::value_dim, batch_size, first, last);
#if defined(NINFER_CUDA_SM75)
        const std::int32_t columns = batch_size * last;
        if (phase == qwen3_6::TextPhase::Verify && columns >= 1 && columns <= 8) {
            WorkspaceLayoutBuilder layout;
            (void)layout.alloc(DType::BF16, {TextConfig::convolution_dim, columns});
            (void)layout.alloc_bytes(ops::detail::gdn_q5_sm75_a8_workspace_bytes(columns));
            return std::max(base, layout.peak_bytes(1));
        }
#endif
        return base;
    }
    case WeightsProfile::Nvfp4:
        return ops::gdn_input_proj_conv_snapshot_workspace_capacity_bytes(
            QType::NVFP4, 16384, TextConfig::hidden, kNvfp4TextPolicy, batch_size, first, last);
    }
    throw std::logic_error("invalid 27B weights profile");
}

std::size_t Variant::gdn_input_projection_record_workspace_capacity_bytes(
    WeightsProfile weights_profile, qwen3_6::TextPhase phase, std::int32_t batch_size, std::int32_t first,
    std::int32_t last) {
    validate_token_interval(first, last);
    switch (weights_profile) {
    case WeightsProfile::GroupwiseInt:
    case WeightsProfile::GroupwiseIntW8Endpoints: {
        std::size_t cap = std::max(kMinimumLeafWorkspaceBytes,
                                   ops::gdn_input_proj_conv_record_workspace_capacity_bytes(
                                       TextConfig::key_dim, TextConfig::key_dim, TextConfig::value_dim,
                                       batch_size, first, last));
#if defined(NINFER_CUDA_SM75)
        const std::int32_t columns = batch_size * last;
        if (phase == qwen3_6::TextPhase::Verify && columns >= 1 && columns <= 16) {
            // Record dispatch uses native T16, otherwise ordered <=T8 slices.
            const auto width = columns == 16 ? 16 : std::min(columns, 8);
            cap = std::max(cap, ops::detail::gdn_q5_sm75_a8_workspace_bytes(width));
        }
#endif
        return cap;
    }
    case WeightsProfile::Nvfp4:
        return std::max(kMinimumLeafWorkspaceBytes,
                        ops::gdn_input_proj_conv_record_workspace_capacity_bytes(
                            QType::NVFP4, 16384, TextConfig::hidden, kNvfp4TextPolicy, batch_size,
                            first, last));
    }
    throw std::logic_error("invalid 27B weights profile");
}

std::size_t Variant::gdn_output_projection_workspace_capacity_bytes(WeightsProfile weights_profile,
                                                                    qwen3_6::TextPhase phase,
                                                                    std::int32_t first,
                                                                    std::int32_t last) {
    validate_token_interval(first, last);
    switch (weights_profile) {
    case WeightsProfile::GroupwiseInt:
    case WeightsProfile::GroupwiseIntW8Endpoints:
#if defined(NINFER_CUDA_SM75)
        if (phase == qwen3_6::TextPhase::Prefill && Sm75PrefillDense::enabled() &&
            first <= 4096 && last >= 64) {
            return Sm75PrefillDense::projection_workspace_bytes(5120, 6144, std::min(last, 4096));
        }
#endif
        return ops::linear_add_workspace_capacity_bytes(QType::Q5G64_F16S, TextConfig::hidden,
                                                        TextConfig::value_dim,
                                                        ops::LinearPolicy::A16Only, first, last);
    case WeightsProfile::Nvfp4:
        return ops::linear_add_workspace_capacity_bytes(
            QType::NVFP4, TextConfig::hidden, TextConfig::value_dim, kNvfp4TextPolicy, first, last);
    }
    throw std::logic_error("invalid 27B weights profile");
}

std::size_t Variant::gdn_norm_control_projection_workspace_capacity_bytes(std::int32_t first,
                                                                          std::int32_t last) {
    return ops::gdn_norm_gating_proj_workspace_capacity_bytes(TextConfig::gdn_value_heads,
                                                              TextConfig::hidden, first, last);
}

std::size_t Variant::post_mixer_workspace_capacity_bytes(WeightsProfile weights_profile,
                                                         qwen3_6::TextPhase phase, std::int32_t first,
                                                         std::int32_t last) {
    validate_token_interval(first, last);
    QType gate_up_qtype;
    QType down_qtype;
    ops::LinearPolicy gate_policy;
    ops::LinearPolicy down_policy;
    switch (weights_profile) {
    case WeightsProfile::GroupwiseInt:
    case WeightsProfile::GroupwiseIntW8Endpoints:
        gate_up_qtype = QType::Q4G64_F16S;
        down_qtype    = QType::Q5G64_F16S;
#if defined(NINFER_CUDA_SM75)
        gate_policy = phase == qwen3_6::TextPhase::Verify ? ops::LinearPolicy::AllowA8
                                                           : ops::LinearPolicy::A16Only;
#else
        gate_policy = ops::LinearPolicy::A16Only;
#endif
        down_policy = ops::LinearPolicy::A16Only;
        break;
    case WeightsProfile::Nvfp4:
        gate_up_qtype = QType::NVFP4;
        down_qtype    = QType::NVFP4;
        gate_policy   = kNvfp4TextPolicy;
        down_policy   = kNvfp4TextPolicy;
        break;
    default:
        throw std::invalid_argument("qwen3_6_27b: invalid weights profile");
    }
    const bool resident_prefill = phase == qwen3_6::TextPhase::Prefill &&
                                   gate_up_qtype == QType::Q4G64_F16S &&
                                   sm75_resident_prefill_enabled();
    WorkspaceLayoutBuilder layout;
    (void)layout.alloc(DType::BF16, {TextConfig::intermediate, last});
    {
        auto scope = layout.scope();
        if (resident_prefill) {
            (void)layout.alloc_bytes(ops::detail::q4_linear_swiglu_sm75_transform_scratch_bytes());
        }
        (void)layout.alloc_bytes(ops::linear_swiglu_workspace_capacity_bytes(
            gate_up_qtype, 2 * TextConfig::intermediate, TextConfig::hidden, gate_policy, first, last));
    }
    {
        auto scope = layout.scope();
        if (resident_prefill) {
            (void)layout.alloc_bytes(ops::detail::q5_sm75_canonical_workspace_bytes(
                TextConfig::hidden, TextConfig::intermediate));
        }
        std::size_t down_workspace = ops::linear_add_workspace_capacity_bytes(
            down_qtype, TextConfig::hidden, TextConfig::intermediate, down_policy, first, last);
#if defined(NINFER_CUDA_SM75)
        if ((weights_profile == WeightsProfile::GroupwiseInt ||
             weights_profile == WeightsProfile::GroupwiseIntW8Endpoints) &&
            phase == qwen3_6::TextPhase::Verify && first <= 8) {
            down_workspace = std::max(
                down_workspace,
                ops::detail::q5_linear_add_sm75_a8_workspace_bytes(std::min(last, 8)));
        }
#endif
        (void)layout.alloc_bytes(down_workspace);
    }
    std::size_t capacity = layout.peak_bytes(1);
#if defined(NINFER_CUDA_SM75)
    if (phase == qwen3_6::TextPhase::Prefill && gate_up_qtype == QType::Q4G64_F16S &&
        Sm75PrefillDense::enabled() && first <= 4096 && last >= 64) {
        capacity = std::max(capacity, Sm75PrefillDense::workspace_bytes(std::min(last, 4096)));
    }
#endif
    return capacity;
}

std::size_t Variant::mtp_post_mixer_workspace_capacity_bytes(std::int32_t first,
                                                             std::int32_t last) {
    validate_token_interval(first, last);
    WorkspaceLayoutBuilder layout;
    (void)layout.alloc(DType::BF16, {TextConfig::mtp_mlp_gate_up_rows, last});
    (void)layout.alloc(DType::BF16, {TextConfig::intermediate, last});
    (void)layout.alloc(DType::BF16, {TextConfig::hidden, last});
    return layout.peak_bytes(1);
}

} // namespace ninfer::targets::qwen3_6_27b::detail
