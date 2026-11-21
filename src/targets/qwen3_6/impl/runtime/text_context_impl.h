#include "targets/qwen3_6/impl/runtime/instance.h"
#include "targets/qwen3_6/impl/runtime/text_context.h"
#include "targets/qwen3_6/impl/runtime/workspace_recipe.h"

#include "core/nvtx.h"
#include "targets/qwen3_6/impl/runtime/visual_scatter.h"
#include "targets/qwen3_6/impl/runtime/vision_context.h"
#include <ninfer/targets/qwen3_6/vision_control.h>
#include "ninfer/ops/argmax.h"
#include "ninfer/ops/attn_input_proj.h"
#include "ninfer/ops/causal_conv1d_silu.h"
#include "ninfer/ops/embedding.h"
#include "ninfer/ops/gated_delta_net.h"
#include "ninfer/ops/gated_rmsnorm.h"
#include "ninfer/ops/gdn_gating.h"
#include "ninfer/ops/gdn_gating_proj.h"
#include "ninfer/ops/gdn_input_proj.h"
#include "ninfer/ops/gqa_attention.h"
#include "ninfer/ops/linear.h"
#include "ninfer/ops/linear_add.h"
#include "ninfer/ops/linear_pair.h"
#include "ninfer/ops/linear_swiglu.h"
#include "ninfer/ops/mtp_pack.h"
#include "ninfer/ops/position.h"
#include "ninfer/ops/residual_add.h"
#include "ninfer/ops/rmsnorm.h"
#include "ninfer/ops/rope.h"
#include "ninfer/ops/scatter.h"
#include "ninfer/ops/scalar.h"
#include "ninfer/ops/sigmoid_mul.h"
#include "ninfer/ops/silu_mul.h"
#include "ops/linear/q4/q4_proposal_b2_sm75.h"
#include "ops/linear/w8/w8_vocab_w4_a8_sm75.h"
#include "ops/launcher/gqa_attention.h"
#include "ops/marlin/mtp_q4_marlin.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <array>
#include <cstdint>
#include <cstdlib>
#include <initializer_list>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace ninfer::targets::qwen3_6::detail::NINFER_QWEN36_RUNTIME_NS::schedule {
namespace {

inline constexpr std::uint32_t kLongContextTurboMinKeys = 32768;
inline constexpr std::uint64_t kLongContextVerifyFfnSkipMask =
    0x1b80001b0001ffb8ULL;
inline constexpr std::uint64_t kLongContextVerifyFfnWarmup = 4;


void copy_i32(const std::int32_t* source, Tensor& destination, cudaStream_t stream) {
    if (source == nullptr || destination.dtype != DType::I32 || !destination.is_contiguous() ||
        destination.data == nullptr) {
        throw std::invalid_argument("copy_i32: invalid host source or I32 destination");
    }
    CUDA_CHECK(cudaMemcpyAsync(destination.data, source, destination.bytes(),
                               cudaMemcpyHostToDevice, stream));
}

void require_tensor_shape(const Tensor& t, DType dtype, std::initializer_list<std::int32_t> shape,
                          const char* label) {
    if (t.dtype != dtype) { throw std::invalid_argument(std::string(label) + " dtype mismatch"); }
    int i = 0;
    for (const std::int32_t dim : shape) {
        if (t.ne[i] != dim) { throw std::invalid_argument(std::string(label) + " shape mismatch"); }
        ++i;
    }
    for (; i < 4; ++i) {
        if (t.ne[i] != 1) { throw std::invalid_argument(std::string(label) + " shape mismatch"); }
    }
    if (!t.is_contiguous()) {
        throw std::invalid_argument(std::string(label) + " must be contiguous");
    }
    if (t.data == nullptr) { throw std::invalid_argument(std::string(label) + " data is null"); }
}

void require_tensor_window(const Tensor& t, DType dtype, std::int32_t rows, std::int32_t cols,
                           const char* label) {
    if (cols <= 0) { throw std::invalid_argument(std::string(label) + " cols must be positive"); }
    if (t.dtype != dtype) { throw std::invalid_argument(std::string(label) + " dtype mismatch"); }
    if (t.ne[0] != rows || t.ne[1] < cols || t.ne[2] != 1 || t.ne[3] != 1) {
        throw std::invalid_argument(std::string(label) + " shape mismatch");
    }
    if (!t.is_contiguous()) {
        throw std::invalid_argument(std::string(label) + " must be contiguous");
    }
    if (t.data == nullptr) { throw std::invalid_argument(std::string(label) + " data is null"); }
}

Tensor matrix_window(Tensor& t, std::int32_t cols) {
    if (cols <= 0) { throw std::invalid_argument("matrix_window cols must be positive"); }
    if (t.ne[1] < cols || t.ne[2] != 1 || t.ne[3] != 1) {
        throw std::invalid_argument("matrix_window shape mismatch");
    }
    return t.slice(1, 0, cols);
}

class ScopedPositions {
public:
    ScopedPositions(const Tensor*& slot, const Tensor& positions) : slot_(slot) {
        slot_ = &positions;
    }

    ScopedPositions(const ScopedPositions&)            = delete;
    ScopedPositions& operator=(const ScopedPositions&) = delete;

    ~ScopedPositions() { slot_ = nullptr; }

private:
    const Tensor*& slot_;
};

class ScopedEnvelope {
public:
    ScopedEnvelope(const ops::GqaExecutionEnvelope*& slot,
                   const ops::GqaExecutionEnvelope& envelope)
        : slot_(slot) {
        slot_ = &envelope;
    }

    ScopedEnvelope(const ScopedEnvelope&)            = delete;
    ScopedEnvelope& operator=(const ScopedEnvelope&) = delete;

    ~ScopedEnvelope() { slot_ = nullptr; }

private:
    const ops::GqaExecutionEnvelope*& slot_;
};

template <class T>
class ScopedValue {
public:
    ScopedValue(T& slot, T value) : slot_(slot), previous_(slot) { slot_ = value; }

    ScopedValue(const ScopedValue&)            = delete;
    ScopedValue& operator=(const ScopedValue&) = delete;

    ~ScopedValue() { slot_ = previous_; }

private:
    T& slot_;
    T previous_;
};

} // namespace

void DFlashFeatureSink::begin(const Tensor& value) {
    const bool prefill = features != nullptr && positions != nullptr && batch_features == nullptr;
    const bool batch   = batch_features != nullptr && batch_lanes != nullptr &&
                       batch_valid_columns != nullptr && batch_width > 0 && batch_size > 0;
    if ((!prefill && !batch) || layers.empty()) {
        throw std::logic_error("DFlash feature sink is incomplete");
    }
    captured_mask = 0;
    active_tokens = batch ? batch_width * batch_size : value.ne[1];
    if (value.ne[1] != active_tokens) {
        throw std::logic_error("DFlash batch feature source has an invalid width");
    }
}

void DFlashFeatureSink::capture_layer(int layer, const Tensor& value, cudaStream_t stream) {
    const auto it = std::find(layers.begin(), layers.end(), layer);
    if (it == layers.end()) { return; }
    const std::size_t index = static_cast<std::size_t>(it - layers.begin());
    Tensor* destination     = batch_features != nullptr ? batch_features : features;
    if (layers.size() > 32 || active_tokens <= 0 || value.dtype != DType::BF16 ||
        destination == nullptr ||
        value.ne[0] * static_cast<std::int32_t>(layers.size()) != destination->ne[0] ||
        value.ne[1] != active_tokens) {
        throw std::logic_error("DFlash feature capture shape is invalid");
    }
    if (batch_features != nullptr) {
        Tensor source = value.view({value.ne[0], batch_width, batch_size});
        Tensor target =
            batch_features->slice(0, static_cast<std::int32_t>(index) * value.ne[0], value.ne[0]);
        ops::scatter_bf16_batch(source, *batch_lanes, *batch_valid_columns, target, stream);
        captured_mask |= 1U << index;
        return;
    }
    if (active_tokens > features->ne[1]) {
        throw std::logic_error("DFlash prefill feature capture exceeds its buffer");
    }
    const std::size_t element_bytes = dtype_size(DType::BF16);
    const std::size_t width_bytes   = static_cast<std::size_t>(value.ne[0]) * element_bytes;
    const std::size_t source_pitch  = static_cast<std::size_t>(value.nb[1]);
    const std::size_t target_pitch  = static_cast<std::size_t>(features->nb[1]);
    auto* target                    = static_cast<std::byte*>(features->data) + index * width_bytes;
    CUDA_CHECK(cudaMemcpy2DAsync(target, target_pitch, value.data, source_pitch, width_bytes,
                                 static_cast<std::size_t>(active_tokens), cudaMemcpyDeviceToDevice,
                                 stream));
    captured_mask |= 1U << index;
}

void DFlashFeatureSink::capture_positions(const Tensor& source, cudaStream_t stream) {
    const std::uint32_t complete_mask = layers.size() == 32 ? ~0U : ((1U << layers.size()) - 1U);
    if (captured_mask != complete_mask) {
        throw std::logic_error("DFlash target call did not publish every feature layer");
    }
    if (batch_features != nullptr) {
        if (source.dtype != DType::I32 || source.ne[0] != batch_width ||
            source.ne[1] != batch_size) {
            throw std::logic_error("DFlash batch feature positions are invalid");
        }
        return;
    }
    if (active_tokens <= 0 || source.dtype != DType::I32 || source.ne[0] != active_tokens ||
        positions == nullptr || active_tokens > positions->ne[0]) {
        throw std::logic_error("DFlash feature positions are invalid");
    }
    CUDA_CHECK(cudaMemcpyAsync(positions->data, source.data,
                               static_cast<std::size_t>(active_tokens) * sizeof(std::int32_t),
                               cudaMemcpyDeviceToDevice, stream));
}

void DFlashFeatureSink::consume_prefill_chunk(std::int32_t tokens, bool turn_checkpoint) {
    if (!consume_prefill || tokens != active_tokens) {
        throw std::logic_error("DFlash prefill feature consumer is unavailable");
    }
    Tensor feature_window  = features->slice(1, 0, tokens);
    Tensor position_window = positions->slice(0, 0, tokens);
    consume_prefill(feature_window, position_window, turn_checkpoint);
}

TextContext::TextContext(DeviceContext& ctx, const LoadedModelData& weights, WorkspaceArena& work,
                         qwen3_6::PagedKVCacheView kv, LinearAttentionStatePool& state,
                         qwen3_6::RoundState& io, Tensor& prefill_hidden,
                         std::uint32_t prefill_chunk, std::uint32_t text_kv_base,
                         qwen3_6::PagedKVCacheView mtp_kv,
                         const qwen3_6::PagedKVCache* batch_text_kv,
                         const qwen3_6::PagedKVCache* batch_mtp_kv,
                         TextLayerRange layer_range)
    : ctx_(ctx), weights_(weights), work_(work), kv_(kv), mtp_kv_(mtp_kv), state_(state), io_(io),
      prefill_hidden_(prefill_hidden), prefill_chunk_(prefill_chunk), text_kv_base_(text_kv_base),
      layer_range_(layer_range), batch_text_kv_(batch_text_kv), batch_mtp_kv_(batch_mtp_kv) {
    if (layer_range_.begin < 0 || layer_range_.begin >= layer_range_.end ||
        layer_range_.end > kCfg.n_layers || layer_range_.begin % 4 != 0 ||
        layer_range_.end % 4 != 0 ||
        (layer_range_.input_embedding && layer_range_.begin != 0) ||
        (layer_range_.output_projection && layer_range_.end != kCfg.n_layers)) {
        throw std::invalid_argument("TextContext layer range is invalid");
    }
    full_attention_base_ = layer_range_.begin / 4;
    gdn_state_base_      = layer_range_.begin - full_attention_base_;
    const std::int32_t full_end = layer_range_.end / 4;
    const std::int32_t expected_full = full_end - full_attention_base_;
    const std::int32_t expected_gdn =
        (layer_range_.end - layer_range_.begin) - expected_full;
    if (batch_text_kv_ == nullptr ||
        batch_text_kv_->layers() != static_cast<std::uint32_t>(expected_full) ||
        state_.layer_count() != static_cast<std::uint32_t>(expected_gdn)) {
        throw std::invalid_argument("TextContext shard state topology does not match its layers");
    }
    if (prefill_chunk_ == 0 ||
        prefill_chunk_ > static_cast<std::uint32_t>(std::numeric_limits<std::int32_t>::max())) {
        throw std::invalid_argument("TextContext effective prefill chunk must fit positive int32");
    }
    if (mtp_enabled() && !io_.mtp_decode && !io_.mtp) {
        throw std::invalid_argument("MTP TextContext requires MTP round state");
    }
    set_linear_state_slots(0, state_.slot_count() > 1 ? 1 : 0);
    bind();
}

TextContext::~TextContext() = default;

void TextContext::set_linear_state_slots(std::int32_t current_slot,
                                         std::int32_t turn_checkpoint_slot) {
    if (current_slot < 0 || current_slot >= state_.slot_count() || turn_checkpoint_slot < 0 ||
        turn_checkpoint_slot >= state_.slot_count() || current_slot == turn_checkpoint_slot) {
        throw std::invalid_argument("TextContext Linear Attention slots are invalid");
    }
    linear_state_current_slot_         = current_slot;
    linear_state_turn_checkpoint_slot_ = turn_checkpoint_slot;
}

void TextContext::set_gdn_state_action(GdnStateAction action,
                                       const GdnReplayRecords* replay_records) {
    if ((action == GdnStateAction::RecordForReplay) != (replay_records != nullptr)) {
        throw std::invalid_argument("TextContext GDN state action has inconsistent records");
    }
    gdn_state_action_ = action;
    replay_records_   = replay_records;
}

void TextContext::bind() {
    using TargetBindings = LoadedModelData;
    using TargetMlp      = MlpWeights;
    const auto bind_mlp  = [](const TargetMlp& source) { return MlpW{&source}; };

    // MTP proposal execution also consumes the shared token embedding on an output-only shard.
    // Ordinary shard entry points still enforce layer_range_.input_embedding independently.
    if (layer_range_.input_embedding || mtp_enabled()) { embed_ = &weights_.token_embedding; }
    if (layer_range_.output_projection || mtp_enabled()) {
        final_norm_ = &weights_.final_norm;
    }
    if (layer_range_.output_projection) { lm_head_ = &weights_.output_head; }
    if (weights_.optimized_proposal) {
        const auto& proposal = *weights_.optimized_proposal;
        set_proposal_head(&proposal.head, static_cast<const std::int32_t*>(proposal.token_ids.data),
                          proposal.head.n);
#if defined(NINFER_CUDA_SM75) && defined(NINFER_QWEN36_SM75_DENSE_A8)
        ops::detail::q4_proposal_b2_prepare(proposal.head, ctx_.stream);
        if (lm_head_ != nullptr && weights_.features.allow_optimized_target_head()) {
            ops::detail::w8_vocab_w4_a8_prepare(*lm_head_, ctx_.stream);
        }
#endif
    }

#if defined(NINFER_CUDA_SM75) && defined(NINFER_QWEN36_SM75_DENSE_A8)
    if (weights_.features.fast_full_vocab_draft) {
        if (lm_head_ == nullptr ||
            (!weights_.features.force_full_target_head && !weights_.features.retained_target_full_vocab_draft)) {
            throw std::logic_error("full-vocabulary drafting requires the native full target head");
        }
        ops::detail::w8_vocab_w4_a8_prepare(*lm_head_, ctx_.stream);
    }
#endif

    if (mtp_enabled()) {
        if (!weights_.mtp) {
            throw std::invalid_argument("MTP state was enabled without materialized MTP weights");
        }
        const auto& source = *weights_.mtp;
        mtp_               = MtpW{&source,
                    &source.input_projection,
                    &source.embedding_norm,
                    &source.hidden_norm,
                    &source.input_norm,
                    &source.query_norm,
                    &source.key_norm,
                    &source.output,
                    &source.post_attention_norm,
                    &source.final_norm};
        const auto require_mtp_tensor = [](const Tensor* tensor, const char* label) {
            if (tensor == nullptr || tensor->data == nullptr || !tensor->is_contiguous()) {
                throw std::invalid_argument(std::string("MTP binding is invalid: ") + label);
            }
        };
        const auto require_mtp_weight = [](const Weight* weight, const char* label) {
            if (weight == nullptr || weight->payload == nullptr || weight->qdata == nullptr ||
                weight->scales == nullptr) {
                throw std::invalid_argument(std::string("MTP weight binding is invalid: ") + label);
            }
        };
        require_mtp_weight(mtp_.fc, "input projection");
        require_mtp_tensor(mtp_.pre_fc_norm_embedding, "embedding norm");
        require_mtp_tensor(mtp_.pre_fc_norm_hidden, "hidden norm");
        require_mtp_tensor(mtp_.input_norm, "input norm");
        require_mtp_tensor(mtp_.q_norm, "query norm");
        require_mtp_tensor(mtp_.k_norm, "key norm");
        require_mtp_weight(mtp_.o_proj, "attention output");
        require_mtp_tensor(mtp_.post_attn_norm, "post-attention norm");
        require_mtp_tensor(mtp_.norm, "final norm");
        const auto prepare_mtp_marlin = [&]<class Source>(const Source& candidate) {
            if constexpr (requires(const Source& value) {
                              value.attention.packed;
                              value.post_mixer.gate_up;
                              value.post_mixer.down;
                          }) {
                if (ops::detail::mtp_q4_marlin_enabled()) {
                    ops::detail::mtp_q4_marlin_prepare(candidate.input_projection, ctx_.stream);
                    ops::detail::mtp_q4_marlin_prepare(candidate.attention.packed, ctx_.stream);
                    ops::detail::mtp_q4_marlin_prepare(candidate.output, ctx_.stream);
                    ops::detail::mtp_q4_marlin_prepare(candidate.post_mixer.gate_up, ctx_.stream);
                    ops::detail::mtp_q4_marlin_prepare(candidate.post_mixer.down, ctx_.stream);
                }
            }
        };
        prepare_mtp_marlin(source);
    }

    for (int layer = layer_range_.begin; layer < layer_range_.end; ++layer) {
        if (ModelConfig::is_full(layer)) {
            FullLayerW& out = full_[static_cast<std::size_t>(ModelConfig::full_idx(layer))];
            const auto& source =
                weights_.full_layers[static_cast<std::size_t>(ModelConfig::full_idx(layer))];
            out.input_norm     = &source.input_norm;
            out.projection     = &source.projection;
            out.o_proj         = &source.output;
            out.q_norm         = &source.query_norm;
            out.k_norm         = &source.key_norm;
            out.post_attn_norm = &source.post_attention_norm;
            out.mlp            = bind_mlp(source.post_mixer);
        } else {
            const std::size_t gidx = static_cast<std::size_t>(ModelConfig::gdn_idx(layer));
            GdnLayerW& out         = gdn_[gidx];
            const auto& source     = weights_.gdn_layers[gidx];
            out.input_norm         = &source.input_norm;
            out.projection         = &source.projection;
            out.conv1d             = &source.convolution;
            out.gdn_norm           = &source.norm;
            out.out_proj           = &source.output;
            out.post_attn_norm     = &source.post_attention_norm;
            out.mlp                = bind_mlp(source.post_mixer);
        }
    }
}

const MtpW& TextContext::mtp_weights() const {
    if (!mtp_enabled()) { throw std::runtime_error("MTP draft weights are not enabled"); }
    return mtp_;
}

void TextContext::mtp_forward_stem(const Tensor& ids, const Tensor& hidden,
                                   const Tensor* input_embeddings, Tensor& x, Tensor& ah) {
    cudaStream_t s     = ctx_.stream;
    const int T        = ids.ne[0] * ids.ne[1];
    Tensor flat_ids    = ids.view({T});
    Tensor flat_hidden = hidden.view({kCfg.hidden, T});

    auto roots = workspace_recipe::mtp_stem<TextConfig>(work_, T, input_embeddings == nullptr);
    Tensor emb;
    if (input_embeddings != nullptr) {
        if (input_embeddings->dtype != DType::BF16 || input_embeddings->ne[0] != kCfg.hidden ||
            input_embeddings->numel() != static_cast<std::int64_t>(kCfg.hidden) * T ||
            !input_embeddings->is_contiguous() || input_embeddings->data == nullptr) {
            throw std::invalid_argument("MTP input embeddings shape mismatch");
        }
        emb = input_embeddings->view({kCfg.hidden, T});
    } else {
        emb = roots.embedding;
        ops::embedding(flat_ids, *embed_, emb, s);
    }

    Tensor e = roots.normalized_embedding;
    Tensor h = roots.normalized_hidden;
    ops::rmsnorm(emb, *mtp_.pre_fc_norm_embedding, kCfg.rms_eps, true, e, s);
    ops::rmsnorm(flat_hidden, *mtp_.pre_fc_norm_hidden, kCfg.rms_eps, true, h, s);

    Tensor fc_in = roots.packed_input;
    ops::mtp_pack_fc_input(e, h, fc_in, s);

    x = roots.residual;
    ops::linear(fc_in, *mtp_.fc, x, s);

    ah = roots.attention_hidden;
    ops::rmsnorm(x, *mtp_.input_norm, kCfg.rms_eps, true, ah, s);
}

void TextContext::mtp_forward_tail(Tensor& x, const Tensor& ah, const Tensor& positions,
                                   const Tensor& rope_positions, ops::GqaExecutionEnvelope envelope,
                                   Tensor& mtp_hidden) {
    cudaStream_t s = ctx_.stream;
    const int T    = x.ne[1];

    const auto projection = workspace_recipe::mtp_attention_projection<TextConfig>(work_, T);
    Tensor q              = projection.query.view({kCfg.head_dim, kCfg.n_q, T});
    Tensor k              = projection.key.view({kCfg.head_dim, kCfg.n_kv, T});
    Tensor gate           = projection.gate.view({kCfg.head_dim, kCfg.n_q, T});
    Tensor v              = projection.value.view({kCfg.head_dim, kCfg.n_kv, T});
    Tensor q_flat         = q.view({kCfg.q_size, T});
    Tensor gate_flat      = gate.view({kCfg.q_size, T});
    Tensor k_flat         = k.view({kCfg.kv_size, T});
    Tensor v_flat         = v.view({kCfg.kv_size, T});
    Variant::mtp_attention_projection(ah, mtp_.payload->attention, q_flat, gate_flat, k_flat,
                                      v_flat, work_, s);

    const auto results = workspace_recipe::mtp_attention_results<TextConfig>(work_, T);
    Tensor qn          = results.normalized_query.view({kCfg.head_dim, kCfg.n_q, T});
    Tensor kn          = results.normalized_key.view({kCfg.head_dim, kCfg.n_kv, T});
    ops::rmsnorm(q, *mtp_.q_norm, kCfg.rms_eps, true, qn, s);
    ops::rmsnorm(k, *mtp_.k_norm, kCfg.rms_eps, true, kn, s);
    Tensor rope_for_op = active_sequence_batch_ != 0 ? rope_positions.view({T}) : rope_positions;
    ops::rope(rope_for_op, kCfg.rotary_dim, kCfg.rope_theta, qn, kn, s);

    Tensor a = results.attention.view({kCfg.head_dim, kCfg.n_q, T});
    const bool long_context_turbo =
        qwen3_6::use_retained_long_context(long_context_computation_,envelope.max_visible_keys);
    const double sink_fraction = long_context_turbo ? 0.001 : 0.0;
    const double tail_fraction = long_context_turbo ? 0.005 : 0.0;
    const bool sparse_single_sequence =
        sink_fraction > 0.0 && tail_fraction > 0.0 &&
        (active_sequence_batch_ == 0 ||
         (active_sequence_batch_ == 1 && active_sequence_width_ == T));
    if(mtp_attention_window_!=0 && !mtp_prefill_active_ &&
       long_context_computation_==qwen3_6::LongContextComputation::Complete &&
       envelope.max_visible_keys>mtp_attention_window_+128U && active_sequence_batch_<=1) {
        if(T>6 || (active_sequence_batch_==1 && active_sequence_width_!=T))
            throw std::logic_error("MTP window requires one contiguous sequence");
        // Aliasing changes only the proposal's visible page set. The fused append
        // keeps original validity masking and writes the exact same physical pages.
        constexpr std::uint32_t sink=128;
        auto window_cache=batch_mtp_kv_->batch_layer_view(0);
        Tensor flat_positions=positions.view({T});
        Tensor local_positions=work_.alloc(DType::I32,{T});
        Tensor local_tables=work_.alloc(DType::I32,{static_cast<int>((mtp_attention_window_+sink)/64+2),1});
        Tensor local_row=work_.alloc(DType::I32,{1});
        const Tensor& original_row=active_backend_kv_table_rows_!=nullptr
            ? *active_backend_kv_table_rows_ : io_.backend_kv_table_row;
        Tensor target_guide,target_guide_position;
        if(std::getenv("NINFER_TARGET_GUIDED_PROPOSAL_ORACLE")!=nullptr &&
           std::getenv("NINFER_TARGET_GUIDE_CAPTURE_ONLY")==nullptr &&
           gqa_execution_override_!=nullptr &&
           gqa_execution_override_->proposal_page_guide(target_guide,target_guide_position)) {
            Tensor scores=work_.alloc(DType::FP32,{window_cache.block_tables.ne[0],24});
            Tensor normalizers=work_.alloc(DType::FP32,{24});
            ops::detail::gqa_target_guided_proposal_view_launch(target_guide,target_guide_position,
                flat_positions,window_cache,original_row,mtp_attention_window_,scores,normalizers,
                local_positions,local_tables,local_row,s,mtp_guide_licensed_columns_);
        } else {
        ops::detail::gqa_recent_view_launch(flat_positions,window_cache.block_tables,original_row,
            mtp_attention_window_,sink,local_positions,local_tables,local_row,s);
        }
        window_cache.block_tables=local_tables;
        const std::uint32_t local_capacity=static_cast<std::uint32_t>(local_tables.ne[0])*64U;
        Tensor valid=active_valid_columns_!=nullptr ? *active_valid_columns_ : Tensor{};
        ops::gqa_attention(qn,kn,v,local_positions,valid,local_row,kAttnScale,window_cache,
            {1U,std::min(envelope.max_visible_keys,local_capacity)},work_,a,s);
    } else if (sparse_single_sequence) {
        Tensor sparse_positions = active_sequence_batch_ == 0 ? positions : positions.view({T});
        PagedKVLayerView cache;
        if (active_sequence_batch_ == 0) {
            cache = mtp_kv_.layer_view(0);
        } else {
            const PagedKVBatchLayerView batch_cache = batch_mtp_kv_->batch_layer_view(0);
            if (batch_cache.block_tables.ne[1] < 1) {
                throw std::logic_error("MTP sink-tail has no batch KV table row");
            }
            cache = {
                .k_pages = batch_cache.k_pages,
                .v_pages = batch_cache.v_pages,
                .k_scale_pages = batch_cache.k_scale_pages,
                .v_scale_pages = batch_cache.v_scale_pages,
                .block_table = batch_cache.block_tables.slice(1, 0, 1).view(
                    {batch_cache.block_tables.ne[0]}),
                .head_dim = batch_cache.head_dim,
                .num_kv_heads = batch_cache.num_kv_heads,
                .dtype = batch_cache.dtype,
                .quant_group = batch_cache.quant_group,
            };
        }
        ops::gqa_kv_append(kn, v, sparse_positions, cache, s);
        const std::int32_t total_splits = ops::detail::gqa_attention_split_capacity(
            kCfg.n_q, T, cache.dtype, envelope);
        if (total_splits < 2) {
            throw std::logic_error("MTP sink-tail needs at least two GQA splits");
        }
        std::int32_t sink_splits = std::clamp(
            static_cast<std::int32_t>(std::llround(total_splits * sink_fraction)),
            1, total_splits - 1);
        std::int32_t tail_splits = std::clamp(
            static_cast<std::int32_t>(std::llround(total_splits * tail_fraction)),
            1, total_splits - sink_splits);
        if (sink_splits + tail_splits > total_splits) {
            tail_splits = total_splits - sink_splits;
        }
        const std::int32_t tail_begin = total_splits - tail_splits;
        Tensor partial_acc = work_.alloc(
            DType::BF16, {kCfg.head_dim, kCfg.n_q, T, total_splits});
        Tensor partial_m = work_.alloc(DType::FP32, {kCfg.n_q, T, total_splits});
        Tensor partial_l = work_.alloc(DType::FP32, {kCfg.n_q, T, total_splits});
        Tensor sink_acc = work_.alloc(DType::FP32, {kCfg.head_dim, kCfg.n_q, T});
        Tensor sink_m = work_.alloc(DType::FP32, {kCfg.n_q, T});
        Tensor sink_l = work_.alloc(DType::FP32, {kCfg.n_q, T});
        Tensor tail_acc = work_.alloc(DType::FP32, {kCfg.head_dim, kCfg.n_q, T});
        Tensor tail_m = work_.alloc(DType::FP32, {kCfg.n_q, T});
        Tensor tail_l = work_.alloc(DType::FP32, {kCfg.n_q, T});
        ops::detail::gqa_attention_cached_small_t_partial_range_launch(
            qn, sparse_positions, kAttnScale, cache, envelope, 0, sink_splits,
            partial_acc, partial_m, partial_l, s);
        ops::detail::gqa_attention_small_t_reduce_range_state_launch(
            kCfg.n_q, T, total_splits, 0, sink_splits,
            partial_acc, partial_m, partial_l, sink_acc, sink_m, sink_l, s);
        ops::detail::gqa_attention_cached_small_t_partial_range_launch(
            qn, sparse_positions, kAttnScale, cache, envelope, tail_begin, tail_splits,
            partial_acc, partial_m, partial_l, s);
        ops::detail::gqa_attention_small_t_reduce_range_state_launch(
            kCfg.n_q, T, total_splits, tail_begin, tail_splits,
            partial_acc, partial_m, partial_l, tail_acc, tail_m, tail_l, s);
        ops::detail::gqa_attention_small_t_merge_two_states_launch(
            kCfg.n_q, T, sink_acc, sink_m, sink_l, tail_acc, tail_m, tail_l, a, s);
    } else if (active_sequence_batch_ != 0) {
        const std::int32_t width = active_sequence_width_;
        if (width <= 0 || width * active_sequence_batch_ != T ||
            active_backend_kv_table_rows_ == nullptr || active_valid_columns_ == nullptr) {
            throw std::logic_error("MTP sequence batch binding is incomplete");
        }
        Tensor q_batch        = qn.view({kCfg.head_dim, kCfg.n_q, width, active_sequence_batch_});
        Tensor k_batch        = kn.view({kCfg.head_dim, kCfg.n_kv, width, active_sequence_batch_});
        Tensor v_batch        = v.view({kCfg.head_dim, kCfg.n_kv, width, active_sequence_batch_});
        Tensor a_batch        = a.view({kCfg.head_dim, kCfg.n_q, width, active_sequence_batch_});
        Tensor position_batch = positions.view({width, active_sequence_batch_});
        ops::gqa_attention(q_batch, k_batch, v_batch, position_batch, *active_valid_columns_,
                           *active_backend_kv_table_rows_, kAttnScale,
                           batch_mtp_kv_->batch_layer_view(0), envelope, work_, a_batch, s);
    } else {
        ops::gqa_attention(qn, kn, v, positions, Tensor{}, io_.backend_kv_table_row, kAttnScale,
                           batch_mtp_kv_->batch_layer_view(0), envelope, work_, a, s);
    }
    ops::sigmoid_mul(gate, a, s);

    const auto post = workspace_recipe::mtp_post_attention<TextConfig>(work_, T);
    Tensor o        = post.output;
    ops::linear(a.view({kCfg.q_size, T}), *mtp_.o_proj, o, s);
    ops::residual_add(o, x, s);

    Tensor mh = post.post_mixer_hidden;
    ops::rmsnorm(x, *mtp_.post_attn_norm, kCfg.rms_eps, true, mh, s);

    {
        auto post_mixer_scope = work_.scope();
        Variant::mtp_post_mixer(mh, mtp_.payload->post_mixer, x, work_, s);
    }

    Tensor flat_mtp_hidden = mtp_hidden.view({kCfg.hidden, T});
    ops::rmsnorm(x, *mtp_.norm, kCfg.rms_eps, true, flat_mtp_hidden, s);
}

void TextContext::mtp_forward_core(const Tensor& ids, const Tensor& hidden, const Tensor& positions,
                                   const Tensor& rope_positions, ops::GqaExecutionEnvelope envelope,
                                   Tensor& mtp_hidden, const Tensor* input_embeddings) {
    if (batch_mtp_kv_ == nullptr) { throw std::runtime_error("MTP forward is not enabled"); }
    auto scratch_scope = work_.scope();
    Tensor x;
    Tensor ah;
    mtp_forward_stem(ids, hidden, input_embeddings, x, ah);
    mtp_forward_tail(x, ah, positions, rope_positions, envelope, mtp_hidden);
}

void TextContext::mtp_prefill_chunk(const Tensor& ids, const Tensor& hidden,
                                    const Tensor* input_embeddings, const Tensor& positions,
                                    const Tensor& rope_positions,
                                    ops::GqaExecutionEnvelope envelope, bool final_chunk,
                                    Tensor* final_hidden, Tensor* logits, Tensor* draft_token) {
    // The proposal window never changes teacher-forced MTP prefix ingestion.
    ScopedValue<bool> prefill_scope(mtp_prefill_active_,true);

    if (!mtp_kv_.valid()) { throw std::runtime_error("MTP prefill is not enabled"); }
    const int T = ids.ne[0];
    if (T <= 0 || static_cast<std::uint32_t>(T) > prefill_chunk_) {
        throw std::invalid_argument("MTP prefill chunk T must be in [1,prefill_chunk]");
    }
    nvtx::ScopedRange mtp_prefill_range(nvtx::Name::PrefillMtpChunk, nvtx::Category::Mtp,
                                        static_cast<std::uint64_t>(T));
    require_tensor_shape(ids, DType::I32, {T}, "MTP prefill ids");
    require_tensor_shape(hidden, DType::BF16, {kCfg.hidden, T}, "MTP prefill hidden");
    require_tensor_shape(positions, DType::I32, {T}, "MTP prefill positions");
    if (rope_positions.dtype != DType::I32 || rope_positions.ne[0] != T ||
        (rope_positions.ne[1] != 1 && rope_positions.ne[1] != 3) || rope_positions.ne[2] != 1 ||
        rope_positions.ne[3] != 1 || !rope_positions.is_contiguous() ||
        rope_positions.data == nullptr) {
        throw std::invalid_argument("MTP prefill rope positions must be [T] or [T,3]");
    }
    if (final_chunk) {
        if (final_hidden == nullptr || logits == nullptr || draft_token == nullptr) {
            throw std::invalid_argument("MTP final prefill outputs are required");
        }
        require_tensor_shape(*final_hidden, DType::BF16, {kCfg.hidden, 1},
                             "MTP final prefill hidden");
        require_tensor_shape(*logits, DType::BF16, {kCfg.vocab, 1}, "MTP final prefill logits");
        require_tensor_shape(*draft_token, DType::I32, {1}, "MTP final prefill draft token");
    }

    cudaStream_t s     = ctx_.stream;
    auto scratch_scope = work_.scope();
    Tensor x_last;
    Tensor ah_last;
    if (final_chunk) {
        x_last  = work_.alloc(DType::BF16, {kCfg.hidden, 1});
        ah_last = work_.alloc(DType::BF16, {kCfg.hidden, 1});
    }

    {
        auto bulk_scope = work_.scope();
        Tensor x;
        Tensor ah;
        mtp_forward_stem(ids, hidden, input_embeddings, x, ah);

        Tensor k_flat = work_.alloc(DType::BF16, {kCfg.kv_size, T});
        Tensor v_flat = work_.alloc(DType::BF16, {kCfg.kv_size, T});
        Variant::mtp_kv_projection(ah, mtp_.payload->attention, k_flat, v_flat, work_, s);
        Tensor k  = k_flat.view({kCfg.head_dim, kCfg.n_kv, T});
        Tensor v  = v_flat.view({kCfg.head_dim, kCfg.n_kv, T});
        Tensor kn = work_.alloc(DType::BF16, {kCfg.head_dim, kCfg.n_kv, T});
        ops::rmsnorm(k, *mtp_.k_norm, kCfg.rms_eps, true, kn, s);
        ops::rope(rope_positions, kCfg.rotary_dim, kCfg.rope_theta, kn, s);
        ops::gqa_kv_append(kn, v, positions, mtp_kv_.layer_view(0), s);

        if (final_chunk) {
            const std::size_t column_bytes =
                static_cast<std::size_t>(kCfg.hidden) * dtype_size(DType::BF16);
            const auto* x_src = static_cast<const unsigned char*>(x.data) +
                                static_cast<std::size_t>(T - 1) * column_bytes;
            const auto* ah_src = static_cast<const unsigned char*>(ah.data) +
                                 static_cast<std::size_t>(T - 1) * column_bytes;
            CUDA_CHECK(
                cudaMemcpyAsync(x_last.data, x_src, column_bytes, cudaMemcpyDeviceToDevice, s));
            CUDA_CHECK(
                cudaMemcpyAsync(ah_last.data, ah_src, column_bytes, cudaMemcpyDeviceToDevice, s));
        }
    }

    if (final_chunk) {
        Tensor q_flat    = work_.alloc(DType::BF16, {kCfg.q_size, 1});
        Tensor gate_flat = work_.alloc(DType::BF16, {kCfg.q_size, 1});
        Variant::mtp_q_gate_projection(ah_last, mtp_.payload->attention, q_flat, gate_flat, work_,
                                       s, weights_.features.retained_mtp_packed_prefix);
        Tensor q    = q_flat.view({kCfg.head_dim, kCfg.n_q, 1});
        Tensor gate = gate_flat.view({kCfg.head_dim, kCfg.n_q, 1});
        Tensor qn   = work_.alloc(DType::BF16, {kCfg.head_dim, kCfg.n_q, 1});
        ops::rmsnorm(q, *mtp_.q_norm, kCfg.rms_eps, true, qn, s);
        Tensor last_position = positions.slice(0, T - 1, 1);
        Tensor last_rope_position;
        if (rope_positions.ne[1] == 1) {
            last_rope_position = rope_positions.slice(0, T - 1, 1);
        } else {
            last_rope_position = work_.alloc(DType::I32, {1, 3});
            for (int axis = 0; axis < 3; ++axis) {
                const auto* src = static_cast<const std::int32_t*>(rope_positions.data) +
                                  static_cast<std::size_t>(axis) * T + (T - 1);
                auto* dst = static_cast<std::int32_t*>(last_rope_position.data) + axis;
                CUDA_CHECK(
                    cudaMemcpyAsync(dst, src, sizeof(std::int32_t), cudaMemcpyDeviceToDevice, s));
            }
        }
        ops::rope(last_rope_position, kCfg.rotary_dim, kCfg.rope_theta, qn, s);

        Tensor a = work_.alloc(DType::BF16, {kCfg.head_dim, kCfg.n_q, 1});
        ops::gqa_attention_cached(qn, last_position, kAttnScale, mtp_kv_.layer_view(0), envelope,
                                  work_, a, s);
        ops::sigmoid_mul(gate, a, s);

        Tensor o = work_.alloc(DType::BF16, {kCfg.hidden, 1});
        ops::linear(a.view({kCfg.q_size, 1}), *mtp_.o_proj, o, s);
        ops::residual_add(o, x_last, s);

        Tensor mh = work_.alloc(DType::BF16, {kCfg.hidden, 1});
        ops::rmsnorm(x_last, *mtp_.post_attn_norm, kCfg.rms_eps, true, mh, s);
        {
            auto post_mixer_scope = work_.scope();
            Variant::mtp_post_mixer(mh, mtp_.payload->post_mixer, x_last, work_, s);
        }
        ops::rmsnorm(x_last, *mtp_.norm, kCfg.rms_eps, true, *final_hidden, s);
        proposal_argmax(*final_hidden, *logits, *draft_token);
    }
}

void TextContext::proposal_argmax(const Tensor& hidden, Tensor& logits, Tensor& proposal_tokens) {
    const int T = hidden.ne[1];
    require_tensor_shape(hidden, DType::BF16, {kCfg.hidden, T}, "proposal hidden");
    require_tensor_shape(proposal_tokens, DType::I32, {T}, "proposal tokens");
    require_tensor_window(logits, DType::BF16, kCfg.vocab, T, "proposal logits");
#if defined(NINFER_CUDA_SM75) && defined(NINFER_QWEN36_SM75_DENSE_A8)
    if (weights_.features.fast_full_vocab_draft) {
        if (T != 1 || lm_head_ == nullptr ||
            !ops::detail::w8_vocab_w4_a8_ready(*lm_head_)) {
            throw std::logic_error("fast full-vocabulary drafting requires a prepared C1 head");
        }
        // Proposal ranking only. Acceptance always uses the separately frozen
        // target policy. The experimental retained-target composition is explicit;
        // preparing this cache never enables a target shortcut implicitly.
        Tensor output_logits = matrix_window(logits, 1);
        ops::detail::w8_vocab_w4_a8_launch_t1(hidden, *lm_head_, output_logits,
                                               proposal_tokens, ctx_.stream);
        return;
    }
#endif
    if (proposal_head_ != nullptr) {
#if defined(NINFER_CUDA_SM75) && defined(NINFER_QWEN36_SM75_DENSE_A8)
        if (T == 1 && ops::detail::q4_proposal_b2_ready(*proposal_head_)) {
            ops::detail::q4_proposal_b2_launch(hidden, *proposal_head_, proposal_tokens,
                                               ctx_.stream);
            ops::proposal_remap_token_ids(proposal_tokens, proposal_head_ids_, proposal_head_n_,
                                          ctx_.stream);
            return;
        }
#endif
        Tensor proposal_logits = work_.alloc(DType::BF16, {proposal_head_n_, T});
        ops::linear(hidden, *proposal_head_, proposal_logits, ctx_.stream);
        ops::argmax(proposal_logits, proposal_tokens, proposal_head_n_, ctx_.stream);
        ops::proposal_remap_token_ids(proposal_tokens, proposal_head_ids_, proposal_head_n_,
                                      ctx_.stream);
    } else {
        Tensor output_logits = matrix_window(logits, T);
        ops::linear(hidden, *lm_head_, output_logits, ctx_.stream);
        ops::argmax(output_logits, proposal_tokens, kCfg.token_domain, ctx_.stream);
    }
}

void TextContext::mtp_forward_batch(const Tensor& ids, const Tensor& hidden,
                                    const Tensor& positions, ops::GqaExecutionEnvelope envelope,
                                    Tensor& mtp_hidden, int logits_column, Tensor* logits,
                                    Tensor* draft_token, const Tensor* explicit_rope_positions,
                                    const Tensor* input_embeddings) {
    if (batch_mtp_kv_ == nullptr) { throw std::runtime_error("MTP forward is not enabled"); }
    const int T = ids.ne[0];
    if (T <= 0 || static_cast<std::uint32_t>(T) > prefill_chunk_) {
        throw std::invalid_argument("MTP batch T must be in [1,prefill_chunk]");
    }
    require_tensor_shape(ids, DType::I32, {T}, "MTP ids");
    require_tensor_shape(positions, DType::I32, {T}, "MTP positions");
    require_tensor_shape(hidden, DType::BF16, {kCfg.hidden, T}, "MTP hidden");
    require_tensor_shape(mtp_hidden, DType::BF16, {kCfg.hidden, T}, "MTP output hidden");
    if (logits_column >= T) { throw std::invalid_argument("MTP logits column out of range"); }
    if (logits_column >= 0) {
        if (logits == nullptr || draft_token == nullptr) {
            throw std::invalid_argument("MTP logits and draft_token outputs are required");
        }
        require_tensor_shape(*logits, DType::BF16, {kCfg.vocab, 1}, "MTP logits");
        require_tensor_shape(*draft_token, DType::I32, {1}, "MTP draft token");
    }

    auto position_scope = work_.scope();
    Tensor generated_rope_positions;
    const Tensor* rope_positions = explicit_rope_positions;
    if (rope_positions == nullptr) {
        generated_rope_positions = work_.alloc(DType::I32, {T});
        ops::offset_i32_positions(positions, io_.rope_delta, generated_rope_positions, ctx_.stream);
        rope_positions = &generated_rope_positions;
    } else if (rope_positions->dtype != DType::I32 || rope_positions->ne[0] != T ||
               (rope_positions->ne[1] != 1 && rope_positions->ne[1] != 3) ||
               rope_positions->ne[2] != 1 || rope_positions->ne[3] != 1 ||
               !rope_positions->is_contiguous() || rope_positions->data == nullptr) {
        throw std::invalid_argument("MTP explicit rope positions must be [T] or [T,3]");
    }
    mtp_forward_core(ids, hidden, positions, *rope_positions, envelope, mtp_hidden,
                     input_embeddings);

    if (logits_column >= 0) {
        auto logits_scope = work_.scope();
        Tensor col        = mtp_hidden.slice(1, logits_column, 1);
        proposal_argmax(col, *logits, *draft_token);
    }
}

void TextContext::mtp_forward_ar_step(const Tensor& token, const Tensor& previous_hidden,
                                      const Tensor& position, ops::GqaExecutionEnvelope envelope,
                                      Tensor& mtp_hidden, Tensor& logits, Tensor& draft_token) {
    if (batch_mtp_kv_ == nullptr) { throw std::runtime_error("MTP forward is not enabled"); }
    require_tensor_shape(token, DType::I32, {1}, "MTP AR token");
    require_tensor_shape(position, DType::I32, {1}, "MTP AR position");
    require_tensor_shape(previous_hidden, DType::BF16, {kCfg.hidden, 1}, "MTP AR previous hidden");
    require_tensor_shape(mtp_hidden, DType::BF16, {kCfg.hidden, 1}, "MTP AR output hidden");
    require_tensor_shape(logits, DType::BF16, {kCfg.vocab, 1}, "MTP AR logits");
    require_tensor_shape(draft_token, DType::I32, {1}, "MTP AR draft token");

    auto position_scope  = work_.scope();
    Tensor rope_position = work_.alloc(DType::I32, {1});
    ops::offset_i32_positions(position, io_.rope_delta, rope_position, ctx_.stream);
    mtp_forward_core(token, previous_hidden, position, rope_position, envelope, mtp_hidden,
                     nullptr);
    auto logits_scope = work_.scope();
    proposal_argmax(mtp_hidden, logits, draft_token);
}

void TextContext::ordinary_decode_batch(const Tensor& ids, const Tensor& cache_positions,
                                        const Tensor& rope_positions, const Tensor& kv_table_rows,
                                        const Tensor& linear_state_slots,
                                        ops::GqaExecutionEnvelope envelope, Tensor& hidden,
                                        Tensor& logits) {
    const std::int32_t batch = ids.ne[0];
    if (batch <= 0 || batch > static_cast<std::int32_t>(kMaximumConcurrency)) {
        throw std::invalid_argument("ordinary decode batch size must be in [1,8]");
    }
    require_tensor_shape(ids, DType::I32, {batch}, "ordinary decode ids");
    require_tensor_shape(cache_positions, DType::I32, {batch}, "ordinary decode cache positions");
    require_tensor_shape(rope_positions, DType::I32, {batch}, "ordinary decode RoPE positions");
    require_tensor_shape(kv_table_rows, DType::I32, {batch}, "ordinary decode KV rows");
    require_tensor_shape(linear_state_slots, DType::I32, {batch},
                         "ordinary decode Linear Attention slots");
    require_tensor_shape(hidden, DType::BF16, {kCfg.hidden, batch}, "ordinary decode hidden");
    require_tensor_shape(logits, DType::BF16, {kCfg.vocab, batch}, "ordinary decode logits");

    cudaStream_t stream = ctx_.stream;
    work_.reset();
    {
        ScopedPositions cache_binding(active_cache_positions_, cache_positions);
        ScopedPositions rope_binding(active_rope_positions_, rope_positions);
        ScopedEnvelope envelope_binding(active_gqa_envelope_, envelope);
        ScopedValue<const Tensor*> kv_binding(active_kv_table_rows_, &kv_table_rows);
        ScopedValue<const Tensor*> state_binding(active_linear_state_slots_, &linear_state_slots);
        ScopedValue<std::int32_t> batch_binding(active_sequence_batch_, batch);
        ScopedValue<std::int32_t> width_binding(active_sequence_width_, 1);

        Tensor x = work_.alloc(DType::BF16, {kCfg.hidden, batch});
        ops::embedding(ids, *embed_, x, stream);
        NullTap tap;
        run_layers(x, Phase::Verify, tap);
        ops::rmsnorm(x, *final_norm_, kCfg.rms_eps, true, hidden, stream);
        ops::linear(hidden, *lm_head_, logits, stream);
    }
    work_.reset();
}

void TextContext::forward_token_shard(const Tensor& id, const Tensor& cache_position,
                                      const Tensor& rope_position, const Tensor& kv_table_row,
                                      const Tensor& linear_state_slot,
                                      ops::GqaExecutionEnvelope envelope, Tensor& residual) {
    if (!layer_range_.input_embedding || embed_ == nullptr) {
        throw std::logic_error("text shard does not own the token embedding");
    }
    require_tensor_shape(id, DType::I32, {1}, "text shard token");
    require_tensor_shape(cache_position, DType::I32, {1}, "text shard cache position");
    require_tensor_shape(rope_position, DType::I32, {1}, "text shard RoPE position");
    require_tensor_shape(kv_table_row, DType::I32, {1}, "text shard KV row");
    require_tensor_shape(linear_state_slot, DType::I32, {1}, "text shard state slot");
    require_tensor_shape(residual, DType::BF16, {kCfg.hidden, 1}, "text shard residual");

    cudaStream_t stream = ctx_.stream;
    work_.reset();
    {
        ScopedPositions cache_binding(active_cache_positions_, cache_position);
        ScopedPositions rope_binding(active_rope_positions_, rope_position);
        ScopedEnvelope envelope_binding(active_gqa_envelope_, envelope);
        ScopedValue<const Tensor*> kv_binding(active_kv_table_rows_, &kv_table_row);
        ScopedValue<const Tensor*> state_binding(active_linear_state_slots_, &linear_state_slot);
        ScopedValue<std::int32_t> batch_binding(active_sequence_batch_, 1);
        ScopedValue<std::int32_t> width_binding(active_sequence_width_, 1);
        ops::embedding(id, *embed_, residual, stream);
        NullTap tap;
        run_layers(residual, Phase::Verify, tap);
    }
    work_.reset();
}

void TextContext::forward_hidden_shard(Tensor& residual, const Tensor& cache_position,
                                       const Tensor& rope_position, const Tensor& kv_table_row,
                                       const Tensor& linear_state_slot,
                                       ops::GqaExecutionEnvelope envelope) {
    require_tensor_shape(cache_position, DType::I32, {1}, "text shard cache position");
    require_tensor_shape(rope_position, DType::I32, {1}, "text shard RoPE position");
    require_tensor_shape(kv_table_row, DType::I32, {1}, "text shard KV row");
    require_tensor_shape(linear_state_slot, DType::I32, {1}, "text shard state slot");
    require_tensor_shape(residual, DType::BF16, {kCfg.hidden, 1}, "text shard residual");

    work_.reset();
    {
        ScopedPositions cache_binding(active_cache_positions_, cache_position);
        ScopedPositions rope_binding(active_rope_positions_, rope_position);
        ScopedEnvelope envelope_binding(active_gqa_envelope_, envelope);
        ScopedValue<const Tensor*> kv_binding(active_kv_table_rows_, &kv_table_row);
        ScopedValue<const Tensor*> state_binding(active_linear_state_slots_, &linear_state_slot);
        ScopedValue<std::int32_t> batch_binding(active_sequence_batch_, 1);
        ScopedValue<std::int32_t> width_binding(active_sequence_width_, 1);
        NullTap tap;
        run_layers(residual, Phase::Verify, tap);
    }
    work_.reset();
}

void TextContext::target_verify_token_shard(
    const Tensor& ids, const Tensor& cache_positions, const Tensor& rope_positions,
    const Tensor& valid_columns, const Tensor& kv_table_row, const Tensor& linear_state_slot,
    const GdnReplayRecords& replay_records, ops::GqaExecutionEnvelope envelope, Tensor& residual) {
    if (!layer_range_.input_embedding || embed_ == nullptr) {
        throw std::logic_error("target verifier shard does not own the token embedding");
    }
    const std::int32_t width = ids.ne[0];
    const std::int32_t batch = ids.ne[1];
    if (width <= 0 || width > static_cast<std::int32_t>(kMtpDecodeMaximumDrafts + 1U) ||
        batch <= 0 || batch > static_cast<std::int32_t>(kMaximumConcurrency)) {
        throw std::invalid_argument("target verifier shard shape is outside MTP3 domain");
    }
    const std::int32_t columns = width * batch;
    require_tensor_shape(ids, DType::I32, {width, batch}, "target verifier shard ids");
    require_tensor_shape(cache_positions, DType::I32, {width, batch},
                         "target verifier shard cache positions");
    require_tensor_shape(rope_positions, DType::I32, {width, batch},
                         "target verifier shard RoPE positions");
    require_tensor_shape(valid_columns, DType::I32, {batch},
                         "target verifier shard valid columns");
    require_tensor_shape(kv_table_row, DType::I32, {batch}, "target verifier shard KV row");
    require_tensor_shape(linear_state_slot, DType::I32, {batch},
                         "target verifier shard state slot");
    require_tensor_shape(residual, DType::BF16, {kCfg.hidden, width, batch},
                         "target verifier shard residual");

    cudaStream_t stream = ctx_.stream;
    work_.reset();
    set_gdn_state_action(GdnStateAction::RecordForReplay, &replay_records);
    try {
        ScopedPositions cache_binding(active_cache_positions_, cache_positions);
        ScopedPositions rope_binding(active_rope_positions_, rope_positions);
        ScopedEnvelope envelope_binding(active_gqa_envelope_, envelope);
        ScopedValue<const Tensor*> kv_binding(active_kv_table_rows_, &kv_table_row);
        ScopedValue<const Tensor*> state_binding(active_linear_state_slots_, &linear_state_slot);
        ScopedValue<const Tensor*> valid_binding(active_valid_columns_, &valid_columns);
        ScopedValue<std::int32_t> batch_binding(active_sequence_batch_, batch);
        ScopedValue<std::int32_t> width_binding(active_sequence_width_, width);
        Tensor flat_ids = ids.view({columns});
        Tensor flat_residual = residual.view({kCfg.hidden, columns});
        ops::embedding(flat_ids, *embed_, flat_residual, stream);
        NullTap tap;
        run_layers(flat_residual, Phase::Verify, tap);
    } catch (...) {
        set_gdn_state_action(GdnStateAction::UpdateInPlace, nullptr);
        work_.reset();
        throw;
    }
    set_gdn_state_action(GdnStateAction::UpdateInPlace, nullptr);
    work_.reset();
}

bool TextContext::target_verify_hidden_shard(
    Tensor& residual, const Tensor& cache_positions, const Tensor& rope_positions,
    const Tensor& valid_columns, const Tensor& kv_table_row, const Tensor& linear_state_slot,
    const GdnReplayRecords& replay_records, ops::GqaExecutionEnvelope envelope, Tensor& hidden,
    Tensor& logits, Tensor& target_tokens, bool licensed_full_round, bool require_logits) {
    if (!layer_range_.output_projection || final_norm_ == nullptr || lm_head_ == nullptr) {
        throw std::logic_error("target verifier shard does not own the output projection");
    }
    const std::int32_t width = residual.ne[1];
    const std::int32_t batch = residual.ne[2];
    if (width <= 0 || width > static_cast<std::int32_t>(kMtpDecodeMaximumDrafts + 1U) ||
        batch <= 0 || batch > static_cast<std::int32_t>(kMaximumConcurrency)) {
        throw std::invalid_argument("target verifier shard shape is outside MTP3 domain");
    }
    const std::int32_t columns = width * batch;
    require_tensor_shape(residual, DType::BF16, {kCfg.hidden, width, batch},
                         "target verifier shard residual");
    require_tensor_shape(cache_positions, DType::I32, {width, batch},
                         "target verifier shard cache positions");
    require_tensor_shape(rope_positions, DType::I32, {width, batch},
                         "target verifier shard RoPE positions");
    require_tensor_shape(valid_columns, DType::I32, {batch},
                         "target verifier shard valid columns");
    require_tensor_shape(kv_table_row, DType::I32, {batch}, "target verifier shard KV row");
    require_tensor_shape(linear_state_slot, DType::I32, {batch},
                         "target verifier shard state slot");
    require_tensor_shape(hidden, DType::BF16, {kCfg.hidden, width, batch},
                         "target verifier shard hidden");
    require_tensor_shape(logits, DType::BF16, {kCfg.vocab, width, batch},
                         "target verifier shard logits");
    require_tensor_shape(target_tokens, DType::I32, {width, batch},
                         "target verifier shard tokens");

    bool logits_materialized = true;
    work_.reset();
    set_gdn_state_action(GdnStateAction::RecordForReplay, &replay_records);
    try {
        ScopedPositions cache_binding(active_cache_positions_, cache_positions);
        ScopedPositions rope_binding(active_rope_positions_, rope_positions);
        ScopedEnvelope envelope_binding(active_gqa_envelope_, envelope);
        ScopedValue<const Tensor*> kv_binding(active_kv_table_rows_, &kv_table_row);
        ScopedValue<const Tensor*> state_binding(active_linear_state_slots_, &linear_state_slot);
        ScopedValue<const Tensor*> valid_binding(active_valid_columns_, &valid_columns);
        ScopedValue<std::int32_t> batch_binding(active_sequence_batch_, batch);
        ScopedValue<std::int32_t> width_binding(active_sequence_width_, width);
        Tensor flat_residual = residual.view({kCfg.hidden, columns});
        NullTap tap;
        run_layers(flat_residual, Phase::Verify, tap);
        for (std::int32_t row = 0; row < batch; ++row) {
            Tensor row_residual = residual.slice(2, row, 1).view({kCfg.hidden, width});
            Tensor row_hidden = hidden.slice(2, row, 1).view({kCfg.hidden, width});
            Tensor row_logits = logits.slice(2, row, 1).view({kCfg.vocab, width});
            Tensor row_tokens = target_tokens.slice(1, row, 1).view({width});
            ops::rmsnorm(row_residual, *final_norm_, kCfg.rms_eps, true, row_hidden, ctx_.stream);
            if (!require_logits && head_execution_override_ != nullptr &&
                head_execution_override_->execute_greedy(row_hidden,*lm_head_,row_tokens,
                                                         kCfg.token_domain,ctx_.stream)) {
                logits_materialized = false;
                continue;
            }

#if defined(NINFER_CUDA_SM75) && defined(NINFER_QWEN36_SM75_DENSE_A8)
            if (licensed_full_round && width == 5 &&
                (weights_.features.allow_optimized_target_head() &&
                 ops::detail::w8_vocab_w4_a8_ready(*lm_head_))) {
                Tensor final_hidden = row_hidden.slice(1, 4, 1);
                Tensor final_logits = row_logits.slice(1, 4, 1);
                Tensor final_token = row_tokens.slice(0, 4, 1);
                ops::detail::w8_vocab_w4_a8_launch_t1(final_hidden, *lm_head_, final_logits,
                                                       final_token, ctx_.stream);
            } else if (width == 4 && (weights_.features.allow_optimized_target_head() &&
                 ops::detail::w8_vocab_w4_a8_ready(*lm_head_))) {
                ops::detail::w8_vocab_w4_a8_launch(row_hidden, *lm_head_, row_logits, row_tokens,
                                                    ctx_.stream);
            } else {
                if (head_execution_override_ == nullptr ||
                    !head_execution_override_->execute(row_hidden,*lm_head_,row_logits,ctx_.stream)) {
                    ops::linear(row_hidden, *lm_head_, row_logits, ctx_.stream);
                }
                ops::argmax(row_logits, row_tokens, kCfg.token_domain, ctx_.stream);
            }
#else
            ops::linear(row_hidden, *lm_head_, row_logits, ctx_.stream);
            ops::argmax(row_logits, row_tokens, kCfg.token_domain, ctx_.stream);
#endif
        }
    } catch (...) {
        set_gdn_state_action(GdnStateAction::UpdateInPlace, nullptr);
        work_.reset();
        throw;
    }
    set_gdn_state_action(GdnStateAction::UpdateInPlace, nullptr);
    work_.reset();
    return logits_materialized;
}

void TextContext::include_target_draft_candidates(const Tensor& hidden,const Tensor& drafts,
                                                   const Tensor& extents,Tensor& logits,Tensor& tokens) {
#if defined(NINFER_CUDA_SM75) && defined(NINFER_QWEN36_SM75_DENSE_A8)
    if(!weights_.features.target_draft_union || !weights_.features.allow_optimized_target_head() ||
       !ops::detail::w8_vocab_w4_a8_ready(*lm_head_)) {
        throw std::logic_error("target candidate union is missing its frozen prepared-head policy");
    }
    Tensor h=hidden.view({kCfg.hidden,4}),d=drafts.view({3});
    Tensor e=extents.view({1}),l=logits.view({kCfg.vocab,4}),t=tokens.view({4});
    ops::detail::w8_vocab_w4_a8_include_drafts(h,*lm_head_,d,e,l,t,ctx_.stream);
#else
    throw std::logic_error("target candidate union is unavailable for this target");
#endif
}

void TextContext::score_target_verify_mandatory_drafts(const Tensor& hidden,
                                                        const Tensor& drafts,
                                                        Tensor& logits, bool enabled) {
    if (!enabled) { return; }
#if defined(NINFER_CUDA_SM75) && defined(NINFER_QWEN36_SM75_DENSE_A8)
    if (lm_head_ != nullptr && (weights_.features.allow_optimized_target_head() &&
                 ops::detail::w8_vocab_w4_a8_ready(*lm_head_))) {
        ops::detail::w8_vocab_w4_a8_score_mandatory_drafts(hidden, *lm_head_, drafts, logits,
                                                            ctx_.stream);
    }
#else
    (void)hidden;
    (void)drafts;
    (void)logits;
#endif
}

void TextContext::prefill_token_shard(const Tensor& ids, const Tensor& cache_positions,
                                      const Tensor& rope_positions, const Tensor& kv_table_row,
                                      ops::GqaExecutionEnvelope envelope, Tensor& residual) {
    if (!layer_range_.input_embedding || embed_ == nullptr) {
        throw std::logic_error("text shard does not own the token embedding");
    }
    const std::int32_t tokens = ids.ne[0];
    if (tokens <= 0 || tokens > static_cast<std::int32_t>(prefill_chunk_)) {
        throw std::invalid_argument("text shard prefill width is outside its configured chunk");
    }
    require_tensor_shape(ids, DType::I32, {tokens}, "text shard prefill ids");
    require_tensor_shape(cache_positions, DType::I32, {tokens},
                         "text shard prefill cache positions");
    require_tensor_shape(rope_positions, DType::I32, {tokens},
                         "text shard prefill RoPE positions");
    require_tensor_shape(kv_table_row, DType::I32, {1}, "text shard prefill KV row");
    require_tensor_shape(residual, DType::BF16, {kCfg.hidden, tokens},
                         "text shard prefill residual");

    cudaStream_t stream = ctx_.stream;
    work_.reset();
    {
        ScopedPositions cache_binding(active_cache_positions_, cache_positions);
        ScopedPositions rope_binding(active_rope_positions_, rope_positions);
        ScopedEnvelope envelope_binding(active_gqa_envelope_, envelope);
        ScopedValue<const Tensor*> kv_binding(active_kv_table_rows_, &kv_table_row);
        ops::embedding(ids, *embed_, residual, stream);
        NullTap tap;
        run_layers(residual, Phase::Prefill, tap);
    }
    work_.reset();
}

void TextContext::prefill_multimodal_token_shard(
    const Tensor& ids, const Tensor& cache_positions, const Tensor& rope_positions,
    const Tensor& visual_embeddings, std::int32_t visual_begin,
    std::span<const std::int32_t> local_scatter_indices, const Tensor& kv_table_row,
    ops::GqaExecutionEnvelope envelope, Tensor& residual) {
    if (!layer_range_.input_embedding || embed_ == nullptr) {
        throw std::logic_error("multimodal text shard does not own the token embedding");
    }
    const std::int32_t tokens = ids.ne[0];
    if (tokens <= 0 || tokens > static_cast<std::int32_t>(prefill_chunk_)) {
        throw std::invalid_argument("multimodal shard prefill width is outside its chunk");
    }
    require_tensor_shape(ids, DType::I32, {tokens}, "multimodal shard ids");
    require_tensor_shape(cache_positions, DType::I32, {tokens},
                         "multimodal shard cache positions");
    require_tensor_shape(rope_positions, DType::I32, {tokens, 3},
                         "multimodal shard mRoPE positions");
    require_tensor_shape(kv_table_row, DType::I32, {1}, "multimodal shard KV row");
    require_tensor_shape(residual, DType::BF16, {kCfg.hidden, tokens},
                         "multimodal shard residual");
    if (!local_scatter_indices.empty() &&
        (visual_begin < 0 || visual_embeddings.dtype != DType::BF16 ||
         visual_embeddings.ne[0] != kCfg.hidden || visual_embeddings.ne[1] < visual_begin ||
         static_cast<std::size_t>(visual_embeddings.ne[1] - visual_begin) <
             local_scatter_indices.size())) {
        throw std::invalid_argument("multimodal shard visual embedding window is invalid");
    }
    for (const std::int32_t column : local_scatter_indices) {
        if (column < 0 || column >= tokens) {
            throw std::invalid_argument("multimodal shard scatter column is outside chunk");
        }
    }

    cudaStream_t stream = ctx_.stream;
    work_.reset();
    {
        ScopedPositions cache_binding(active_cache_positions_, cache_positions);
        ScopedPositions rope_binding(active_rope_positions_, rope_positions);
        ScopedEnvelope envelope_binding(active_gqa_envelope_, envelope);
        ScopedValue<const Tensor*> kv_binding(active_kv_table_rows_, &kv_table_row);
        ops::embedding(ids, *embed_, residual, stream);
        if (!local_scatter_indices.empty()) {
            Tensor indices = work_.alloc(
                DType::I32, {static_cast<std::int32_t>(local_scatter_indices.size())});
            copy_i32(local_scatter_indices.data(), indices, stream);
            Tensor embeddings = visual_embeddings.slice(
                1, visual_begin, static_cast<std::int32_t>(local_scatter_indices.size()));
            ops::scatter(embeddings, indices, residual, stream);
        }
        NullTap tap;
        run_layers(residual, Phase::Prefill, tap);
    }
    work_.reset();
}

void TextContext::prefill_hidden_shard(Tensor& residual, const Tensor& cache_positions,
                                       const Tensor& rope_positions, const Tensor& kv_table_row,
                                       ops::GqaExecutionEnvelope envelope) {
    const std::int32_t tokens = residual.ne[1];
    if (tokens <= 0 || tokens > static_cast<std::int32_t>(prefill_chunk_)) {
        throw std::invalid_argument("text shard prefill width is outside its configured chunk");
    }
    require_tensor_shape(residual, DType::BF16, {kCfg.hidden, tokens},
                         "text shard prefill residual");
    require_tensor_shape(cache_positions, DType::I32, {tokens},
                         "text shard prefill cache positions");
    const bool linear_rope = rope_positions.dtype == DType::I32 &&
                             rope_positions.ne[0] == tokens && rope_positions.ne[1] == 1 &&
                             rope_positions.ne[2] == 1 && rope_positions.ne[3] == 1 &&
                             rope_positions.is_contiguous() && rope_positions.data != nullptr;
    const bool multimodal_rope = rope_positions.dtype == DType::I32 &&
                                 rope_positions.ne[0] == tokens && rope_positions.ne[1] == 3 &&
                                 rope_positions.ne[2] == 1 && rope_positions.ne[3] == 1 &&
                                 rope_positions.is_contiguous() && rope_positions.data != nullptr;
    if (!linear_rope && !multimodal_rope) {
        throw std::invalid_argument("text shard prefill RoPE positions shape mismatch");
    }
    require_tensor_shape(kv_table_row, DType::I32, {1}, "text shard prefill KV row");

    work_.reset();
    {
        ScopedPositions cache_binding(active_cache_positions_, cache_positions);
        ScopedPositions rope_binding(active_rope_positions_, rope_positions);
        ScopedEnvelope envelope_binding(active_gqa_envelope_, envelope);
        ScopedValue<const Tensor*> kv_binding(active_kv_table_rows_, &kv_table_row);
        NullTap tap;
        run_layers(residual, Phase::Prefill, tap);
    }
    work_.reset();
}

void TextContext::project_shard_head(const Tensor& residual, Tensor& normalized, Tensor& logits,
                                     Tensor& token) {
    if (!layer_range_.output_projection || final_norm_ == nullptr || lm_head_ == nullptr) {
        throw std::logic_error("text shard does not own the output projection");
    }
    const std::int32_t columns = residual.ne[1];
    if (columns <= 0 || columns > static_cast<std::int32_t>(kMaximumConcurrency)) {
        throw std::invalid_argument("text shard output width must be in [1,8]");
    }
    require_tensor_shape(residual, DType::BF16, {kCfg.hidden, columns}, "text shard residual");
    require_tensor_shape(normalized, DType::BF16, {kCfg.hidden, columns}, "text shard normalized");
    require_tensor_shape(logits, DType::BF16, {kCfg.vocab, columns}, "text shard logits");
    require_tensor_shape(token, DType::I32, {columns}, "text shard output token");
    ops::rmsnorm(residual, *final_norm_, kCfg.rms_eps, true, normalized, ctx_.stream);
    if (head_execution_override_ == nullptr ||
        !head_execution_override_->execute(normalized,*lm_head_,logits,ctx_.stream)) {
        ops::linear(normalized, *lm_head_, logits, ctx_.stream);
    }
    ops::argmax(logits, token, kCfg.token_domain, ctx_.stream);
}

template <class Tap>
void TextContext::target_verify_batch_impl(const Tensor& ids, const Tensor& cache_positions,
                                           const Tensor& rope_positions,
                                           const Tensor& valid_columns, const Tensor& kv_table_rows,
                                           const Tensor& linear_state_slots,
                                           ops::GqaExecutionEnvelope envelope, Tensor& hidden,
                                           Tensor& logits, Tensor& target_tokens, Tap& tap) {
    const std::int32_t width = ids.ne[0];
    const std::int32_t batch = ids.ne[1];
    if (width <= 0 || width > static_cast<std::int32_t>(kDFlashDecodeMaximumWidth) || batch <= 0 ||
        batch > static_cast<std::int32_t>(kMaximumConcurrency)) {
        throw std::invalid_argument("target verify batch shape is outside the supported domain");
    }
    const std::int32_t columns = width * batch;
    require_tensor_shape(ids, DType::I32, {width, batch}, "target verify batch ids");
    require_tensor_shape(cache_positions, DType::I32, {width, batch},
                         "target verify batch cache positions");
    require_tensor_shape(rope_positions, DType::I32, {width, batch},
                         "target verify batch RoPE positions");
    require_tensor_shape(valid_columns, DType::I32, {batch}, "target verify batch valid columns");
    require_tensor_shape(kv_table_rows, DType::I32, {batch}, "target verify batch KV rows");
    require_tensor_shape(linear_state_slots, DType::I32, {batch},
                         "target verify batch Linear Attention slots");
    require_tensor_shape(hidden, DType::BF16, {kCfg.hidden, width, batch},
                         "target verify batch hidden");
    require_tensor_shape(logits, DType::BF16, {kCfg.vocab, width, batch},
                         "target verify batch logits");
    require_tensor_shape(target_tokens, DType::I32, {width, batch}, "target verify batch tokens");

    cudaStream_t stream = ctx_.stream;
    work_.reset();
    {
        ScopedPositions cache_binding(active_cache_positions_, cache_positions);
        ScopedPositions rope_binding(active_rope_positions_, rope_positions);
        ScopedEnvelope envelope_binding(active_gqa_envelope_, envelope);
        ScopedValue<const Tensor*> kv_binding(active_kv_table_rows_, &kv_table_rows);
        ScopedValue<const Tensor*> state_binding(active_linear_state_slots_, &linear_state_slots);
        ScopedValue<const Tensor*> valid_binding(active_valid_columns_, &valid_columns);
        ScopedValue<std::int32_t> batch_binding(active_sequence_batch_, batch);
        ScopedValue<std::int32_t> width_binding(active_sequence_width_, width);

        Tensor x        = work_.alloc(DType::BF16, {kCfg.hidden, columns});
        Tensor flat_ids = ids.view({columns});
        ops::embedding(flat_ids, *embed_, x, stream);
        if constexpr (Tap::enabled) { tap.begin(x); }
        run_layers(x, Phase::Verify, tap);
        if constexpr (requires { tap.capture_positions(cache_positions, stream); }) {
            tap.capture_positions(cache_positions, stream);
        }
        Tensor flat_hidden = hidden.view({kCfg.hidden, columns});
        Tensor flat_logits = logits.view({kCfg.vocab, columns});
        Tensor flat_tokens = target_tokens.view({columns});
        ops::rmsnorm(x, *final_norm_, kCfg.rms_eps, true, flat_hidden, stream);
#if defined(NINFER_CUDA_SM75) && defined(NINFER_QWEN36_SM75_DENSE_A8)
        if (columns == 4 && (weights_.features.allow_optimized_target_head() &&
                 ops::detail::w8_vocab_w4_a8_ready(*lm_head_))) {
            ops::detail::w8_vocab_w4_a8_launch(flat_hidden, *lm_head_, flat_logits,
                                                flat_tokens, stream);
        } else {
            ops::linear(flat_hidden, *lm_head_, flat_logits, stream);
            ops::argmax(flat_logits, flat_tokens, kCfg.token_domain, stream);
        }
#else
        ops::linear(flat_hidden, *lm_head_, flat_logits, stream);
        ops::argmax(flat_logits, flat_tokens, kCfg.token_domain, stream);
#endif
    }
    work_.reset();
}

void TextContext::target_verify_batch(const Tensor& ids, const Tensor& cache_positions,
                                      const Tensor& rope_positions, const Tensor& valid_columns,
                                      const Tensor& kv_table_rows, const Tensor& linear_state_slots,
                                      ops::GqaExecutionEnvelope envelope, Tensor& hidden,
                                      Tensor& logits, Tensor& target_tokens) {
    NullTap tap;
    target_verify_batch_impl(ids, cache_positions, rope_positions, valid_columns, kv_table_rows,
                             linear_state_slots, envelope, hidden, logits, target_tokens, tap);
}

void TextContext::target_verify_batch(const Tensor& ids, const Tensor& cache_positions,
                                      const Tensor& rope_positions, const Tensor& valid_columns,
                                      const Tensor& kv_table_rows, const Tensor& linear_state_slots,
                                      ops::GqaExecutionEnvelope envelope, Tensor& hidden,
                                      Tensor& logits, Tensor& target_tokens,
                                      DFlashFeatureSink& sink) {
    target_verify_batch_impl(ids, cache_positions, rope_positions, valid_columns, kv_table_rows,
                             linear_state_slots, envelope, hidden, logits, target_tokens, sink);
}

void TextContext::mtp_prefill_shard_chunk(
    std::span<const TokenId> full_ids, std::uint32_t chunk_begin,
    const Tensor& target_residual, const Tensor& positions, const Tensor& rope_positions,
    bool final_chunk, TokenId generated_token, bool build_proposal,
    const Tensor* visual_embeddings,
    std::span<const std::int32_t> visual_scatter_indices) {
    if (!mtp_enabled() || !io_.mtp) {
        throw std::logic_error("sharded MTP prefill requires canonical MTP state");
    }
    const std::int32_t columns = target_residual.ne[1];
    if (columns <= 0 || columns > static_cast<std::int32_t>(prefill_chunk_) ||
        chunk_begin >= full_ids.size() ||
        static_cast<std::uint64_t>(chunk_begin) + static_cast<std::uint32_t>(columns) >
            full_ids.size()) {
        throw std::invalid_argument("sharded MTP prefill window is outside the prompt");
    }
    require_tensor_shape(target_residual, DType::BF16, {kCfg.hidden, columns},
                         "sharded MTP target residual");
    require_tensor_shape(positions, DType::I32, {columns}, "sharded MTP positions");
    if (rope_positions.dtype != DType::I32 || rope_positions.ne[0] != columns ||
        (rope_positions.ne[1] != 1 && rope_positions.ne[1] != 3) ||
        rope_positions.ne[2] != 1 || rope_positions.ne[3] != 1 ||
        !rope_positions.is_contiguous() || rope_positions.data == nullptr) {
        throw std::invalid_argument(
            "sharded MTP RoPE positions must be [T] or [T,3]");
    }
    const qwen3_6::MtpAlignmentWindow window = qwen3_6::plan_mtp_alignment_window(
        static_cast<std::uint32_t>(full_ids.size()), chunk_begin,
        static_cast<std::uint32_t>(columns));
    if (window.final_column_uses_generated_token != final_chunk ||
        (build_proposal && !final_chunk)) {
        throw std::invalid_argument("sharded MTP final-column contract is inconsistent");
    }

    cudaStream_t stream = ctx_.stream;
    work_.reset();
    Tensor target_hidden = matrix_window(prefill_hidden_, columns);
    ops::rmsnorm(target_residual, *final_norm_, kCfg.rms_eps, true, target_hidden, stream);

    std::vector<std::int32_t> shifted_ids(static_cast<std::size_t>(columns));
    const std::int32_t prompt_columns = columns - static_cast<std::int32_t>(final_chunk);
    for (std::int32_t column = 0; column < prompt_columns; ++column) {
        shifted_ids[static_cast<std::size_t>(column)] =
            full_ids[static_cast<std::size_t>(window.shifted_embedding_begin) +
                     static_cast<std::size_t>(column)];
    }
    if (final_chunk) { shifted_ids.back() = generated_token; }
    Tensor mtp_ids = work_.alloc(DType::I32, {columns});
    copy_i32(shifted_ids.data(), mtp_ids, stream);
    Tensor mtp_input_embeddings;
    const Tensor* mtp_input_embeddings_ptr = nullptr;
    if (visual_embeddings != nullptr) {
        if (visual_embeddings->dtype != DType::BF16 ||
            visual_embeddings->ne[0] != kCfg.hidden ||
            visual_embeddings->ne[1] <= 0 || visual_embeddings->data == nullptr) {
            throw std::invalid_argument("sharded MTP visual embeddings are invalid");
        }
        mtp_input_embeddings = work_.alloc(DType::BF16, {kCfg.hidden, columns});
        ops::embedding(mtp_ids, *embed_, mtp_input_embeddings, stream);
        const qwen3_6::MtpVisualOverlap overlap = qwen3_6::shifted_visual_overlap(
            visual_scatter_indices, static_cast<std::uint32_t>(full_ids.size()), window);
        if (!overlap.empty()) {
            Tensor shifted_indices = workspace_recipe::visual_scatter_indices(
                work_, static_cast<std::int32_t>(overlap.size()));
            qwen3_6::detail::scatter_shifted_visual_embeddings(
                mtp_input_embeddings, *visual_embeddings, overlap, shifted_indices, stream);
        }
        mtp_input_embeddings_ptr = &mtp_input_embeddings;
    } else if (!visual_scatter_indices.empty()) {
        throw std::invalid_argument(
            "sharded MTP visual scatter indices require visual embeddings");
    }
    const auto visible = chunk_begin + static_cast<std::uint32_t>(columns);
    const ops::GqaExecutionEnvelope envelope{visible, visible};

    if (!build_proposal) {
        mtp_prefill_chunk(mtp_ids, target_hidden, mtp_input_embeddings_ptr, positions,
                          rope_positions, envelope, false, nullptr, nullptr, nullptr);
        work_.reset();
        return;
    }

    if (mtp_proposal_extent_ == 0 ||
        mtp_proposal_extent_ > static_cast<std::uint32_t>(io_.mtp->draft_tokens.ne[0])) {
        throw std::logic_error("sharded MTP proposal extent is outside the configured state");
    }
    Tensor logits = matrix_window(io_.logits, 1);
    Tensor draft0 = io_.mtp->draft_tokens.slice(0, 0, 1);
    mtp_prefill_chunk(mtp_ids, target_hidden, mtp_input_embeddings_ptr, positions,
                      rope_positions, envelope, true, &io_.mtp->ar_hidden, &logits, &draft0);

    Tensor ar_position = io_.mtp->position.slice(0, 0, 1);
    ops::set_i32_scalar(ar_position, static_cast<std::int32_t>(visible), stream);
    for (std::uint32_t index = 1; index < mtp_proposal_extent_; ++index) {
        Tensor previous = io_.mtp->draft_tokens.slice(0, static_cast<std::int32_t>(index - 1U), 1);
        Tensor next = io_.mtp->draft_tokens.slice(0, static_cast<std::int32_t>(index), 1);
        Tensor next_hidden = work_.alloc(DType::BF16, {kCfg.hidden, 1});
        const auto ar_visible = visible + index;
        mtp_forward_ar_step(previous, io_.mtp->ar_hidden, ar_position,
                            ops::GqaExecutionEnvelope{ar_visible, ar_visible}, next_hidden, logits,
                            next);
        CUDA_CHECK(cudaMemcpyAsync(io_.mtp->ar_hidden.data, next_hidden.data,
                                   io_.mtp->ar_hidden.bytes(), cudaMemcpyDeviceToDevice, stream));
        ops::increment_i32_scalar(ar_position, stream);
    }
    work_.reset();
}

void TextContext::mtp_forward_decode_batch(const Tensor& ids, const Tensor& hidden,
                                           const Tensor& cache_positions,
                                           const Tensor& rope_positions,
                                           const Tensor& valid_columns, const Tensor& kv_table_rows,
                                           ops::GqaExecutionEnvelope envelope, Tensor& mtp_hidden) {
    if (batch_mtp_kv_ == nullptr) { throw std::runtime_error("MTP forward is not enabled"); }
    const std::int32_t width = ids.ne[0];
    const std::int32_t batch = ids.ne[1];
    if (width <= 0 || width > static_cast<std::int32_t>(kMaximumMtpDraftTokens + 1) || batch <= 0 ||
        batch > static_cast<std::int32_t>(kMaximumConcurrency)) {
        throw std::invalid_argument("MTP decode batch shape is outside the supported domain");
    }
    require_tensor_shape(ids, DType::I32, {width, batch}, "MTP decode batch ids");
    require_tensor_shape(hidden, DType::BF16, {kCfg.hidden, width, batch},
                         "MTP decode batch target hidden");
    require_tensor_shape(cache_positions, DType::I32, {width, batch},
                         "MTP decode batch cache positions");
    require_tensor_shape(rope_positions, DType::I32, {width, batch},
                         "MTP decode batch RoPE positions");
    require_tensor_shape(valid_columns, DType::I32, {batch}, "MTP decode batch valid columns");
    require_tensor_shape(kv_table_rows, DType::I32, {batch}, "MTP decode batch KV rows");
    require_tensor_shape(mtp_hidden, DType::BF16, {kCfg.hidden, width, batch},
                         "MTP decode batch hidden");

    ScopedValue<const Tensor*> backend_binding(active_backend_kv_table_rows_, &kv_table_rows);
    ScopedValue<const Tensor*> valid_binding(active_valid_columns_, &valid_columns);
    ScopedValue<std::int32_t> batch_binding(active_sequence_batch_, batch);
    ScopedValue<std::int32_t> width_binding(active_sequence_width_, width);
    mtp_forward_core(ids, hidden, cache_positions, rope_positions, envelope, mtp_hidden, nullptr);
}

void TextContext::mtp_propose_batch(const Tensor& hidden, Tensor& logits, Tensor& draft_tokens) {
    const std::int32_t batch = hidden.ne[1];
    require_tensor_shape(hidden, DType::BF16, {kCfg.hidden, batch}, "MTP proposal batch hidden");
    require_tensor_shape(logits, DType::BF16, {kCfg.vocab, batch}, "MTP proposal batch logits");
    require_tensor_shape(draft_tokens, DType::I32, {batch}, "MTP proposal batch tokens");
    proposal_argmax(hidden, logits, draft_tokens);
}

void TextContext::attn_mix(const FullLayerW& w, Tensor& x, int global_layer, int fidx,
                           Phase ph) {
    cudaStream_t s = ctx_.stream;
    const int T    = x.ne[1];
    if (active_gqa_envelope_ == nullptr) {
        throw std::logic_error("Text GQA execution envelope is not set");
    }

    const auto projection = workspace_recipe::text_attention_projection<TextConfig>(work_, T);
    Tensor h              = projection.hidden;
    ops::rmsnorm(x, *w.input_norm, kCfg.rms_eps, true, h, s);

    Tensor q         = projection.query.view({kCfg.head_dim, kCfg.n_q, T});
    Tensor gate      = projection.gate.view({kCfg.head_dim, kCfg.n_q, T});
    Tensor k         = projection.key.view({kCfg.head_dim, kCfg.n_kv, T});
    Tensor v         = projection.value.view({kCfg.head_dim, kCfg.n_kv, T});
    Tensor q_flat    = q.view({kCfg.q_size, T});
    Tensor gate_flat = gate.view({kCfg.q_size, T});
    Tensor k_flat    = k.view({kCfg.kv_size, T});
    Tensor v_flat    = v.view({kCfg.kv_size, T});
    Variant::attention_projection(h, *w.projection, q_flat, gate_flat, k_flat, v_flat, ph, work_,
                                  s, ph == Phase::Prefill ? prefill_mlp_override_ : nullptr);

    const auto results = workspace_recipe::text_attention_results<TextConfig>(work_, T);
    Tensor qn          = results.normalized_query.view({kCfg.head_dim, kCfg.n_q, T});
    Tensor kn          = results.normalized_key.view({kCfg.head_dim, kCfg.n_kv, T});
    ops::rmsnorm(q, *w.q_norm, kCfg.rms_eps, true, qn, s);
    ops::rmsnorm(k, *w.k_norm, kCfg.rms_eps, true, kn, s);
    const Tensor& cache_positions =
        active_cache_positions_ != nullptr ? *active_cache_positions_ : io_.pos;
    const Tensor& rope_positions =
        active_rope_positions_ != nullptr ? *active_rope_positions_ : io_.rope_pos;
    Tensor rope_for_op = active_sequence_batch_ != 0 ? rope_positions.view({T}) : rope_positions;
    ops::rope(rope_for_op, kCfg.rotary_dim, kCfg.rope_theta, qn, kn, s);

    Tensor a = results.attention.view({kCfg.head_dim, kCfg.n_q, T});
    const Tensor& kv_table_rows =
        active_kv_table_rows_ != nullptr ? *active_kv_table_rows_ : io_.text_kv_table_row;
    const Tensor valid = active_valid_columns_ != nullptr ? *active_valid_columns_ : Tensor{};
    const PagedKVBatchLayerView cache =
        batch_text_kv_->batch_layer_view(fidx - full_attention_base_);
    bool handled = false;
    if (!active_packed_text_.empty()) {
        std::int32_t off=0;
        for(const auto& chunk:active_packed_text_) {
            const auto width=static_cast<std::int32_t>(chunk.width);
            Tensor qi=qn.slice(2,off,width),ki=kn.slice(2,off,width),vi=v.slice(2,off,width);
            Tensor oi=a.slice(2,off,width),pi=cache_positions.slice(0,off,width);
            ops::set_i32_scalar(io_.text_kv_table_row,chunk.text_kv_row,s);
            auto temporary=work_.scope();
            const auto visible=chunk.begin+chunk.width;
            ops::gqa_attention(qi,ki,vi,pi,Tensor{},io_.text_kv_table_row,kAttnScale,
                              cache,{visible,visible},work_,oi,s);
            off += width;
        }
        handled=true;
    }
    if (!handled && gqa_execution_override_ != nullptr && active_sequence_batch_ <= 1) {
        handled = gqa_execution_override_->execute(
            global_layer, qn, kn, v, cache_positions.view({T}), valid, kv_table_rows,
            kAttnScale, cache, *active_gqa_envelope_, work_, a, s);
    }
    if (!handled && active_sequence_batch_ != 0) {
        const std::int32_t width = active_sequence_width_;
        if (width <= 0 || width * active_sequence_batch_ != T) {
            throw std::logic_error("Text sequence batch binding does not match aggregate columns");
        }
        Tensor q_batch        = qn.view({kCfg.head_dim, kCfg.n_q, width, active_sequence_batch_});
        Tensor k_batch        = kn.view({kCfg.head_dim, kCfg.n_kv, width, active_sequence_batch_});
        Tensor v_batch        = v.view({kCfg.head_dim, kCfg.n_kv, width, active_sequence_batch_});
        Tensor a_batch        = a.view({kCfg.head_dim, kCfg.n_q, width, active_sequence_batch_});
        Tensor position_batch = cache_positions.view({width, active_sequence_batch_});
        ops::gqa_attention(q_batch, k_batch, v_batch, position_batch, valid, kv_table_rows,
                           kAttnScale, cache, *active_gqa_envelope_, work_, a_batch, s);
    } else if (!handled) {
        ops::gqa_attention(qn, kn, v, cache_positions, Tensor{}, kv_table_rows, kAttnScale,
                           cache, *active_gqa_envelope_, work_, a, s);
    }
    ops::sigmoid_mul(gate, a, s);

    Tensor output_hidden = a.view({kCfg.q_size, T});
    if (!(ph == Phase::Prefill && prefill_mlp_override_ != nullptr &&
          prefill_mlp_override_->residual(output_hidden, *w.o_proj, x, s))) {
        Variant::attention_output_projection(output_hidden, *w.o_proj, x, ph, work_, s);
    }
}

void TextContext::gdn_mix(const GdnLayerW& w, Tensor& x, int gidx, Phase ph) {
    cudaStream_t s = ctx_.stream;
    const int T    = x.ne[1];

    const auto control = workspace_recipe::gdn_control<TextConfig>(work_, T);
    Tensor h           = control.hidden;
    Tensor g           = control.g;
    Tensor beta        = control.beta;
    Variant::gdn_norm_control_projection(x, *w.input_norm, kCfg.rms_eps, *w.projection, h, g, beta,
                                         work_, s);

    const auto projection = workspace_recipe::gdn_projection<TextConfig>(work_, T);
    Tensor z              = projection.output_gate.view({kCfg.gdn_v_dim, kCfg.gdn_v_heads, T});
    Tensor qc             = projection.query;
    Tensor kc             = projection.key;
    Tensor vc             = projection.value;
    if (ph == Phase::Verify) {
        if (active_sequence_batch_ == 0 || active_linear_state_slots_ == nullptr) {
            throw std::logic_error(
                "Verify GDN requires an explicit sequence batch and state slots");
        }
        const std::int32_t width = active_sequence_width_;
        if (width <= 0 || width * active_sequence_batch_ != T) {
            throw std::logic_error("GDN sequence batch binding does not match aggregate columns");
        }
        Tensor projection_input = h.view({kCfg.hidden, width, active_sequence_batch_});
        Tensor query_output     = qc.view({kCfg.key_dim, width, active_sequence_batch_});
        Tensor key_output       = kc.view({kCfg.key_dim, width, active_sequence_batch_});
        Tensor value_output     = vc.view({kCfg.value_dim, width, active_sequence_batch_});
        Tensor gate_output      = z.view({kCfg.value_dim, width, active_sequence_batch_});
        Tensor& conv_states     = state_.conv.at(static_cast<std::size_t>(gidx - gdn_state_base_));
        const Tensor valid = active_valid_columns_ != nullptr ? *active_valid_columns_ : Tensor{};
        if (gdn_state_action_ == GdnStateAction::RecordForReplay) {
            if (replay_records_ == nullptr) {
                throw std::logic_error("Replay-record GDN has no record storage");
            }
            GdnReplayRecordLayer records =
                replay_records_->layer(gidx - gdn_state_base_, active_sequence_batch_);
            Variant::gdn_input_projection_record(projection_input, *w.projection, *w.conv1d,
                                                 conv_states, valid, *active_linear_state_slots_,
                                                 records.conv, query_output, key_output,
                                                 value_output, gate_output, ph, work_, s);
        } else {
            Variant::gdn_input_projection_snapshot(
                projection_input, *w.projection, *w.conv1d, conv_states, valid,
                *active_linear_state_slots_, *active_linear_state_slots_, query_output, key_output,
                value_output, gate_output, ph, work_, s);
        }
    } else {
        // Q/K/V and output-gate roots above own the surviving data. The
        // projected/convolved copies are dead after the ordered extraction
        // kernels below; later work on this same stream can reuse their arena.
        auto conv_scope=work_.scope();
        const auto conv = workspace_recipe::gdn_prefill_conv<TextConfig>(work_, T);
        Tensor qkv      = conv.projected;
        Variant::gdn_input_projection(h, *w.projection, qkv, z, ph, work_, s,
                                       ph == Phase::Prefill ? prefill_mlp_override_ : nullptr);
        Tensor qkv_c = conv.convolved;
        if(active_packed_text_.empty()) {
            Tensor conv_state=state_.conv_slot(static_cast<std::uint32_t>(gidx-gdn_state_base_),
                                               linear_state_current_slot_);
            ops::causal_conv1d_silu(qkv,*w.conv1d,conv_state,conv_state,qkv_c,s);
        } else {
            int off=0;
            for(const auto& chunk:active_packed_text_) {
                const int width=static_cast<int>(chunk.width);
                Tensor input=qkv.slice(1,off,width);
                Tensor output=qkv_c.slice(1,off,width);
                Tensor state=state_.conv_slot(static_cast<std::uint32_t>(gidx-gdn_state_base_),
                                              chunk.state_slot);
                ops::causal_conv1d_silu(input,*w.conv1d,state,state,output,s);
                off += width;
            }
        }
        ops::extract_bf16_columns(qkv_c, 0, qc, s);
        ops::extract_bf16_columns(qkv_c, kCfg.key_dim, kc, s);
        ops::extract_bf16_columns(qkv_c, 2 * kCfg.key_dim, vc, s);
    }

    Tensor q_recurrent = qc.view({kCfg.gdn_k_dim, kCfg.gdn_k_heads, T});
    Tensor k_recurrent = kc.view({kCfg.gdn_k_dim, kCfg.gdn_k_heads, T});

    Tensor vv = vc.view({kCfg.gdn_v_dim, kCfg.gdn_v_heads, T});
    Tensor o  = workspace_recipe::gdn_recurrent_output<TextConfig>(work_, T).view(
        {kCfg.gdn_v_dim, kCfg.gdn_v_heads, T});
    if (ph == Phase::Verify) {
        Tensor& recurrent_states = state_.recurrent.at(static_cast<std::size_t>(gidx - gdn_state_base_));
        const std::int32_t width = active_sequence_width_;
        Tensor q_batch =
            q_recurrent.view({kCfg.gdn_k_dim, kCfg.gdn_k_heads, width, active_sequence_batch_});
        Tensor k_batch =
            k_recurrent.view({kCfg.gdn_k_dim, kCfg.gdn_k_heads, width, active_sequence_batch_});
        Tensor v_batch = vv.view({kCfg.gdn_v_dim, kCfg.gdn_v_heads, width, active_sequence_batch_});
        Tensor g_batch = g.view({kCfg.gdn_v_heads, width, active_sequence_batch_});
        Tensor beta_batch = beta.view({kCfg.gdn_v_heads, width, active_sequence_batch_});
        Tensor out_batch =
            o.view({kCfg.gdn_v_dim, kCfg.gdn_v_heads, width, active_sequence_batch_});
        const Tensor valid = active_valid_columns_ != nullptr ? *active_valid_columns_ : Tensor{};
        if (gdn_state_action_ == GdnStateAction::RecordForReplay) {
            GdnReplayRecordLayer records =
                replay_records_->layer(gidx - gdn_state_base_, active_sequence_batch_);
            ops::gated_delta_net_replay_record(q_batch, k_batch, v_batch, g_batch, beta_batch,
                                               kGdnScale, recurrent_states, valid,
                                               *active_linear_state_slots_, records.key,
                                               records.value, records.gate, out_batch, s);
        } else {
            ops::gated_delta_net_snapshot(q_batch, k_batch, v_batch, g_batch, beta_batch, kGdnScale,
                                          /*normalize_qk=*/true, recurrent_states, valid,
                                          *active_linear_state_slots_, *active_linear_state_slots_,
                                          out_batch, s);
        }
    } else {
        if(active_packed_text_.empty()) {
            Tensor state=state_.recurrent_slot(static_cast<std::uint32_t>(gidx-gdn_state_base_),
                                               linear_state_current_slot_);
            ops::gated_delta_net(q_recurrent,k_recurrent,vv,g,beta,kGdnScale,true,work_,state,o,s);
        } else {
            int off=0;
            for(const auto& chunk:active_packed_text_) {
                const int width=static_cast<int>(chunk.width);
                Tensor qi=q_recurrent.slice(2,off,width),ki=k_recurrent.slice(2,off,width);
                Tensor vi=vv.slice(2,off,width),oi=o.slice(2,off,width);
                Tensor gi=g.slice(1,off,width),bi=beta.slice(1,off,width);
                Tensor state=state_.recurrent_slot(static_cast<std::uint32_t>(gidx-gdn_state_base_),
                                                   chunk.state_slot);
                auto temporary=work_.scope();
                ops::gated_delta_net(qi,ki,vi,gi,bi,kGdnScale,true,work_,state,oi,s);
                off += width;
            }
        }
    }

    Tensor on = workspace_recipe::gdn_normalized_output<TextConfig>(work_, T).view(
        {kCfg.gdn_v_dim, kCfg.gdn_v_heads, T});
    ops::gated_rmsnorm(o, *w.gdn_norm, z, kCfg.rms_eps, on, s);

    Tensor output_hidden = on.view({kCfg.value_dim, T});
    if (!(ph == Phase::Prefill && prefill_mlp_override_ != nullptr &&
          prefill_mlp_override_->residual(output_hidden, *w.out_proj, x, s))) {
        Variant::gdn_output_projection(output_hidden, *w.out_proj, x, ph, work_, s);
    }
}

void TextContext::mlp_tail(int global_layer, const Tensor* post_norm, const MlpW& m, Tensor& x,
                           Phase ph) {
    cudaStream_t s = ctx_.stream;
    const int T    = x.ne[1];
    // Preserve the canonical workspace allocation while replacing the separate
    // RMSNorm + Q4 input-quantize pair with an exact fused SM75 path.
    Tensor h = workspace_recipe::post_mixer_hidden<TextConfig>(work_, T);
    bool normalized = false;
    const auto try_override = [&]<class Payload>(const Payload& payload) {
        if constexpr (requires(const Payload& value) { value.gate_up; value.down; }) {
            if (mlp_execution_override_ != nullptr && ph == Phase::Verify && (T == 4 || T == 5)) {
                ops::rmsnorm(x, *post_norm, kCfg.rms_eps, true, h, s);
                normalized = true;
                return mlp_execution_override_->execute(
                    global_layer, h, payload.gate_up, payload.down, x, s);
            }
        }
        return false;
    };
    if (try_override(*m.payload)) { return; }
#if defined(NINFER_CUDA_SM75) && defined(NINFER_QWEN36_SM75_DENSE_A8)
    if (!normalized && Variant::try_post_mixer_rmsnorm_sm75(
                           x, *post_norm, kCfg.rms_eps, *m.payload, ph, work_, s)) {
        return;
    }
#endif
    if (!normalized) { ops::rmsnorm(x, *post_norm, kCfg.rms_eps, true, h, s); }
    const auto try_prefill = [&]<class Payload>(const Payload& payload) {
        if constexpr (requires { payload.gate_up; payload.down; }) {
            if (ph == Phase::Prefill && prefill_mlp_override_ != nullptr && T >= 32) {
                return prefill_mlp_override_->execute(global_layer, h, payload.gate_up,
                                                       payload.down, x, s);
            }
        }
        return false;
    };
    if (!try_prefill(*m.payload)) { Variant::post_mixer(h, *m.payload, x, ph, work_, s); }
}

template <class Tap>
void TextContext::run_layers(Tensor& x, Phase ph, Tap& tap) {
    const bool prefill = ph == Phase::Prefill;
    const bool long_context_verify =
        ph == Phase::Verify && TextConfig::hidden == 5120 && TextConfig::layers == 64 &&
        x.ne[1] == 5 && active_gqa_envelope_ != nullptr &&
        qwen3_6::use_retained_long_context(long_context_computation_,active_gqa_envelope_->max_visible_keys);
    const std::uint64_t verify_pass = long_context_verify ? verify_pass_count_++ : 0;
    const auto skip_ffn = [&](int layer) {
        return long_context_verify && verify_pass >= kLongContextVerifyFfnWarmup &&
               ((kLongContextVerifyFfnSkipMask >> layer) & 1ULL) != 0ULL;
    };
    for (int layer = layer_range_.begin; layer < layer_range_.end; ++layer) {
        if (ModelConfig::is_full(layer)) {
            const int fidx         = ModelConfig::full_idx(layer);
            const FullLayerW& full = full_.at(static_cast<std::size_t>(fidx));
            nvtx::ScopedRange layer_range(
                prefill ? nvtx::Name::PrefillLayerFull : nvtx::Name::VerifyLayerFull,
                nvtx::Category::Attention, static_cast<std::uint64_t>(layer));
            {
                nvtx::ScopedRange mixer_range(
                    prefill ? nvtx::Name::PrefillAttention : nvtx::Name::VerifyAttention,
                    nvtx::Category::Attention, static_cast<std::uint64_t>(layer));
                auto mixer_scope = work_.scope();
                attn_mix(full, x, layer, fidx, ph);
            }
            {
                nvtx::ScopedRange post_mixer_range(
                    prefill ? nvtx::Name::PrefillPostMixer : nvtx::Name::VerifyPostMixer,
                    nvtx::Category::PostMixer, static_cast<std::uint64_t>(layer));
                auto mlp_scope = work_.scope();
                if (!skip_ffn(layer)) { mlp_tail(layer, full.post_attn_norm, full.mlp, x, ph); }
                if constexpr (Tap::enabled) { tap.capture_layer(layer, x, ctx_.stream); }
            }
        } else {
            const int gidx       = ModelConfig::gdn_idx(layer);
            const GdnLayerW& gdn = gdn_.at(static_cast<std::size_t>(gidx));
            nvtx::ScopedRange layer_range(prefill ? nvtx::Name::PrefillLayerGdn
                                                  : nvtx::Name::VerifyLayerGdn,
                                          nvtx::Category::Gdn, static_cast<std::uint64_t>(layer));
            {
                nvtx::ScopedRange mixer_range(
                    prefill ? nvtx::Name::PrefillGdn : nvtx::Name::VerifyGdn, nvtx::Category::Gdn,
                    static_cast<std::uint64_t>(layer));
                auto mixer_scope = work_.scope();
                gdn_mix(gdn, x, gidx, ph);
            }
            {
                nvtx::ScopedRange post_mixer_range(
                    prefill ? nvtx::Name::PrefillPostMixer : nvtx::Name::VerifyPostMixer,
                    nvtx::Category::PostMixer, static_cast<std::uint64_t>(layer));
                auto mlp_scope = work_.scope();
                if (!skip_ffn(layer)) { mlp_tail(layer, gdn.post_attn_norm, gdn.mlp, x, ph); }
                if constexpr (Tap::enabled) { tap.capture_layer(layer, x, ctx_.stream); }
            }
        }
    }
}

void TextContext::run_layers(Tensor& x, Phase ph) {
    NullTap tap;
    run_layers(x, ph, tap);
}

void TextContext::prefill_packed_text(std::span<const PackedTextChunk> chunks) {
    if(chunks.size()<2 || chunks.size()>kMaximumConcurrency ||
       !mtp_enabled() || prefill_hidden_.data==nullptr || !prefill_mlp_override_) {
        throw std::invalid_argument("packed nonfinal prefill shape or MTP storage is invalid");
    }
    std::uint64_t total_columns=0;
    for(std::size_t i=0;i<chunks.size();++i) {
        const auto& c=chunks[i];
        total_columns += c.width;
        if(c.width==0 || c.begin>=c.token_ids.size() ||
           c.width>=c.token_ids.size()-c.begin || !c.mtp_cache.valid()) {
            throw std::invalid_argument("packed prefill crossed its nonfinal boundary");
        }
        for(std::size_t j=0;j<i;++j) {
            if(c.state_slot==chunks[j].state_slot || c.text_kv_row==chunks[j].text_kv_row) {
                throw std::invalid_argument("packed requests cannot share state or cache rows");
            }
        }
    }
    if(total_columns<prefill_mlp_override_->minimum_batch_tokens() || total_columns>prefill_chunk_)
        throw std::invalid_argument("packed prefill exceeds its reserved work unit");
    const int total=static_cast<int>(total_columns);
    cudaStream_t stream=ctx_.stream;
    work_.reset();
    {
        auto roots=workspace_recipe::text_prefill_roots<TextConfig>(work_,total,0,0);
        std::vector<std::int32_t> ids(static_cast<std::size_t>(total)),positions(ids.size());
        std::size_t offset=0;
        for(const auto& c:chunks) {
            for(std::uint32_t j=0;j<c.width;++j) {
                ids[offset+j]=c.token_ids[c.begin+j];
                positions[offset+j]=static_cast<std::int32_t>(c.begin+j);
            }
            offset+=c.width;
        }
        copy_i32(ids.data(),roots.ids,stream);
        copy_i32(positions.data(),roots.positions,stream);
        Tensor x=roots.residual;
        ops::embedding(roots.ids,*embed_,x,stream);
        {
            ScopedPositions cache_binding(active_cache_positions_,roots.positions);
            ScopedPositions rope_binding(active_rope_positions_,roots.positions);
            const auto max_visible=*std::max_element(positions.begin(),positions.end())+1;
            const ops::GqaExecutionEnvelope envelope{static_cast<std::uint32_t>(max_visible),
                                                     static_cast<std::uint32_t>(max_visible)};
            ScopedEnvelope envelope_binding(active_gqa_envelope_,envelope);
            ScopedValue<std::span<const PackedTextChunk>> pack_binding(active_packed_text_,chunks);
            NullTap tap;
            run_layers(x,Phase::Prefill,tap);
        }
        Tensor normalized=matrix_window(prefill_hidden_,total);
        ops::rmsnorm(x,*final_norm_,kCfg.rms_eps,true,normalized,stream);
        // Every shifted MTP token comes from its own owner's actual prompt.
        // The last prompt token stays on the ordinary sampling/checkpoint path.
        int off=0;
        for(const auto& c:chunks) {
            auto temporary=work_.scope();
            const int width=static_cast<int>(c.width);
            Tensor shifted=work_.alloc(DType::I32,{width});
            copy_i32(c.token_ids.data()+c.begin+1,shifted,stream);
            Tensor h=normalized.slice(1,off,width);
            Tensor pos=roots.positions.slice(0,off,width);
            ScopedValue<qwen3_6::PagedKVCacheView> owner(mtp_kv_,c.mtp_cache);
            const auto visible=c.begin+c.width;
            mtp_prefill_chunk(shifted,h,nullptr,pos,pos,{visible,visible},false,nullptr,nullptr,nullptr);
            off+=width;
        }
        ctx_.synchronize();
    }
    work_.reset();
}

template <class Tap>
PrefillChunkResult
TextContext::prefill_impl(std::span<const int> ids, const TextPrefill* text_prefill,
                          const MultimodalPrefill* multimodal, Tap& tap, bool finalize_at_end) {
    if (ids.empty()) { throw std::invalid_argument("TextContext::prefill requires tokens"); }
    if (ids.size() > static_cast<std::size_t>(std::numeric_limits<std::int32_t>::max())) {
        throw std::overflow_error("TextContext::prefill token count exceeds int32");
    }
    cudaStream_t s           = ctx_.stream;
    const int T              = static_cast<int>(ids.size());
    const int chunk          = static_cast<int>(prefill_chunk_);
    const std::uint32_t base = text_kv_base_;
    if (base == 0) { verify_pass_count_ = 0; }

    if (text_prefill != nullptr) {
        if (multimodal != nullptr || base != text_prefill->begin ||
            text_prefill->token_ids.size() < static_cast<std::size_t>(base) + ids.size()) {
            throw std::invalid_argument("text prefill chunk does not match its full prompt");
        }
    }
    if (multimodal != nullptr) {
        if (base != multimodal->begin ||
            multimodal->token_ids.size() < static_cast<std::size_t>(base) + ids.size()) {
            throw std::invalid_argument("multimodal prefill suffix does not match its cache base");
        }
        if (multimodal->positions.size() != 3 * multimodal->token_ids.size()) {
            throw std::invalid_argument("multimodal positions must have shape [3,T]");
        }
        if (multimodal->vision == nullptr) {
            throw std::invalid_argument("multimodal prefill requires a Vision session");
        }
        rope_delta_ = multimodal->rope_delta;
    } else if (text_kv_base_ == 0) {
        rope_delta_ = 0;
    }
    ops::set_i32_scalar(io_.rope_delta, rope_delta_, s);

    // Prefix-append prefill continues an existing cache: positions are absolute (start at the
    // resident length) and KV/GDN state is not reset. For a reset prefill base == 0.
    if (static_cast<std::uint64_t>(base) + static_cast<std::uint64_t>(T) >
        static_cast<std::uint64_t>(std::numeric_limits<std::int32_t>::max())) {
        throw std::overflow_error("TextContext::prefill absolute position exceeds int32");
    }
    const int base_i = static_cast<int>(base);

    const std::int64_t base64         = static_cast<std::int64_t>(base);
    const std::int64_t checkpoint_abs = prefill_turn_checkpoint_frontier_;
    const bool has_turn_checkpoint =
        checkpoint_abs > base64 && checkpoint_abs <= base64 + static_cast<std::int64_t>(T);
    const int checkpoint_rel = has_turn_checkpoint ? static_cast<int>(checkpoint_abs - base64) : -1;
    const std::int32_t turn_checkpoint_slot = linear_state_turn_checkpoint_slot_;

    const bool prepare_mtp_prompt = mtp_enabled() && io_.mtp.has_value();
    if (prepare_mtp_prompt &&
        mtp_proposal_extent_ > static_cast<std::uint32_t>(io_.mtp->draft_tokens.ne[0])) {
        throw std::logic_error("MTP proposal extent exceeds the configured draft window");
    }
    int t0 = 0;
    for (; t0 < T;) {
        int len = std::min(chunk, T - t0);
        if (checkpoint_rel > 0 && t0 < checkpoint_rel && t0 + len > checkpoint_rel) {
            len = checkpoint_rel - t0;
        }
        work_.reset();

        VisionChunk vision_chunk;
        const std::uint32_t prompt_t0 = base + static_cast<std::uint32_t>(t0);
        if (multimodal != nullptr) {
            if (multimodal->vision == nullptr) {
                throw std::logic_error("multimodal prefill has no Vision session");
            }
            vision_chunk =
                multimodal->vision->prepare_chunk(prompt_t0, static_cast<std::uint32_t>(len));
            len = vision_chunk.length;
        }
        const bool is_last = finalize_at_end && (t0 + len == T);
        nvtx::ScopedRange chunk_range(nvtx::Name::PrefillChunk, nvtx::Category::Prefill,
                                      static_cast<std::uint64_t>(len));

        {
            std::vector<std::int32_t> local_scatter_indices;
            std::int32_t visual_begin = 0;
            if (vision_chunk.control != nullptr) {
                const auto scatter =
                    std::span<const std::int32_t>(vision_chunk.control->scatter_indices);
                const auto begin = std::lower_bound(scatter.begin(), scatter.end(), prompt_t0);
                const auto end   = std::lower_bound(begin, scatter.end(), prompt_t0 + len);
                const auto count = static_cast<std::int32_t>(end - begin);
                visual_begin     = static_cast<std::int32_t>(begin - scatter.begin());
                local_scatter_indices.resize(static_cast<std::size_t>(count));
                for (std::int32_t i = 0; i < count; ++i) {
                    local_scatter_indices[static_cast<std::size_t>(i)] =
                        begin[i] - static_cast<std::int32_t>(prompt_t0);
                }
            }

            const std::int32_t rope_axes = multimodal != nullptr ? 3 : (rope_delta_ != 0 ? 1 : 0);
            const auto roots             = workspace_recipe::text_prefill_roots<TextConfig>(
                work_, len, rope_axes, static_cast<std::int32_t>(local_scatter_indices.size()));
            Tensor ids_device = roots.ids;
            copy_i32(ids.data() + t0, ids_device, s);

            Tensor positions = roots.positions;
            ops::fill_i32_positions(positions, base_i + t0, s);

            Tensor rope_positions = positions;
            std::vector<std::int32_t> rope_positions_host;
            if (multimodal != nullptr) {
                rope_positions = roots.rope_positions;
                rope_positions_host.resize(static_cast<std::size_t>(3) * len);
                const std::size_t prompt_tokens = multimodal->token_ids.size();
                for (int axis = 0; axis < 3; ++axis) {
                    const auto* src = multimodal->positions.data() +
                                      static_cast<std::size_t>(axis) * prompt_tokens + prompt_t0;
                    std::copy_n(src, len,
                                rope_positions_host.data() + static_cast<std::size_t>(axis) * len);
                }
                copy_i32(rope_positions_host.data(), rope_positions, s);
            } else if (rope_delta_ != 0) {
                rope_positions = roots.rope_positions;
                ops::offset_i32_positions(positions, io_.rope_delta, rope_positions, s);
            }
            ScopedPositions scoped_cache(active_cache_positions_, positions);
            ScopedPositions scoped_rope(active_rope_positions_, rope_positions);
            const auto visible = static_cast<std::uint32_t>(base_i + t0 + len);
            const ops::GqaExecutionEnvelope chunk_envelope{visible, visible};
            ScopedEnvelope scoped_envelope(active_gqa_envelope_, chunk_envelope);

            Tensor x = roots.residual;
            ops::embedding(ids_device, *embed_, x, s);
            if (!local_scatter_indices.empty()) {
                Tensor indices_device = roots.scatter_indices;
                copy_i32(local_scatter_indices.data(), indices_device, s);
                Tensor embeddings = vision_chunk.embeddings.slice(
                    1, visual_begin, static_cast<std::int32_t>(local_scatter_indices.size()));
                ops::scatter(embeddings, indices_device, x, s);
            }
            if constexpr (Tap::enabled) { tap.begin(x); }
            run_layers(x, Phase::Prefill, tap);
            if constexpr (requires { tap.capture_positions(positions, s); }) {
                tap.capture_positions(positions, s);
            }

            Tensor xf = prefill_hidden_.data != nullptr
                            ? matrix_window(prefill_hidden_, len)
                            : work_.alloc(DType::BF16, {kCfg.hidden, len});
            ops::rmsnorm(x, *final_norm_, kCfg.rms_eps, true, xf, s);

            if (is_last) {
                Tensor last_xf = xf.slice(1, len - 1, 1);
                Tensor logits  = matrix_window(io_.logits, 1);
                ops::linear(last_xf, *lm_head_, logits, s);
                // Set io_.pos to the bonus token's absolute position (base + T) before picking so
                // the sampler RNG is keyed by it (prefill purpose keeps it distinct from the first
                // decode step, which reuses the same io_.pos).
                ops::set_i32_scalar(io_.pos, base_i + T, s);
                ops::set_i32_scalar(io_.rope_pos, base_i + T + rope_delta_, s);
                if (sampling_config_ != nullptr) {
                    ops::sample(logits, io_.token, kCfg.token_domain, sampling_config_, io_.pos,
                                ops::kSamplePurposePrefill, work_, s);
                } else {
                    ops::argmax(logits, io_.token, kCfg.token_domain, s);
                }
            }

            if (prepare_mtp_prompt) {
                const std::uint32_t alignment_tokens =
                    multimodal != nullptr ? static_cast<std::uint32_t>(multimodal->token_ids.size())
                    : text_prefill != nullptr
                        ? static_cast<std::uint32_t>(text_prefill->token_ids.size())
                        : static_cast<std::uint32_t>(T);
                const std::uint32_t alignment_begin =
                    multimodal != nullptr || text_prefill != nullptr
                        ? prompt_t0
                        : static_cast<std::uint32_t>(t0);
                const qwen3_6::MtpAlignmentWindow mtp_window = qwen3_6::plan_mtp_alignment_window(
                    alignment_tokens, alignment_begin, static_cast<std::uint32_t>(len));
                const std::span<const int> alignment_ids =
                    multimodal != nullptr     ? multimodal->token_ids
                    : text_prefill != nullptr ? text_prefill->token_ids
                                              : ids;
                std::vector<int> mtp_ids_host(static_cast<std::size_t>(len));
                const int prompt_columns =
                    len - static_cast<int>(mtp_window.final_column_uses_generated_token);
                for (int j = 0; j < prompt_columns; ++j) {
                    mtp_ids_host[static_cast<std::size_t>(j)] =
                        alignment_ids[static_cast<std::size_t>(mtp_window.shifted_embedding_begin) +
                                      static_cast<std::size_t>(j)];
                }
                if (mtp_window.final_column_uses_generated_token) {
                    int next_token = 0;
                    CUDA_CHECK(cudaStreamSynchronize(s));
                    CUDA_CHECK(cudaMemcpy(&next_token, io_.token.data, sizeof(next_token),
                                          cudaMemcpyDeviceToHost));
                    mtp_ids_host[static_cast<std::size_t>(len - 1)] = next_token;
                }

                Tensor mtp_ids = work_.alloc(DType::I32, {len});
                copy_i32(mtp_ids_host.data(), mtp_ids, s);
                Tensor mtp_input_embeddings;
                const Tensor* mtp_input_embeddings_ptr = nullptr;
                if (multimodal != nullptr) {
                    mtp_input_embeddings = work_.alloc(DType::BF16, {kCfg.hidden, len});
                    ops::embedding(mtp_ids, *embed_, mtp_input_embeddings, s);
                    if (vision_chunk.control != nullptr) {
                        const qwen3_6::MtpVisualOverlap overlap = qwen3_6::shifted_visual_overlap(
                            vision_chunk.control->scatter_indices, alignment_tokens, mtp_window);
                        if (!overlap.empty()) {
                            Tensor shifted_indices = workspace_recipe::visual_scatter_indices(
                                work_, static_cast<std::int32_t>(overlap.size()));
                            qwen3_6::detail::scatter_shifted_visual_embeddings(
                                mtp_input_embeddings, vision_chunk.embeddings, overlap,
                                shifted_indices, s);
                        }
                    }
                    mtp_input_embeddings_ptr = &mtp_input_embeddings;
                }
                if (is_last && mtp_proposal_extent_ != 0) {
                    Tensor logits = matrix_window(io_.logits, 1);
                    Tensor draft0 = io_.mtp->draft_tokens.slice(0, 0, 1);
                    mtp_prefill_chunk(mtp_ids, xf, mtp_input_embeddings_ptr, positions,
                                      rope_positions, chunk_envelope, true, &io_.mtp->ar_hidden,
                                      &logits, &draft0);

                    Tensor ar_position = io_.mtp->position.slice(0, 0, 1);
                    ops::set_i32_scalar(ar_position, base_i + T, s);
                    for (int i = 1; i < static_cast<int>(mtp_proposal_extent_); ++i) {
                        Tensor prev_token     = io_.mtp->draft_tokens.slice(0, i - 1, 1);
                        Tensor next_token     = io_.mtp->draft_tokens.slice(0, i, 1);
                        Tensor next_hidden    = work_.alloc(DType::BF16, {kCfg.hidden, 1});
                        const auto ar_visible = static_cast<std::uint32_t>(base_i + T + i);
                        const ops::GqaExecutionEnvelope ar_envelope{ar_visible, ar_visible};
                        mtp_forward_ar_step(prev_token, io_.mtp->ar_hidden, ar_position,
                                            ar_envelope, next_hidden, logits, next_token);
                        CUDA_CHECK(cudaMemcpyAsync(io_.mtp->ar_hidden.data, next_hidden.data,
                                                   io_.mtp->ar_hidden.bytes(),
                                                   cudaMemcpyDeviceToDevice, s));
                        ops::increment_i32_scalar(ar_position, s);
                    }
                } else {
                    mtp_prefill_chunk(mtp_ids, xf, mtp_input_embeddings_ptr, positions,
                                      rope_positions, chunk_envelope, false, nullptr, nullptr,
                                      nullptr);
                }
            }

            if (checkpoint_rel > 0 && t0 + len == checkpoint_rel &&
                turn_checkpoint_hidden_output_ != nullptr) {
                require_tensor_shape(*turn_checkpoint_hidden_output_, DType::BF16, {kCfg.hidden, 1},
                                     "turn checkpoint hidden output");
                const Tensor turn_hidden = xf.slice(1, len - 1, 1);
                CUDA_CHECK(cudaMemcpyAsync(turn_checkpoint_hidden_output_->data, turn_hidden.data,
                                           turn_hidden.bytes(), cudaMemcpyDeviceToDevice, s));
            }
        }

        if constexpr (requires { tap.consume_prefill_chunk(len, false); }) {
            work_.reset();
            tap.consume_prefill_chunk(len, checkpoint_rel > 0 && t0 + len == checkpoint_rel);
        }

        if (checkpoint_rel > 0 && t0 + len == checkpoint_rel) {
            state_.copy_slot(linear_state_current_slot_, turn_checkpoint_slot, s);
        }

        t0 += len;
        break;
    }

    prefill_turn_checkpoint_frontier_ = -1;

    ctx_.synchronize();
    work_.reset();
    return PrefillChunkResult{.processed_tokens = static_cast<std::uint32_t>(t0),
                              .finalized        = finalize_at_end && t0 == T};
}

PrefillChunkResult TextContext::prefill_chunk(std::span<const int> full_ids, std::uint32_t begin,
                                              std::uint32_t nominal_length, bool finalize_at_end) {
    if (begin >= full_ids.size() || nominal_length == 0 ||
        nominal_length > full_ids.size() - begin) {
        throw std::invalid_argument("text prefill chunk is outside the prompt");
    }
    const TextPrefill text_prefill{full_ids, begin};
    NullTap tap;
    return prefill_impl(full_ids.subspan(begin, nominal_length), &text_prefill, nullptr, tap,
                        finalize_at_end);
}

PrefillChunkResult TextContext::prefill_chunk(std::span<const int> full_ids, std::uint32_t begin,
                                              std::uint32_t nominal_length, bool finalize_at_end,
                                              DFlashFeatureSink& sink) {
    if (begin >= full_ids.size() || nominal_length == 0 ||
        nominal_length > full_ids.size() - begin) {
        throw std::invalid_argument("text prefill chunk is outside the prompt");
    }
    const TextPrefill text_prefill{full_ids, begin};
    return prefill_impl(full_ids.subspan(begin, nominal_length), &text_prefill, nullptr, sink,
                        finalize_at_end);
}

PrefillChunkResult TextContext::prefill_chunk(const qwen3_6::PreparedPromptData& input,
                                              std::uint32_t begin, std::uint32_t nominal_length,
                                              VisionPrefillSession& vision, bool finalize_at_end) {
    if (begin >= input.token_ids.size() || nominal_length == 0 ||
        nominal_length > input.token_ids.size() - begin) {
        throw std::invalid_argument("multimodal prefill chunk is outside the prompt");
    }
    const std::span<const int> tokens(input.token_ids);
    const MultimodalPrefill multimodal{tokens, input.positions, &vision, begin, input.rope_delta};
    NullTap tap;
    return prefill_impl(tokens.subspan(begin, nominal_length), nullptr, &multimodal, tap,
                        finalize_at_end);
}

} // namespace ninfer::targets::qwen3_6::detail::NINFER_QWEN36_RUNTIME_NS::schedule
