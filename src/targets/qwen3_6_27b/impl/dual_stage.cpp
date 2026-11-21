#include "targets/qwen3_6_27b/impl/dual_stage.h"
#include "targets/qwen3_6_27b/impl/variant.h"

#if defined(NINFER_CUDA_SM75)
#define NINFER_QWEN36_SM75_DENSE_A8 1
#endif
#include <ninfer/targets/qwen3_6_27b/package.h>
#include <ninfer/targets/qwen3_6/frontend.h>
#include <ninfer/targets/qwen3_6/prepared_prompt.h>
#include <ninfer/targets/qwen3_6/vision_control.h>
#include "runtime/contract/transient_region.h"
#include <ninfer/targets/qwen3_6/decoder_state.h>
#include "core/arena.h"
#include "core/decode_graph.h"
#include "core/device.h"
#include "core/gdn_replay_records.h"
#include "core/layout.h"
#include "ninfer/ops/gdn_replay.h"
#include "ninfer/ops/mtp_round.h"
#include "ninfer/ops/position.h"
#include "ninfer/ops/speculative_round.h"
#include "ops/gdn_input_proj/q4_q5/q5_a8_sm75.h"
#include "ops/linear_add/q5/q5_linear_add_a8_sm75.h"
#include "ops/linear_swiglu/q4/q4_linear_swiglu_a8_sm75.h"
#include "ops/linear/w8/w8_vocab_w4_a8_sm75.h"

#define NINFER_QWEN36_VARIANT ::ninfer::targets::qwen3_6_27b::detail::Variant
#define NINFER_QWEN36_RUNTIME_NS qwen3_6_27b_runtime
#include "targets/qwen3_6/impl/runtime/text_context.h"
#include "targets/qwen3_6/impl/runtime/vision_context.h"
#include "targets/qwen3_6/impl/runtime/vision_prefill.h"
#undef NINFER_QWEN36_RUNTIME_NS
#undef NINFER_QWEN36_VARIANT

#include <algorithm>
#include <array>
#include <cstdlib>
#include <limits>
#include <optional>
#include <cstring>
#include <cstddef>
#include <utility>
#include <stdexcept>
#include <string_view>

namespace ninfer::targets::qwen3_6_27b::detail {
namespace {

inline constexpr std::uint32_t kMtp3VerifyWidth = 4;
struct FullVocabularyDraftOwner {
    Weight source;cudaStream_t stream=nullptr;int device=0;
    ~FullVocabularyDraftOwner() {
        (void)cudaSetDevice(device);
        if(stream)(void)cudaStreamSynchronize(stream);
        ops::detail::w8_vocab_w4_a8_release(source);
    }
};

std::uint64_t align_up(std::uint64_t value, std::uint64_t alignment) {
    if (alignment == 0 || (alignment & (alignment - 1)) != 0) {
        throw std::invalid_argument("text shard materialization has invalid alignment");
    }
    const std::uint64_t mask = alignment - 1;
    if (value > std::numeric_limits<std::uint64_t>::max() - mask) {
        throw std::overflow_error("text shard materialization size overflow");
    }
    return (value + mask) & ~mask;
}

bool parse_layer(std::string_view name, std::size_t& layer) {
    constexpr std::string_view prefix = "text/layers/";
    if (!name.starts_with(prefix)) { return false; }
    name.remove_prefix(prefix.size());
    const std::size_t slash = name.find('/');
    if (slash == std::string_view::npos || slash == 0) { return false; }
    std::size_t value = 0;
    for (const char ch : name.substr(0, slash)) {
        if (ch < '0' || ch > '9') { return false; }
        value = value * 10 + static_cast<std::size_t>(ch - '0');
    }
    layer = value;
    return true;
}

bool shard_owns(std::string_view name, const TextShardSpec& shard,
                qwen3_6::StartupFeatures features) {
    if (name == "text/token_embedding") {
        return shard.token_embedding || features.mtp();
    }
    if (name == "text/final_norm") { return shard.output_head || features.mtp(); }
    if (name == "text/output_head") { return shard.output_head; }
    if (features.requires_optimized_proposal_payload() &&
        (name == "text/draft_head" || name == "text/draft_head_token_ids")) {
        return true;
    }
    if (features.mtp() && name.starts_with("mtp/")) { return true; }
    if (features.vision && name.starts_with("vision/")) { return true; }
    std::size_t layer = 0;
    return parse_layer(name, layer) && layer >= shard.layer_begin && layer < shard.layer_end;
}

} // namespace

ArtifactLoadPlan bind_text_shard_artifact(artifact::Binder& binder,
                                          const artifact::Reader& reader,
                                          WeightsProfile weights_profile,
                                          TextShardSpec shard,
                                          qwen3_6::StartupFeatures features) {
    if (shard.layer_begin >= shard.layer_end || shard.layer_end > kTextLayers ||
        shard.layer_begin % 4 != 0 || shard.layer_end % 4 != 0) {
        throw std::invalid_argument("qwen3_6_27b: invalid text shard layer interval");
    }

    if (features.dflash() ||
        (features.speculative_enabled() && !features.mtp())) {
        throw std::invalid_argument("qwen3_6_27b: unsupported text-shard startup features");
    }
    ArtifactLoadPlan out = bind_artifact(binder, weights_profile, features);
    artifact::MaterializationPlan selected;
    selected.object_count = out.materialization.object_count;
    selected.host_objects = out.materialization.host_objects;

    std::uint64_t cursor = 0;
    for (const artifact::DeviceMaterialization& source : out.materialization.device_objects) {
        const std::string_view name = artifact::object_name(reader.objects().at(source.object.index));
        if (!shard_owns(name, shard, features)) { continue; }
        const std::uint64_t offset = align_up(cursor, source.alignment);
        if (source.bytes > std::numeric_limits<std::uint64_t>::max() - offset) {
            throw std::overflow_error("text shard materialization size overflow");
        }
        selected.device_objects.push_back(
            artifact::DeviceMaterialization{source.object, offset, source.bytes, source.alignment});
        cursor = offset + source.bytes;
    }
    if (selected.device_objects.empty()) {
        throw std::invalid_argument("qwen3_6_27b: text shard selected no device tensors");
    }
    selected.device_capacity_bytes = cursor;
    out.materialization            = std::move(selected);
    return out;
}

namespace shard_runtime = qwen3_6::detail::qwen3_6_27b_runtime;
using StageTextContext = shard_runtime::schedule::TextContext;
using StageTextRange   = shard_runtime::schedule::TextLayerRange;
using StageTextConfig  = shard_runtime::TextConfig;
using StageVisionContext = shard_runtime::schedule::VisionContext;
using StageVisionPrefillSession = shard_runtime::schedule::VisionPrefillSession;
using StageVisionChunk = shard_runtime::schedule::VisionChunk;
using StageVisionPrefillPlan = shard_runtime::VisionPrefillPlan;
using StageVisionUseSpan = shard_runtime::VisionUseSpan;

struct PlannedStageState {
    qwen3_6::DecoderStateLayout decoder;
    std::vector<qwen3_6::RoundStateLayout> rounds;
    std::vector<GdnReplayRecordLayout> replay;
    std::vector<GdnReplayRecordLayout> batch_replay;
    std::uint32_t logical_pages = 0;
    std::uint32_t text_pages = 0;
    std::size_t suffix_begin = 0;
    std::size_t bytes = 0;
};

std::size_t region_begin(const LayoutRegion& region) noexcept { return region.offset; }
std::size_t region_begin(const TensorRegion& region) noexcept { return region.region.offset; }

std::size_t decoder_suffix_begin(const qwen3_6::DecoderStateLayout& layout) {
    std::size_t begin = std::numeric_limits<std::size_t>::max();
    const auto consider = [&](std::size_t value) { begin = std::min(begin, value); };
    if (layout.mtp_kv) {
        for (const PagedKVPlaneLayout& plane : layout.mtp_kv->pool.planes) {
            consider(region_begin(plane.storage));
        }
        consider(region_begin(layout.mtp_kv->pool.block_tables));
    } else {
        for (const LayoutRegion& region : layout.linear_attention.conv) {
            consider(region_begin(region));
        }
        for (const LayoutRegion& region : layout.linear_attention.recurrent) {
            consider(region_begin(region));
        }
    }
    if (begin == std::numeric_limits<std::size_t>::max()) {
        throw std::logic_error("dual text shard state has no suffix after text KV");
    }
    return begin;
}

PlannedStageState plan_stage_state(std::uint32_t capacity, const TextShardSpec& shard,
                                   KvCacheStorage kv_cache, std::uint32_t sessions,
                                   std::uint32_t verify_width, bool enable_mtp,
                                   TextKvPageOwnership text_ownership) {
    const std::uint32_t full_begin = static_cast<std::uint32_t>(shard.layer_begin / 4);
    const std::uint32_t full_end = static_cast<std::uint32_t>(shard.layer_end / 4);
    const std::uint32_t full_layers = full_end - full_begin;
    const std::uint32_t gdn_layers =
        static_cast<std::uint32_t>(shard.layer_end - shard.layer_begin) - full_layers;
    const std::uint32_t logical_pages =
        (capacity + static_cast<std::uint32_t>(kPagedKVPageSize) - 1U) /
        static_cast<std::uint32_t>(kPagedKVPageSize);
    if (text_ownership.enabled &&
        (sessions != 1 || text_ownership.logical_page_count == 0 ||
         text_ownership.logical_page_begin > logical_pages ||
         text_ownership.logical_page_count >
             logical_pages - text_ownership.logical_page_begin)) {
        throw std::invalid_argument("sharded text KV ownership is outside the logical domain");
    }
    const std::uint64_t text_physical64 = text_ownership.enabled
        ? text_ownership.logical_page_count
        : static_cast<std::uint64_t>(logical_pages) * sessions;
    const std::uint64_t mtp_physical64 =
        enable_mtp ? static_cast<std::uint64_t>(logical_pages) * sessions : 0ULL;
    if (text_physical64 > std::numeric_limits<std::uint32_t>::max() ||
        mtp_physical64 > std::numeric_limits<std::uint32_t>::max()) {
        throw std::overflow_error("dual text shard KV page inventory overflows uint32");
    }

    DType kv_dtype;
    std::int32_t kv_quant_group;
    switch (kv_cache) {
    case KvCacheStorage::BFloat16:
        kv_dtype = DType::BF16;
        kv_quant_group = 0;
        break;
    case KvCacheStorage::Int8Group64:
        kv_dtype = DType::I8;
        kv_quant_group = qwen3_6::kKvQuantGroup;
        break;
    default:
        throw std::invalid_argument("text shard has an unsupported KV cache format");
    }

    LayoutBuilder builder;
    qwen3_6::DecoderStateSpec state_spec{
        .full_attention_layers = full_layers,
        .mtp_layers = enable_mtp ? static_cast<std::uint32_t>(StageTextConfig::mtp_layers) : 0U,
        .capacity = capacity,
        .kv_heads = StageTextConfig::kv_heads,
        .attention_head_dim = StageTextConfig::head_dim,
        .kv_dtype = kv_dtype,
        .kv_quant_group = kv_quant_group,
        .enable_mtp = enable_mtp,
        .kv_table_rows = static_cast<std::int32_t>(sessions),
        .text_physical_page_groups = static_cast<std::uint32_t>(text_physical64),
        .mtp_physical_page_groups = static_cast<std::uint32_t>(mtp_physical64),
        .linear_attention =
            {
                .layers = gdn_layers,
                .conv_channels = StageTextConfig::convolution_dim,
                .conv_width = StageTextConfig::gdn_conv_state_width,
                .value_heads = StageTextConfig::gdn_value_heads,
                .value_head_dim = StageTextConfig::gdn_value_head_dim,
                .key_head_dim = StageTextConfig::gdn_key_head_dim,
                .slot_count = static_cast<std::int32_t>(2U * sessions),
                .conv_dtype = DType::BF16,
            },
    };

    PlannedStageState out;
    out.decoder = qwen3_6::plan_decoder_state(builder, state_spec);
    if (enable_mtp) {
        out.rounds.reserve(sessions);
        for (std::uint32_t session = 0; session < sessions; ++session) {
            out.rounds.push_back(qwen3_6::begin_round_state_layout(
                builder,
                qwen3_6::RoundStateSpec{
                    .hidden = StageTextConfig::hidden,
                    .output_rows = StageTextConfig::output_rows,
                    .batch_capacity = 1,
                    .draft_window = verify_width - 1U,
                    .enable_mtp = true,
                    .enable_dflash = false,
                }));
            qwen3_6::complete_round_state_layout(builder, out.rounds.back());
        }
    }
    if (verify_width != 0) {
        const auto replay_spec = [&](std::int32_t record_capacity) {
            return GdnReplayRecordSpec{
                .layers = static_cast<std::int32_t>(gdn_layers),
                .record_capacity = record_capacity,
                .width = static_cast<std::int32_t>(verify_width),
                .conv_channels = StageTextConfig::convolution_dim,
                .qk_heads = StageTextConfig::gdn_key_heads,
                .value_heads = StageTextConfig::gdn_value_heads,
                .key_dim = StageTextConfig::gdn_key_head_dim,
                .value_dim = StageTextConfig::gdn_value_head_dim,
            };
        };
        out.replay.reserve(sessions);
        for (std::uint32_t session = 0; session < sessions; ++session) {
            out.replay.push_back(plan_gdn_replay_records(builder, replay_spec(1)));
        }
        if (sessions > 1) {
            const std::uint32_t pair_count = sessions / 2U;
            out.batch_replay.reserve(pair_count);
            for (std::uint32_t pair = 0; pair < pair_count; ++pair) {
                out.batch_replay.push_back(plan_gdn_replay_records(builder, replay_spec(2)));
            }
        }
    }
    out.logical_pages = logical_pages;
    out.text_pages = text_ownership.enabled ? text_ownership.logical_page_count
                                            : logical_pages;
    out.suffix_begin = decoder_suffix_begin(out.decoder);
    out.bytes = builder.finish(256, "dual text shard state");
    if (out.suffix_begin > out.bytes) {
        throw std::logic_error("dual text shard suffix lies outside state layout");
    }
    return out;
}

class TextShardStage::Impl {
public:
    Impl(std::string artifact_path, int device_id, std::uint32_t capacity,
         TextShardSpec shard, artifact::LoadProgress* progress, std::uint32_t prefill_chunk,
         KvCacheStorage kv_cache, std::uint32_t sessions, std::uint32_t verify_width,
         bool enable_mtp, TextKvPageOwnership text_kv_ownership, bool enable_vision,
         qwen3_6::LongContextComputation computation, std::uint32_t mtp_attention_window,
         bool full_vocab_draft)
        : device_id_(device_id), capacity_(capacity), prefill_chunk_(prefill_chunk),
          verify_width_(verify_width), shard_(shard), kv_cache_(kv_cache),
          session_count_(sessions), enable_mtp_(enable_mtp), enable_vision_(enable_vision),
          text_kv_ownership_(text_kv_ownership), computation_(computation), device_(device_id) {
        if (capacity_ == 0 || capacity_ > Variant::maximum_context || prefill_chunk_ == 0 ||
            prefill_chunk_ > 128 || prefill_chunk_ > capacity_ ||
            prefill_chunk_ > static_cast<std::uint32_t>(std::numeric_limits<std::int32_t>::max()) ||
            session_count_ == 0 || session_count_ > kMaximumConcurrency ||
            (verify_width_ != 0 &&
             (verify_width_ < kMtp3VerifyWidth ||
              verify_width_ > qwen3_6::kMtpDecodeMaximumWidth)) ||
            (enable_mtp_ &&
             (verify_width_ < kMtp3VerifyWidth ||
              verify_width_ > qwen3_6::kMtpDecodeMaximumWidth)) ||
            (enable_vision_ && session_count_ != 1) ||
            shard_.layer_begin >= shard_.layer_end || shard_.layer_end > kTextLayers ||
            shard_.layer_begin % 4 != 0 || shard_.layer_end % 4 != 0) {
            throw std::invalid_argument("text shard configuration is outside the model domain");
        }
        if(full_vocab_draft && mtp_attention_window==0)
            throw std::invalid_argument("full-vocabulary draft composition requires a proposal window");
        if(mtp_attention_window && (computation_!=qwen3_6::LongContextComputation::Complete ||
           !enable_mtp_ || !shard_.output_head || sessions!=1 || verify_width!=4 ||
           mtp_attention_window<128 || mtp_attention_window>262144 || mtp_attention_window%64)) {
            throw std::invalid_argument("MTP attention window is outside the complete D3 proposal scope");
        }
        CUDA_CHECK(cudaSetDevice(device_id_));

        artifact::Reader reader(artifact_path);
        weights_profile_ = Package::resolve_weights(reader.identity());
        artifact::Binder binder(reader);
        const qwen3_6::StartupFeatures features{
            .vision = enable_vision_,
            .speculative = enable_mtp_ ? SpeculativeBackend::Mtp : SpeculativeBackend::None,
            .proposal_head = enable_mtp_ ? ProposalHead::Optimized : ProposalHead::Full,
            .force_full_target_head = computation_==qwen3_6::LongContextComputation::Complete,
            .fast_full_vocab_draft=full_vocab_draft,
        };
        ArtifactLoadPlan plan =
            bind_text_shard_artifact(binder, reader, weights_profile_, shard_, features);
        artifact::MaterializedArtifact materialized =
            artifact::materialize(reader, plan.materialization, device_, progress);
        weights_bytes_ = materialized.stats().device_capacity_bytes;
        model_ = std::make_unique<LoadedModelData>(
            std::move(plan.bindings), std::move(materialized), shard_);

        state_plan_ = plan_stage_state(capacity_, shard_, kv_cache_, session_count_,
                                       verify_width_, enable_mtp_, text_kv_ownership_);
        legacy_full_state_plan_ = plan_stage_state(capacity_, shard_, kv_cache_, session_count_,
                                                   verify_width_, enable_mtp_, {});
        snapshot_source_state_plan_ = plan_stage_state(
            capacity_, shard_, kv_cache_, session_count_,
            verify_width_ != 0 ? kMtp3VerifyWidth : 0U, enable_mtp_, {});
        const std::size_t current_suffix_bytes = state_plan_.bytes - state_plan_.suffix_begin;
        const std::size_t legacy_suffix_bytes =
            legacy_full_state_plan_.bytes - legacy_full_state_plan_.suffix_begin;
        if (current_suffix_bytes != legacy_suffix_bytes) {
            throw std::logic_error("text KV sharding changed the non-text state suffix");
        }
        state_bytes_ = state_plan_.bytes;
        kv_payload_bytes_ = state_plan_.decoder.kv_payload_bytes();
        state_storage_ = std::make_unique<DeviceArena>(state_bytes_);
        CUDA_CHECK(cudaMemsetAsync(state_storage_->base(), 0, state_storage_->capacity(),
                                   device_.stream));
        const DeviceSpan state_backing{state_storage_->base(), state_storage_->capacity()};
        decoder_ = std::make_unique<qwen3_6::DecoderState>(state_backing, state_plan_.decoder);
        rounds_.reserve(state_plan_.rounds.size());
        for (const qwen3_6::RoundStateLayout& layout : state_plan_.rounds) {
            rounds_.emplace_back(state_backing, layout);
        }
        replay_layouts_ = state_plan_.replay;
        batch_replay_layouts_ = state_plan_.batch_replay;
        replay_records_.reserve(replay_layouts_.size());
        for (const GdnReplayRecordLayout& layout : replay_layouts_) {
            replay_records_.emplace_back(state_backing, layout);
        }
        batch_replay_records_.reserve(batch_replay_layouts_.size());
        for (const GdnReplayRecordLayout& layout : batch_replay_layouts_) {
            batch_replay_records_.emplace_back(state_backing, layout);
        }
        kv_allocations_.reserve(session_count_);
        mtp_allocations_.reserve(enable_mtp_ ? session_count_ : 0U);
        for (std::uint32_t session = 0; session < session_count_; ++session) {
            PagedKVAllocation allocation =
                decoder_->text_kv.pool().reserve(state_plan_.text_pages);
            allocation.bind_row_range(
                static_cast<std::int32_t>(session),
                text_kv_ownership_.enabled ? text_kv_ownership_.logical_page_begin : 0U,
                device_.stream);
            allocation.materialize_pages(state_plan_.text_pages, device_.stream);
            kv_allocations_.push_back(std::move(allocation));
            if (enable_mtp_) {
                qwen3_6::PagedKVCache* mtp_cache = decoder_->mtp_cache();
                if (mtp_cache == nullptr) { throw std::logic_error("MTP cache was not planned"); }
                PagedKVAllocation mtp_allocation =
                    mtp_cache->pool().reserve(state_plan_.logical_pages);
                mtp_allocation.bind_row(static_cast<std::int32_t>(session), device_.stream);
                mtp_allocation.materialize_tokens(capacity_, device_.stream);
                mtp_allocations_.push_back(std::move(mtp_allocation));
            }
        }

        workspace_bytes_ = enable_mtp_ ? (256ULL << 20) : (64ULL << 20);
        workspace_       = std::make_unique<DeviceArena>(workspace_bytes_);
        const std::size_t io_bytes = verify_width_ == 0 ? (16ULL << 20) : (64ULL << 20);
        io_storage_      = std::make_unique<DeviceArena>(io_bytes);
        const auto sessions_i32 = static_cast<std::int32_t>(session_count_);
        id_              = io_storage_->alloc(DType::I32, {sessions_i32});
        position_        = io_storage_->alloc(DType::I32, {sessions_i32});
        rope_position_   = io_storage_->alloc(DType::I32, {sessions_i32});
        kv_table_row_    = io_storage_->alloc(DType::I32, {sessions_i32});
        state_slot_      = io_storage_->alloc(DType::I32, {sessions_i32});
        residual_        = io_storage_->alloc(DType::BF16, {StageTextConfig::hidden, sessions_i32});
        prefill_ids_ = io_storage_->alloc(DType::I32,
                                          {static_cast<std::int32_t>(prefill_chunk_)});
        prefill_positions_ = io_storage_->alloc(DType::I32,
                                                {static_cast<std::int32_t>(prefill_chunk_)});
        prefill_rope_positions_ = io_storage_->alloc(
            DType::I32, {static_cast<std::int32_t>(prefill_chunk_)});
        prefill_residual_ = io_storage_->alloc(
            DType::BF16, {StageTextConfig::hidden, static_cast<std::int32_t>(prefill_chunk_)});
        prefill_hidden_ = io_storage_->alloc(
            DType::BF16, {StageTextConfig::hidden, static_cast<std::int32_t>(prefill_chunk_)});
        if (shard_.output_head) {
            normalized_ = io_storage_->alloc(DType::BF16, {StageTextConfig::hidden, sessions_i32});
            logits_     = io_storage_->alloc(DType::BF16, {StageTextConfig::output_rows, sessions_i32});
            output_token_ = io_storage_->alloc(DType::I32, {sessions_i32});
        }
        snapshot_source_io_prefix_bytes_ = io_storage_->used();
        if (verify_width_ != 0) {
            const auto verify_i32 = static_cast<std::int32_t>(verify_width_);
            verify_ids_ = io_storage_->alloc(DType::I32, {verify_i32, sessions_i32});
            verify_positions_ = io_storage_->alloc(DType::I32, {verify_i32, sessions_i32});
            verify_rope_positions_ = io_storage_->alloc(DType::I32, {verify_i32, sessions_i32});
            verify_valid_columns_ = io_storage_->alloc(DType::I32, {sessions_i32});
            verify_residual_ = io_storage_->alloc(
                DType::BF16, {StageTextConfig::hidden, verify_i32, sessions_i32});
            if (shard_.output_head) {
                verify_hidden_ = io_storage_->alloc(
                    DType::BF16, {StageTextConfig::hidden, verify_i32, sessions_i32});
                verify_logits_ = io_storage_->alloc(
                    DType::BF16, {StageTextConfig::output_rows, verify_i32, sessions_i32});
                verify_tokens_ = io_storage_->alloc(DType::I32, {verify_i32, sessions_i32});
            }
        }
        std::vector<std::int32_t> kv_rows(session_count_);
        std::vector<std::int32_t> state_slots(session_count_);
        for (std::uint32_t session = 0; session < session_count_; ++session) {
            kv_rows[session] = static_cast<std::int32_t>(session);
            state_slots[session] = static_cast<std::int32_t>(2U * session);
        }
        const std::size_t index_bytes = kv_rows.size() * sizeof(std::int32_t);
        CUDA_CHECK(cudaMemcpyAsync(kv_table_row_.data, kv_rows.data(), index_bytes,
                                   cudaMemcpyHostToDevice, device_.stream));
        CUDA_CHECK(cudaMemcpyAsync(state_slot_.data, state_slots.data(), index_bytes,
                                   cudaMemcpyHostToDevice, device_.stream));
        if (verify_width_ != 0) {
            std::vector<std::int32_t> valid_columns(session_count_,
                                                    static_cast<std::int32_t>(verify_width_));
            CUDA_CHECK(cudaMemcpyAsync(verify_valid_columns_.data, valid_columns.data(), index_bytes,
                                       cudaMemcpyHostToDevice, device_.stream));
        }

        if(full_vocab_draft) {
            const auto bytes=ops::detail::w8_vocab_w4_a8_reservation_bytes();
            std::size_t free_bytes=0,total_bytes=0;CUDA_CHECK(cudaMemGetInfo(&free_bytes,&total_bytes));
            if(bytes>free_bytes)throw std::runtime_error("full-vocabulary draft cache exceeds device capacity");
            full_vocab_owner_=std::make_unique<FullVocabularyDraftOwner>();
            full_vocab_owner_->source=model_->runtime.output_head;full_vocab_owner_->stream=device_.stream;
            full_vocab_owner_->device=device_id_;
            ops::detail::w8_vocab_w4_a8_prepare(model_->runtime.output_head,device_.stream);
            weights_bytes_+=bytes;
        }
        StageTextRange range{
            .begin             = static_cast<std::int32_t>(shard_.layer_begin),
            .end               = static_cast<std::int32_t>(shard_.layer_end),
            .input_embedding   = shard_.token_embedding,
            .output_projection = shard_.output_head,
        };
        qwen3_6::PagedKVCache* mtp_cache = decoder_->mtp_cache();
        qwen3_6::RoundState& primary_round = rounds_.empty() ? round_ : rounds_.front();
        text_ = std::make_unique<StageTextContext>(
            device_, model_->runtime, *workspace_, qwen3_6::PagedKVCacheView{},
            decoder_->linear_attention, primary_round, prefill_hidden_, prefill_chunk_, 0,
            qwen3_6::PagedKVCacheView{}, &decoder_->text_kv, mtp_cache, range);
        text_->set_long_context_computation(computation_);
        text_->set_mtp_attention_window(mtp_attention_window);
        if (verify_width_ != 0) {
            verify_token_graph_definitions_.resize(session_count_);
            verify_token_graphs_.resize(session_count_);
            verify_token_graph_phases_.assign(session_count_, 0);
            if (shard_.output_head) {
                verify_hidden_graph_definitions_.resize(session_count_);
                verify_hidden_graphs_.resize(session_count_);
                verify_hidden_graph_phases_.assign(session_count_, 0);
            }
        }
        if (enable_mtp_) {
            if (mtp_cache == nullptr || mtp_allocations_.size() != session_count_) {
                throw std::logic_error("MTP stage cache ownership is incomplete");
            }
            mtp_texts_.reserve(session_count_);
            for (std::uint32_t session = 0; session < session_count_; ++session) {
                auto context = std::make_unique<StageTextContext>(
                    device_, model_->runtime, *workspace_,
                    decoder_->text_kv.execution_view(kv_allocations_.at(session)),
                    decoder_->linear_attention, rounds_.at(session), prefill_hidden_, prefill_chunk_, 0,
                    mtp_cache->execution_view(mtp_allocations_.at(session)), &decoder_->text_kv,
                    mtp_cache, range);
                context->set_linear_state_slots(static_cast<std::int32_t>(2U * session),
                                                static_cast<std::int32_t>(2U * session + 1U));
                context->set_mtp_proposal_extent(verify_width_ - 1U);
                context->set_long_context_computation(computation_);
                context->set_mtp_attention_window(mtp_attention_window);
                mtp_texts_.push_back(std::move(context));
            }
        }
        if (shard_.token_embedding) {
            frontend_ = std::make_unique<qwen3_6::Frontend>(
                qwen3_6::make_frontend(model_->frontend, enable_vision_));
        }
        if (enable_vision_) {
            vision_io_ = std::make_unique<DeviceArena>(
                static_cast<std::size_t>(prefill_chunk_) * 3U * sizeof(std::int32_t));
        }
        device_.synchronize();
    }

    ~Impl() {
        (void)cudaSetDevice(device_id_);
        if (device_.stream != nullptr) { (void)cudaStreamSynchronize(device_.stream); }
    }

    void ensure_vision_session(const TextPromptSequence& prompt, std::uint32_t begin) {
        if (!enable_vision_ || prompt.prepared == nullptr || !prompt.prepared->has_media()) {
            throw std::logic_error("multimodal prefill requested without stage-0 vision resources");
        }
        if (vision_session_ != nullptr && active_vision_prompt_ == prompt.prepared.get()) {
            return;
        }
        if (begin != 0) {
            throw std::invalid_argument("new multimodal prompt must begin at token zero");
        }
        qwen3_6::PreparedPromptData& data = *prompt.prepared;
        vision_plan_ = {};
        vision_plan_.control =
            std::make_shared<qwen3_6::VisionControl>(qwen3_6::build_vision_control(data));
        std::size_t max_merged = 0;
        std::size_t max_workspace = 0;
        vision_plan_.uses.reserve(vision_plan_.control->items.size());
        for (std::size_t item_index = 0;
             item_index < vision_plan_.control->items.size(); ++item_index) {
            const qwen3_6::VisionItemControl& item =
                vision_plan_.control->items[item_index];
            if (item.scatter_indices.empty()) {
                throw std::invalid_argument("vision item has no scatter positions");
            }
            const auto [first, last] = std::minmax_element(
                item.scatter_indices.begin(), item.scatter_indices.end());
            vision_plan_.uses.push_back(StageVisionUseSpan{
                .begin = static_cast<std::uint32_t>(*first),
                .end = static_cast<std::uint32_t>(*last + 1),
                .item_index = static_cast<std::uint32_t>(item_index),
            });
            max_merged = std::max(max_merged, item.merged_count);
            max_workspace = std::max(
                max_workspace, StageVisionContext::workspace_bytes(item));
        }
        if (max_merged <= 0 || max_workspace == 0) {
            throw std::invalid_argument("vision prompt produced an empty runtime plan");
        }
        vision_workspace_ = std::make_unique<DeviceArena>(max_workspace);
        vision_transient_ = std::make_unique<DeviceArena>(
            StageVisionContext::output_transient_bytes(max_merged));
        const runtime::TransientRegion transient{
            .data = static_cast<std::byte*>(vision_transient_->base()),
            .size = vision_transient_->capacity(),
            .alignment = 256,
        };
        vision_session_ = std::make_unique<StageVisionPrefillSession>(
            device_, model_->runtime, *vision_workspace_, data, vision_plan_, transient);
        active_vision_prompt_ = prompt.prepared.get();
    }

    void ensure_sm75_a8_decode_layout() {
#if defined(NINFER_CUDA_SM75) && defined(NINFER_QWEN36_SM75_DENSE_A8)
        const char* mode = std::getenv("NINFER_DUAL_SM75_A8");
        if (mode == nullptr || verify_width_ == 0 || sm75_a8_layout_active_) { return; }
        const bool stage0_mlp = std::strcmp(mode, "stage0-mlp") == 0;
        const bool stage1_mlp = std::strcmp(mode, "stage1-mlp") == 0;
        const bool both_mlp = std::strcmp(mode, "both-mlp") == 0;
        const bool stage1_all = std::strcmp(mode, "stage1-all") == 0;
        const bool both_mlp_stage1_all = std::strcmp(mode, "both-mlp-stage1-all") == 0;
        const bool legacy_all = mode[0] == '1' && mode[1] == '\0';
        if (!stage0_mlp && !stage1_mlp && !both_mlp && !stage1_all &&
            !both_mlp_stage1_all && !legacy_all) {
            return;
        }
        if ((stage1_mlp || stage1_all) && !shard_.output_head) { return; }
        if (stage0_mlp && !shard_.token_embedding) { return; }

        const std::size_t scratch_bytes = std::max({
            ops::detail::q4_linear_swiglu_sm75_transform_scratch_bytes(),
            ops::detail::q5_linear_add_sm75_transform_scratch_bytes(),
            ops::detail::gdn_q5_sm75_transform_scratch_bytes(),
        });
        DeviceArena transform_scratch(scratch_bytes);
        void* scratch = transform_scratch.base();
        const auto transform_post_mixer = [&](auto& post_mixer) {
            if (post_mixer.gate_up.qtype == QType::Q4G64_F16S &&
                post_mixer.gate_up.qdata != nullptr) {
                ops::detail::q4_linear_swiglu_sm75_transform_layout(
                    post_mixer.gate_up, QuantLayout::RowSplitSm75Fragment, scratch,
                    scratch_bytes, device_.stream);
            }
            if (post_mixer.down.qtype == QType::Q5G64_F16S &&
                post_mixer.down.qdata != nullptr) {
                ops::detail::q5_linear_add_sm75_transform_layout(
                    post_mixer.down, QuantLayout::RowSplitSm75Fragment, scratch, scratch_bytes,
                    device_.stream);
            }
        };
        for (int global_layer = static_cast<int>(shard_.layer_begin);
             global_layer < static_cast<int>(shard_.layer_end); ++global_layer) {
            if (qwen3_6::is_full_attention_layer(global_layer)) {
                transform_post_mixer(model_->runtime.full_layers.at(
                    static_cast<std::size_t>(qwen3_6::full_attention_index(global_layer)))
                                         .post_mixer);
            } else {
                transform_post_mixer(model_->runtime.gdn_layers.at(
                    static_cast<std::size_t>(qwen3_6::gdn_index(global_layer)))
                                         .post_mixer);
            }
        }
        if (stage1_all || legacy_all ||
            (both_mlp_stage1_all && shard_.output_head)) {
            Variant::sm75_transform_gdn_value_z(
                model_->runtime, QuantLayout::RowSplitSm75Fragment, scratch, device_.stream);
        }
        device_.synchronize();
        sm75_a8_layout_active_ = true;
#endif
    }
    void activate() const { CUDA_CHECK(cudaSetDevice(device_id_)); }
    void validate_session(std::uint32_t session) const {
        if (session >= session_count_) { throw std::out_of_range("text shard session is out of range"); }
    }
    [[nodiscard]] Tensor scalar(const Tensor& tensor, std::uint32_t session) const {
        validate_session(session);
        return tensor.slice(0, static_cast<std::int32_t>(session), 1);
    }
    [[nodiscard]] Tensor column(const Tensor& tensor, std::uint32_t session) const {
        validate_session(session);
        return tensor.slice(1, static_cast<std::int32_t>(session), 1);
    }
    void require_verify() const {
        if (verify_width_ == 0 || replay_records_.size() != session_count_) {
            throw std::logic_error("text shard verifier state is not enabled");
        }
    }
    [[nodiscard]] Tensor verify_vector(const Tensor& tensor, std::uint32_t session) const {
        require_verify();
        validate_session(session);
        return tensor.slice(1, static_cast<std::int32_t>(session), 1);
    }
    [[nodiscard]] Tensor verify_cube(const Tensor& tensor, std::uint32_t session) const {
        require_verify();
        validate_session(session);
        return tensor.slice(2, static_cast<std::int32_t>(session), 1);
    }
    void validate_verify_pair(std::uint32_t session_begin, std::size_t count) const {
        require_verify();
        if (count != 2 || (session_begin & 1U) != 0U ||
            session_begin + count > session_count_) {
            throw std::invalid_argument("target-verifier pair must be an aligned two-session range");
        }
    }
    [[nodiscard]] Tensor verify_vector_pair(const Tensor& tensor,
                                             std::uint32_t session_begin) const {
        validate_verify_pair(session_begin, 2);
        return tensor.slice(1, static_cast<std::int32_t>(session_begin), 2);
    }
    [[nodiscard]] Tensor verify_cube_pair(const Tensor& tensor,
                                           std::uint32_t session_begin) const {
        validate_verify_pair(session_begin, 2);
        return tensor.slice(2, static_cast<std::int32_t>(session_begin), 2);
    }
    [[nodiscard]] Tensor scalar_pair(const Tensor& tensor,
                                      std::uint32_t session_begin) const {
        validate_verify_pair(session_begin, 2);
        return tensor.slice(0, static_cast<std::int32_t>(session_begin), 2);
    }
    [[nodiscard]] GdnReplayRecords& verify_records(std::uint32_t session) {
        require_verify();
        validate_session(session);
        return replay_records_.at(session);
    }
    [[nodiscard]] GdnReplayRecords& batch_verify_records(std::uint32_t pair) {
        require_verify();
        if (pair >= batch_replay_records_.size()) {
            throw std::logic_error("batched target-verifier pair records are not enabled");
        }
        return batch_replay_records_[pair];
    }
    [[nodiscard]] StageTextContext& mtp_text(std::uint32_t session) {
        if (!enable_mtp_ || mtp_texts_.size() != session_count_) {
            throw std::logic_error("text shard MTP context is not enabled");
        }
        validate_session(session);
        return *mtp_texts_.at(session);
    }
    [[nodiscard]] qwen3_6::RoundState& mtp_round(std::uint32_t session) {
        if (!enable_mtp_ || rounds_.size() != session_count_) {
            throw std::logic_error("text shard MTP round state is not enabled");
        }
        validate_session(session);
        return rounds_.at(session);
    }
    void set_verify_positions(std::int32_t position, std::int32_t rope_position,
                              std::uint32_t session) {
        Tensor positions = verify_vector(verify_positions_, session);
        Tensor rope_positions = verify_vector(verify_rope_positions_, session);
        ops::fill_i32_positions(positions, position, device_.stream);
        ops::fill_i32_positions(rope_positions, rope_position, device_.stream);
    }
    void set_position(std::int32_t position, std::int32_t rope_position, std::uint32_t session) {
        Tensor position_tensor = scalar(position_, session);
        Tensor rope_tensor = scalar(rope_position_, session);
        CUDA_CHECK(cudaMemcpyAsync(position_tensor.data, &position, sizeof(position),
                                   cudaMemcpyHostToDevice, device_.stream));
        CUDA_CHECK(cudaMemcpyAsync(rope_tensor.data, &rope_position, sizeof(rope_position),
                                   cudaMemcpyHostToDevice, device_.stream));
    }

    int device_id_ = 0;
    std::uint32_t capacity_ = 0;
    std::uint32_t prefill_chunk_ = 0;
    std::uint32_t verify_width_ = 0;
    std::uint32_t session_count_ = 1;
    TextShardSpec shard_;
    KvCacheStorage kv_cache_ = KvCacheStorage::BFloat16;
    bool enable_mtp_ = false;
    bool enable_vision_ = false;
    float mtp_greedy_margin_delta_ = -1.0F;
    TextKvPageOwnership text_kv_ownership_;
    bool sm75_a8_layout_active_ = false;
    const qwen3_6::LongContextComputation computation_;
    DeviceContext device_;
    WeightsProfile weights_profile_ = WeightsProfile::GroupwiseIntW8Endpoints;
    std::unique_ptr<LoadedModelData> model_;
    std::unique_ptr<FullVocabularyDraftOwner> full_vocab_owner_;
    PlannedStageState state_plan_;
    PlannedStageState legacy_full_state_plan_;
    PlannedStageState snapshot_source_state_plan_;
    std::unique_ptr<DeviceArena> state_storage_;
    std::unique_ptr<qwen3_6::DecoderState> decoder_;
    std::vector<GdnReplayRecordLayout> replay_layouts_;
    std::vector<GdnReplayRecords> replay_records_;
    std::vector<GdnReplayRecordLayout> batch_replay_layouts_;
    std::vector<GdnReplayRecords> batch_replay_records_;
    std::vector<PagedKVAllocation> kv_allocations_;
    std::vector<PagedKVAllocation> mtp_allocations_;
    std::unique_ptr<DeviceArena> workspace_;
    std::unique_ptr<DeviceArena> io_storage_;
    std::unique_ptr<DeviceArena> vision_io_;
    std::unique_ptr<DeviceArena> vision_workspace_;
    std::unique_ptr<DeviceArena> vision_transient_;
    StageVisionPrefillPlan vision_plan_;
    std::unique_ptr<StageVisionPrefillSession> vision_session_;
    const qwen3_6::PreparedPromptData* active_vision_prompt_ = nullptr;
    qwen3_6::RoundState round_;
    std::vector<qwen3_6::RoundState> rounds_;
    Tensor id_;
    Tensor position_;
    Tensor rope_position_;
    Tensor kv_table_row_;
    Tensor state_slot_;
    Tensor residual_;
    Tensor prefill_ids_;
    Tensor prefill_positions_;
    Tensor prefill_rope_positions_;
    Tensor prefill_residual_;
    Tensor prefill_hidden_;
    Tensor normalized_;
    Tensor logits_;
    Tensor output_token_;
    Tensor verify_ids_;
    Tensor verify_positions_;
    Tensor verify_rope_positions_;
    Tensor verify_valid_columns_;
    Tensor verify_residual_;
    Tensor verify_hidden_;
    Tensor verify_logits_;
    Tensor verify_tokens_;
    std::unique_ptr<StageTextContext> text_;
    std::vector<std::unique_ptr<StageTextContext>> mtp_texts_;
    std::vector<DecodeGraphDefinition> verify_token_graph_definitions_;
    std::vector<DecodeGraphExecutable> verify_token_graphs_;
    std::vector<std::uint8_t> verify_token_graph_phases_;
    std::vector<DecodeGraphDefinition> verify_hidden_graph_definitions_;
    std::vector<DecodeGraphExecutable> verify_hidden_graphs_;
    std::vector<std::uint8_t> verify_hidden_graph_phases_;
    bool greedy_head_tokens_only_ = false;
    std::array<bool, kMaximumConcurrency> verify_logits_materialized_{};
    std::unique_ptr<qwen3_6::Frontend> frontend_;
    std::size_t weights_bytes_ = 0;
    std::size_t state_bytes_ = 0;
    std::size_t workspace_bytes_ = 0;
    std::size_t kv_payload_bytes_ = 0;
    std::size_t snapshot_source_io_prefix_bytes_ = 0;
};


TextShardStage::TextShardStage(std::string artifact_path, int device_id, std::uint32_t capacity,
                               TextShardSpec shard, artifact::LoadProgress* progress,
                               std::uint32_t prefill_chunk, KvCacheStorage kv_cache,
                               std::uint32_t sessions, std::uint32_t verify_width,
                               bool enable_mtp, TextKvPageOwnership text_kv_ownership,
                               bool enable_vision,qwen3_6::LongContextComputation computation,
                               std::uint32_t mtp_attention_window,bool full_vocab_draft)
    : impl_(std::make_unique<Impl>(std::move(artifact_path), device_id, capacity, shard, progress,
                                   prefill_chunk, kv_cache, sessions, verify_width, enable_mtp,
                                   text_kv_ownership, enable_vision, computation, mtp_attention_window, full_vocab_draft)) {}

TextShardStage::~TextShardStage() = default;

TextPromptSequence TextShardStage::prepare_text_prompt(std::string_view text,
                                                       bool enable_thinking) const {
    PromptInput input;
    input.options.enable_thinking = enable_thinking;
    ChatMessage message;
    message.role = "user";
    MessagePart part;
    part.kind = MessagePartKind::Text;
    part.text.assign(text);
    message.parts.push_back(std::move(part));
    input.messages.push_back(std::move(message));
    return prepare_prompt(std::move(input));
}

TextPromptSequence TextShardStage::prepare_prompt(PromptInput input) const {
    if (impl_ == nullptr || impl_->frontend_ == nullptr) {
        throw std::logic_error("text shard does not own frontend resources");
    }
    qwen3_6::PreparedPrompt prepared = impl_->frontend_->prepare(std::move(input));
    qwen3_6::PreparedPromptData data =
        qwen3_6::PreparedPromptAccess::take(std::move(prepared));
    TextPromptSequence out;
    out.token_ids = data.token_ids;
    out.rope_delta = data.rope_delta;
    if (data.has_media()) {
        if (!impl_->enable_vision_) {
            throw std::invalid_argument("media prompt requires a vision-enabled stage-0 shard");
        }
        out.prepared = std::make_shared<qwen3_6::PreparedPromptData>(std::move(data));
    }
    return out;
}

void TextShardStage::forward_token(TokenId token, std::int32_t position,
                                   std::int32_t rope_position, std::uint32_t session) {
    if (impl_ == nullptr) { throw std::logic_error("text shard is empty"); }
    if (position < 0 || static_cast<std::uint32_t>(position) >= impl_->capacity_) {
        throw std::out_of_range("text shard token position is outside capacity");
    }
    impl_->activate();
    Tensor id = impl_->scalar(impl_->id_, session);
    Tensor position_tensor = impl_->scalar(impl_->position_, session);
    Tensor rope_tensor = impl_->scalar(impl_->rope_position_, session);
    Tensor kv_row = impl_->scalar(impl_->kv_table_row_, session);
    Tensor state_slot = impl_->scalar(impl_->state_slot_, session);
    Tensor residual = impl_->column(impl_->residual_, session);
    CUDA_CHECK(cudaMemcpyAsync(id.data, &token, sizeof(token), cudaMemcpyHostToDevice,
                               impl_->device_.stream));
    impl_->set_position(position, rope_position, session);
    const std::uint32_t visible = static_cast<std::uint32_t>(position) + 1U;
    impl_->text_->forward_token_shard(id, position_tensor, rope_tensor, kv_row, state_slot,
                                      ops::GqaExecutionEnvelope{visible, visible}, residual);
}

void TextShardStage::forward_hidden(std::int32_t position, std::int32_t rope_position,
                                    std::uint32_t session) {
    if (impl_ == nullptr) { throw std::logic_error("text shard is empty"); }
    if (position < 0 || static_cast<std::uint32_t>(position) >= impl_->capacity_) {
        throw std::out_of_range("text shard hidden position is outside capacity");
    }
    impl_->activate();
    Tensor position_tensor = impl_->scalar(impl_->position_, session);
    Tensor rope_tensor = impl_->scalar(impl_->rope_position_, session);
    Tensor kv_row = impl_->scalar(impl_->kv_table_row_, session);
    Tensor state_slot = impl_->scalar(impl_->state_slot_, session);
    Tensor residual = impl_->column(impl_->residual_, session);
    impl_->set_position(position, rope_position, session);
    const std::uint32_t visible = static_cast<std::uint32_t>(position) + 1U;
    impl_->text_->forward_hidden_shard(residual, position_tensor, rope_tensor, kv_row, state_slot,
                                       ops::GqaExecutionEnvelope{visible, visible});
}

void TextShardStage::prefill_tokens(std::span<const TokenId> tokens, std::int32_t position,
                                    std::int32_t rope_position, std::uint32_t session) {
    if (impl_ == nullptr || tokens.empty() || tokens.size() > impl_->prefill_chunk_ || position < 0 ||
        rope_position < 0 || static_cast<std::uint64_t>(position) + tokens.size() > impl_->capacity_) {
        throw std::invalid_argument("invalid text shard token prefill");
    }
    impl_->activate();
    impl_->validate_session(session);
    impl_->text_->set_linear_state_slots(static_cast<std::int32_t>(2U * session),
                                         static_cast<std::int32_t>(2U * session + 1U));
    const auto width = static_cast<std::int32_t>(tokens.size());
    Tensor ids       = impl_->prefill_ids_.slice(0, 0, width);
    Tensor positions = impl_->prefill_positions_.slice(0, 0, width);
    Tensor rope_positions = impl_->prefill_rope_positions_.slice(0, 0, width);
    Tensor residual       = impl_->prefill_residual_.slice(1, 0, width);
    Tensor kv_row         = impl_->scalar(impl_->kv_table_row_, session);
    CUDA_CHECK(cudaMemcpyAsync(ids.data, tokens.data(), tokens.size_bytes(), cudaMemcpyHostToDevice,
                               impl_->device_.stream));
    ops::fill_i32_positions(positions, position, impl_->device_.stream);
    ops::fill_i32_positions(rope_positions, rope_position, impl_->device_.stream);
    const auto visible = static_cast<std::uint32_t>(position) + static_cast<std::uint32_t>(width);
    impl_->text_->prefill_token_shard(ids, positions, rope_positions, kv_row,
                                      ops::GqaExecutionEnvelope{visible, visible}, residual);
    if (impl_->shard_.output_head) {
        Tensor last = residual.slice(1, width - 1, 1);
        Tensor session_residual = impl_->column(impl_->residual_, session);
        CUDA_CHECK(cudaMemcpyAsync(session_residual.data, last.data, session_residual.bytes(),
                                   cudaMemcpyDeviceToDevice, impl_->device_.stream));
    }
}

void TextShardStage::copy_prefill_residual_to_host(void* destination, std::size_t tokens) {
    if (impl_ == nullptr || destination == nullptr || tokens == 0 || tokens > impl_->prefill_chunk_) {
        throw std::invalid_argument("invalid text shard prefill residual download");
    }
    impl_->activate();
    const std::size_t bytes = residual_bytes(tokens);
    CUDA_CHECK(cudaMemcpyAsync(destination, impl_->prefill_residual_.data, bytes,
                               cudaMemcpyDeviceToHost, impl_->device_.stream));
    impl_->device_.synchronize();
}

void TextShardStage::copy_prefill_residual_from_host(const void* source, std::size_t tokens) {
    if (impl_ == nullptr || source == nullptr || tokens == 0 || tokens > impl_->prefill_chunk_) {
        throw std::invalid_argument("invalid text shard prefill residual upload");
    }
    impl_->activate();
    CUDA_CHECK(cudaMemcpyAsync(impl_->prefill_residual_.data, source, residual_bytes(tokens),
                               cudaMemcpyHostToDevice, impl_->device_.stream));
}

void TextShardStage::prefill_hidden(std::size_t tokens, std::int32_t position,
                                    std::int32_t rope_position, std::uint32_t session) {
    if (impl_ == nullptr || tokens == 0 || tokens > impl_->prefill_chunk_ || position < 0 ||
        rope_position < 0 || static_cast<std::uint64_t>(position) + tokens > impl_->capacity_) {
        throw std::invalid_argument("invalid text shard hidden prefill");
    }
    impl_->activate();
    impl_->validate_session(session);
    impl_->text_->set_linear_state_slots(static_cast<std::int32_t>(2U * session),
                                         static_cast<std::int32_t>(2U * session + 1U));
    const auto width = static_cast<std::int32_t>(tokens);
    Tensor positions = impl_->prefill_positions_.slice(0, 0, width);
    Tensor rope_positions = impl_->prefill_rope_positions_.slice(0, 0, width);
    Tensor residual       = impl_->prefill_residual_.slice(1, 0, width);
    Tensor kv_row         = impl_->scalar(impl_->kv_table_row_, session);
    ops::fill_i32_positions(positions, position, impl_->device_.stream);
    ops::fill_i32_positions(rope_positions, rope_position, impl_->device_.stream);
    const auto visible = static_cast<std::uint32_t>(position) + static_cast<std::uint32_t>(width);
    impl_->text_->prefill_hidden_shard(residual, positions, rope_positions, kv_row,
                                       ops::GqaExecutionEnvelope{visible, visible});
    Tensor last = residual.slice(1, width - 1, 1);
    Tensor session_residual = impl_->column(impl_->residual_, session);
    CUDA_CHECK(cudaMemcpyAsync(session_residual.data, last.data, session_residual.bytes(),
                               cudaMemcpyDeviceToDevice, impl_->device_.stream));
}

std::size_t TextShardStage::prefill_multimodal_tokens(
    const TextPromptSequence& prompt, std::uint32_t begin, std::uint32_t nominal_length,
    std::uint32_t session) {
    if (impl_ == nullptr || prompt.prepared == nullptr || !prompt.prepared->has_media() ||
        nominal_length == 0 || begin >= prompt.token_ids.size() ||
        nominal_length > prompt.token_ids.size() - begin) {
        throw std::invalid_argument("invalid multimodal stage-0 prefill request");
    }
    impl_->activate();
    impl_->validate_session(session);
    impl_->ensure_vision_session(prompt, begin);
    StageVisionChunk chunk = impl_->vision_session_->prepare_chunk(begin, nominal_length);
    if (chunk.length <= 0 || chunk.length > static_cast<std::int32_t>(impl_->prefill_chunk_)) {
        throw std::logic_error("vision prefill returned an invalid chunk length");
    }
    const auto width = chunk.length;
    Tensor ids = impl_->prefill_ids_.slice(0, 0, width);
    Tensor positions = impl_->prefill_positions_.slice(0, 0, width);
    Tensor rope_positions(
        impl_->vision_io_->base(), DType::I32, {width, 3});
    Tensor residual = impl_->prefill_residual_.slice(1, 0, width);
    Tensor kv_row = impl_->scalar(impl_->kv_table_row_, session);
    const qwen3_6::PreparedPromptData& data = *prompt.prepared;
    CUDA_CHECK(cudaMemcpyAsync(ids.data, data.token_ids.data() + begin,
                               static_cast<std::size_t>(width) * sizeof(TokenId),
                               cudaMemcpyHostToDevice, impl_->device_.stream));
    ops::fill_i32_positions(positions, static_cast<std::int32_t>(begin),
                            impl_->device_.stream);
    const std::size_t total = data.token_ids.size();
    for (std::int32_t axis = 0; axis < 3; ++axis) {
        CUDA_CHECK(cudaMemcpyAsync(
            static_cast<std::byte*>(rope_positions.data) +
                static_cast<std::size_t>(axis) * rope_positions.nb[1],
            data.positions.data() + static_cast<std::size_t>(axis) * total + begin,
            static_cast<std::size_t>(width) * sizeof(std::int32_t),
            cudaMemcpyHostToDevice, impl_->device_.stream));
    }

    std::vector<std::int32_t> local_scatter;
    std::int32_t visual_begin = 0;
    if (chunk.control != nullptr) {
        const auto& scatter = chunk.control->scatter_indices;
        const auto first = std::lower_bound(scatter.begin(), scatter.end(),
                                            static_cast<std::int32_t>(begin));
        const auto last = std::lower_bound(scatter.begin(), scatter.end(),
                                           static_cast<std::int32_t>(begin + width));
        visual_begin = static_cast<std::int32_t>(std::distance(scatter.begin(), first));
        local_scatter.reserve(static_cast<std::size_t>(std::distance(first, last)));
        for (auto it = first; it != last; ++it) {
            local_scatter.push_back(*it - static_cast<std::int32_t>(begin));
        }
    }
    impl_->text_->set_linear_state_slots(static_cast<std::int32_t>(2U * session),
                                         static_cast<std::int32_t>(2U * session + 1U));
    const auto visible = begin + static_cast<std::uint32_t>(width);
    impl_->text_->prefill_multimodal_token_shard(
        ids, positions, rope_positions, chunk.embeddings, visual_begin, local_scatter, kv_row,
        ops::GqaExecutionEnvelope{visible, visible}, residual);
    return static_cast<std::size_t>(width);
}

void TextShardStage::prefill_multimodal_hidden(
    const TextPromptSequence& prompt, std::uint32_t begin, std::size_t tokens,
    std::uint32_t session) {
    if (impl_ == nullptr || prompt.prepared == nullptr || tokens == 0 ||
        tokens > impl_->prefill_chunk_ || begin > prompt.token_ids.size() - tokens) {
        throw std::invalid_argument("invalid multimodal stage-1 prefill request");
    }
    impl_->activate();
    impl_->validate_session(session);
    const auto width = static_cast<std::int32_t>(tokens);
    Tensor positions = impl_->prefill_positions_.slice(0, 0, width);
    if (impl_->vision_io_ == nullptr) {
        impl_->vision_io_ = std::make_unique<DeviceArena>(
            static_cast<std::size_t>(impl_->prefill_chunk_) * 3U * sizeof(std::int32_t));
    }
    Tensor rope_positions(impl_->vision_io_->base(), DType::I32, {width, 3});
    Tensor residual = impl_->prefill_residual_.slice(1, 0, width);
    Tensor kv_row = impl_->scalar(impl_->kv_table_row_, session);
    ops::fill_i32_positions(positions, static_cast<std::int32_t>(begin),
                            impl_->device_.stream);
    const qwen3_6::PreparedPromptData& data = *prompt.prepared;
    const std::size_t total = data.token_ids.size();
    for (std::int32_t axis = 0; axis < 3; ++axis) {
        CUDA_CHECK(cudaMemcpyAsync(
            static_cast<std::byte*>(rope_positions.data) +
                static_cast<std::size_t>(axis) * rope_positions.nb[1],
            data.positions.data() + static_cast<std::size_t>(axis) * total + begin,
            tokens * sizeof(std::int32_t), cudaMemcpyHostToDevice,
            impl_->device_.stream));
    }
    impl_->text_->set_linear_state_slots(static_cast<std::int32_t>(2U * session),
                                         static_cast<std::int32_t>(2U * session + 1U));
    const auto visible = begin + static_cast<std::uint32_t>(tokens);
    impl_->text_->prefill_hidden_shard(
        residual, positions, rope_positions, kv_row,
        ops::GqaExecutionEnvelope{visible, visible});
    Tensor last = residual.slice(1, width - 1, 1);
    Tensor session_residual = impl_->column(impl_->residual_, session);
    CUDA_CHECK(cudaMemcpyAsync(session_residual.data, last.data, session_residual.bytes(),
                               cudaMemcpyDeviceToDevice, impl_->device_.stream));
}

void TextShardStage::prepare_sm75_a8_decode_layout() {
    if (impl_ == nullptr) { throw std::logic_error("text shard is empty"); }
    impl_->activate();
    impl_->ensure_sm75_a8_decode_layout();
}

void TextShardStage::set_mlp_execution_override(
    qwen3_6::detail::MlpExecutionOverride* override) {
    if (impl_ == nullptr) { throw std::logic_error("text shard is empty"); }
    impl_->activate();
    impl_->device_.synchronize();
    impl_->text_->set_mlp_execution_override(override);
    if (override != nullptr) {
        // An external coordinator performs cross-device host staging and therefore cannot execute
        // inside CUDA stream capture. Disable only this stage's target-verifier graph wrappers;
        // the canonical eager body, prefill, proposal recursion, and non-overridden stages remain.
        impl_->verify_token_graphs_.clear();
        impl_->verify_token_graph_definitions_.clear();
        impl_->verify_token_graph_phases_.clear();
        impl_->verify_hidden_graphs_.clear();
        impl_->verify_hidden_graph_definitions_.clear();
        impl_->verify_hidden_graph_phases_.clear();
    }
}

void TextShardStage::set_head_execution_override(qwen3_6::detail::HeadExecutionOverride* value,
                                                 bool greedy_tokens_only) {
    if (impl_ == nullptr || !impl_->shard_.output_head || impl_->session_count_ != 1 ||
        impl_->verify_width_ != 4 || impl_->computation_ != qwen3_6::LongContextComputation::Complete)
        throw std::invalid_argument("parallel vocabulary override requires complete C1 D3 output stage");
    impl_->activate();
    // Detach the borrowed pointer even if draining reports an asynchronous error.
    // A failed teardown must not leave future dispatch pointing at freed storage.
    if (value == nullptr) impl_->text_->set_head_execution_override(nullptr);
    impl_->device_.synchronize();
    if (value != nullptr) value->prepare(impl_->model_->runtime.output_head,impl_->device_.stream);
    impl_->text_->set_head_execution_override(value);
    impl_->greedy_head_tokens_only_ = value != nullptr && greedy_tokens_only;
    // An existing graph must never retain a pointer to an earlier coordinator.
    // The head coordinator records its own bounded multi-device operation.
    impl_->verify_hidden_graphs_.clear();
    impl_->verify_hidden_graph_definitions_.clear();
    impl_->verify_hidden_graph_phases_.clear();
}

void TextShardStage::set_mtp_greedy_margin(float delta) {
    if (impl_ == nullptr) { throw std::logic_error("text shard is empty"); }
    if (delta != -1.0F && !(delta >= 0.0F && delta <= 128.0F)) {
        throw std::invalid_argument("MTP greedy margin must be disabled (-1) or in [0,128]");
    }
    impl_->mtp_greedy_margin_delta_ = delta;
}

void TextShardStage::prepare_gqa_prefix(
    qwen3_6::detail::GqaExecutionOverride& executor, std::uint32_t populated_tokens) {
    if (!executor.requires_prefix_preparation()) { return; }
    if (impl_ == nullptr || impl_->session_count_ != 1 || populated_tokens == 0 ||
        populated_tokens > impl_->capacity_) {
        throw std::invalid_argument("attention prefix preparation needs a populated single session");
    }
    impl_->activate();
    impl_->device_.synchronize();
    const auto cache = impl_->decoder_->text_kv.execution_view(impl_->kv_allocations_.at(0));
    const auto first = qwen3_6::full_attention_layers(impl_->shard_.layer_begin);
    for (std::uint32_t layer = impl_->shard_.layer_begin; layer < impl_->shard_.layer_end; ++layer) {
        if (!qwen3_6::is_full_attention_layer(layer)) { continue; }
        const auto local = qwen3_6::full_attention_index(layer) - first;
        executor.prepare_prefix_layer(layer, cache.layer_view(local), populated_tokens,
                                      impl_->device_.stream);
    }
    impl_->activate();
    impl_->device_.synchronize();
}

void TextShardStage::set_gqa_execution_override(
    qwen3_6::detail::GqaExecutionOverride* override) {
    if (impl_ == nullptr) { throw std::logic_error("text shard is empty"); }
    impl_->activate();
    impl_->device_.synchronize();
    impl_->text_->set_gqa_execution_override(override);
    if(std::getenv("NINFER_TARGET_GUIDED_PROPOSAL_ORACLE")!=nullptr) {
        // The drafter has separate request contexts. Share only the coordinator's
        // read-only guide seam; MTP attention still uses its own cache and kernels.
        for(auto& context:impl_->mtp_texts_)context->set_gqa_execution_override(override);
    }

    if (override != nullptr) {
        // Cross-device host staging is not graph-capture-safe. Preserve the canonical eager target
        // body and disable only this stage's target-verifier graph wrappers.
        impl_->verify_token_graphs_.clear();
        impl_->verify_token_graph_definitions_.clear();
        impl_->verify_token_graph_phases_.clear();
        impl_->verify_hidden_graphs_.clear();
        impl_->verify_hidden_graph_definitions_.clear();
        impl_->verify_hidden_graph_phases_.clear();
    }
}

void TextShardStage::mtp_prefill_prompt_chunk(std::span<const TokenId> full_prompt,
                                                  std::uint32_t chunk_begin,
                                                  std::size_t tokens, bool final_chunk,
                                                  TokenId generated_token,
                                                  std::uint32_t session) {
    if (impl_ == nullptr || tokens == 0 || tokens > impl_->prefill_chunk_ ||
        chunk_begin >= full_prompt.size() || tokens > full_prompt.size() - chunk_begin) {
        throw std::invalid_argument("invalid sharded MTP prompt chunk");
    }
    impl_->activate();
    const auto width = static_cast<std::int32_t>(tokens);
    Tensor residual = impl_->prefill_residual_.slice(1, 0, width);
    Tensor positions = impl_->prefill_positions_.slice(0, 0, width);
    Tensor rope_positions = impl_->prefill_rope_positions_.slice(0, 0, width);
    impl_->mtp_text(session).mtp_prefill_shard_chunk(
        full_prompt, chunk_begin, residual, positions, rope_positions, final_chunk,
        generated_token, final_chunk);
}

void TextShardStage::mtp_prefill_multimodal_prompt_chunk(
    const TextPromptSequence& prompt, std::uint32_t chunk_begin, std::size_t tokens,
    bool final_chunk, TokenId generated_token, std::uint32_t session) {
    if (impl_ == nullptr || prompt.prepared == nullptr || !prompt.prepared->has_media() ||
        tokens == 0 || tokens > impl_->prefill_chunk_ ||
        chunk_begin >= prompt.token_ids.size() ||
        tokens > prompt.token_ids.size() - chunk_begin || !impl_->enable_mtp_ ||
        !impl_->enable_vision_) {
        throw std::invalid_argument("invalid sharded multimodal MTP prompt chunk");
    }
    impl_->activate();
    impl_->validate_session(session);
    impl_->ensure_vision_session(prompt, chunk_begin);
    StageVisionChunk chunk = impl_->vision_session_->prepare_chunk(
        chunk_begin, static_cast<std::uint32_t>(tokens));
    if (chunk.length != static_cast<std::int32_t>(tokens)) {
        throw std::logic_error("stage-1 vision chunk disagrees with stage-0 chunk geometry");
    }
    const auto width = static_cast<std::int32_t>(tokens);
    Tensor residual = impl_->prefill_residual_.slice(1, 0, width);
    Tensor positions = impl_->prefill_positions_.slice(0, 0, width);
    if (impl_->vision_io_ == nullptr) {
        throw std::logic_error("stage-1 multimodal RoPE storage is unavailable");
    }
    Tensor rope_positions(impl_->vision_io_->base(), DType::I32, {width, 3});
    const Tensor* visual_embeddings = chunk.control != nullptr ? &chunk.embeddings : nullptr;
    const std::span<const std::int32_t> scatter =
        chunk.control != nullptr
            ? std::span<const std::int32_t>(chunk.control->scatter_indices)
            : std::span<const std::int32_t>{};
    impl_->mtp_text(session).mtp_prefill_shard_chunk(
        prompt.token_ids, chunk_begin, residual, positions, rope_positions, final_chunk,
        generated_token, final_chunk, visual_embeddings, scatter);
    (void)impl_->vision_session_->release_consumed_media_payload();
}

std::vector<TokenId> TextShardStage::mtp_initial_drafts(std::uint32_t session) {
    if (impl_ == nullptr || !impl_->enable_mtp_) {
        throw std::logic_error("text shard MTP proposal state is unavailable");
    }
    impl_->activate();
    qwen3_6::RoundState& round = impl_->mtp_round(session);
    if (!round.mtp) { throw std::logic_error("text shard MTP proposal tensors are unavailable"); }
    const std::size_t count = impl_->verify_width_ - 1U;
    std::vector<TokenId> drafts(count);
    Tensor source = round.mtp->draft_tokens.slice(0, 0, static_cast<std::int32_t>(count));
    CUDA_CHECK(cudaMemcpyAsync(drafts.data(), source.data, drafts.size() * sizeof(TokenId),
                               cudaMemcpyDeviceToHost, impl_->device_.stream));
    impl_->device_.synchronize();
    return drafts;
}

Mtp3VerifyInput TextShardStage::mtp_prepare_verify(const Mtp3SessionState& state,
                                                     std::uint32_t session) {
    if (state.draft_extent < 0 ||
        state.draft_extent > static_cast<std::int32_t>(impl_->verify_width_ - 1U)) {
        throw std::invalid_argument("MTP draft extent is outside the configured width");
    }
    const std::int32_t valid_columns = state.draft_extent + 1;
    if (impl_ == nullptr || !impl_->enable_mtp_ || state.frontier < 0 ||
        state.remaining_budget < valid_columns ||
        static_cast<std::uint64_t>(state.frontier) +
                static_cast<std::uint32_t>(valid_columns) >
            impl_->capacity_) {
        throw std::invalid_argument("MTP verify preparation is outside the qualified domain");
    }
    impl_->activate();
    qwen3_6::RoundState& round = impl_->mtp_round(session);
    if (!round.mtp_decode) { throw std::logic_error("text shard MTP decode tensors are unavailable"); }
    qwen3_6::MtpDecodeState& frame = *round.mtp_decode;
    qwen3_6::MtpDecodeIngress ingress{};
    ingress.anchors[0] = state.anchor;
    ingress.base_frontiers[0] = state.frontier;
    ingress.remaining_budgets[0] = state.remaining_budget;
    ingress.current_extents[0] = state.draft_extent;
    ingress.target_valid_columns[0] = valid_columns;
    for (std::size_t index = 0; index < state.drafts.size(); ++index) {
        ingress.current_drafts[index] =
            index < static_cast<std::size_t>(state.draft_extent) ? state.drafts[index]
                                                                  : state.anchor;
    }
    for (std::size_t column = 0; column < impl_->verify_width_; ++column) {
        const std::int32_t offset = std::min<std::int32_t>(
            static_cast<std::int32_t>(column), state.draft_extent);
        ingress.target_rope_positions[column] = state.frontier + offset + state.rope_delta;
    }
    ingress.text_kv_table_rows[0] = static_cast<std::int32_t>(session);
    ingress.mtp_kv_table_rows[0] = static_cast<std::int32_t>(session);
    ingress.lanes[0] = static_cast<std::int32_t>(session);
    ingress.rope_deltas[0] = state.rope_delta;
    ingress.sampling[0] = {};
    CUDA_CHECK(cudaMemcpyAsync(frame.ingress.data, &ingress, sizeof(ingress),
                               cudaMemcpyHostToDevice, impl_->device_.stream));
    ops::speculative_prepare_verify_inputs(frame.anchors, frame.current_drafts,
                                           frame.base_frontiers, frame.current_extents,
                                           frame.verify_ids, frame.target_positions,
                                           impl_->device_.stream);

    Mtp3VerifyInput out;
    out.valid_columns = valid_columns;
    for (std::size_t column = 0; column < impl_->verify_width_; ++column) {
        const std::int32_t offset = std::min<std::int32_t>(
            static_cast<std::int32_t>(column), state.draft_extent);
        out.ids[column] = column == 0 || static_cast<std::int32_t>(column) > state.draft_extent
                              ? state.anchor
                              : state.drafts[column - 1U];
        out.positions[column] = state.frontier + offset;
        out.rope_positions[column] = out.positions[column] + state.rope_delta;
    }
    return out;
}

Mtp3RoundResult TextShardStage::mtp_accept_round(const Mtp3SessionState& state,
                                                 std::uint32_t session) {
    if (impl_ == nullptr || !impl_->enable_mtp_) {
        throw std::invalid_argument("MTP3 round acceptance is outside the qualified domain");
    }
    impl_->activate();
    qwen3_6::RoundState& round = impl_->mtp_round(session);
    if (!round.mtp_decode) { throw std::logic_error("text shard MTP decode tensors are unavailable"); }
    qwen3_6::MtpDecodeState& frame = *round.mtp_decode;
    Tensor local_tokens = impl_->verify_vector(impl_->verify_tokens_, session);
    Tensor local_hidden = impl_->verify_cube(impl_->verify_hidden_, session);
    Tensor local_logits = impl_->verify_cube(impl_->verify_logits_, session);
    const bool licensed_full_round =
        qwen3_6::license_retained_d4(impl_->computation_,impl_->verify_width_,state.draft_extent);
    if (licensed_full_round) {
        Tensor forced = local_tokens.slice(0, 0, 4).view({4});
        Tensor drafts = frame.current_drafts.view({4});
        CUDA_CHECK(cudaMemcpyAsync(forced.data, drafts.data, 4 * sizeof(std::int32_t),
                                   cudaMemcpyDeviceToDevice, impl_->device_.stream));
    }
    if (impl_->verify_width_ == 4) {
        Tensor flat_hidden = local_hidden.view({StageTextConfig::hidden, 4});
        Tensor flat_logits = local_logits.view({StageTextConfig::output_rows, 4});
        Tensor flat_drafts = frame.current_drafts.view({3});
        impl_->text_->score_target_verify_mandatory_drafts(
            flat_hidden, flat_drafts, flat_logits,
            impl_->mtp_greedy_margin_delta_ >= 0.0F);
    }
    CUDA_CHECK(cudaMemcpyAsync(frame.target_argmax.data, local_tokens.data,
                               frame.target_argmax.bytes(), cudaMemcpyDeviceToDevice,
                               impl_->device_.stream));
    CUDA_CHECK(cudaMemcpyAsync(frame.target_hidden.data, local_hidden.data,
                               frame.target_hidden.bytes(), cudaMemcpyDeviceToDevice,
                               impl_->device_.stream));
    if (licensed_full_round) {
        ops::speculative_accept_all_greedy_drafts(
            frame.target_argmax, frame.current_drafts, frame.current_extents,
            frame.base_frontiers, frame.anchors, frame.licensed_tokens,
            frame.licensed_counts, frame.accepted_drafts, impl_->device_.stream);
    } else {
    CUDA_CHECK(cudaMemcpyAsync(frame.target_logits.data, local_logits.data,
                               frame.target_logits.bytes(), cudaMemcpyDeviceToDevice,
                               impl_->device_.stream));

    impl_->workspace_->reset();
    ops::speculative_accept_greedy_drafts(
        frame.target_argmax, frame.target_logits, frame.current_drafts, frame.current_extents,
        frame.base_frontiers, frame.anchors, frame.licensed_tokens, frame.licensed_counts,
        frame.accepted_drafts, StageTextConfig::token_domain, frame.sampling, *impl_->workspace_,
        impl_->device_.stream, impl_->mtp_greedy_margin_delta_);
    }
    ops::mtp_prepare_next_round(
        frame.verify_ids, frame.anchors, frame.accepted_drafts, frame.base_frontiers,
        frame.remaining_budgets, frame.licensed_counts, frame.rope_deltas, frame.alignment_ids,
        frame.next_extents, frame.ar_positions, frame.ar_rope_positions, frame.ar_valid_columns,
        static_cast<std::int32_t>(impl_->capacity_), impl_->device_.stream);

    Mtp3RoundResult result;
    CUDA_CHECK(cudaMemcpyAsync(result.licensed_tokens.data(), frame.licensed_tokens.data,
                               result.licensed_tokens.size() * sizeof(TokenId),
                               cudaMemcpyDeviceToHost, impl_->device_.stream));
    CUDA_CHECK(cudaMemcpyAsync(&result.licensed_count, frame.licensed_counts.data,
                               sizeof(result.licensed_count), cudaMemcpyDeviceToHost,
                               impl_->device_.stream));
    CUDA_CHECK(cudaMemcpyAsync(&result.accepted_drafts, frame.accepted_drafts.data,
                               sizeof(result.accepted_drafts), cudaMemcpyDeviceToHost,
                               impl_->device_.stream));
    CUDA_CHECK(cudaMemcpyAsync(&result.next_extent, frame.next_extents.data,
                               sizeof(result.next_extent), cudaMemcpyDeviceToHost,
                               impl_->device_.stream));
    CUDA_CHECK(cudaMemcpyAsync(&result.next_anchor, frame.anchors.data,
                               sizeof(result.next_anchor), cudaMemcpyDeviceToHost,
                               impl_->device_.stream));
    CUDA_CHECK(cudaMemcpyAsync(&result.next_frontier, frame.base_frontiers.data,
                               sizeof(result.next_frontier), cudaMemcpyDeviceToHost,
                               impl_->device_.stream));
    impl_->device_.synchronize();

    if (result.licensed_count <= 0 ||
        result.licensed_count > static_cast<std::int32_t>(impl_->verify_width_) ||
        result.accepted_drafts < 0 || result.accepted_drafts >= result.licensed_count ||
        result.next_extent < 0 ||
        result.next_extent > static_cast<std::int32_t>(impl_->verify_width_ - 1U)) {
        throw std::runtime_error("canonical MTP acceptance produced an invalid host egress");
    }
    impl_->workspace_->reset();
    return result;
}

std::array<TokenId, kDualMtpMaximumDrafts>
TextShardStage::mtp_propose_round(const Mtp3SessionState& state,
                                  const Mtp3RoundResult& accepted,
                                  const void* target_hidden_bf16,
                                  std::size_t target_hidden_bytes,
                                  std::uint32_t session) {
    const bool local_target_hidden = target_hidden_bf16 == nullptr;
    if (impl_ == nullptr || !impl_->enable_mtp_ ||
        (local_target_hidden ? target_hidden_bytes != 0
                             : target_hidden_bytes != verify_residual_bytes())) {
        throw std::invalid_argument("MTP3 proposal continuation is outside the qualified domain");
    }
    if (state.draft_extent < 0 ||
        state.draft_extent > static_cast<std::int32_t>(impl_->verify_width_ - 1U) ||
        accepted.licensed_count <= 0 ||
        accepted.licensed_count > static_cast<std::int32_t>(impl_->verify_width_) ||
        accepted.accepted_drafts < 0 || accepted.accepted_drafts >= accepted.licensed_count ||
        accepted.next_extent < 0 ||
        accepted.next_extent > static_cast<std::int32_t>(impl_->verify_width_ - 1U)) {
        throw std::invalid_argument("MTP proposal received invalid extent or acceptance metadata");
    }
    const std::int32_t proposal_extent = accepted.next_extent;
    impl_->activate();
    qwen3_6::RoundState& round = impl_->mtp_round(session);
    if (!round.mtp_decode) { throw std::logic_error("text shard MTP decode tensors are unavailable"); }
    qwen3_6::MtpDecodeState& frame = *round.mtp_decode;
    if (!local_target_hidden) {
        CUDA_CHECK(cudaMemcpyAsync(frame.target_hidden.data, target_hidden_bf16,
                                   target_hidden_bytes, cudaMemcpyHostToDevice,
                                   impl_->device_.stream));
    }
    CUDA_CHECK(cudaMemcpyAsync(frame.accepted_drafts.data, &accepted.accepted_drafts,
                               sizeof(accepted.accepted_drafts), cudaMemcpyHostToDevice,
                               impl_->device_.stream));
    CUDA_CHECK(cudaMemcpyAsync(frame.licensed_counts.data, &accepted.licensed_count,
                               sizeof(accepted.licensed_count), cudaMemcpyHostToDevice,
                               impl_->device_.stream));
    CUDA_CHECK(cudaMemcpyAsync(frame.anchors.data, &accepted.next_anchor,
                               sizeof(accepted.next_anchor), cudaMemcpyHostToDevice,
                               impl_->device_.stream));
    CUDA_CHECK(cudaMemcpyAsync(frame.base_frontiers.data, &accepted.next_frontier,
                               sizeof(accepted.next_frontier), cudaMemcpyHostToDevice,
                               impl_->device_.stream));
    ops::speculative_select_accepted_hidden(frame.target_hidden, frame.accepted_drafts,
                                            frame.target_continuation_hidden,
                                            impl_->device_.stream);
    ops::mtp_prepare_next_round(
        frame.verify_ids, frame.anchors, frame.accepted_drafts, frame.base_frontiers,
        frame.remaining_budgets, frame.licensed_counts, frame.rope_deltas, frame.alignment_ids,
        frame.next_extents, frame.ar_positions, frame.ar_rope_positions, frame.ar_valid_columns,
        static_cast<std::int32_t>(impl_->capacity_), impl_->device_.stream);

    if (accepted.next_extent > 0) {
        StageTextContext& mtp = impl_->mtp_text(session);
        if(std::getenv("NINFER_TARGET_GUIDED_PROPOSAL_ORACLE")!=nullptr)
            mtp.set_mtp_guide_licensed_columns(&frame.licensed_counts);
        const ops::GqaExecutionEnvelope alignment_envelope{
            static_cast<std::uint32_t>(accepted.next_frontier),
            static_cast<std::uint32_t>(accepted.next_frontier)};
        mtp.mtp_forward_decode_batch(
            frame.alignment_ids, frame.target_hidden, frame.target_positions,
            frame.target_rope_positions, frame.licensed_counts, frame.mtp_kv_table_rows,
            alignment_envelope, frame.alignment_hidden);
        ops::speculative_select_accepted_hidden(frame.alignment_hidden, frame.accepted_drafts,
                                                frame.ar_hidden, impl_->device_.stream);
        if (proposal_extent > 0) {
            Tensor first_draft = frame.next_drafts.slice(1, 0, 1).view({1});
            mtp.mtp_propose_batch(frame.ar_hidden, frame.proposal_logits, first_draft);
        }

        for (std::int32_t step = 0; step + 1 < proposal_extent; ++step) {
            Tensor previous = frame.next_drafts.slice(1, step, 1).view({1});
            Tensor position = frame.ar_positions.slice(1, step, 1).view({1});
            Tensor rope_position = frame.ar_rope_positions.slice(1, step, 1).view({1});
            Tensor valid = frame.ar_valid_columns.slice(1, step, 1).view({1});
            const auto visible = static_cast<std::uint32_t>(accepted.next_frontier + step + 1);
            Tensor previous_batch = previous.view({1, 1});
            Tensor hidden_batch = frame.ar_hidden.view({StageTextConfig::hidden, 1, 1});
            Tensor position_batch = position.view({1, 1});
            Tensor rope_batch = rope_position.view({1, 1});
            Tensor next_hidden_batch = frame.next_hidden.view({StageTextConfig::hidden, 1, 1});
            mtp.mtp_forward_decode_batch(
                previous_batch, hidden_batch, position_batch, rope_batch, valid,
                frame.mtp_kv_table_rows, ops::GqaExecutionEnvelope{visible, visible},
                next_hidden_batch);
            Tensor next = frame.next_drafts.slice(1, step + 1, 1).view({1});
            mtp.mtp_propose_batch(frame.next_hidden, frame.proposal_logits, next);
            CUDA_CHECK(cudaMemcpyAsync(frame.ar_hidden.data, frame.next_hidden.data,
                                       frame.ar_hidden.bytes(), cudaMemcpyDeviceToDevice,
                                       impl_->device_.stream));
        }
    }

    std::array<TokenId, kDualMtpMaximumDrafts> drafts{};
    if (proposal_extent > 0) {
        CUDA_CHECK(cudaMemcpyAsync(drafts.data(), frame.next_drafts.data,
                                   static_cast<std::size_t>(proposal_extent) * sizeof(TokenId),
                                   cudaMemcpyDeviceToHost, impl_->device_.stream));
    }
    impl_->device_.synchronize();
    impl_->workspace_->reset();
    return drafts;
}

Mtp3RoundResult TextShardStage::mtp_finish_round(const Mtp3SessionState& state,
                                                 std::uint32_t session) {
    Mtp3RoundResult result = mtp_accept_round(state, session);
    result.next_drafts = mtp_propose_round(state, result, nullptr, 0, session);
    return result;
}

void TextShardStage::copy_residual_to_host_async(void* destination, std::size_t bytes,
                                                     std::uint32_t session) {
    if (impl_ == nullptr || destination == nullptr || bytes != residual_bytes()) {
        throw std::invalid_argument("invalid text shard residual download");
    }
    impl_->activate();
    Tensor residual = impl_->column(impl_->residual_, session);
    CUDA_CHECK(cudaMemcpyAsync(destination, residual.data, bytes, cudaMemcpyDeviceToHost,
                               impl_->device_.stream));
}

void TextShardStage::copy_residual_to_host(void* destination, std::size_t bytes,
                                           std::uint32_t session) {
    copy_residual_to_host_async(destination, bytes, session);
    impl_->device_.synchronize();
}

void TextShardStage::copy_residual_from_host(const void* source, std::size_t bytes,
                                             std::uint32_t session) {
    if (impl_ == nullptr || source == nullptr || bytes != residual_bytes()) {
        throw std::invalid_argument("invalid text shard residual upload");
    }
    impl_->activate();
    Tensor residual = impl_->column(impl_->residual_, session);
    CUDA_CHECK(cudaMemcpyAsync(residual.data, source, bytes, cudaMemcpyHostToDevice,
                               impl_->device_.stream));
}

TokenId TextShardStage::project_token(std::uint32_t session) {
    if (impl_ == nullptr) { throw std::logic_error("text shard is empty"); }
    impl_->activate();
    Tensor residual = impl_->column(impl_->residual_, session);
    Tensor normalized = impl_->column(impl_->normalized_, session);
    Tensor logits = impl_->column(impl_->logits_, session);
    Tensor output_token = impl_->scalar(impl_->output_token_, session);
    impl_->text_->project_shard_head(residual, normalized, logits, output_token);
    TokenId token = 0;
    CUDA_CHECK(cudaMemcpyAsync(&token, output_token.data, sizeof(token),
                               cudaMemcpyDeviceToHost, impl_->device_.stream));
    impl_->device_.synchronize();
    return token;
}

void TextShardStage::target_verify_tokens(std::span<const TokenId> tokens,
                                               std::int32_t position,
                                               std::int32_t rope_position,
                                               std::uint32_t session) {
    if (tokens.size() != kMtp3VerifyWidth) {
        throw std::invalid_argument("fixed target-verifier input must have width 4");
    }
    Mtp3VerifyInput input;
    input.valid_columns = static_cast<std::int32_t>(kMtp3VerifyWidth);
    std::copy(tokens.begin(), tokens.end(), input.ids.begin());
    for (std::size_t column = 0; column < kMtp3VerifyWidth; ++column) {
        input.positions[column] = position + static_cast<std::int32_t>(column);
        input.rope_positions[column] = rope_position + static_cast<std::int32_t>(column);
    }
    target_verify_tokens(input, session);
}

void TextShardStage::target_verify_tokens(const Mtp3VerifyInput& input,
                                          std::uint32_t session) {
    if (impl_ == nullptr || input.valid_columns <= 0 ||
        input.valid_columns > static_cast<std::int32_t>(impl_->verify_width_)) {
        throw std::invalid_argument("invalid text shard target-verifier input");
    }
    const std::size_t last = static_cast<std::size_t>(input.valid_columns - 1);
    if (input.positions[0] < 0 || input.rope_positions[0] < 0 ||
        input.positions[last] < input.positions[0] ||
        static_cast<std::uint64_t>(input.positions[last]) >= impl_->capacity_) {
        throw std::invalid_argument("target-verifier positions are outside capacity");
    }
    impl_->activate();
    impl_->ensure_sm75_a8_decode_layout();
    Tensor ids = impl_->verify_vector(impl_->verify_ids_, session);
    Tensor positions = impl_->verify_vector(impl_->verify_positions_, session);
    Tensor rope_positions = impl_->verify_vector(impl_->verify_rope_positions_, session);
    Tensor valid_columns = impl_->scalar(impl_->verify_valid_columns_, session);
    Tensor kv_row = impl_->scalar(impl_->kv_table_row_, session);
    Tensor state_slot = impl_->scalar(impl_->state_slot_, session);
    Tensor residual = impl_->verify_cube(impl_->verify_residual_, session);
    const std::size_t configured_width = impl_->verify_width_;
    CUDA_CHECK(cudaMemcpyAsync(ids.data, input.ids.data(),
                               configured_width * sizeof(TokenId),
                               cudaMemcpyHostToDevice, impl_->device_.stream));
    CUDA_CHECK(cudaMemcpyAsync(positions.data, input.positions.data(),
                               configured_width * sizeof(std::int32_t),
                               cudaMemcpyHostToDevice, impl_->device_.stream));
    CUDA_CHECK(cudaMemcpyAsync(rope_positions.data, input.rope_positions.data(),
                               configured_width * sizeof(std::int32_t),
                               cudaMemcpyHostToDevice, impl_->device_.stream));
    CUDA_CHECK(cudaMemcpyAsync(valid_columns.data, &input.valid_columns,
                               sizeof(input.valid_columns), cudaMemcpyHostToDevice,
                               impl_->device_.stream));
    const std::uint32_t visible = static_cast<std::uint32_t>(input.positions[last]) + 1U;
    const auto body = [&](ops::GqaExecutionEnvelope envelope) {
        impl_->text_->target_verify_token_shard(
            ids, positions, rope_positions, valid_columns, kv_row, state_slot,
            impl_->verify_records(session), envelope, residual);
    };
    if (impl_->verify_token_graphs_.empty()) {
        body(ops::GqaExecutionEnvelope{visible, visible});
        return;
    }
    std::uint8_t& phase = impl_->verify_token_graph_phases_.at(session);
    if (phase == 0) {
        body(ops::GqaExecutionEnvelope{visible, visible});
        phase = 1;
        return;
    }
    DecodeGraphExecutable& executable = impl_->verify_token_graphs_.at(session);
    if (phase == 1) {
        impl_->device_.synchronize();
        DecodeGraphDefinition& definition = impl_->verify_token_graph_definitions_.at(session);
        definition.capture(impl_->device_.stream, [&]() {
            body(ops::GqaExecutionEnvelope{1U, impl_->capacity_});
        });
        executable.instantiate(definition);
        executable.upload(impl_->device_.stream);
        phase = 2;
    }
    executable.launch(impl_->device_.stream);
}

void TextShardStage::target_verify_tokens_batch(std::span<const Mtp3VerifyInput> inputs,
                                                   std::uint32_t session_begin) {
    if (impl_ == nullptr) { throw std::logic_error("text shard is empty"); }
    impl_->validate_verify_pair(session_begin, inputs.size());
    impl_->activate();
    impl_->ensure_sm75_a8_decode_layout();
    constexpr std::size_t batch = 2;
    std::vector<TokenId> ids(batch * kMtp3VerifyWidth);
    std::vector<std::int32_t> positions(batch * kMtp3VerifyWidth);
    std::vector<std::int32_t> rope_positions(batch * kMtp3VerifyWidth);
    std::array<std::int32_t, batch> valid_columns{};
    std::uint32_t maximum_visible = 0;
    for (std::size_t row = 0; row < batch; ++row) {
        if (inputs[row].valid_columns < 0 ||
            inputs[row].valid_columns > static_cast<std::int32_t>(kMtp3VerifyWidth)) {
            throw std::invalid_argument("target-verifier pair valid-column extent is invalid");
        }
        valid_columns[row] = inputs[row].valid_columns;
        for (std::size_t column = 0; column < kMtp3VerifyWidth; ++column) {
            const std::size_t index = row * kMtp3VerifyWidth + column;
            ids[index] = inputs[row].ids[column];
            positions[index] = inputs[row].positions[column];
            rope_positions[index] = inputs[row].rope_positions[column];
            maximum_visible = std::max(maximum_visible,
                                       static_cast<std::uint32_t>(positions[index] + 1));
        }
    }
    Tensor id_view = impl_->verify_vector_pair(impl_->verify_ids_, session_begin);
    Tensor position_view = impl_->verify_vector_pair(impl_->verify_positions_, session_begin);
    Tensor rope_view = impl_->verify_vector_pair(impl_->verify_rope_positions_, session_begin);
    Tensor valid_view = impl_->scalar_pair(impl_->verify_valid_columns_, session_begin);
    Tensor kv_view = impl_->scalar_pair(impl_->kv_table_row_, session_begin);
    Tensor state_view = impl_->scalar_pair(impl_->state_slot_, session_begin);
    Tensor residual_view = impl_->verify_cube_pair(impl_->verify_residual_, session_begin);
    CUDA_CHECK(cudaMemcpyAsync(id_view.data, ids.data(), ids.size() * sizeof(TokenId),
                               cudaMemcpyHostToDevice, impl_->device_.stream));
    CUDA_CHECK(cudaMemcpyAsync(position_view.data, positions.data(),
                               positions.size() * sizeof(std::int32_t), cudaMemcpyHostToDevice,
                               impl_->device_.stream));
    CUDA_CHECK(cudaMemcpyAsync(rope_view.data, rope_positions.data(),
                               rope_positions.size() * sizeof(std::int32_t),
                               cudaMemcpyHostToDevice, impl_->device_.stream));
    CUDA_CHECK(cudaMemcpyAsync(valid_view.data, valid_columns.data(),
                               valid_columns.size() * sizeof(std::int32_t),
                               cudaMemcpyHostToDevice, impl_->device_.stream));
    impl_->text_->target_verify_token_shard(
        id_view, position_view, rope_view, valid_view, kv_view, state_view,
        impl_->batch_verify_records(session_begin / 2U),
        ops::GqaExecutionEnvelope{1, maximum_visible}, residual_view);
}

void TextShardStage::copy_verify_residual_batch_to_host(void* destination, std::size_t bytes,
                                                             std::uint32_t session_begin) {
    constexpr std::size_t batch = 2;
    const std::size_t expected = verify_residual_bytes() * batch;
    if (impl_ == nullptr || destination == nullptr || bytes != expected) {
        throw std::invalid_argument("invalid target-verifier pair residual download");
    }
    impl_->activate();
    Tensor residual = impl_->verify_cube_pair(impl_->verify_residual_, session_begin);
    CUDA_CHECK(cudaMemcpyAsync(destination, residual.data, bytes, cudaMemcpyDeviceToHost,
                               impl_->device_.stream));
    impl_->device_.synchronize();
}

void TextShardStage::copy_verify_residual_batch_from_host(const void* source, std::size_t bytes,
                                                           std::uint32_t session_begin) {
    constexpr std::size_t batch = 2;
    const std::size_t expected = verify_residual_bytes() * batch;
    if (impl_ == nullptr || source == nullptr || bytes != expected) {
        throw std::invalid_argument("invalid target-verifier pair residual upload");
    }
    impl_->activate();
    Tensor residual = impl_->verify_cube_pair(impl_->verify_residual_, session_begin);
    CUDA_CHECK(cudaMemcpyAsync(residual.data, source, bytes, cudaMemcpyHostToDevice,
                               impl_->device_.stream));
}

std::vector<std::array<TokenId, 4>>
TextShardStage::target_verify_hidden_batch(std::span<const Mtp3VerifyInput> inputs,
                                           std::uint32_t session_begin) {
    if (impl_ == nullptr || !impl_->shard_.output_head) {
        throw std::logic_error("text shard does not own target-verifier output");
    }
    impl_->validate_verify_pair(session_begin, inputs.size());
    impl_->activate();
    impl_->ensure_sm75_a8_decode_layout();
    constexpr std::size_t batch = 2;
    std::vector<std::int32_t> positions(batch * kMtp3VerifyWidth);
    std::vector<std::int32_t> rope_positions(batch * kMtp3VerifyWidth);
    std::array<std::int32_t, batch> valid_columns{};
    std::uint32_t maximum_visible = 0;
    for (std::size_t row = 0; row < batch; ++row) {
        if (inputs[row].valid_columns < 0 ||
            inputs[row].valid_columns > static_cast<std::int32_t>(kMtp3VerifyWidth)) {
            throw std::invalid_argument("target-verifier pair valid-column extent is invalid");
        }
        valid_columns[row] = inputs[row].valid_columns;
        for (std::size_t column = 0; column < kMtp3VerifyWidth; ++column) {
            const std::size_t index = row * kMtp3VerifyWidth + column;
            positions[index] = inputs[row].positions[column];
            rope_positions[index] = inputs[row].rope_positions[column];
            maximum_visible = std::max(maximum_visible,
                                       static_cast<std::uint32_t>(positions[index] + 1));
        }
    }
    Tensor position_view = impl_->verify_vector_pair(impl_->verify_positions_, session_begin);
    Tensor rope_view = impl_->verify_vector_pair(impl_->verify_rope_positions_, session_begin);
    Tensor valid_view = impl_->scalar_pair(impl_->verify_valid_columns_, session_begin);
    Tensor kv_view = impl_->scalar_pair(impl_->kv_table_row_, session_begin);
    Tensor state_view = impl_->scalar_pair(impl_->state_slot_, session_begin);
    Tensor residual_view = impl_->verify_cube_pair(impl_->verify_residual_, session_begin);
    Tensor hidden_view = impl_->verify_cube_pair(impl_->verify_hidden_, session_begin);
    Tensor logits_view = impl_->verify_cube_pair(impl_->verify_logits_, session_begin);
    Tensor token_view = impl_->verify_vector_pair(impl_->verify_tokens_, session_begin);
    CUDA_CHECK(cudaMemcpyAsync(position_view.data, positions.data(),
                               positions.size() * sizeof(std::int32_t), cudaMemcpyHostToDevice,
                               impl_->device_.stream));
    CUDA_CHECK(cudaMemcpyAsync(rope_view.data, rope_positions.data(),
                               rope_positions.size() * sizeof(std::int32_t),
                               cudaMemcpyHostToDevice, impl_->device_.stream));
    CUDA_CHECK(cudaMemcpyAsync(valid_view.data, valid_columns.data(),
                               valid_columns.size() * sizeof(std::int32_t),
                               cudaMemcpyHostToDevice, impl_->device_.stream));
    impl_->text_->target_verify_hidden_shard(
        residual_view, position_view, rope_view, valid_view, kv_view, state_view,
        impl_->batch_verify_records(session_begin / 2U),
        ops::GqaExecutionEnvelope{1, maximum_visible}, hidden_view, logits_view, token_view);
    std::vector<TokenId> flat(batch * kMtp3VerifyWidth);
    CUDA_CHECK(cudaMemcpyAsync(flat.data(), token_view.data,
                               flat.size() * sizeof(TokenId), cudaMemcpyDeviceToHost,
                               impl_->device_.stream));
    impl_->device_.synchronize();
    std::vector<std::array<TokenId, 4>> output(batch);
    for (std::size_t row = 0; row < batch; ++row) {
        for (std::size_t column = 0; column < kMtp3VerifyWidth; ++column) {
            output[row][column] = flat[row * kMtp3VerifyWidth + column];
        }
    }
    return output;
}

void TextShardStage::fold_verify_records_batch(
    std::span<const std::int32_t> commit_columns, std::uint32_t session_begin) {
    if (impl_ == nullptr) { throw std::logic_error("text shard is empty"); }
    impl_->validate_verify_pair(session_begin, commit_columns.size());
    impl_->activate();
    std::array<ops::GdnReplayFoldRow, 2> rows{};
    for (std::size_t row = 0; row < rows.size(); ++row) {
        if (commit_columns[row] < 0 ||
            commit_columns[row] > static_cast<std::int32_t>(impl_->verify_width_)) {
            throw std::invalid_argument("target-verifier pair commit extent is invalid");
        }
        const std::uint32_t session = session_begin + static_cast<std::uint32_t>(row);
        rows[row] = {
            .linear_state_slot = static_cast<std::int32_t>(2U * session),
            .commit_columns = commit_columns[row],
        };
    }
    ops::gdn_replay_fold(impl_->batch_verify_records(session_begin / 2U),
                         impl_->decoder_->linear_attention.all_layers_view(), rows,
                         impl_->device_.stream);
}

void TextShardStage::copy_verify_residual_to_host(void* destination, std::size_t bytes,
                                                   std::uint32_t session) {
    if (impl_ == nullptr || destination == nullptr || bytes != verify_residual_bytes()) {
        throw std::invalid_argument("invalid target-verifier residual download");
    }
    impl_->activate();
    Tensor residual = impl_->verify_cube(impl_->verify_residual_, session);
    CUDA_CHECK(cudaMemcpyAsync(destination, residual.data, bytes, cudaMemcpyDeviceToHost,
                               impl_->device_.stream));
    impl_->device_.synchronize();
}

void TextShardStage::copy_verify_residual_from_host(const void* source, std::size_t bytes,
                                                     std::uint32_t session) {
    if (impl_ == nullptr || source == nullptr || bytes != verify_residual_bytes()) {
        throw std::invalid_argument("invalid target-verifier residual upload");
    }
    impl_->activate();
    Tensor residual = impl_->verify_cube(impl_->verify_residual_, session);
    CUDA_CHECK(cudaMemcpyAsync(residual.data, source, bytes, cudaMemcpyHostToDevice,
                               impl_->device_.stream));
}

std::vector<TokenId> TextShardStage::target_verify_hidden(std::int32_t position,
                                                          std::int32_t rope_position,
                                                          std::uint32_t session) {
    Mtp3VerifyInput input;
    input.valid_columns = static_cast<std::int32_t>(kMtp3VerifyWidth);
    for (std::size_t column = 0; column < kMtp3VerifyWidth; ++column) {
        input.positions[column] = position + static_cast<std::int32_t>(column);
        input.rope_positions[column] = rope_position + static_cast<std::int32_t>(column);
    }
    return target_verify_hidden(input, session);
}

std::vector<TokenId> TextShardStage::target_verify_hidden(const Mtp3VerifyInput& input,
                                                          std::uint32_t session) {
    launch_target_verify_hidden(input, session);
    return collect_verify_tokens(session);
}

void TextShardStage::launch_target_verify_hidden(std::int32_t position,
                                                  std::int32_t rope_position,
                                                  std::uint32_t session) {
    Mtp3VerifyInput input;
    input.valid_columns = static_cast<std::int32_t>(kMtp3VerifyWidth);
    for (std::size_t column = 0; column < kMtp3VerifyWidth; ++column) {
        input.positions[column] = position + static_cast<std::int32_t>(column);
        input.rope_positions[column] = rope_position + static_cast<std::int32_t>(column);
    }
    launch_target_verify_hidden(input, session);
}

void TextShardStage::launch_target_verify_hidden(const Mtp3VerifyInput& input,
                                                  std::uint32_t session) {
    if (impl_ == nullptr || !impl_->shard_.output_head || input.valid_columns <= 0 ||
        input.valid_columns > static_cast<std::int32_t>(impl_->verify_width_)) {
        throw std::invalid_argument("invalid target-verifier output-stage input");
    }
    const std::size_t last = static_cast<std::size_t>(input.valid_columns - 1);
    if (input.positions[0] < 0 || input.rope_positions[0] < 0 ||
        input.positions[last] < input.positions[0] ||
        static_cast<std::uint64_t>(input.positions[last]) >= impl_->capacity_) {
        throw std::invalid_argument("target-verifier output positions are outside capacity");
    }
    impl_->activate();
    impl_->ensure_sm75_a8_decode_layout();
    Tensor positions = impl_->verify_vector(impl_->verify_positions_, session);
    Tensor rope_positions = impl_->verify_vector(impl_->verify_rope_positions_, session);
    Tensor valid_columns = impl_->scalar(impl_->verify_valid_columns_, session);
    Tensor kv_row = impl_->scalar(impl_->kv_table_row_, session);
    Tensor state_slot = impl_->scalar(impl_->state_slot_, session);
    Tensor residual = impl_->verify_cube(impl_->verify_residual_, session);
    Tensor hidden = impl_->verify_cube(impl_->verify_hidden_, session);
    Tensor logits = impl_->verify_cube(impl_->verify_logits_, session);
    Tensor target_tokens = impl_->verify_vector(impl_->verify_tokens_, session);
    const std::size_t configured_width = impl_->verify_width_;
    CUDA_CHECK(cudaMemcpyAsync(positions.data, input.positions.data(),
                               configured_width * sizeof(std::int32_t),
                               cudaMemcpyHostToDevice, impl_->device_.stream));
    CUDA_CHECK(cudaMemcpyAsync(rope_positions.data, input.rope_positions.data(),
                               configured_width * sizeof(std::int32_t),
                               cudaMemcpyHostToDevice, impl_->device_.stream));
    CUDA_CHECK(cudaMemcpyAsync(valid_columns.data, &input.valid_columns,
                               sizeof(input.valid_columns), cudaMemcpyHostToDevice,
                               impl_->device_.stream));
    const std::uint32_t visible = static_cast<std::uint32_t>(input.positions[last]) + 1U;
    const bool licensed_policy = qwen3_6::license_retained_d4(impl_->computation_,configured_width,4);
    const bool licensed_full_round =
        licensed_policy && input.valid_columns == static_cast<std::int32_t>(configured_width);
    const auto body = [&](ops::GqaExecutionEnvelope envelope, bool force) {
        impl_->verify_logits_materialized_.at(session) = false;
        impl_->verify_logits_materialized_.at(session) = impl_->text_->target_verify_hidden_shard(
            residual, positions, rope_positions, valid_columns, kv_row, state_slot,
            impl_->verify_records(session), envelope, hidden, logits, target_tokens, force,
            !impl_->greedy_head_tokens_only_);
    };
    if (licensed_policy && !licensed_full_round) {
        body(ops::GqaExecutionEnvelope{visible, visible}, false);
        return;
    }
    if (impl_->verify_hidden_graphs_.empty()) {
        body(ops::GqaExecutionEnvelope{visible, visible}, licensed_full_round);
        return;
    }
    std::uint8_t& phase = impl_->verify_hidden_graph_phases_.at(session);
    if (phase == 0) {
        body(ops::GqaExecutionEnvelope{visible, visible}, licensed_full_round);
        phase = 1;
        return;
    }
    DecodeGraphExecutable& executable = impl_->verify_hidden_graphs_.at(session);
    if (phase == 1) {
        impl_->device_.synchronize();
        DecodeGraphDefinition& definition = impl_->verify_hidden_graph_definitions_.at(session);
        definition.capture(impl_->device_.stream, [&]() {
            body(ops::GqaExecutionEnvelope{1U, impl_->capacity_}, licensed_full_round);
        });
        executable.instantiate(definition);
        executable.upload(impl_->device_.stream);
        phase = 2;
    }
    executable.launch(impl_->device_.stream);
}

std::vector<TokenId> TextShardStage::collect_verify_tokens(std::uint32_t session) {
    if (impl_ == nullptr || !impl_->shard_.output_head) {
        throw std::logic_error("text shard does not own target-verifier output");
    }
    impl_->activate();
    Tensor target_tokens = impl_->verify_vector(impl_->verify_tokens_, session);
    std::vector<TokenId> tokens(impl_->verify_width_);
    CUDA_CHECK(cudaMemcpyAsync(tokens.data(), target_tokens.data,
                               tokens.size() * sizeof(TokenId), cudaMemcpyDeviceToHost,
                               impl_->device_.stream));
    impl_->device_.synchronize();
    return tokens;
}

std::vector<TokenId> TextShardStage::project_verify_tokens(std::uint32_t session) {
    if (impl_ == nullptr || !impl_->shard_.output_head) {
        throw std::logic_error("text shard does not own target-verifier output");
    }
    impl_->activate();
    Tensor residual = impl_->verify_cube(impl_->verify_residual_, session)
                          .view({StageTextConfig::hidden,
                                 static_cast<std::int32_t>(impl_->verify_width_)});
    Tensor hidden = impl_->verify_cube(impl_->verify_hidden_, session)
                        .view({StageTextConfig::hidden,
                               static_cast<std::int32_t>(impl_->verify_width_)});
    Tensor logits = impl_->verify_cube(impl_->verify_logits_, session)
                        .view({StageTextConfig::output_rows,
                               static_cast<std::int32_t>(impl_->verify_width_)});
    Tensor target_tokens = impl_->verify_vector(impl_->verify_tokens_, session)
                               .view({static_cast<std::int32_t>(impl_->verify_width_)});
    impl_->text_->project_shard_head(residual, hidden, logits, target_tokens);
    impl_->verify_logits_materialized_.at(session) = true;
    std::vector<TokenId> tokens(impl_->verify_width_);
    CUDA_CHECK(cudaMemcpyAsync(tokens.data(), target_tokens.data,
                               tokens.size() * sizeof(TokenId), cudaMemcpyDeviceToHost,
                               impl_->device_.stream));
    impl_->device_.synchronize();
    return tokens;
}

void TextShardStage::copy_verify_hidden_to_host_async(void* destination, std::size_t bytes,
                                                       std::uint32_t session) {
    if (impl_ == nullptr || !impl_->shard_.output_head || destination == nullptr ||
        bytes != verify_residual_bytes()) {
        throw std::invalid_argument("invalid target-verifier hidden download");
    }
    impl_->activate();
    Tensor hidden = impl_->verify_cube(impl_->verify_hidden_, session);
    CUDA_CHECK(cudaMemcpyAsync(destination, hidden.data, bytes, cudaMemcpyDeviceToHost,
                               impl_->device_.stream));
}

void TextShardStage::copy_verify_hidden_to_host(void* destination, std::size_t bytes,
                                                 std::uint32_t session) {
    copy_verify_hidden_to_host_async(destination, bytes, session);
    impl_->device_.synchronize();
}

void TextShardStage::copy_verify_logits_to_host(void* destination, std::size_t bytes,
                                                 std::uint32_t session) {
    if (impl_ == nullptr || !impl_->shard_.output_head || destination == nullptr ||
        bytes != verify_logits_bytes()) {
        throw std::invalid_argument("invalid target-verifier logits download");
    }
    impl_->activate();
    if (!impl_->verify_logits_materialized_.at(session)) {
        // A tokens-only verification deliberately left no valid logit buffer.
        // Reproject the current stored residual on demand; never export the
        // previous round's logits. This also works after detaching the helper.
        (void)project_verify_tokens(session);
    }

    Tensor logits = impl_->verify_cube(impl_->verify_logits_, session);
    CUDA_CHECK(cudaMemcpyAsync(destination, logits.data, bytes, cudaMemcpyDeviceToHost,
                               impl_->device_.stream));
    impl_->device_.synchronize();
}

std::vector<std::uint64_t>
TextShardStage::verify_record_layer_digests(std::uint32_t session) {
    if (impl_ == nullptr) { throw std::logic_error("text shard is empty"); }
    impl_->activate();
    GdnReplayRecords& records = impl_->verify_records(session);
    std::vector<std::uint64_t> digests;
    digests.reserve(static_cast<std::size_t>(records.spec.layers));
    const auto update = [](std::uint64_t hash, const unsigned char* data,
                           std::size_t size) {
        constexpr std::uint64_t prime = 1099511628211ULL;
        for (std::size_t i = 0; i < size; ++i) {
            hash ^= data[i];
            hash *= prime;
        }
        return hash;
    };
    for (std::int32_t layer_index = 0; layer_index < records.spec.layers; ++layer_index) {
        const GdnReplayRecordLayer layer = records.layer(layer_index, 1);
        const Tensor planes[] = {layer.conv, layer.key, layer.value, layer.gate};
        std::uint64_t hash = 1469598103934665603ULL;
        for (const Tensor& plane : planes) {
            std::vector<unsigned char> host(plane.bytes());
            CUDA_CHECK(cudaMemcpyAsync(host.data(), plane.data, host.size(), cudaMemcpyDeviceToHost,
                                       impl_->device_.stream));
            impl_->device_.synchronize();
            hash = update(hash, host.data(), host.size());
        }
        digests.push_back(hash);
    }
    return digests;
}

void TextShardStage::fold_verify_records(std::int32_t commit_columns,
                                         std::uint32_t session) {
    if (impl_ == nullptr || commit_columns < 0 ||
        commit_columns > static_cast<std::int32_t>(impl_->verify_width_)) {
        throw std::invalid_argument("invalid target-verifier commit extent");
    }
    impl_->activate();
    impl_->validate_session(session);
    const std::array<ops::GdnReplayFoldRow, 1> rows{{
        {.linear_state_slot = static_cast<std::int32_t>(2U * session),
         .commit_columns = commit_columns},
    }};
    ops::gdn_replay_fold(impl_->verify_records(session),
                         impl_->decoder_->linear_attention.all_layers_view(), rows,
                         impl_->device_.stream);
}

void TextShardStage::synchronize() {
    if (impl_ == nullptr) { throw std::logic_error("text shard is empty"); }
    impl_->activate();
    impl_->device_.synchronize();
}

std::size_t TextShardStage::prefix_snapshot_bytes() const noexcept {
    if (impl_ == nullptr || impl_->state_storage_ == nullptr || impl_->io_storage_ == nullptr) {
        return 0;
    }
    return impl_->state_bytes_ + impl_->io_storage_->used();
}

void TextShardStage::copy_prefix_snapshot_to_host(void* destination, std::size_t bytes) {
    const std::size_t expected = prefix_snapshot_bytes();
    if (impl_ == nullptr || destination == nullptr || bytes != expected) {
        throw std::invalid_argument("invalid text-shard prefix snapshot download");
    }
    impl_->activate();
    impl_->device_.synchronize();
    auto* host = static_cast<std::uint8_t*>(destination);
    const std::size_t state_bytes = impl_->state_bytes_;
    CUDA_CHECK(cudaMemcpy(host, impl_->state_storage_->base(), state_bytes,
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(host + state_bytes, impl_->io_storage_->base(),
                          impl_->io_storage_->used(), cudaMemcpyDeviceToHost));
}

void TextShardStage::copy_prefix_snapshot_from_host(const void* source, std::size_t bytes) {
    const std::size_t expected = prefix_snapshot_bytes();
    if (impl_ == nullptr || source == nullptr || bytes != expected) {
        throw std::invalid_argument("invalid text-shard prefix snapshot upload");
    }
    impl_->activate();
    impl_->device_.synchronize();
    const auto* host = static_cast<const std::uint8_t*>(source);
    const std::size_t state_bytes = impl_->state_bytes_;
    CUDA_CHECK(cudaMemcpy(impl_->state_storage_->base(), host, state_bytes,
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(impl_->io_storage_->base(), host + state_bytes,
                          impl_->io_storage_->used(), cudaMemcpyHostToDevice));
    impl_->device_.synchronize();
}

std::size_t TextShardStage::legacy_full_prefix_snapshot_bytes() const noexcept {
    if (impl_ == nullptr || impl_->io_storage_ == nullptr) { return 0; }
    return impl_->snapshot_source_state_plan_.bytes +
           impl_->snapshot_source_io_prefix_bytes_;
}

std::size_t TextShardStage::legacy_text_kv_range_bytes(
    std::uint32_t logical_page_begin, std::uint32_t logical_page_count) const {
    if (impl_ == nullptr || logical_page_count == 0 ||
        logical_page_begin > impl_->snapshot_source_state_plan_.logical_pages ||
        logical_page_count >
            impl_->snapshot_source_state_plan_.logical_pages - logical_page_begin) {
        throw std::invalid_argument("legacy text KV range is outside the logical domain");
    }
    std::size_t bytes = 0;
    for (const PagedKVPlaneLayout& plane :
         impl_->snapshot_source_state_plan_.decoder.text_kv.pool.planes) {
        const std::size_t plane_bytes = plane.storage.region.bytes;
        if (plane_bytes % impl_->snapshot_source_state_plan_.logical_pages != 0) {
            throw std::logic_error("legacy text KV plane is not page-contiguous");
        }
        const std::size_t page_bytes =
            plane_bytes / impl_->snapshot_source_state_plan_.logical_pages;
        if (page_bytes > (std::numeric_limits<std::size_t>::max() - bytes) /
                             logical_page_count) {
            throw std::overflow_error("legacy text KV range byte count overflows");
        }
        bytes += page_bytes * logical_page_count;
    }
    return bytes;
}

std::vector<std::uint8_t> TextShardStage::pack_legacy_text_kv_range(
    const void* source, std::size_t bytes, std::uint32_t logical_page_begin,
    std::uint32_t logical_page_count) const {
    const std::size_t expected = legacy_full_prefix_snapshot_bytes();
    if (impl_ == nullptr || source == nullptr || bytes < expected) {
        throw std::invalid_argument("invalid legacy text-shard prefix snapshot");
    }
    std::vector<std::uint8_t> packed(
        legacy_text_kv_range_bytes(logical_page_begin, logical_page_count));
    const auto* host = static_cast<const std::uint8_t*>(source);
    std::size_t cursor = 0;
    for (const PagedKVPlaneLayout& plane :
         impl_->snapshot_source_state_plan_.decoder.text_kv.pool.planes) {
        const std::size_t page_bytes =
            plane.storage.region.bytes / impl_->snapshot_source_state_plan_.logical_pages;
        const std::size_t range_bytes = page_bytes * logical_page_count;
        const std::size_t source_offset = plane.storage.region.offset +
            static_cast<std::size_t>(logical_page_begin) * page_bytes;
        std::memcpy(packed.data() + cursor, host + source_offset, range_bytes);
        cursor += range_bytes;
    }
    return packed;
}

void TextShardStage::copy_legacy_full_prefix_snapshot_from_host(
    const void* source, std::size_t bytes) {
    const std::size_t expected = legacy_full_prefix_snapshot_bytes();
    if (impl_ == nullptr || source == nullptr || bytes < expected ||
        impl_->state_storage_ == nullptr || impl_->io_storage_ == nullptr) {
        throw std::invalid_argument("invalid legacy text-shard prefix snapshot upload");
    }
    impl_->activate();
    impl_->device_.synchronize();
    const auto* host = static_cast<const std::uint8_t*>(source);
    auto* device_base = static_cast<std::uint8_t*>(impl_->state_storage_->base());
    const PlannedStageState& source_plan = impl_->snapshot_source_state_plan_;
    const PlannedStageState& target_plan = impl_->state_plan_;

    const auto copy_region = [&](const LayoutRegion& source_region,
                                 const LayoutRegion& target_region,
                                 std::string_view label) {
        if (source_region.bytes != target_region.bytes) {
            throw std::logic_error(std::string(label) + " snapshot region size differs");
        }
        CUDA_CHECK(cudaMemcpyAsync(device_base + target_region.offset,
                                   host + source_region.offset,
                                   source_region.bytes, cudaMemcpyHostToDevice,
                                   impl_->device_.stream));
    };
    const auto copy_tensor_region = [&](const TensorRegion& source_region,
                                        const TensorRegion& target_region,
                                        std::string_view label) {
        if (source_region.dtype != target_region.dtype ||
            source_region.shape != target_region.shape) {
            throw std::logic_error(std::string(label) + " snapshot tensor shape differs");
        }
        copy_region(source_region.region, target_region.region, label);
    };

    // Text KV is copied by owned logical page range. Its block table remains the freshly bound
    // range-aware table from PagedKVAllocation::bind_row_range.
    if (impl_->text_kv_ownership_.enabled) {
        if (impl_->session_count_ != 1 || impl_->kv_allocations_.size() != 1) {
            throw std::logic_error("legacy sharded snapshot import requires one text session");
        }
        const std::span<const std::int32_t> page_ids =
            impl_->kv_allocations_.front().page_ids();
        if (page_ids.size() != impl_->text_kv_ownership_.logical_page_count) {
            throw std::logic_error("sharded text KV physical page inventory is incomplete");
        }
        for (std::size_t index = 0; index < page_ids.size(); ++index) {
            if (page_ids[index] != static_cast<std::int32_t>(index)) {
                throw std::logic_error("legacy snapshot import requires contiguous physical pages");
            }
        }
        const auto& source_planes = source_plan.decoder.text_kv.pool.planes;
        if (source_planes.size() != impl_->decoder_->text_kv.pool().plane_count()) {
            throw std::logic_error("legacy/current text KV plane inventory differs");
        }
        for (std::size_t index = 0; index < source_planes.size(); ++index) {
            const PagedKVPlaneLayout& source_plane = source_planes[index];
            const Tensor& target_plane = impl_->decoder_->text_kv.pool().plane(index);
            const std::size_t page_bytes =
                source_plane.storage.region.bytes / source_plan.logical_pages;
            const std::size_t range_bytes =
                page_bytes * impl_->text_kv_ownership_.logical_page_count;
            if (target_plane.bytes() != range_bytes) {
                throw std::logic_error("legacy/current text KV plane size differs");
            }
            const std::size_t source_offset = source_plane.storage.region.offset +
                static_cast<std::size_t>(impl_->text_kv_ownership_.logical_page_begin) *
                    page_bytes;
            CUDA_CHECK(cudaMemcpyAsync(target_plane.data, host + source_offset,
                                       range_bytes, cudaMemcpyHostToDevice,
                                       impl_->device_.stream));
        }
    } else {
        const auto& source_planes = source_plan.decoder.text_kv.pool.planes;
        const auto& target_planes = target_plan.decoder.text_kv.pool.planes;
        if (source_planes.size() != target_planes.size()) {
            throw std::logic_error("legacy/current full text KV plane inventory differs");
        }
        for (std::size_t index = 0; index < source_planes.size(); ++index) {
            copy_tensor_region(source_planes[index].storage, target_planes[index].storage,
                               "text KV plane");
        }
    }

    // The proposal-model KV cache is independent of text sequence sharding and remains full.
    if (source_plan.decoder.mtp_kv.has_value() !=
        target_plan.decoder.mtp_kv.has_value()) {
        throw std::logic_error("legacy/current MTP KV presence differs");
    }
    if (source_plan.decoder.mtp_kv && target_plan.decoder.mtp_kv) {
        const auto& source_mtp = *source_plan.decoder.mtp_kv;
        const auto& target_mtp = *target_plan.decoder.mtp_kv;
        if (source_mtp.pool.planes.size() != target_mtp.pool.planes.size()) {
            throw std::logic_error("legacy/current MTP KV plane inventory differs");
        }
        for (std::size_t index = 0; index < source_mtp.pool.planes.size(); ++index) {
            copy_tensor_region(source_mtp.pool.planes[index].storage,
                               target_mtp.pool.planes[index].storage,
                               "MTP KV plane");
        }
        copy_tensor_region(source_mtp.pool.block_tables, target_mtp.pool.block_tables,
                           "MTP KV block table");
    }

    // Mutable recurrent/conv state is width-independent. Round scratch and replay records are
    // deliberately left zeroed; every verifier round reconstructs them from host ingress.
    if (source_plan.decoder.linear_attention.conv.size() !=
            target_plan.decoder.linear_attention.conv.size() ||
        source_plan.decoder.linear_attention.recurrent.size() !=
            target_plan.decoder.linear_attention.recurrent.size()) {
        throw std::logic_error("legacy/current linear-attention state inventory differs");
    }
    for (std::size_t index = 0;
         index < source_plan.decoder.linear_attention.conv.size(); ++index) {
        copy_region(source_plan.decoder.linear_attention.conv[index],
                    target_plan.decoder.linear_attention.conv[index],
                    "linear-attention conv state");
    }
    for (std::size_t index = 0;
         index < source_plan.decoder.linear_attention.recurrent.size(); ++index) {
        copy_region(source_plan.decoder.linear_attention.recurrent[index],
                    target_plan.decoder.linear_attention.recurrent[index],
                    "linear-attention recurrent state");
    }

    CUDA_CHECK(cudaMemcpyAsync(
        impl_->io_storage_->base(), host + source_plan.bytes,
        impl_->snapshot_source_io_prefix_bytes_, cudaMemcpyHostToDevice,
        impl_->device_.stream));
    impl_->device_.synchronize();
}

TextKvPageOwnership TextShardStage::text_kv_ownership() const noexcept {
    return impl_ == nullptr ? TextKvPageOwnership{} : impl_->text_kv_ownership_;
}

int TextShardStage::device_id() const noexcept { return impl_ == nullptr ? -1 : impl_->device_id_; }

std::uint32_t TextShardStage::capacity() const noexcept {
    return impl_ == nullptr ? 0 : impl_->capacity_;
}

std::size_t TextShardStage::weights_bytes() const noexcept {
    return impl_ == nullptr ? 0 : impl_->weights_bytes_;
}

std::size_t TextShardStage::state_bytes() const noexcept {
    return impl_ == nullptr ? 0 : impl_->state_bytes_;
}

std::size_t TextShardStage::workspace_bytes() const noexcept {
    return impl_ == nullptr ? 0 : impl_->workspace_bytes_;
}

std::size_t TextShardStage::kv_payload_bytes() const noexcept {
    return impl_ == nullptr ? 0 : impl_->kv_payload_bytes_;
}

KvCacheStorage TextShardStage::kv_cache() const noexcept {
    return impl_ == nullptr ? KvCacheStorage::BFloat16 : impl_->kv_cache_;
}

std::uint32_t TextShardStage::session_count() const noexcept {
    return impl_ == nullptr ? 0 : impl_->session_count_;
}

std::uint32_t TextShardStage::verify_width() const noexcept {
    return impl_ == nullptr ? 0 : impl_->verify_width_;
}

std::uint32_t TextShardStage::prefill_chunk() const noexcept {
    return impl_ == nullptr ? 0 : impl_->prefill_chunk_;
}

} // namespace ninfer::targets::qwen3_6_27b::detail
