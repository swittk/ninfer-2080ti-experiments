#include "targets/qwen3_6_27b/impl/dual_stage.h"
#include "targets/qwen3_6_27b/impl/variant.h"
#include <ninfer/targets/qwen3_6_27b/package.h>
#include <ninfer/targets/qwen3_6/frontend.h>
#include <ninfer/targets/qwen3_6/prepared_prompt.h>
#include <ninfer/targets/qwen3_6/decoder_state.h>
#include "core/arena.h"
#include "core/device.h"
#include "core/layout.h"
#include "ninfer/ops/position.h"

#define NINFER_QWEN36_VARIANT ::ninfer::targets::qwen3_6_27b::detail::Variant
#define NINFER_QWEN36_RUNTIME_NS qwen3_6_27b_runtime
#include "targets/qwen3_6/impl/runtime/text_context.h"
#undef NINFER_QWEN36_RUNTIME_NS
#undef NINFER_QWEN36_VARIANT

#include <limits>
#include <cstring>
#include <utility>
#include <stdexcept>
#include <string_view>

namespace ninfer::targets::qwen3_6_27b::detail {
namespace {

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

bool shard_owns(std::string_view name, const TextShardSpec& shard) {
    if (name == "text/token_embedding") { return shard.token_embedding; }
    if (name == "text/final_norm" || name == "text/output_head") { return shard.output_head; }
    std::size_t layer = 0;
    return parse_layer(name, layer) && layer >= shard.layer_begin && layer < shard.layer_end;
}

} // namespace

ArtifactLoadPlan bind_text_shard_artifact(artifact::Binder& binder,
                                          const artifact::Reader& reader,
                                          WeightsProfile weights_profile,
                                          TextShardSpec shard) {
    if (shard.layer_begin >= shard.layer_end || shard.layer_end > kTextLayers ||
        shard.layer_begin % 4 != 0 || shard.layer_end % 4 != 0) {
        throw std::invalid_argument("qwen3_6_27b: invalid text shard layer interval");
    }

    ArtifactLoadPlan out = bind_artifact(binder, weights_profile, qwen3_6::StartupFeatures{});
    artifact::MaterializationPlan selected;
    selected.object_count = out.materialization.object_count;
    selected.host_objects = out.materialization.host_objects;

    std::uint64_t cursor = 0;
    for (const artifact::DeviceMaterialization& source : out.materialization.device_objects) {
        const std::string_view name = artifact::object_name(reader.objects().at(source.object.index));
        if (!shard_owns(name, shard)) { continue; }
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

class TextShardStage::Impl {
public:
    Impl(std::string artifact_path, int device_id, std::uint32_t capacity,
         TextShardSpec shard, artifact::LoadProgress* progress, std::uint32_t prefill_chunk)
        : device_id_(device_id), capacity_(capacity), prefill_chunk_(prefill_chunk), shard_(shard),
          device_(device_id) {
        if (capacity_ == 0 || capacity_ > Variant::maximum_context || prefill_chunk_ == 0 ||
            prefill_chunk_ > 128 || prefill_chunk_ > capacity_ ||
            prefill_chunk_ > static_cast<std::uint32_t>(std::numeric_limits<std::int32_t>::max()) ||
            shard_.layer_begin >= shard_.layer_end || shard_.layer_end > kTextLayers ||
            shard_.layer_begin % 4 != 0 || shard_.layer_end % 4 != 0) {
            throw std::invalid_argument("text shard configuration is outside the model domain");
        }
        CUDA_CHECK(cudaSetDevice(device_id_));

        artifact::Reader reader(artifact_path);
        weights_profile_ = Package::resolve_weights(reader.identity());
        artifact::Binder binder(reader);
        ArtifactLoadPlan plan =
            bind_text_shard_artifact(binder, reader, weights_profile_, shard_);
        artifact::MaterializedArtifact materialized =
            artifact::materialize(reader, plan.materialization, device_, progress);
        weights_bytes_ = materialized.stats().device_capacity_bytes;
        model_ = std::make_unique<LoadedModelData>(
            std::move(plan.bindings), std::move(materialized), shard_);

        const std::uint32_t full_begin = static_cast<std::uint32_t>(shard_.layer_begin / 4);
        const std::uint32_t full_end   = static_cast<std::uint32_t>(shard_.layer_end / 4);
        const std::uint32_t full_layers = full_end - full_begin;
        const std::uint32_t gdn_layers =
            static_cast<std::uint32_t>(shard_.layer_end - shard_.layer_begin) - full_layers;
        const std::uint32_t logical_pages =
            (capacity_ + static_cast<std::uint32_t>(kPagedKVPageSize) - 1U) /
            static_cast<std::uint32_t>(kPagedKVPageSize);

        LayoutBuilder state_builder;
        qwen3_6::DecoderStateSpec state_spec{
            .full_attention_layers     = full_layers,
            .mtp_layers                = 0,
            .capacity                  = capacity_,
            .kv_heads                  = StageTextConfig::kv_heads,
            .attention_head_dim        = StageTextConfig::head_dim,
            .kv_dtype                  = DType::BF16,
            .kv_quant_group            = 0,
            .enable_mtp                = false,
            .kv_table_rows             = 1,
            .text_physical_page_groups = logical_pages,
            .mtp_physical_page_groups  = 0,
            .linear_attention =
                {
                    .layers         = gdn_layers,
                    .conv_channels  = StageTextConfig::convolution_dim,
                    .conv_width     = StageTextConfig::gdn_conv_state_width,
                    .value_heads    = StageTextConfig::gdn_value_heads,
                    .value_head_dim = StageTextConfig::gdn_value_head_dim,
                    .key_head_dim   = StageTextConfig::gdn_key_head_dim,
                    .slot_count     = 2,
                    .conv_dtype     = DType::BF16,
                },
        };
        const qwen3_6::DecoderStateLayout state_layout =
            qwen3_6::plan_decoder_state(state_builder, state_spec);
        state_bytes_   = state_builder.finish(256, "dual text shard state");
        state_storage_ = std::make_unique<DeviceArena>(state_bytes_);
        CUDA_CHECK(cudaMemsetAsync(state_storage_->base(), 0, state_storage_->capacity(),
                                   device_.stream));
        decoder_ = std::make_unique<qwen3_6::DecoderState>(
            DeviceSpan{state_storage_->base(), state_storage_->capacity()}, state_layout);
        kv_allocation_ = decoder_->text_kv.pool().reserve(logical_pages);
        kv_allocation_.bind_row(0, device_.stream);
        kv_allocation_.materialize_tokens(capacity_, device_.stream);

        workspace_bytes_ = 64ULL << 20;
        workspace_       = std::make_unique<DeviceArena>(workspace_bytes_);
        io_storage_      = std::make_unique<DeviceArena>(8ULL << 20);
        id_              = io_storage_->alloc(DType::I32, {1});
        position_        = io_storage_->alloc(DType::I32, {1});
        rope_position_   = io_storage_->alloc(DType::I32, {1});
        kv_table_row_    = io_storage_->alloc(DType::I32, {1});
        state_slot_      = io_storage_->alloc(DType::I32, {1});
        residual_        = io_storage_->alloc(DType::BF16, {StageTextConfig::hidden, 1});
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
            normalized_ = io_storage_->alloc(DType::BF16, {StageTextConfig::hidden, 1});
            logits_     = io_storage_->alloc(DType::BF16, {StageTextConfig::output_rows, 1});
            output_token_ = io_storage_->alloc(DType::I32, {1});
        }
        const std::int32_t zero = 0;
        CUDA_CHECK(cudaMemcpyAsync(kv_table_row_.data, &zero, sizeof(zero),
                                   cudaMemcpyHostToDevice, device_.stream));
        CUDA_CHECK(cudaMemcpyAsync(state_slot_.data, &zero, sizeof(zero),
                                   cudaMemcpyHostToDevice, device_.stream));

        StageTextRange range{
            .begin             = static_cast<std::int32_t>(shard_.layer_begin),
            .end               = static_cast<std::int32_t>(shard_.layer_end),
            .input_embedding   = shard_.token_embedding,
            .output_projection = shard_.output_head,
        };
        text_ = std::make_unique<StageTextContext>(
            device_, model_->runtime, *workspace_, qwen3_6::PagedKVCacheView{},
            decoder_->linear_attention, round_, prefill_hidden_, prefill_chunk_, 0,
            qwen3_6::PagedKVCacheView{}, &decoder_->text_kv, nullptr, range);
        if (shard_.token_embedding) {
            frontend_ = std::make_unique<qwen3_6::Frontend>(
                qwen3_6::make_frontend(model_->frontend, false));
        }
        device_.synchronize();
    }

    ~Impl() {
        (void)cudaSetDevice(device_id_);
        if (device_.stream != nullptr) { (void)cudaStreamSynchronize(device_.stream); }
    }

    void activate() const { CUDA_CHECK(cudaSetDevice(device_id_)); }
    void set_position(std::int32_t position, std::int32_t rope_position) {
        CUDA_CHECK(cudaMemcpyAsync(position_.data, &position, sizeof(position),
                                   cudaMemcpyHostToDevice, device_.stream));
        CUDA_CHECK(cudaMemcpyAsync(rope_position_.data, &rope_position, sizeof(rope_position),
                                   cudaMemcpyHostToDevice, device_.stream));
    }

    int device_id_ = 0;
    std::uint32_t capacity_ = 0;
    std::uint32_t prefill_chunk_ = 0;
    TextShardSpec shard_;
    DeviceContext device_;
    WeightsProfile weights_profile_ = WeightsProfile::GroupwiseIntW8Endpoints;
    std::unique_ptr<LoadedModelData> model_;
    std::unique_ptr<DeviceArena> state_storage_;
    std::unique_ptr<qwen3_6::DecoderState> decoder_;
    PagedKVAllocation kv_allocation_;
    std::unique_ptr<DeviceArena> workspace_;
    std::unique_ptr<DeviceArena> io_storage_;
    qwen3_6::RoundState round_;
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
    std::unique_ptr<StageTextContext> text_;
    std::unique_ptr<qwen3_6::Frontend> frontend_;
    std::size_t weights_bytes_ = 0;
    std::size_t state_bytes_ = 0;
    std::size_t workspace_bytes_ = 0;
};


TextShardStage::TextShardStage(std::string artifact_path, int device_id, std::uint32_t capacity,
                               TextShardSpec shard, artifact::LoadProgress* progress,
                               std::uint32_t prefill_chunk)
    : impl_(std::make_unique<Impl>(std::move(artifact_path), device_id, capacity, shard, progress,
                                   prefill_chunk)) {}

TextShardStage::~TextShardStage() = default;

TextPromptSequence TextShardStage::prepare_text_prompt(std::string_view text,
                                                       bool enable_thinking) const {
    if (impl_ == nullptr || impl_->frontend_ == nullptr) {
        throw std::logic_error("text shard does not own frontend resources");
    }
    PromptInput input;
    input.options.enable_thinking = enable_thinking;
    ChatMessage message;
    message.role = "user";
    MessagePart part;
    part.kind = MessagePartKind::Text;
    part.text.assign(text);
    message.parts.push_back(std::move(part));
    input.messages.push_back(std::move(message));
    qwen3_6::PreparedPrompt prepared = impl_->frontend_->prepare(std::move(input));
    qwen3_6::PreparedPromptData data =
        qwen3_6::PreparedPromptAccess::take(std::move(prepared));
    if (data.has_media()) {
        throw std::invalid_argument("dual text shard driver does not support media prompts");
    }
    return TextPromptSequence{.token_ids = std::move(data.token_ids),
                              .rope_delta = data.rope_delta};
}

void TextShardStage::forward_token(TokenId token, std::int32_t position,
                                   std::int32_t rope_position) {
    if (impl_ == nullptr) { throw std::logic_error("text shard is empty"); }
    if (position < 0 || static_cast<std::uint32_t>(position) >= impl_->capacity_) {
        throw std::out_of_range("text shard token position is outside capacity");
    }
    impl_->activate();
    CUDA_CHECK(cudaMemcpyAsync(impl_->id_.data, &token, sizeof(token), cudaMemcpyHostToDevice,
                               impl_->device_.stream));
    impl_->set_position(position, rope_position);
    const std::uint32_t visible = static_cast<std::uint32_t>(position) + 1U;
    impl_->text_->forward_token_shard(
        impl_->id_, impl_->position_, impl_->rope_position_, impl_->kv_table_row_,
        impl_->state_slot_, ops::GqaExecutionEnvelope{visible, visible}, impl_->residual_);
}

void TextShardStage::forward_hidden(std::int32_t position, std::int32_t rope_position) {
    if (impl_ == nullptr) { throw std::logic_error("text shard is empty"); }
    if (position < 0 || static_cast<std::uint32_t>(position) >= impl_->capacity_) {
        throw std::out_of_range("text shard hidden position is outside capacity");
    }
    impl_->activate();
    impl_->set_position(position, rope_position);
    const std::uint32_t visible = static_cast<std::uint32_t>(position) + 1U;
    impl_->text_->forward_hidden_shard(
        impl_->residual_, impl_->position_, impl_->rope_position_, impl_->kv_table_row_,
        impl_->state_slot_, ops::GqaExecutionEnvelope{visible, visible});
}

void TextShardStage::prefill_tokens(std::span<const TokenId> tokens, std::int32_t position,
                                    std::int32_t rope_position) {
    if (impl_ == nullptr || tokens.empty() || tokens.size() > impl_->prefill_chunk_ || position < 0 ||
        rope_position < 0 || static_cast<std::uint64_t>(position) + tokens.size() > impl_->capacity_) {
        throw std::invalid_argument("invalid text shard token prefill");
    }
    impl_->activate();
    const auto width = static_cast<std::int32_t>(tokens.size());
    Tensor ids       = impl_->prefill_ids_.slice(0, 0, width);
    Tensor positions = impl_->prefill_positions_.slice(0, 0, width);
    Tensor rope_positions = impl_->prefill_rope_positions_.slice(0, 0, width);
    Tensor residual       = impl_->prefill_residual_.slice(1, 0, width);
    CUDA_CHECK(cudaMemcpyAsync(ids.data, tokens.data(), tokens.size_bytes(), cudaMemcpyHostToDevice,
                               impl_->device_.stream));
    ops::fill_i32_positions(positions, position, impl_->device_.stream);
    ops::fill_i32_positions(rope_positions, rope_position, impl_->device_.stream);
    const auto visible = static_cast<std::uint32_t>(position) + static_cast<std::uint32_t>(width);
    impl_->text_->prefill_token_shard(ids, positions, rope_positions, impl_->kv_table_row_,
                                      ops::GqaExecutionEnvelope{visible, visible}, residual);
    if (impl_->shard_.output_head) {
        Tensor last = residual.slice(1, width - 1, 1);
        CUDA_CHECK(cudaMemcpyAsync(impl_->residual_.data, last.data, impl_->residual_.bytes(),
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
                                    std::int32_t rope_position) {
    if (impl_ == nullptr || tokens == 0 || tokens > impl_->prefill_chunk_ || position < 0 ||
        rope_position < 0 || static_cast<std::uint64_t>(position) + tokens > impl_->capacity_) {
        throw std::invalid_argument("invalid text shard hidden prefill");
    }
    impl_->activate();
    const auto width = static_cast<std::int32_t>(tokens);
    Tensor positions = impl_->prefill_positions_.slice(0, 0, width);
    Tensor rope_positions = impl_->prefill_rope_positions_.slice(0, 0, width);
    Tensor residual       = impl_->prefill_residual_.slice(1, 0, width);
    ops::fill_i32_positions(positions, position, impl_->device_.stream);
    ops::fill_i32_positions(rope_positions, rope_position, impl_->device_.stream);
    const auto visible = static_cast<std::uint32_t>(position) + static_cast<std::uint32_t>(width);
    impl_->text_->prefill_hidden_shard(residual, positions, rope_positions, impl_->kv_table_row_,
                                       ops::GqaExecutionEnvelope{visible, visible});
    Tensor last = residual.slice(1, width - 1, 1);
    CUDA_CHECK(cudaMemcpyAsync(impl_->residual_.data, last.data, impl_->residual_.bytes(),
                               cudaMemcpyDeviceToDevice, impl_->device_.stream));
}

void TextShardStage::copy_residual_to_host(void* destination, std::size_t bytes) {
    if (impl_ == nullptr || destination == nullptr || bytes != residual_bytes()) {
        throw std::invalid_argument("invalid text shard residual download");
    }
    impl_->activate();
    CUDA_CHECK(cudaMemcpyAsync(destination, impl_->residual_.data, bytes, cudaMemcpyDeviceToHost,
                               impl_->device_.stream));
    impl_->device_.synchronize();
}

void TextShardStage::copy_residual_from_host(const void* source, std::size_t bytes) {
    if (impl_ == nullptr || source == nullptr || bytes != residual_bytes()) {
        throw std::invalid_argument("invalid text shard residual upload");
    }
    impl_->activate();
    CUDA_CHECK(cudaMemcpyAsync(impl_->residual_.data, source, bytes, cudaMemcpyHostToDevice,
                               impl_->device_.stream));
}

TokenId TextShardStage::project_token() {
    if (impl_ == nullptr) { throw std::logic_error("text shard is empty"); }
    impl_->activate();
    impl_->text_->project_shard_head(impl_->residual_, impl_->normalized_, impl_->logits_,
                                     impl_->output_token_);
    TokenId token = 0;
    CUDA_CHECK(cudaMemcpyAsync(&token, impl_->output_token_.data, sizeof(token),
                               cudaMemcpyDeviceToHost, impl_->device_.stream));
    impl_->device_.synchronize();
    return token;
}

void TextShardStage::synchronize() {
    if (impl_ == nullptr) { throw std::logic_error("text shard is empty"); }
    impl_->activate();
    impl_->device_.synchronize();
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

std::uint32_t TextShardStage::prefill_chunk() const noexcept {
    return impl_ == nullptr ? 0 : impl_->prefill_chunk_;
}

} // namespace ninfer::targets::qwen3_6_27b::detail
