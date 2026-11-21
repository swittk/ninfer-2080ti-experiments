#include "dual/attention_packet.cuh"
#include <cstdio>
#include "dual/context_parallel_gqa.h"
#include "dual/headscale_gqa.cuh"
#include "dual/joint_kv/runtime.cuh"
#include "../../.chatgpt/production_serve_20260918/target_guided_proposal/headscale_guide.cuh"
#include "../../.chatgpt/production_serve_20260918/target_guided_proposal/guide_metadata.cuh"

#include "core/arena.h"
#include "core/device.h"
#include "core/layout.h"
#include "core/paged_kv_cache.h"
#include "ninfer/ops/gqa_attention.h"
#include "ops/launcher/gqa_attention.h"
#include "targets/qwen3_6_27b/impl/config.h"

#include <ninfer/targets/qwen3_6/decoder_state.h>
#include <ninfer/targets/qwen3_6/hybrid_topology.h>

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>

namespace ninfer::targets::qwen3_6_27b::detail {
namespace {

using Clock = std::chrono::steady_clock;
inline constexpr std::int32_t kMaximumTokens = 6;
inline constexpr std::int32_t kKeyTile = 32;
inline constexpr std::uint64_t kSnapshotMagic = 0x315648534b5653ULL;
inline constexpr std::uint32_t kSnapshotInt8Version = 1;
inline constexpr std::uint32_t kSnapshotBf16Version = 2;
using TextConfig = qwen3_6_27b::detail::TextConfig;
namespace packet = ninfer::apps::dual::attention_packet;
static_assert(TextConfig::head_dim==packet::HeadDim && TextConfig::query_heads==packet::QueryHeads && TextConfig::kv_heads==packet::KvHeads);


DType storage_dtype(KvCacheStorage storage) {
    switch (storage) {
    case KvCacheStorage::BFloat16:
        return DType::BF16;
    case KvCacheStorage::Int8Group64:
        return DType::I8;
    }
    throw std::invalid_argument("unsupported sequence-sharded KV storage");
}

inline constexpr std::uint32_t kLongContextTurboMinKeys = 32768;
inline constexpr double kLongContextSinkFraction = 0.01;
inline constexpr double kLongContextTailFraction = 0.05;

bool tail_fast_ownership_enabled(std::uint32_t capacity) noexcept {
    return capacity >= kLongContextTurboMinKeys;
}

std::int32_t storage_quant_group(KvCacheStorage storage) {
    return storage == KvCacheStorage::Int8Group64 ? qwen3_6::kKvQuantGroup : 0;
}

std::uint32_t snapshot_version(KvCacheStorage storage) {
    return storage == KvCacheStorage::Int8Group64 ? kSnapshotInt8Version : kSnapshotBf16Version;
}

struct SnapshotHeader {
    std::uint64_t magic = kSnapshotMagic;
    std::uint32_t version = 0;
    std::uint32_t populated_tokens = 0;
    std::uint64_t first_bytes = 0;
    std::uint64_t second_bytes = 0;
};

void check(cudaError_t status, const char* operation) {
    if (status != cudaSuccess) {
        throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(status));
    }
}

std::uint32_t logical_pages(std::uint32_t capacity) {
    if (capacity == 0) { throw std::invalid_argument("sequence-sharded capacity is zero"); }
    return 1U + (capacity - 1U) / static_cast<std::uint32_t>(kPagedKVPageSize);
}

std::size_t checked_product(std::initializer_list<std::size_t> factors) {
    std::size_t out = 1;
    for (std::size_t factor : factors) {
        if (factor != 0 && out > std::numeric_limits<std::size_t>::max() / factor) {
            throw std::overflow_error("sequence-sharded allocation size overflow");
        }
        out *= factor;
    }
    return out;
}

PagedKVLayerView row_zero_view(PagedKVBatchLayerView cache) {
    if (cache.block_tables.data == nullptr || cache.block_tables.ne[1] != 1) {
        throw std::invalid_argument("sequence-sharded GQA requires one KV table row");
    }
    return {
        .k_pages = cache.k_pages,
        .v_pages = cache.v_pages,
        .k_scale_pages = cache.k_scale_pages,
        .v_scale_pages = cache.v_scale_pages,
        .block_table = cache.block_tables.slice(1, 0, 1).view({cache.block_tables.ne[0]}),
        .head_dim = cache.head_dim,
        .num_kv_heads = cache.num_kv_heads,
        .dtype = cache.dtype,
        .quant_group = cache.quant_group,
    };
}

qwen3_6::PagedKVCacheLayout plan_helper_cache(LayoutBuilder& builder,
                                               std::uint32_t layers,
                                               std::uint32_t capacity,
                                               TextKvPageOwnership ownership,
                                               KvCacheStorage storage) {
    const std::uint32_t pages = logical_pages(capacity);
    if (layers == 0 || !ownership.enabled || ownership.logical_page_count == 0 ||
        ownership.logical_page_begin > pages ||
        ownership.logical_page_count > pages - ownership.logical_page_begin) {
        throw std::invalid_argument("sequence-sharded helper cache geometry is invalid");
    }
    const DType dtype = storage_dtype(storage);
    const std::int32_t quant_group = storage_quant_group(storage);
    const bool quantized = dtype == DType::I8;
    PagedKVPoolSpec pool_spec;
    pool_spec.page_group_count = ownership.logical_page_count;
    pool_spec.logical_page_capacity = pages;
    pool_spec.table_rows = 1;
    pool_spec.plane_order = PagedKVPlaneOrder::PageMajor;
    pool_spec.planes.reserve(static_cast<std::size_t>(layers) * (quantized ? 4U : 2U));
    for (std::uint32_t layer = 0; layer < layers; ++layer) {
        pool_spec.planes.push_back(
            {dtype, TextConfig::head_dim, TextConfig::kv_heads, 256});
        pool_spec.planes.push_back(
            {dtype, TextConfig::head_dim, TextConfig::kv_heads, 256});
        if (quantized) {
            pool_spec.planes.push_back(
                {DType::FP16, TextConfig::head_dim / quant_group,
                 TextConfig::kv_heads, 256});
            pool_spec.planes.push_back(
                {DType::FP16, TextConfig::head_dim / quant_group,
                 TextConfig::kv_heads, 256});
        }
    }
    return {
        .pool = plan_paged_kv_pool(builder, pool_spec),
        .layers = layers,
        .max_context = capacity,
        .kv_heads = TextConfig::kv_heads,
        .head_dim = TextConfig::head_dim,
        .dtype = dtype,
        .quant_group = quant_group,
    };
}

struct SplitRange {
    std::int32_t begin = 0;
    std::int32_t count = 0;
    std::int32_t total = 0;
};

std::int32_t retained_split_subdivide(std::uint32_t capacity,qwen3_6::LongContextComputation policy) noexcept {
    return static_cast<std::int32_t>(qwen3_6::long_context_split_subdivision(policy,capacity));
}

struct DeviceScratch {
    DeviceScratch(int device, std::uint32_t capacity, DType kv_dtype,qwen3_6::LongContextComputation policy)
        : device_id(device), subdivision(retained_split_subdivide(capacity,policy)),
          split_capacity(ops::detail::gqa_attention_split_capacity(
                             TextConfig::query_heads, kMaximumTokens, kv_dtype,
                             {capacity, capacity}) * subdivision) {
        check(cudaSetDevice(device_id), "cudaSetDevice sequence scratch");
        const std::size_t partial_acc_bytes = checked_product({
            static_cast<std::size_t>(TextConfig::head_dim),
            static_cast<std::size_t>(TextConfig::query_heads),
            static_cast<std::size_t>(kMaximumTokens),
            static_cast<std::size_t>(split_capacity), sizeof(__nv_bfloat16)});
        const std::size_t partial_stat_bytes = checked_product({
            static_cast<std::size_t>(TextConfig::query_heads),
            static_cast<std::size_t>(kMaximumTokens),
            static_cast<std::size_t>(split_capacity), sizeof(float)});
        const std::size_t state_acc_bytes = checked_product({
            static_cast<std::size_t>(TextConfig::head_dim),
            static_cast<std::size_t>(TextConfig::query_heads),
            static_cast<std::size_t>(kMaximumTokens), sizeof(float)});
        const std::size_t state_stat_bytes = checked_product({
            static_cast<std::size_t>(TextConfig::query_heads),
            static_cast<std::size_t>(kMaximumTokens), sizeof(float)});
        partial_acc = std::make_unique<DeviceBuffer>(partial_acc_bytes);
        partial_m = std::make_unique<DeviceBuffer>(partial_stat_bytes);
        partial_l = std::make_unique<DeviceBuffer>(partial_stat_bytes);
        const auto state_bytes=packet::state_layout(kMaximumTokens).bytes;
        if(state_bytes!=state_acc_bytes+2*state_stat_bytes)throw std::logic_error("state packet size differs from original storage");
        state_packet = std::make_unique<DeviceBuffer>(state_bytes);
        uploaded_packet = std::make_unique<DeviceBuffer>(state_bytes);
    }

    Tensor partial_acc_tensor(std::int32_t tokens, std::int32_t total) const {
        return Tensor(partial_acc->p, DType::BF16,
                      {TextConfig::head_dim, TextConfig::query_heads, tokens, total});
    }
    Tensor partial_m_tensor(std::int32_t tokens, std::int32_t total) const {
        return Tensor(partial_m->p, DType::FP32,
                      {TextConfig::query_heads, tokens, total});
    }
    Tensor partial_l_tensor(std::int32_t tokens, std::int32_t total) const {
        return Tensor(partial_l->p, DType::FP32,
                      {TextConfig::query_heads, tokens, total});
    }
    Tensor state_acc_tensor(std::int32_t tokens) const {
        return Tensor(state_packet->p, DType::FP32,
                      {TextConfig::head_dim, TextConfig::query_heads, tokens});
    }
    Tensor state_m_tensor(std::int32_t tokens) const {
        return Tensor(static_cast<char*>(state_packet->p)+packet::state_layout(tokens).maximum_offset, DType::FP32, {TextConfig::query_heads, tokens});
    }
    Tensor state_l_tensor(std::int32_t tokens) const {
        return Tensor(static_cast<char*>(state_packet->p)+packet::state_layout(tokens).normalizer_offset, DType::FP32, {TextConfig::query_heads, tokens});
    }
    Tensor uploaded_acc_tensor(std::int32_t tokens) const {
        return Tensor(uploaded_packet->p, DType::FP32,
                      {TextConfig::head_dim, TextConfig::query_heads, tokens});
    }
    Tensor uploaded_m_tensor(std::int32_t tokens) const {
        return Tensor(static_cast<char*>(uploaded_packet->p)+packet::state_layout(tokens).maximum_offset, DType::FP32, {TextConfig::query_heads, tokens});
    }
    Tensor uploaded_l_tensor(std::int32_t tokens) const {
        return Tensor(static_cast<char*>(uploaded_packet->p)+packet::state_layout(tokens).normalizer_offset, DType::FP32, {TextConfig::query_heads, tokens});
    }

    int device_id = 0;
    std::int32_t subdivision = 1;
    std::int32_t split_capacity = 0;
    std::unique_ptr<DeviceBuffer> partial_acc;
    std::unique_ptr<DeviceBuffer> partial_m;
    std::unique_ptr<DeviceBuffer> partial_l;
    std::unique_ptr<DeviceBuffer> state_packet;
    std::unique_ptr<DeviceBuffer> uploaded_packet;
};

struct HelperDevice {
    HelperDevice(int device, std::uint32_t capacity, std::uint32_t layers,
                 TextKvPageOwnership ownership, KvCacheStorage storage,qwen3_6::LongContextComputation policy)
        : device_id(device), context(device), scratch(device, capacity, storage_dtype(storage),policy),
          ownership(ownership) {
        check(cudaSetDevice(device_id), "cudaSetDevice sequence helper");
        const std::size_t q_bytes = checked_product({
            static_cast<std::size_t>(TextConfig::head_dim),
            static_cast<std::size_t>(TextConfig::query_heads),
            static_cast<std::size_t>(kMaximumTokens), sizeof(__nv_bfloat16)});
        const std::size_t kv_bytes = checked_product({
            static_cast<std::size_t>(TextConfig::head_dim),
            static_cast<std::size_t>(TextConfig::kv_heads),
            static_cast<std::size_t>(kMaximumTokens), sizeof(__nv_bfloat16)});
        const auto input_layout=packet::input_layout(kMaximumTokens,true);
        if(input_layout.query_bytes!=q_bytes||input_layout.kv_bytes!=kv_bytes)throw std::logic_error("helper packet geometry changed");
        input_packet = std::make_unique<DeviceBuffer>(input_layout.bytes);

        LayoutBuilder builder;
        const qwen3_6::PagedKVCacheLayout layout =
            plan_helper_cache(builder, layers, capacity, ownership, storage);
        const std::size_t bytes = builder.finish(256, "sequence-sharded helper KV");
        backing = std::make_unique<DeviceBuffer>(bytes);
        cache = std::make_unique<qwen3_6::PagedKVCache>(
            DeviceSpan{backing->p, backing->bytes}, layout);
        allocation.emplace(cache->pool().reserve(ownership.logical_page_count));
        allocation->bind_row_range(0, ownership.logical_page_begin, context.stream);
        allocation->materialize_pages(ownership.logical_page_count, context.stream);
        check(cudaEventCreateWithFlags(&input_ready, cudaEventDisableTiming),
              "cudaEventCreate sequence helper input");
        context.synchronize();
    }

    ~HelperDevice() {
        (void)cudaSetDevice(device_id);
        if (input_ready != nullptr) { (void)cudaEventDestroy(input_ready); }
    }

    Tensor q_tensor(std::int32_t tokens) const {
        return Tensor(input_packet->p, DType::BF16,
                      {TextConfig::head_dim, TextConfig::query_heads, tokens});
    }
    Tensor k_tensor(std::int32_t tokens) const {
        return Tensor(static_cast<char*>(input_packet->p)+packet::input_layout(tokens,true).key_offset, DType::BF16,
                      {TextConfig::head_dim, TextConfig::kv_heads, tokens});
    }
    Tensor v_tensor(std::int32_t tokens) const {
        return Tensor(static_cast<char*>(input_packet->p)+packet::input_layout(tokens,true).value_offset, DType::BF16,
                      {TextConfig::head_dim, TextConfig::kv_heads, tokens});
    }
    Tensor positions_tensor(std::int32_t tokens) const {
        return Tensor(static_cast<char*>(input_packet->p)+packet::input_layout(tokens,true).position_offset, DType::I32, {tokens});
    }
    PagedKVLayerView layer(std::uint32_t local_layer) const {
        return cache->execution_view(*allocation).layer_view(local_layer);
    }
    std::size_t packed_plane_bytes() const noexcept {
        std::size_t bytes = 0;
        for (std::size_t index = 0; index < cache->pool().plane_count(); ++index) {
            bytes += cache->pool().plane(index).bytes();
        }
        return bytes;
    }
    void load_packed(std::span<const std::uint8_t> packed) {
        if (packed.size() != packed_plane_bytes()) {
            throw std::invalid_argument("sequence helper packed KV size mismatch");
        }
        check(cudaSetDevice(device_id), "cudaSetDevice sequence helper load");
        std::size_t cursor = 0;
        for (std::size_t index = 0; index < cache->pool().plane_count(); ++index) {
            const Tensor& plane = cache->pool().plane(index);
            check(cudaMemcpyAsync(plane.data, packed.data() + cursor, plane.bytes(),
                                  cudaMemcpyHostToDevice, context.load_stream),
                  "cudaMemcpyAsync sequence helper plane");
            cursor += plane.bytes();
        }
        context.synchronize();
        loaded = true;
    }

    int device_id = 0;
    DeviceContext context;
    DeviceScratch scratch;
    TextKvPageOwnership ownership{};
    std::unique_ptr<DeviceBuffer> input_packet;
    std::unique_ptr<DeviceBuffer> backing;
    std::unique_ptr<qwen3_6::PagedKVCache> cache;
    std::optional<PagedKVAllocation> allocation;
    cudaEvent_t input_ready = nullptr;
    bool loaded = false;
};

class StageEngine {
public:
    StageEngine(int owner_device, int helper_device, bool owner_is_fast,
                std::int32_t layer_begin, std::int32_t layer_end,
                std::uint32_t capacity, std::uint32_t threshold,
                const SequenceShardGeometry& geometry, double fast_fraction,
                qwen3_6::LongContextComputation computation,bool headscale_qk,bool value_fp16)
        : owner_device_(owner_device), helper_device_(helper_device),
          owner_is_fast_(owner_is_fast), layer_begin_(layer_begin), layer_end_(layer_end),
          full_begin_(qwen3_6::full_attention_layers(layer_begin)),
          full_layers_(qwen3_6::full_attention_layers(layer_end) - full_begin_),
          capacity_(capacity), threshold_(threshold), geometry_(geometry), computation_(computation),
          headscale_qk_(headscale_qk),value_fp16_(value_fp16),
          storage_(geometry.storage), kv_dtype_(geometry.dtype), quant_group_(geometry.quant_group),
          fast_fraction_(fast_fraction), owner_scratch_(owner_device, capacity, kv_dtype_,computation),
          helper_(helper_device, capacity, static_cast<std::uint32_t>(full_layers_),
                  owner_is_fast ? geometry.slow_pages : geometry.fast_pages, storage_,computation),
          host_input_packet_(packet::input_layout(kMaximumTokens,true).bytes),
          host_state_packet_(packet::state_layout(kMaximumTokens).bytes) {
        if (layer_begin_ < 0 || layer_end_ > TextConfig::layers || layer_begin_ >= layer_end_ ||
            layer_begin_ % qwen3_6::kHybridAttentionInterval != 0 ||
            layer_end_ % qwen3_6::kHybridAttentionInterval != 0 || full_layers_ <= 0) {
            throw std::invalid_argument("sequence-sharded stage layer range is invalid");
        }
        if (joint_requested_ && (!headscale_qk_ || !value_fp16_ ||
            computation_ != qwen3_6::LongContextComputation::Complete || capacity_ < 32768))
            throw std::invalid_argument("joint-KV research requires complete long-context headscale/direct-value attention");
        check(cudaSetDevice(owner_device_), "cudaSetDevice sequence owner");
        if(joint_requested_)owner_input_packet_=std::make_unique<DeviceBuffer>(packet::input_layout(kMaximumTokens,true).bytes);
        check(cudaStreamCreateWithFlags(&owner_copy_stream_, cudaStreamNonBlocking),
              "cudaStreamCreate sequence owner copy");
        check(cudaEventCreateWithFlags(&owner_inputs_ready_, cudaEventDisableTiming),
              "cudaEventCreate sequence owner inputs");
        check(cudaEventCreateWithFlags(&owner_state_ready_, cudaEventDisableTiming),
              "cudaEventCreate sequence owner state");
    }

    ~StageEngine() {
        joint_.reset();
        if(guide_owner_log_)std::fprintf(stderr,"TARGET_GUIDE_EXPERIMENT owner=%d captures=%llu views=%llu\n",owner_device_,static_cast<unsigned long long>(guide_captures_),static_cast<unsigned long long>(guide_reads_));
        (void)cudaSetDevice(owner_device_);
        if (owner_inputs_ready_ != nullptr) { (void)cudaEventDestroy(owner_inputs_ready_); }
        if (owner_state_ready_ != nullptr) { (void)cudaEventDestroy(owner_state_ready_); }
        if (owner_copy_stream_ != nullptr) { (void)cudaStreamDestroy(owner_copy_stream_); }
    }

    bool owns(std::int32_t global_layer) const noexcept {
        return global_layer >= layer_begin_ && global_layer < layer_end_ &&
               qwen3_6::is_full_attention_layer(global_layer);
    }
    TextKvPageOwnership helper_ownership() const noexcept { return helper_.ownership; }
    std::size_t helper_bytes() const noexcept {
        return helper_.backing == nullptr ? 0 : helper_.backing->bytes;
    }
    bool loaded() const noexcept { return helper_.loaded; }
    void load_helper(std::span<const std::uint8_t> packed) { helper_.load_packed(packed); }
    void activate_prefix(std::uint32_t populated_tokens) {
        if (!helper_.loaded || populated_tokens < static_cast<std::uint32_t>(geometry_.boundary_key) ||
            populated_tokens > capacity_) {
            throw std::invalid_argument("sequence-sharded prefix is not inside owned-page geometry");
        }
        joint_.reset(); // Prepared coordinates belong to exactly this committed prefix.
        populated_tokens_ = populated_tokens;
        owner_headscale_ready_.fill(false);helper_headscale_ready_.fill(false);
        guide_ready_ = false;
        ready_ = true;
    }
    bool ready() const noexcept { return ready_; }
    bool requires_prefix_preparation() const noexcept { return joint_requested_; }
    void prepare_prefix_layer(std::int32_t global_layer, PagedKVLayerView local_cache,
                              std::uint32_t populated_tokens, cudaStream_t owner_stream) {
        if (!joint_requested_) { return; }
        if (!ready_ || populated_tokens != populated_tokens_ || !owns(global_layer))
            throw std::logic_error("jointKV prefix preparation is not bound to the active prefix");
        if (populated_tokens_ < 32768) { return; } // Keep the native short-prefix fallback.
        // The final layer's proposal guidance is a different output contract.
        if (global_layer == 63 && std::getenv("NINFER_TARGET_GUIDED_PROPOSAL_ORACLE")) { return; }
        const int index = qwen3_6::full_attention_index(global_layer) - full_begin_;
        const auto helper_cache = helper_.layer(static_cast<std::uint32_t>(index));
        if (!owner_headscale_ready_[index]) {
            check(cudaSetDevice(owner_device_), "prefix owner prepare");
            ops::headscale_qk::prepare_headscale_keys(local_cache,nullptr,populated_tokens_,owner_stream);
            owner_headscale_ready_[index] = true;
        }
        if (!helper_headscale_ready_[index]) {
            check(cudaSetDevice(helper_device_), "prefix helper prepare");
            ops::headscale_qk::prepare_headscale_keys(helper_cache,nullptr,populated_tokens_,helper_.context.stream);
            helper_headscale_ready_[index] = true;
        }
        if (!joint_)
            joint_ = std::make_unique<ninfer::apps::dual::joint_runtime::Collection>(
                owner_device_,helper_device_,capacity_,populated_tokens_,full_layers_,
                local_cache,helper_cache,owner_stream,helper_.context.stream);
        joint_->prepare_prefix(index,local_cache,helper_cache);
        check(cudaSetDevice(owner_device_), "restore prefix owner");
    }

    bool guide(Tensor& score,Tensor& pos) const {
        if(!guide_ready_)return false;
        ++guide_reads_;
        score=Tensor(guide_pages_->p,DType::FP32,{static_cast<int>(geometry_.logical_pages),24,guide_columns_});
        pos=Tensor(guide_position_->p,DType::I32,{guide_columns_});
        return true;
    }


    void copy_snapshot_to_host(void* destination, std::size_t bytes) const {
        if (headscale_qk_)throw std::logic_error("experimental headscale cache cannot be exported as a canonical snapshot");
        if (destination == nullptr || bytes != helper_bytes()) {
            throw std::invalid_argument("invalid sequence helper snapshot download");
        }
        check(cudaSetDevice(helper_device_), "cudaSetDevice helper snapshot download");
        helper_.context.synchronize();
        check(cudaMemcpy(destination, helper_.backing->p, bytes, cudaMemcpyDeviceToHost),
              "cudaMemcpy sequence helper snapshot D2H");
    }
    void copy_snapshot_from_host(const void* source, std::size_t bytes) {
        if (source == nullptr || bytes != helper_bytes()) {
            throw std::invalid_argument("invalid sequence helper snapshot upload");
        }
        check(cudaSetDevice(helper_device_), "cudaSetDevice helper snapshot upload");
        check(cudaMemcpy(helper_.backing->p, source, bytes, cudaMemcpyHostToDevice),
              "cudaMemcpy sequence helper snapshot H2D");
        helper_.context.synchronize();
        helper_.loaded = true;
    }

    ContextParallelGqaStats stats() const noexcept { return stats_; }

    bool execute(std::int32_t global_layer, const Tensor& q, const Tensor& k,
                 const Tensor& v, const Tensor& positions, const Tensor& valid_columns,
                 const Tensor& kv_table_rows, float scale, PagedKVBatchLayerView batch_cache,
                 ops::GqaExecutionEnvelope envelope, Tensor& out,
                 cudaStream_t owner_stream) {
        if (!owns(global_layer)) { return false; }
        if (!ready_) {
            throw std::logic_error("sequence-sharded GQA ran before prefix activation");
        }
        if (q.ne[2] <= 0 || q.ne[2] > kMaximumTokens || k.ne[2] != q.ne[2] ||
            v.ne[2] != q.ne[2] || positions.ne[0] != q.ne[2] ||
            kv_table_rows.numel() != 1 || batch_cache.dtype != kv_dtype_ ||
            batch_cache.quant_group != quant_group_) {
            throw std::invalid_argument("sequence-sharded C1 GQA invocation is unsupported");
        }
        const std::int32_t tokens = q.ne[2];
        std::int32_t active_tokens = tokens;
        if (global_layer == layer_begin_ + qwen3_6::kHybridAttentionInterval - 1) {
            if (valid_columns.data != nullptr) {
                check(cudaSetDevice(owner_device_), "cudaSetDevice sequence valid columns");
                check(cudaMemcpyAsync(&active_tokens, valid_columns.data, sizeof(active_tokens),
                                      cudaMemcpyDeviceToHost, owner_stream),
                      "cudaMemcpyAsync sequence valid columns");
                check(cudaStreamSynchronize(owner_stream),
                      "cudaStreamSynchronize sequence valid columns");
            }
            if (active_tokens <= 0 || active_tokens > tokens) {
                throw std::invalid_argument("sequence-sharded valid-column extent is invalid");
            }
            active_storage_tokens_ = tokens;
            active_tokens_ = active_tokens;
        } else {
            if (active_storage_tokens_ != tokens || active_tokens_ <= 0 ||
                active_tokens_ > tokens) {
                throw std::logic_error("sequence-sharded valid-column cache is stale");
            }
            active_tokens = active_tokens_;
        }
        Tensor active_q = active_tokens == tokens ? q : q.slice(2, 0, active_tokens);
        Tensor active_k = active_tokens == tokens ? k : k.slice(2, 0, active_tokens);
        Tensor active_v = active_tokens == tokens ? v : v.slice(2, 0, active_tokens);
        Tensor active_positions = active_tokens == tokens
            ? positions : positions.slice(0, 0, active_tokens);
        Tensor active_out = active_tokens == tokens ? out : out.slice(2, 0, active_tokens);
        const std::int32_t total = ops::detail::gqa_attention_split_capacity(
            TextConfig::query_heads, active_tokens, kv_dtype_, envelope);
        if (total != geometry_.total_splits) {
            throw std::logic_error("sequence-sharded split inventory changed after ownership plan");
        }
        const std::int32_t fast_count = std::clamp(
            static_cast<std::int32_t>(std::llround(total * fast_fraction_)), 1, total - 1);
        if (fast_count != geometry_.fast_splits) {
            throw std::logic_error("sequence-sharded calibrated split count changed");
        }
        const bool tail_fast = tail_fast_ownership_enabled(capacity_);
        const std::int32_t slow_count = total - fast_count;
        const SplitRange fast = tail_fast
            ? SplitRange{slow_count, fast_count, total}
            : SplitRange{0, fast_count, total};
        const SplitRange slow = tail_fast
            ? SplitRange{0, slow_count, total}
            : SplitRange{fast_count, slow_count, total};
        const bool long_context_turbo =
            qwen3_6::use_retained_long_context(computation_,envelope.max_visible_keys);
        const double sink_fraction = long_context_turbo ? kLongContextSinkFraction : 0.0;
        const double tail_fraction = long_context_turbo ? kLongContextTailFraction : 0.0;
        SplitRange retained_fast = fast;
        SplitRange retained_slow = slow;
        if (sink_fraction > 0.0 || tail_fraction > 0.0) {
            const std::int32_t sink_count = std::clamp(
                static_cast<std::int32_t>(std::llround(total * sink_fraction)), 1, fast.count);
            const std::int32_t tail_count = std::clamp(
                static_cast<std::int32_t>(std::llround(total * tail_fraction)), 1, slow.count);
            if (tail_fast) {
                retained_slow = SplitRange{0, sink_count, total};
                retained_fast = SplitRange{total - tail_count, tail_count, total};
            } else {
                retained_fast = SplitRange{0, sink_count, total};
                retained_slow = SplitRange{total - tail_count, tail_count, total};
            }
        }
        const SplitRange owner_range = owner_is_fast_ ? retained_fast : retained_slow;
        const SplitRange helper_range = owner_is_fast_ ? retained_slow : retained_fast;
        const std::int32_t local_index =
            qwen3_6::full_attention_index(global_layer) - full_begin_;
        if (local_index < 0 || local_index >= full_layers_) {
            throw std::logic_error("sequence-sharded local layer mapping is invalid");
        }
        const PagedKVLayerView local_cache = row_zero_view(batch_cache);
        const PagedKVLayerView helper_cache =
            helper_.layer(static_cast<std::uint32_t>(local_index));
        const TextKvPageOwnership expected_local =
            owner_is_fast_ ? geometry_.fast_pages : geometry_.slow_pages;
        if (local_cache.k_pages.ne[3] !=
                static_cast<std::int32_t>(expected_local.logical_page_count) ||
            helper_cache.k_pages.ne[3] !=
                static_cast<std::int32_t>(helper_.ownership.logical_page_count)) {
            throw std::logic_error("sequence-sharded physical cache size is inconsistent");
        }
        if (envelope.max_visible_keys < threshold_) {
            throw std::logic_error("sequence-sharded runtime cannot fall back to a missing full cache");
        }

        if(headscale_qk_ && (computation_!=qwen3_6::LongContextComputation::Complete ||
            local_cache.dtype!=DType::I8 || tokens>4))
            throw std::invalid_argument("headscale requires complete INT8 D3 target execution");
        const bool collect_guide=global_layer==63 && std::getenv("NINFER_TARGET_GUIDED_PROPOSAL_ORACLE")!=nullptr;
        const bool try_joint = joint_requested_ && !collect_guide && active_tokens == 4 &&
            populated_tokens_ >= 32768 && scale == .0625f;
        if(collect_guide) {
            if(!headscale_qk_ || !value_fp16_ || computation_!=qwen3_6::LongContextComputation::Complete)
                throw std::logic_error("target guide requires the complete headscale/direct-value path");
            if(!guide_owner_log_) {
                const std::size_t page_bytes=static_cast<std::size_t>(geometry_.logical_pages)*24*4*sizeof(float);
                check(cudaSetDevice(owner_device_),"guide owner allocation");
                guide_owner_log_=std::make_unique<DeviceBuffer>(page_bytes*4);
                guide_pages_=std::make_unique<DeviceBuffer>(page_bytes);
                guide_remote_pages_=std::make_unique<DeviceBuffer>(page_bytes);
                guide_position_=std::make_unique<DeviceBuffer>(4*sizeof(int));
                guide_host_=std::make_unique<PinnedHostBuffer>(page_bytes);
                check(cudaSetDevice(helper_device_),"guide helper allocation");
                guide_helper_log_=std::make_unique<DeviceBuffer>(page_bytes*4);
                guide_helper_pages_=std::make_unique<DeviceBuffer>(page_bytes);
            }
            check(cudaSetDevice(owner_device_),"guide owner clear");
            ops::guide_metadata::clear<<<(guide_owner_log_->bytes/4+255)/256,256,0,owner_stream>>>(
                static_cast<float*>(guide_owner_log_->p),guide_owner_log_->bytes/4);
            check(cudaSetDevice(helper_device_),"guide helper clear");
            ops::guide_metadata::clear<<<(guide_helper_log_->bytes/4+255)/256,256,0,helper_.context.stream>>>(
                static_cast<float*>(guide_helper_log_->p),guide_helper_log_->bytes/4);
            check(cudaGetLastError(),"guide clear launches");
        }
        const bool append_on_helper = tail_fast ? !owner_is_fast_ : owner_is_fast_;
        const bool owner_has_prefix = tail_fast ? !owner_is_fast_ : owner_is_fast_;
        const auto packet_started = Clock::now();
        check(cudaSetDevice(owner_device_), "cudaSetDevice sequence owner input");
        check(cudaEventRecord(owner_inputs_ready_, owner_stream),
              "cudaEventRecord sequence owner input");
        check(cudaStreamWaitEvent(owner_copy_stream_, owner_inputs_ready_, 0),
              "cudaStreamWaitEvent sequence owner input");
        const auto input_layout=packet::input_layout(active_tokens,append_on_helper);
        if(try_joint) {
            packet::pack_input<<<static_cast<unsigned>((input_layout.bytes/16+255)/256),256,0,owner_copy_stream_>>>(
                static_cast<const uint4*>(active_q.data),static_cast<const int*>(active_positions.data),
                append_on_helper?static_cast<const uint4*>(active_k.data):nullptr,
                append_on_helper?static_cast<const uint4*>(active_v.data):nullptr,
                static_cast<uint4*>(owner_input_packet_->p),active_tokens,append_on_helper);
            check(cudaGetLastError(),"pack sequence input without numerical conversion");
            check(cudaMemcpyAsync(host_input_packet_.data(),owner_input_packet_->p,input_layout.bytes,
                                  cudaMemcpyDeviceToHost,owner_copy_stream_),"sequence input packet D2H");
        } else {
            // Native QK can overlap DMA on the copy engine. Adding an input
            // packing kernel here would compete for its SM/memory resources.
            // Keep direct source copies for native, short-context and guide work.
            auto* packet_bytes=static_cast<char*>(host_input_packet_.data());
            check(cudaMemcpyAsync(packet_bytes,active_q.data,active_q.bytes(),cudaMemcpyDeviceToHost,owner_copy_stream_),"native sequence Q D2H");
            check(cudaMemcpyAsync(packet_bytes+input_layout.position_offset,active_positions.data,active_positions.bytes(),cudaMemcpyDeviceToHost,owner_copy_stream_),"native sequence positions D2H");
            if(append_on_helper) {
                check(cudaMemcpyAsync(packet_bytes+input_layout.key_offset,active_k.data,active_k.bytes(),cudaMemcpyDeviceToHost,owner_copy_stream_),"native sequence K D2H");
                check(cudaMemcpyAsync(packet_bytes+input_layout.value_offset,active_v.data,active_v.bytes(),cudaMemcpyDeviceToHost,owner_copy_stream_),"native sequence V D2H");
            }
        }
        if(headscale_qk_ && !owner_headscale_ready_[local_index]) {
            ops::headscale_qk::prepare_headscale_keys(local_cache,nullptr,populated_tokens_,owner_stream);
            owner_headscale_ready_[local_index]=true;
        }
        if (!append_on_helper) {
            ops::gqa_kv_append(active_k, active_v, active_positions, local_cache,
                               owner_stream);
            if(headscale_qk_)ops::headscale_qk::prepare_headscale_keys(local_cache,&active_positions,0,owner_stream);
        }
        if (!try_joint) {
        launch_range(owner_scratch_, active_q, active_positions, scale, local_cache,
                     envelope, owner_range, owner_stream,collect_guide?static_cast<float*>(guide_owner_log_->p):nullptr);
        }
        check(cudaStreamSynchronize(owner_copy_stream_),
              "cudaStreamSynchronize sequence input D2H");
        stats_.packet_seconds +=
            std::chrono::duration<double>(Clock::now() - packet_started).count();
        stats_.q_bytes += active_q.bytes();

        const auto helper_started = Clock::now();
        check(cudaSetDevice(helper_device_), "cudaSetDevice sequence helper execute");
        Tensor helper_q = helper_.q_tensor(active_tokens);
        Tensor helper_positions = helper_.positions_tensor(active_tokens);
        if(try_joint) {
            check(cudaMemcpyAsync(helper_.input_packet->p,host_input_packet_.data(),input_layout.bytes,
                                  cudaMemcpyHostToDevice,helper_.context.load_stream),"sequence input packet H2D");
        } else {
            const auto* packet_bytes=static_cast<const char*>(host_input_packet_.data());
            check(cudaMemcpyAsync(helper_q.data,packet_bytes,active_q.bytes(),cudaMemcpyHostToDevice,helper_.context.load_stream),"native sequence Q H2D");
            check(cudaMemcpyAsync(helper_positions.data,packet_bytes+input_layout.position_offset,active_positions.bytes(),cudaMemcpyHostToDevice,helper_.context.load_stream),"native sequence positions H2D");
            if(append_on_helper) {
                check(cudaMemcpyAsync(helper_.k_tensor(active_tokens).data,packet_bytes+input_layout.key_offset,active_k.bytes(),cudaMemcpyHostToDevice,helper_.context.load_stream),"native sequence K H2D");
                check(cudaMemcpyAsync(helper_.v_tensor(active_tokens).data,packet_bytes+input_layout.value_offset,active_v.bytes(),cudaMemcpyHostToDevice,helper_.context.load_stream),"native sequence V H2D");
            }
        }
        check(cudaEventRecord(helper_.input_ready, helper_.context.load_stream),
              "cudaEventRecord sequence helper input");
        check(cudaStreamWaitEvent(helper_.context.stream, helper_.input_ready, 0),
              "cudaStreamWaitEvent sequence helper input");
        if(headscale_qk_ && !helper_headscale_ready_[local_index]) {
            ops::headscale_qk::prepare_headscale_keys(helper_cache,nullptr,populated_tokens_,helper_.context.stream);
            helper_headscale_ready_[local_index]=true;
        }
        if (append_on_helper) {
            ops::gqa_kv_append(helper_.k_tensor(active_tokens),
                               helper_.v_tensor(active_tokens), helper_positions,
                               helper_cache, helper_.context.stream);
            if(headscale_qk_)ops::headscale_qk::prepare_headscale_keys(helper_cache,&helper_positions,0,helper_.context.stream);
        }
        if (try_joint) {
            if (!joint_) {
                joint_ = std::make_unique<ninfer::apps::dual::joint_runtime::Collection>(
                    owner_device_, helper_device_, capacity_, populated_tokens_, full_layers_,
                    local_cache, helper_cache, owner_stream, helper_.context.stream);
            }
            const int visible = reinterpret_cast<const int*>(static_cast<const char*>(host_input_packet_.data())+input_layout.position_offset)[active_tokens - 1] + 1;
            const bool used = joint_->execute(local_index, local_cache, helper_cache,
                active_q, active_positions, helper_q, helper_positions, visible,
                owner_scratch_.state_acc_tensor(active_tokens), owner_scratch_.state_m_tensor(active_tokens), owner_scratch_.state_l_tensor(active_tokens),
                helper_.scratch.state_acc_tensor(active_tokens), helper_.scratch.state_m_tensor(active_tokens), helper_.scratch.state_l_tensor(active_tokens));
            if (!used) {
                // A global selection-count, source-tail or numerical guard failure
                // recomputes BOTH original partitions. Never merge one approximate
                // partition with a stale or independently accepted peer result.
                check(cudaSetDevice(owner_device_), "joint fallback owner");
                launch_range(owner_scratch_, active_q, active_positions, scale, local_cache,
                    envelope, owner_range, owner_stream);
                check(cudaSetDevice(helper_device_), "joint fallback helper");
                launch_range(helper_.scratch, helper_q, helper_positions, scale, helper_cache,
                    envelope, helper_range, helper_.context.stream);
            }
            check(cudaSetDevice(helper_device_), "joint pipeline restore helper");
        } else {
        launch_range(helper_.scratch, helper_q, helper_positions, scale, helper_cache,
                     envelope, helper_range, helper_.context.stream,collect_guide?static_cast<float*>(guide_helper_log_->p):nullptr);
        }
        if(collect_guide) {
            const int count=geometry_.logical_pages*24*active_tokens;
            ops::guide_metadata::pages<<<(count+255)/256,256,0,helper_.context.stream>>>(
                static_cast<const float*>(guide_helper_log_->p),nullptr,static_cast<float*>(guide_helper_pages_->p),count);
            check(cudaMemcpyAsync(guide_host_->data(),guide_helper_pages_->p,count*sizeof(float),
                  cudaMemcpyDeviceToHost,helper_.context.stream),"guide helper pages D2H");
        }
        download_helper_state(active_tokens);
        stats_.helper_seconds +=
            std::chrono::duration<double>(Clock::now() - helper_started).count();

        const auto transfer_started = Clock::now();
        upload_helper_state(active_tokens, owner_stream);
        if(collect_guide) {
            // The existing owner-input event covers the previous round's guide
            // H2D. Its synchronized input copy precedes this helper D2H, protecting
            // pinned-buffer reuse without an additional host synchronization.
            const int count=geometry_.logical_pages*24*active_tokens;
            check(cudaMemcpyAsync(guide_remote_pages_->p,guide_host_->data(),count*sizeof(float),
                  cudaMemcpyHostToDevice,owner_stream),"guide helper pages H2D");
            ops::guide_metadata::pages<<<(count+255)/256,256,0,owner_stream>>>(
                static_cast<const float*>(guide_owner_log_->p),static_cast<const float*>(guide_remote_pages_->p),
                static_cast<float*>(guide_pages_->p),count);
            check(cudaMemcpyAsync(guide_position_->p,active_positions.data,active_tokens*sizeof(int),cudaMemcpyDeviceToDevice,owner_stream),"guide anchor position");
            check(cudaGetLastError(),"guide page merge");
            guide_columns_=active_tokens;
            guide_ready_=true;
            ++guide_captures_;
        }

        stats_.state_transfer_seconds +=
            std::chrono::duration<double>(Clock::now() - transfer_started).count();

        const auto merge_started = Clock::now();
        check(cudaSetDevice(owner_device_), "cudaSetDevice sequence merge");
        Tensor owner_acc = owner_scratch_.state_acc_tensor(active_tokens);
        Tensor owner_m = owner_scratch_.state_m_tensor(active_tokens);
        Tensor owner_l = owner_scratch_.state_l_tensor(active_tokens);
        Tensor helper_acc = owner_scratch_.uploaded_acc_tensor(active_tokens);
        Tensor helper_m = owner_scratch_.uploaded_m_tensor(active_tokens);
        Tensor helper_l = owner_scratch_.uploaded_l_tensor(active_tokens);
        if (owner_has_prefix) {
            ops::detail::gqa_attention_small_t_merge_two_states_launch(
                TextConfig::query_heads, active_tokens, owner_acc, owner_m, owner_l,
                helper_acc, helper_m, helper_l, active_out, owner_stream);
        } else {
            ops::detail::gqa_attention_small_t_merge_two_states_launch(
                TextConfig::query_heads, active_tokens, helper_acc, helper_m, helper_l,
                owner_acc, owner_m, owner_l, active_out, owner_stream);
        }
        stats_.merge_seconds +=
            std::chrono::duration<double>(Clock::now() - merge_started).count();
        ++stats_.calls;
        ++stats_.parallel_calls;
        if(headscale_qk_)++stats_.headscale_calls;
        if(value_fp16_)++stats_.value_fp16_calls;
        stats_.routed_tokens += static_cast<std::uint64_t>(active_tokens);
        stats_.state_bytes += helper_.scratch.state_acc_tensor(active_tokens).bytes() +
                              2U * helper_.scratch.state_m_tensor(active_tokens).bytes();
        return true;
    }

private:
    void launch_range(DeviceScratch& scratch, const Tensor& q,
                             const Tensor& positions, float scale,
                             const PagedKVLayerView& cache,
                             ops::GqaExecutionEnvelope envelope, SplitRange range,
                             cudaStream_t stream,float* guide_log=nullptr) {
        const std::int32_t subdivision = scratch.subdivision;
        const std::int32_t compute_total = range.total * subdivision;
        const std::int32_t compute_begin = range.begin * subdivision;
        const std::int32_t compute_count = range.count * subdivision;
        Tensor partial_acc = scratch.partial_acc_tensor(q.ne[2], compute_total);
        Tensor partial_m = scratch.partial_m_tensor(q.ne[2], compute_total);
        Tensor partial_l = scratch.partial_l_tensor(q.ne[2], compute_total);
        if(headscale_qk_) {
            if(subdivision!=1)throw std::logic_error("headscale must not prune or oversplit target keys");
            if(guide_log!=nullptr) {
                ops::guide_headscale::launch_headscale_range(q,positions,scale,cache,range.begin,range.count,
                    range.total,capacity_,partial_acc,partial_m,partial_l,stream,value_fp16_,guide_log,geometry_.logical_pages*4);
            } else {
            ops::headscale_qk::launch_headscale_range(q,positions,scale,cache,range.begin,range.count,
                range.total,capacity_,partial_acc,partial_m,partial_l,stream,value_fp16_);
            }
        } else {
        ops::detail::gqa_attention_cached_small_t_partial_range_launch(
            q, positions, scale, cache, envelope, range.begin, range.count,
            partial_acc, partial_m, partial_l, stream, subdivision,
            computation_==qwen3_6::LongContextComputation::Complete ? static_cast<std::int32_t>(capacity_) : 0);
        }
        Tensor state_acc = scratch.state_acc_tensor(q.ne[2]);
        Tensor state_m = scratch.state_m_tensor(q.ne[2]);
        Tensor state_l = scratch.state_l_tensor(q.ne[2]);
        ops::detail::gqa_attention_small_t_reduce_range_state_launch(
            TextConfig::query_heads, q.ne[2], compute_total, compute_begin, compute_count,
            partial_acc, partial_m, partial_l, state_acc, state_m, state_l, stream);
    }

    void download_helper_state(std::int32_t tokens) {
        check(cudaMemcpyAsync(host_state_packet_.data(),helper_.scratch.state_packet->p,
                              packet::state_layout(tokens).bytes,cudaMemcpyDeviceToHost,helper_.context.stream),
              "sequence complete state packet D2H");
        helper_.context.synchronize();
    }

    void upload_helper_state(std::int32_t tokens,cudaStream_t owner_stream) {
        check(cudaSetDevice(owner_device_),"sequence state upload device");
        check(cudaMemcpyAsync(owner_scratch_.uploaded_packet->p,host_state_packet_.data(),
                              packet::state_layout(tokens).bytes,cudaMemcpyHostToDevice,owner_copy_stream_),
              "sequence complete state packet H2D");
        check(cudaEventRecord(owner_state_ready_,owner_copy_stream_),"sequence state upload ready");
        check(cudaStreamWaitEvent(owner_stream,owner_state_ready_,0),"sequence state upload dependency");
    }

    int owner_device_ = 0;
    int helper_device_ = 0;
    bool owner_is_fast_ = false;
    std::int32_t layer_begin_ = 0;
    std::int32_t layer_end_ = 0;
    std::int32_t full_begin_ = 0;
    std::int32_t full_layers_ = 0;
    std::uint32_t capacity_ = 0;
    std::uint32_t threshold_ = 0;
    SequenceShardGeometry geometry_{};
    const qwen3_6::LongContextComputation computation_;
    const bool headscale_qk_;
    const bool value_fp16_;
    std::array<bool,16> owner_headscale_ready_{};
    std::array<bool,16> helper_headscale_ready_{};
    KvCacheStorage storage_   = KvCacheStorage::Int8Group64;
    DType kv_dtype_           = DType::I8;
    std::int32_t quant_group_ = qwen3_6::kKvQuantGroup;
    double fast_fraction_ = 0.60;
    DeviceScratch owner_scratch_;
    HelperDevice helper_;
    std::unique_ptr<DeviceBuffer> owner_input_packet_;
    PinnedHostBuffer host_input_packet_;
    PinnedHostBuffer host_state_packet_;
    cudaStream_t owner_copy_stream_ = nullptr;
    cudaEvent_t owner_inputs_ready_ = nullptr;
    cudaEvent_t owner_state_ready_ = nullptr;
    std::uint32_t populated_tokens_ = 0;
    std::int32_t active_storage_tokens_ = 0;
    std::int32_t active_tokens_ = 0;
    bool ready_ = false;
    std::unique_ptr<DeviceBuffer> guide_owner_log_,guide_helper_log_,guide_pages_,guide_remote_pages_,guide_helper_pages_,guide_position_;
    std::unique_ptr<PinnedHostBuffer> guide_host_;
    bool guide_ready_=false;
    int guide_columns_=0;
    std::uint64_t guide_captures_=0;
    mutable std::uint64_t guide_reads_=0;
    const bool joint_requested_ = std::getenv("NINFER_JOINT_KV_RESIDUAL_ORACLE") != nullptr;
    std::unique_ptr<ninfer::apps::dual::joint_runtime::Collection> joint_;
    ContextParallelGqaStats stats_{};
};

} // namespace

SequenceShardGeometry plan_sequence_sharded_kv(std::uint32_t capacity,
                                               KvCacheStorage storage,
                                               double fast_fraction,
                                               std::uint32_t verifier_width) {
    if (capacity == 0 || verifier_width == 0 || verifier_width > 6 ||
        !(fast_fraction > 0.0 && fast_fraction < 1.0)) {
        throw std::invalid_argument("sequence-sharded page geometry is invalid");
    }
    SequenceShardGeometry out;
    out.storage       = storage;
    out.dtype         = storage_dtype(storage);
    out.quant_group   = storage_quant_group(storage);
    out.logical_pages = logical_pages(capacity);
    out.total_splits = ops::detail::gqa_attention_split_capacity(
        TextConfig::query_heads, static_cast<std::int32_t>(verifier_width), out.dtype,
        {capacity, capacity});
    if (out.total_splits < 2) {
        throw std::logic_error("sequence-sharded page geometry needs two split ranges");
    }
    out.fast_splits = std::clamp(
        static_cast<std::int32_t>(std::llround(out.total_splits * fast_fraction)),
        1, out.total_splits - 1);
    out.slow_splits = out.total_splits - out.fast_splits;
    const std::int32_t logical_tiles =
        (static_cast<std::int32_t>(capacity) + kKeyTile - 1) / kKeyTile;
    const bool tile_split = logical_tiles >= out.total_splits;
    const std::int32_t units = tile_split
        ? (logical_tiles + out.total_splits - 1) / out.total_splits
        : (static_cast<std::int32_t>(capacity) + out.total_splits - 1) /
              out.total_splits;
    out.split_unit_keys = units * (tile_split ? kKeyTile : 1);
    const bool tail_fast = tail_fast_ownership_enabled(capacity);
    const std::int32_t prefix_splits = tail_fast ? out.slow_splits : out.fast_splits;
    out.boundary_key = std::min(static_cast<std::int32_t>(capacity),
                                prefix_splits * out.split_unit_keys);
    if (out.boundary_key <= 0 || out.boundary_key >= static_cast<std::int32_t>(capacity)) {
        throw std::logic_error("sequence-sharded boundary is empty");
    }
    const std::uint32_t boundary_page =
        static_cast<std::uint32_t>(out.boundary_key) /
        static_cast<std::uint32_t>(kPagedKVPageSize);
    const std::uint32_t boundary_offset =
        static_cast<std::uint32_t>(out.boundary_key) %
        static_cast<std::uint32_t>(kPagedKVPageSize);
    const std::uint32_t prefix_pages =
        boundary_page + (boundary_offset != 0 ? 1U : 0U);
    const std::uint32_t suffix_begin = boundary_page;
    const TextKvPageOwnership prefix{
        .enabled = true,
        .logical_page_begin = 0,
        .logical_page_count = prefix_pages,
    };
    const TextKvPageOwnership suffix{
        .enabled = true,
        .logical_page_begin = suffix_begin,
        .logical_page_count = out.logical_pages - suffix_begin,
    };
    out.fast_pages = tail_fast ? suffix : prefix;
    out.slow_pages = tail_fast ? prefix : suffix;
    out.duplicate_boundary_pages = boundary_offset != 0 ? 1U : 0U;
    return out;
}

class ContextParallelGqaCoordinator::Impl {
public:
    Impl(int fast_device, int slow_device, std::uint32_t capacity,
         std::int32_t split_layer, KvCacheStorage storage, std::uint32_t threshold,
         std::uint32_t verifier_width, double fast_fraction,qwen3_6::LongContextComputation computation,bool headscale_qk,bool value_fp16)
        : storage(storage),
          geometry(plan_sequence_sharded_kv(capacity, storage, fast_fraction, verifier_width)),
          first(fast_device, slow_device, true, 0, split_layer, capacity, threshold,
                geometry, fast_fraction, computation,headscale_qk,value_fp16),
          second(slow_device, fast_device, false, split_layer, TextConfig::layers,
                 capacity, threshold, geometry, fast_fraction, computation,headscale_qk,value_fp16) {}

    KvCacheStorage storage;
    SequenceShardGeometry geometry;
    StageEngine first;
    StageEngine second;
    std::uint32_t populated_tokens = 0;
};

ContextParallelGqaCoordinator::ContextParallelGqaCoordinator(
    int fast_device, int slow_device, std::uint32_t capacity,
    std::int32_t split_layer, KvCacheStorage storage, std::uint32_t parallel_threshold,
    std::uint32_t verifier_width, double fast_fraction,qwen3_6::LongContextComputation computation,bool headscale_qk,bool value_fp16)
    : impl_(nullptr) {
    if(value_fp16 && (!headscale_qk || computation!=qwen3_6::LongContextComputation::Complete ||
                       storage!=KvCacheStorage::Int8Group64 || verifier_width!=4)) {
        throw std::invalid_argument("packed value arithmetic requires complete headscale INT8 D3 context");
    }
    impl_=std::make_unique<Impl>(fast_device,slow_device,capacity,split_layer,storage,
        parallel_threshold,verifier_width,fast_fraction,computation,headscale_qk,value_fp16);
}

ContextParallelGqaCoordinator::~ContextParallelGqaCoordinator() = default;

bool ContextParallelGqaCoordinator::execute(
    std::int32_t global_layer, const Tensor& q, const Tensor& k, const Tensor& v,
    const Tensor& positions, const Tensor& valid_columns, const Tensor& kv_table_rows,
    float scale, PagedKVBatchLayerView local_cache, ops::GqaExecutionEnvelope envelope,
    WorkspaceArena&, Tensor& out, cudaStream_t owner_stream) {
    StageEngine* engine = impl_->first.owns(global_layer) ? &impl_->first
                         : impl_->second.owns(global_layer) ? &impl_->second
                                                           : nullptr;
    return engine != nullptr &&
           engine->execute(global_layer, q, k, v, positions, valid_columns, kv_table_rows,
                           scale, local_cache, envelope, out, owner_stream);
}

const SequenceShardGeometry& ContextParallelGqaCoordinator::geometry() const noexcept {
    return impl_->geometry;
}

TextKvPageOwnership ContextParallelGqaCoordinator::helper_ownership(
    std::uint32_t stage) const {
    if (stage == 0) { return impl_->first.helper_ownership(); }
    if (stage == 1) { return impl_->second.helper_ownership(); }
    throw std::out_of_range("sequence-sharded stage index is invalid");
}

void ContextParallelGqaCoordinator::load_legacy_helper_pages(
    std::uint32_t stage, std::span<const std::uint8_t> packed_planes) {
    if (stage == 0) { impl_->first.load_helper(packed_planes); return; }
    if (stage == 1) { impl_->second.load_helper(packed_planes); return; }
    throw std::out_of_range("sequence-sharded stage index is invalid");
}

void ContextParallelGqaCoordinator::activate_legacy_prefix(
    std::uint32_t populated_tokens) {
    impl_->first.activate_prefix(populated_tokens);
    impl_->second.activate_prefix(populated_tokens);
    impl_->populated_tokens = populated_tokens;
}

bool ContextParallelGqaCoordinator::decode_ready() const noexcept {
    return impl_ != nullptr && impl_->first.ready() && impl_->second.ready();
}

bool ContextParallelGqaCoordinator::requires_prefix_preparation() const noexcept {
    return impl_ && (impl_->first.requires_prefix_preparation() || impl_->second.requires_prefix_preparation());
}

void ContextParallelGqaCoordinator::prepare_prefix_layer(
    std::int32_t global_layer, PagedKVLayerView local_cache, std::uint32_t populated_tokens,
    cudaStream_t owner_stream) {
    StageEngine* engine = impl_->first.owns(global_layer) ? &impl_->first
                         : impl_->second.owns(global_layer) ? &impl_->second : nullptr;
    if (!engine) { throw std::invalid_argument("prefix layer is outside both stage ranges"); }
    engine->prepare_prefix_layer(global_layer,local_cache,populated_tokens,owner_stream);
}

bool ContextParallelGqaCoordinator::proposal_page_guide(Tensor& score,Tensor& pos) const {
    return impl_!=nullptr && impl_->second.guide(score,pos);
}

ContextParallelGqaStats ContextParallelGqaCoordinator::stats() const noexcept {
    const ContextParallelGqaStats a = impl_->first.stats();
    const ContextParallelGqaStats b = impl_->second.stats();
    return {
        .calls = a.calls + b.calls,
        .parallel_calls = a.parallel_calls + b.parallel_calls,
        .headscale_calls = a.headscale_calls + b.headscale_calls,
        .value_fp16_calls = a.value_fp16_calls + b.value_fp16_calls,
        .routed_tokens = a.routed_tokens + b.routed_tokens,
        .state_bytes = a.state_bytes + b.state_bytes,
        .q_bytes = a.q_bytes + b.q_bytes,
        .physical_helper_bytes = impl_->first.helper_bytes() + impl_->second.helper_bytes(),
        .fast_pages = impl_->geometry.fast_pages.logical_page_count,
        .slow_pages = impl_->geometry.slow_pages.logical_page_count,
        .duplicate_boundary_pages = impl_->geometry.duplicate_boundary_pages,
        .packet_seconds = a.packet_seconds + b.packet_seconds,
        .helper_seconds = a.helper_seconds + b.helper_seconds,
        .state_transfer_seconds = a.state_transfer_seconds + b.state_transfer_seconds,
        .merge_seconds = a.merge_seconds + b.merge_seconds,
    };
}

std::size_t ContextParallelGqaCoordinator::prefix_snapshot_bytes() const noexcept {
    return impl_ == nullptr ? 0 : sizeof(SnapshotHeader) + impl_->first.helper_bytes() +
                                      impl_->second.helper_bytes();
}

void ContextParallelGqaCoordinator::copy_prefix_snapshot_to_host(void* destination,
                                                                  std::size_t bytes) {
    const std::size_t expected = prefix_snapshot_bytes();
    if (impl_ == nullptr || destination == nullptr || bytes != expected ||
        !decode_ready()) {
        throw std::invalid_argument("invalid sequence-sharded snapshot download");
    }
    SnapshotHeader header;
    header.version          = snapshot_version(impl_->storage);
    header.populated_tokens = impl_->populated_tokens;
    header.first_bytes = impl_->first.helper_bytes();
    header.second_bytes = impl_->second.helper_bytes();
    auto* host = static_cast<std::uint8_t*>(destination);
    std::memcpy(host, &header, sizeof(header));
    impl_->first.copy_snapshot_to_host(host + sizeof(header), header.first_bytes);
    impl_->second.copy_snapshot_to_host(host + sizeof(header) + header.first_bytes,
                                        header.second_bytes);
}

void ContextParallelGqaCoordinator::copy_prefix_snapshot_from_host(
    const void* source, std::size_t bytes) {
    const std::size_t expected = prefix_snapshot_bytes();
    if (impl_ == nullptr || source == nullptr || bytes != expected) {
        throw std::invalid_argument("invalid sequence-sharded snapshot upload");
    }
    SnapshotHeader header;
    const auto* host = static_cast<const std::uint8_t*>(source);
    std::memcpy(&header, host, sizeof(header));
    if (header.magic != kSnapshotMagic || header.version != snapshot_version(impl_->storage) ||
        header.first_bytes != impl_->first.helper_bytes() ||
        header.second_bytes != impl_->second.helper_bytes()) {
        throw std::runtime_error("sequence-sharded snapshot contract mismatch");
    }
    impl_->first.copy_snapshot_from_host(host + sizeof(header), header.first_bytes);
    impl_->second.copy_snapshot_from_host(host + sizeof(header) + header.first_bytes,
                                          header.second_bytes);
    activate_legacy_prefix(header.populated_tokens);
}

} // namespace ninfer::targets::qwen3_6_27b::detail
