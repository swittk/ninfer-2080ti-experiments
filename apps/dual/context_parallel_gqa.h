#pragma once

#include "targets/qwen3_6/impl/runtime/gqa_execution_override.h"
#include "targets/qwen3_6_27b/impl/dual_stage.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>

namespace ninfer::targets::qwen3_6_27b::detail {

struct SequenceShardGeometry {
    KvCacheStorage storage       = KvCacheStorage::Int8Group64;
    DType dtype                  = DType::I8;
    std::int32_t quant_group     = 0;
    std::uint32_t logical_pages = 0;
    std::int32_t total_splits = 0;
    std::int32_t fast_splits = 0;
    std::int32_t slow_splits = 0;
    std::int32_t split_unit_keys = 0;
    std::int32_t boundary_key = 0;
    TextKvPageOwnership fast_pages{};
    TextKvPageOwnership slow_pages{};
    std::uint32_t duplicate_boundary_pages = 0;
};

[[nodiscard]] SequenceShardGeometry plan_sequence_sharded_kv(std::uint32_t capacity,
                                                             KvCacheStorage storage,
                                                             double fast_fraction         = 0.60,
                                                             std::uint32_t verifier_width = 4);

struct ContextParallelGqaStats {
    std::uint64_t calls = 0;
    std::uint64_t parallel_calls = 0;
    std::uint64_t headscale_calls = 0;
    std::uint64_t value_fp16_calls = 0;
    std::uint64_t routed_tokens = 0;
    std::uint64_t state_bytes = 0;
    std::uint64_t q_bytes = 0;
    std::uint64_t physical_helper_bytes = 0;
    std::uint32_t fast_pages = 0;
    std::uint32_t slow_pages = 0;
    std::uint32_t duplicate_boundary_pages = 0;
    double packet_seconds = 0.0;
    double helper_seconds = 0.0;
    double state_transfer_seconds = 0.0;
    double merge_seconds = 0.0;
};

class ContextParallelGqaCoordinator final
    : public qwen3_6::detail::GqaExecutionOverride {
public:
    ContextParallelGqaCoordinator(int fast_device, int slow_device, std::uint32_t capacity,
                                  std::int32_t split_layer, KvCacheStorage storage,
                                  std::uint32_t parallel_threshold = 32768,
                                  std::uint32_t verifier_width = 4, double fast_fraction = 0.60,
                                  qwen3_6::LongContextComputation computation =
                                      qwen3_6::LongContextComputation::RetainedThroughput,
                                  bool headscale_qk = false,bool value_fp16 = false);
    ~ContextParallelGqaCoordinator() override;

    ContextParallelGqaCoordinator(const ContextParallelGqaCoordinator&) = delete;
    ContextParallelGqaCoordinator& operator=(const ContextParallelGqaCoordinator&) = delete;

    bool execute(std::int32_t global_layer, const Tensor& q, const Tensor& k,
                 const Tensor& v, const Tensor& positions, const Tensor& valid_columns,
                 const Tensor& kv_table_rows, float scale, PagedKVBatchLayerView local_cache,
                 ops::GqaExecutionEnvelope envelope, WorkspaceArena& workspace, Tensor& out,
                 cudaStream_t owner_stream) override;

    bool requires_prefix_preparation() const noexcept override;
    void prepare_prefix_layer(std::int32_t global_layer, PagedKVLayerView local_cache,
                              std::uint32_t populated_tokens, cudaStream_t owner_stream) override;

    bool proposal_page_guide(Tensor& log_mass,Tensor& anchor_position) const override;

    [[nodiscard]] const SequenceShardGeometry& geometry() const noexcept;
    [[nodiscard]] TextKvPageOwnership helper_ownership(std::uint32_t stage) const;
    void load_legacy_helper_pages(std::uint32_t stage,
                                  std::span<const std::uint8_t> packed_planes);
    void activate_legacy_prefix(std::uint32_t populated_tokens);
    [[nodiscard]] bool decode_ready() const noexcept;

    [[nodiscard]] ContextParallelGqaStats stats() const noexcept;
    [[nodiscard]] std::size_t prefix_snapshot_bytes() const noexcept;
    void copy_prefix_snapshot_to_host(void* destination, std::size_t bytes);
    void copy_prefix_snapshot_from_host(const void* source, std::size_t bytes);

private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace ninfer::targets::qwen3_6_27b::detail
