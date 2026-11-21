#pragma once

#include "targets/qwen3_6/impl/runtime/mlp_execution_override.h"

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <memory>

namespace ninfer::apps::dual {

struct Tp2MlpStats {
    std::uint64_t calls = 0;
    std::uint64_t width4_calls = 0;
    std::uint64_t width5_calls = 0;
    std::uint64_t input_transfer_bytes = 0;
    std::uint64_t partial_transfer_bytes = 0;
    double input_seconds = 0.0;
    double engine_seconds = 0.0;
    double reduction_seconds = 0.0;
    std::size_t fast_device_bytes = 0;
    std::size_t slow_device_bytes = 0;
    std::size_t fast_canonical_alias_bytes = 0;
    std::size_t slow_canonical_alias_bytes = 0;
};

class Tp2MlpCoordinator final : public targets::qwen3_6::detail::MlpExecutionOverride {
public:
    Tp2MlpCoordinator(std::filesystem::path weights_root, int fast_device, int slow_device,
                      std::int32_t split_layer);
    ~Tp2MlpCoordinator() override;

    Tp2MlpCoordinator(const Tp2MlpCoordinator&) = delete;
    Tp2MlpCoordinator& operator=(const Tp2MlpCoordinator&) = delete;

    bool execute(std::int32_t global_layer, const Tensor& normalized_hidden,
                 const Weight& canonical_gate_up, const Weight& canonical_down,
                 Tensor& residual, cudaStream_t owner_stream) override;

    [[nodiscard]] Tp2MlpStats stats() const noexcept;

private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace ninfer::apps::dual
