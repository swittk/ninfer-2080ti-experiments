#pragma once

// Product-side adapter between HTTP protocol requests and the public NInfer
// engine. It owns independent per-device Engines and keeps protocol concerns (aliases, usage,
// streaming callbacks, and tool-call parsing) outside the target package.

#include "ninfer/engine.h"
#include "serve/request.h"
#include "serve/replica_router.h"
#include "serve/serve_options.h"

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace ninfer::serve {

struct RequestLifetime;
struct RequestCapacity;
struct MediaInputCapacity;

struct StreamEmission {
    std::size_t content_bytes = 0;
    std::uint32_t committed_tokens = 0;
};

struct GenerationMetrics {
    // Optional, bounded by the output budget; emitted after successful writes.
    // The HTTP benchmark pairs these with client timestamps by UTF-8 byte count.
    std::vector<StreamEmission> stream_trace;
    double prepare_seconds = 0.0;
    double ttft_seconds    = 0.0;
    double vision_seconds  = 0.0;
    double prefill_seconds = 0.0;
    double decode_seconds  = 0.0;
    double total_seconds   = 0.0;

    SpeculativeBackend speculative_backend    = SpeculativeBackend::None;
    std::uint32_t speculative_draft_window    = 0;
    std::uint64_t speculative_rounds          = 0;
    std::uint64_t speculative_draft_tokens    = 0;
    std::uint64_t speculative_accepted_tokens = 0;
    std::uint64_t speculative_fallback_steps  = 0;
    std::uint64_t session_copy_rounds = 0;
    std::uint64_t session_copy_drafted_tokens = 0;
    std::uint64_t session_copy_accepted_tokens = 0;
    std::uint64_t session_copy_factory_resumes = 0;
    std::vector<std::uint64_t> speculative_accepted_per_position;
    std::uint32_t prefix_cache_hit_tokens     = 0;
    ninfer::PrefixReusePath prefix_reuse_path = ninfer::PrefixReusePath::FullReset;
};

struct GenerationOutcome {
    std::string text;
    std::string reasoning;
    std::vector<ToolCall> tool_calls;
    int prompt_tokens                  = 0;
    int completion_tokens              = 0;
    int reasoning_tokens               = 0;
    std::size_t streamed_content_bytes = 0;
    ninfer::FinishReason finish_reason = ninfer::FinishReason::OutputLimit;
    GenerationMetrics metrics;
};

struct StreamSink {
    std::function<void(const std::string& delta_text)> on_content;
    std::function<void(const std::string& delta_text)> on_reasoning;
    std::function<bool()> is_cancelled;
};

// Preparation ends by synchronously submitting the owning prompt to the Engine FIFO. The returned
// request keeps its ingress/response lifetime reservation until the HTTP response is released and
// is consumed exactly once by run().
struct PreparedRequest {
    // Declared first so an abandoned handle cancels before its routing lease is released.
    std::unique_ptr<ReplicaRouter::Reservation> replica_reservation;
    ninfer::GenerationHandle generation;
    std::size_t replica_index = 0;
    int device = 0;
    ninfer::ResolvedSamplingParameters sampling;
    double prepare_seconds                 = 0.0;
    int prompt_tokens                      = 0;
    bool include_usage                     = false;
    bool tool_capable                      = false;
    std::size_t tool_name_max_length       = 64;
    bool enable_thinking                   = true;
    bool preserve_thinking                 = false;
    bool preserve_thinking_semantic_change = false;
    std::shared_ptr<RequestLifetime> lifetime;
};

class GenerationService {
public:
    explicit GenerationService(ServeOptions options, LoadProgress load_progress = {});

    [[nodiscard]] const ServeOptions& options() const noexcept { return options_; }

    [[nodiscard]] ninfer::LoadSummary load_summary() const { return engines_.front()->load_summary(); }

    [[nodiscard]] ninfer::MemorySummary memory_summary(std::size_t replica = 0) const {
        return engines_.at(replica)->memory_summary();
    }

    [[nodiscard]] ninfer::RuntimeStats runtime_stats() const;

    [[nodiscard]] ninfer::ModelSamplingDefaults sampling_defaults() const {
        return engines_.front()->sampling_defaults();
    }

    [[nodiscard]] PreparedRequest prepare(const GenerationRequest& req,
                                          std::function<bool()> is_cancelled = {}) const;
    [[nodiscard]] int count_prompt_tokens(const GenerationRequest& req,
                                          std::function<bool()> is_cancelled = {}) const;

    // Consumes prepared.generation. A PreparedRequest is single-use.
    GenerationOutcome run(PreparedRequest& prepared, const StreamSink* sink,
                          std::function<bool()> is_cancelled = {});

    void warmup();

private:
    [[nodiscard]] std::shared_ptr<RequestLifetime> acquire_request_lifetime() const;
    [[nodiscard]] HostInputLease
    acquire_media_input(std::chrono::steady_clock::time_point deadline,
                        const std::function<bool()>& is_cancelled, std::size_t replica = 0) const;

    ServeOptions options_;
    std::vector<std::unique_ptr<ninfer::Engine>> engines_;
    std::unique_ptr<ReplicaRouter> replica_router_;
    ninfer::PromptCapabilities prompt_capabilities_;
    std::shared_ptr<RequestCapacity> request_capacity_;
    std::vector<std::shared_ptr<MediaInputCapacity>> media_input_capacities_;
};

} // namespace ninfer::serve
