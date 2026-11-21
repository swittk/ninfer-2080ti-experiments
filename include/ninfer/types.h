#pragma once

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <functional>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace ninfer {

using TokenId = std::int32_t;

inline constexpr std::uint32_t kMaximumConcurrency = 8;

enum class KvCacheStorage : std::uint8_t {
    BFloat16,
    Int8Group64,
};

enum class KvCapacityMode : std::uint8_t {
    Explicit,
    Automatic,
};

inline constexpr std::size_t kDefaultKvCapacityHeadroomBytes = 1024ULL * 1024ULL * 1024ULL;

struct KvCapacityPolicy {
    KvCapacityMode mode                  = KvCapacityMode::Explicit;
    std::uint32_t explicit_tokens        = 2048;
    std::size_t automatic_headroom_bytes = 0;

    [[nodiscard]] static constexpr KvCapacityPolicy
    explicit_capacity(std::uint32_t tokens) noexcept {
        return KvCapacityPolicy{KvCapacityMode::Explicit, tokens, 0};
    }

    [[nodiscard]] static constexpr KvCapacityPolicy
    automatic(std::size_t headroom_bytes = kDefaultKvCapacityHeadroomBytes) noexcept {
        return KvCapacityPolicy{KvCapacityMode::Automatic, 0, headroom_bytes};
    }
};

enum class ProposalHead : std::uint8_t {
    Full,
    Optimized,
};

enum class SpeculativeBackend : std::uint8_t {
    None,
    Mtp,
    DFlash,
};

struct SpeculativeOptions {
    SpeculativeBackend backend = SpeculativeBackend::None;
    std::uint32_t draft_tokens = 0;
    ProposalHead proposal_head = ProposalHead::Full;
    // Pin target verification to the complete native vocabulary even when an
    // optimized draft head is selected. False preserves existing profile behavior.
    bool force_full_target_head = false;
    // SM75 C1 research profile: rank drafts over all native vocabulary rows with
    // the existing fast projection, while retaining full target verification.
    bool fast_full_vocab_draft = false;
    // Explicit experimental composition. The retained target policy is not
    // inferred from prepared-cache existence or silently selected by the drafter.
    bool retained_target_full_vocab_draft = false;
    // Explicit retained C1/D3 policy: accept a greedy draft whose native target
    // score trails the selected target by at most this delta. -1 is strict.
    // This changes decoding policy; it is not exact-greedy equivalence.
    float greedy_margin_delta = -1.0F;
    // Accepted historical draft-only packed-prefix interpretation, not an exact
    // logical row slice. The selected target verifier remains the authority.
    bool retained_mtp_packed_prefix = false;
    // Opt-in two-limb projections, including shared-weight D3 widths up to 16.
    // Does not relax acceptance; wider capacity is reserved during preparation.
    bool sm75_radix_projections = false;
    // Exact resident permutation; requires radix projections.
    bool sm75_resident_radix = false;
    // Experimental numerical whole-MLP representation. Fixed nonuniform 3-bit
    // levels with retained A8G64 boundaries; source artifact and acceptance stay unchanged.
    bool sm75_nu3_mlp = false;
    // Opt-in target candidate coverage: score current proposals with the native
    // target head and merge them with its existing shortlist before argmax.
    // This is not a full-vocabulary argmax certificate or relaxed acceptance.
    bool target_draft_union = false;
    // Request-local exact-history drafting. Fixed C1/D3/full-target contract;
    // no copied token becomes output until the unchanged target accepts it.
    bool session_copy_draft = false;
};

struct LoadProgress {
    std::function<void(std::string_view phase, std::uint64_t done, std::uint64_t total)> callback;
};

struct EngineOptions {
    std::filesystem::path artifact_path;
    int device                         = 0;
    std::uint32_t max_context          = 2048; // Exact logical ceiling of each request.
    KvCapacityPolicy kv_capacity       = KvCapacityPolicy::explicit_capacity(2048);
    std::uint32_t max_concurrency      = 1;
    std::uint32_t max_pending_requests = 16;
    std::uint32_t pending_timeout_ms   = 30000;
    std::uint32_t prefill_chunk        = 1024;
    KvCacheStorage kv_cache            = KvCacheStorage::BFloat16;
    SpeculativeOptions speculative;
    bool enable_vision  = false;
    bool use_cuda_graph = true;
    LoadProgress load_progress;
    // At most this many decode rounds between substantial prefill chunks (1..64).
    // Genuinely short suffixes retain one-round alternation. Larger values
    // protect active streams but can increase a cold request's TTFT.
    // Long-prefill burst protects requests already streaming at the incoming request's arrival.
    std::uint32_t prefill_decode_rounds = 1;
    // Optional timing-aware burst target for each established stream. Zero keeps
    // fixed-round scheduling. This is a measured target, not a hard SLA; the
    // prefill_decode_rounds cap always bounds delay of the incoming request.
    double prefill_target_decode_tps = 0.0;
    // Optional smaller GPU work unit while serving requests that were already
    // streaming at this request's arrival. Zero inherits prefill_chunk. Memory
    // is always planned for prefill_chunk; service admission uses the smaller bound.
    std::uint32_t prefill_live_chunk = 0;
    // Maximum simultaneous staged prefill owners; all share the already planned
    // request-transient allocation. One retains the historical scheduling policy.
    std::uint32_t max_prefill_lanes = 1;
    // Compatible nonfinal text chunks may share dense projection work. Whole
    // existing work units are packed within prefill_chunk, without splitting
    // the all-cold reference or changing decoder acceptance.
    bool pack_prefill = false;
};

enum class SamplingMode : std::uint8_t {
    Thinking,
    NonThinking,
};

// Immutable model-owned values used when a request does not override a sampling field. Seed is
// deliberately excluded: it is an execution choice rather than a model recommendation.
struct SamplingPreset {
    float temperature       = 0.0F;
    std::int32_t top_k      = 0;
    float top_p             = 1.0F;
    float min_p             = 0.0F;
    float presence_penalty  = 0.0F;
    float frequency_penalty = 0.0F;
};

struct ModelSamplingDefaults {
    SamplingPreset thinking;
    SamplingPreset non_thinking;

    [[nodiscard]] constexpr const SamplingPreset& for_mode(SamplingMode mode) const noexcept {
        return mode == SamplingMode::Thinking ? thinking : non_thinking;
    }
};

// Public request-side overrides. std::nullopt means "use the registered model/mode default";
// explicit zero remains a real override (including temperature=0 for exact argmax).
struct SamplingOverrides {
    std::optional<float> temperature;
    std::optional<std::int32_t> top_k;
    std::optional<float> top_p;
    std::optional<float> min_p;
    std::optional<float> presence_penalty;
    std::optional<float> frequency_penalty;
    std::optional<std::uint64_t> seed;
};

// Complete parameters after Engine resolution. Target runtimes consume only this type.
struct ResolvedSamplingParameters {
    float temperature       = 0.0F;
    std::int32_t top_k      = 0;
    float top_p             = 1.0F;
    float min_p             = 0.0F;
    float presence_penalty  = 0.0F;
    float frequency_penalty = 0.0F;
    std::uint64_t seed      = 0;
};

enum class OutputChannel : std::uint8_t {
    Content,
    Reasoning,
};

struct StopString {
    std::string text;
    OutputChannel channel  = OutputChannel::Content;
    bool include_in_output = false;
};

struct StopPolicy {
    std::vector<TokenId> token_ids;
    std::vector<StopString> strings;
    bool include_model_defaults = true;
    bool publish_stop_token     = false;
};

struct ExecutionOptions {
    SamplingOverrides sampling;
    std::uint32_t requested_output_tokens = 0;
    bool allow_prefix_reuse               = true;
};

struct OutputOptions {
    bool raw                     = false;
    bool preserve_special_tokens = false;
};

struct RequestOptions {
    ExecutionOptions execution;
    StopPolicy stop;
    OutputOptions output;
};

// Owns a bounded host-input reservation whose lifetime may cross from request preparation into
// Engine execution. The concrete reservation is product-owned; Engine only releases it once the
// target reports that the retained host payload is no longer needed.
class HostInputLease {
public:
    HostInputLease() noexcept = default;

    explicit HostInputLease(std::shared_ptr<void> owner) noexcept : owner_(std::move(owner)) {}

    HostInputLease(HostInputLease&&) noexcept            = default;
    HostInputLease& operator=(HostInputLease&&) noexcept = default;

    HostInputLease(const HostInputLease&)            = delete;
    HostInputLease& operator=(const HostInputLease&) = delete;

    void reset() noexcept { owner_.reset(); }

    [[nodiscard]] explicit operator bool() const noexcept { return owner_ != nullptr; }

private:
    std::shared_ptr<void> owner_;
};

enum class MediaKind : std::uint8_t {
    Image,
    Video,
};

struct OwnedMedia {
    MediaKind kind = MediaKind::Image;
    std::vector<std::uint8_t> bytes;
    std::string media_type;
    std::string source_name;
};

struct ToolCall {
    std::string id;
    std::string name;
    std::string arguments_json;
};

enum class MessagePartKind : std::uint8_t {
    Text,
    Media,
};

struct MessagePart {
    MessagePartKind kind = MessagePartKind::Text;
    std::string text;
    OwnedMedia media;
};

struct ChatMessage {
    std::string role;
    std::vector<MessagePart> parts;
    std::string reasoning_content;
    std::vector<ToolCall> tool_calls;
    std::string tool_call_id;
};

enum class ReasoningEffort : std::uint8_t {
    Low,
    Medium,
    XHigh,
};

struct ReasoningEffortCapabilities {
    bool low    = false;
    bool medium = false;
    bool xhigh  = false;
    std::optional<ReasoningEffort> default_effort;

    [[nodiscard]] constexpr bool supports(ReasoningEffort effort) const noexcept {
        switch (effort) {
        case ReasoningEffort::Low:
            return low;
        case ReasoningEffort::Medium:
            return medium;
        case ReasoningEffort::XHigh:
            return xhigh;
        }
        return false;
    }
};

struct PromptCapabilities {
    bool enable_thinking = false;
    ReasoningEffortCapabilities reasoning_effort;
};

struct PromptOptions {
    bool add_generation_prompt = true;
    bool enable_thinking       = true;
    std::optional<ReasoningEffort> reasoning_effort;
    bool preserve_thinking = false;
    bool add_vision_id     = false;
    std::vector<std::string> tool_jsons;
};

struct PromptInput {
    std::vector<ChatMessage> messages;
    PromptOptions options;
};

enum class RequestErrorKind : std::uint8_t {
    ContextLengthExceeded,
    MediaBudgetExceeded,
    Overloaded,
    QueueTimeout,
    Unavailable,
};

class RequestError final : public std::invalid_argument {
public:
    RequestError(RequestErrorKind kind, std::string message)
        : std::invalid_argument(std::move(message)), kind_(kind) {}

    [[nodiscard]] RequestErrorKind kind() const noexcept { return kind_; }

private:
    RequestErrorKind kind_;
};

struct PromptSummary {
    std::uint32_t prompt_tokens = 0;
    bool has_media              = false;
};

enum class FinishReason : std::uint8_t {
    None,
    OutputLimit,
    ContextCapacity,
    StopToken,
    StopString,
    Cancelled,
};

struct OutputDelta {
    OutputChannel channel = OutputChannel::Content;
    std::string text;
    // Cumulative accepted generation tokens at this publication boundary.
    // Several buffered deltas can share one boundary. This is not an estimate
    // from decoded text; it includes accepted control/reasoning tokens, like
    // GenerationResult::generated_token_ids. Zero means unavailable.
    std::uint32_t committed_tokens = 0;
};

class OutputSink {
public:
    virtual ~OutputSink()                   = default;
    virtual void publish(OutputDelta delta) = 0;
};

class CancellationView {
public:
    CancellationView() = default;
    explicit CancellationView(std::function<bool()> requested);

    [[nodiscard]] bool requested() const;

private:
    std::function<bool()> requested_;
};

struct GenerationTimings {
    double prepare_seconds     = 0.0;
    double first_token_seconds = 0.0;
    double vision_seconds      = 0.0;
    double prefill_seconds     = 0.0;
    double decode_seconds      = 0.0;
    double total_seconds       = 0.0;
};

struct SpeculativeStats {
    SpeculativeBackend backend    = SpeculativeBackend::None;
    bool enabled                  = false;
    std::uint32_t draft_window    = 0;
    std::uint64_t rounds          = 0;
    std::uint64_t drafted_tokens  = 0;
    std::uint64_t accepted_tokens = 0;
    std::uint64_t fallback_steps  = 0;
    std::uint64_t session_copy_rounds = 0;
    std::uint64_t session_copy_drafted_tokens = 0;
    std::uint64_t session_copy_accepted_tokens = 0;
    std::uint64_t session_copy_factory_resumes = 0;
    // FNV-1a over the ordered MTP ingress/egress transaction stream. This catches proposal or
    // acceptance transients that can reconverge and therefore leave final generated text unchanged.
    std::uint64_t trajectory_digest = 0;
    std::vector<std::uint64_t> accepted_per_position;
};

enum class PrefixReusePath : std::uint8_t {
    FullReset,
    AppendAtFrontier,
    RestoreTurnCheckpoint,
};

struct GenerationResult {
    PromptSummary prompt;
    std::vector<TokenId> generated_token_ids;
    std::string content;
    std::string reasoning;
    std::uint32_t reasoning_tokens     = 0;
    FinishReason finish_reason         = FinishReason::None;
    std::uint32_t reused_prompt_tokens = 0;
    PrefixReusePath prefix_reuse_path  = PrefixReusePath::FullReset;
    GenerationTimings timings;
    SpeculativeStats speculative;
};

struct ArenaMemorySummary {
    std::size_t capacity_bytes  = 0;
    std::size_t used_bytes      = 0;
    std::size_t peak_used_bytes = 0;
};

struct MemorySummary {
    int device                                = 0;
    std::uint32_t max_context                 = 0;
    KvCapacityMode kv_capacity_mode           = KvCapacityMode::Explicit;
    std::uint32_t kv_capacity                 = 0; // Resolved page-aligned Main KV capacity.
    std::uint32_t kv_capacity_page_groups     = 0;
    std::uint32_t kv_capacity_max_page_groups = 0;
    KvCacheStorage kv_cache                   = KvCacheStorage::BFloat16;
    ArenaMemorySummary weights;
    ArenaMemorySummary sequence;
    ArenaMemorySummary workspace;
    ArenaMemorySummary request_transient;
    std::size_t minimum_runtime_reservation_bytes = 0;
    std::size_t kv_capacity_increment_bytes       = 0;
    std::size_t runtime_reservation_bytes         = 0;
    std::size_t available_after_weights_bytes     = 0;
    std::size_t available_after_startup_bytes     = 0;
    std::size_t kv_capacity_headroom_bytes        = 0;
    std::size_t planned_slack_bytes               = 0;
    std::size_t workspace_logical_peak_bytes      = 0;
    std::size_t cuda_graph_allowance_bytes        = 0;
    std::size_t cuda_graph_observed_bytes         = 0;
    std::size_t kv_payload_bytes                  = 0;
};

// Monotonic execution counters plus one boundary-consistent scheduler snapshot. Consumers derive
// interval throughput by subtracting two snapshots and dividing by their own monotonic wall time.
struct RuntimeStats {
    // Actual prompt tokens evaluated by prefill; resident prefix hits are excluded.
    std::uint64_t computed_prefill_tokens = 0;
    std::uint64_t packed_prefill_rounds = 0;
    std::uint64_t packed_prefill_tokens = 0;
    // Tokens committed by decode rounds; the first token emitted by prefill is excluded.
    std::uint64_t committed_decode_tokens = 0;
    // Decode batch executions and the sum of their batch sizes.
    std::uint64_t decode_rounds         = 0;
    std::uint64_t decode_row_rounds     = 0;
    std::uint32_t running_requests      = 0;
    std::uint32_t prefilling_requests   = 0;
    std::uint32_t decode_ready_requests = 0;
    std::uint32_t waiting_requests      = 0;
};

struct LoadSummary {
    std::string target;
    std::string model_id;
    std::string weights_id;
    double load_seconds                = 0.0;
    double upload_seconds              = 0.0;
    std::uint64_t artifact_bytes_read  = 0;
    std::uint64_t host_to_device_bytes = 0;
    std::uint64_t peak_staging_bytes   = 0;
    std::size_t tensor_count           = 0;
    std::size_t resource_count         = 0;
};

} // namespace ninfer
