#include "serve/generation_service.h"

#include "product/media_acquire/acquire.h"
#include "serve/console_log.h"
#include "serve/replica_affinity.h"
#include "serve/tool_call_parser.h"
#include "serve/translate.h"

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <condition_variable>
#include <mutex>
#include <stdexcept>
#include <string>
#include <utility>

namespace ninfer::serve {

struct RequestCapacity {
    explicit RequestCapacity(std::size_t limit) : maximum(limit) {}

    std::mutex mutex;
    std::size_t active = 0;
    const std::size_t maximum;
};

struct RequestLifetime {
    RequestLifetime(std::shared_ptr<RequestCapacity> owner,
                    std::chrono::steady_clock::time_point begin,
                    std::chrono::steady_clock::time_point limit)
        : capacity(std::move(owner)), started(begin), deadline(limit) {}

    ~RequestLifetime() {
        std::lock_guard lock(capacity->mutex);
        --capacity->active;
    }

    std::shared_ptr<RequestCapacity> capacity;
    std::chrono::steady_clock::time_point started;
    std::chrono::steady_clock::time_point deadline;
};

struct MediaInputCapacity {
    std::mutex mutex;
    std::condition_variable cv;
    bool occupied = false;
};

struct MediaInputPermit {
    explicit MediaInputPermit(std::shared_ptr<MediaInputCapacity> owner)
        : capacity(std::move(owner)) {}

    ~MediaInputPermit() {
        {
            std::lock_guard lock(capacity->mutex);
            capacity->occupied = false;
        }
        capacity->cv.notify_one();
    }

    std::shared_ptr<MediaInputCapacity> capacity;
};

namespace {

using Clock                              = std::chrono::steady_clock;
constexpr std::size_t kMaximumMediaItems = 16;

[[noreturn]] void throw_preparation_cancelled();

[[noreturn]] void throw_media_error(const ninfer::product::media_acquire::Error& exception) {
    ApiError error;
    error.param   = "messages";
    error.message = exception.what();
    switch (exception.kind()) {
    case ninfer::product::media_acquire::ErrorKind::BudgetExceeded:
        error.status = 413;
        error.code   = "media_budget_exceeded";
        break;
    case ninfer::product::media_acquire::ErrorKind::RemoteUnavailable:
        error.status = 502;
        error.type   = "server_error";
        error.code   = "media_fetch_failed";
        break;
    case ninfer::product::media_acquire::ErrorKind::RemoteTimeout:
        error.status = 504;
        error.type   = "server_error";
        error.code   = "media_fetch_timeout";
        break;
    case ninfer::product::media_acquire::ErrorKind::DeadlineExceeded:
        error.status = 503;
        error.type   = "server_error";
        error.code   = "request_queue_timeout";
        break;
    case ninfer::product::media_acquire::ErrorKind::Cancelled:
        throw_preparation_cancelled();
    }
    throw ApiException(std::move(error));
}

[[noreturn]] void throw_invalid_input(const std::exception& exception,
                                      const char* code = "invalid_media") {
    ApiError error;
    error.status  = 400;
    error.param   = "messages";
    error.code    = code;
    error.message = exception.what();
    throw ApiException(std::move(error));
}

[[noreturn]] void throw_preparation_cancelled() {
    ApiError error;
    error.status  = 499;
    error.type    = "request_cancelled";
    error.code    = "client_disconnected";
    error.message = "client disconnected during media preparation";
    throw ApiException(std::move(error));
}

std::size_t media_item_count(const GenerationRequest& request) {
    std::size_t count = 0;
    for (const ChatTurn& message : request.messages) {
        for (const ContentPart& part : message.content) {
            if (part.kind == ContentKind::Image || part.kind == ContentKind::Video) { ++count; }
        }
    }
    return count;
}

ninfer::OwnedMedia acquire_media(const ContentPart& part, Clock::time_point deadline,
                                 const std::function<bool()>& is_cancelled,
                                 std::size_t& remaining_bytes) {
    if (remaining_bytes == 0) {
        throw_media_error(ninfer::product::media_acquire::Error(
            ninfer::product::media_acquire::ErrorKind::BudgetExceeded,
            "request media exceeds aggregate byte limit"));
    }
    ninfer::product::media_acquire::Policy policy;
    policy.max_bytes    = std::min(policy.max_bytes, remaining_bytes);
    policy.deadline     = deadline;
    policy.is_cancelled = is_cancelled;
    std::vector<std::uint8_t> source_bytes;
    try {
        source_bytes = ninfer::product::media_acquire::acquire_bytes(part.source, policy);
    } catch (const ninfer::product::media_acquire::Error& exception) {
        throw_media_error(exception);
    } catch (const std::invalid_argument& exception) { throw_invalid_input(exception); }

    remaining_bytes -= source_bytes.size();
    ninfer::OwnedMedia media;
    media.kind =
        part.kind == ContentKind::Image ? ninfer::MediaKind::Image : ninfer::MediaKind::Video;
    media.media_type = part.source.media_type;
    switch (part.source.kind) {
    case ninfer::product::media_acquire::SourceKind::Path:
    case ninfer::product::media_acquire::SourceKind::Url:
        media.source_name = part.source.value;
        break;
    case ninfer::product::media_acquire::SourceKind::Data:
        media.source_name = "inline-data";
        break;
    case ninfer::product::media_acquire::SourceKind::Bytes:
        media.source_name = "inline-bytes";
        break;
    }
    media.bytes = std::move(source_bytes);
    return media;
}

[[noreturn]] void throw_request_error(const ninfer::RequestError& exception) {
    ApiError error;
    error.param   = "messages";
    error.message = exception.what();
    switch (exception.kind()) {
    case ninfer::RequestErrorKind::ContextLengthExceeded:
        error.status = 400;
        error.code   = "context_length_exceeded";
        break;
    case ninfer::RequestErrorKind::MediaBudgetExceeded:
        error.status = 413;
        error.code   = "media_budget_exceeded";
        break;
    case ninfer::RequestErrorKind::Overloaded:
        error.param.clear();
        error.status = 429;
        error.type   = "rate_limit_error";
        error.code   = "server_overloaded";
        break;
    case ninfer::RequestErrorKind::QueueTimeout:
        error.param.clear();
        error.status = 503;
        error.type   = "server_error";
        error.code   = "request_queue_timeout";
        break;
    case ninfer::RequestErrorKind::Unavailable:
        error.param.clear();
        error.status = 503;
        error.type   = "server_error";
        error.code   = "service_unavailable";
        break;
    }
    throw ApiException(std::move(error));
}

void check_preparation_control(Clock::time_point deadline,
                               const std::function<bool()>& is_cancelled) {
    if (is_cancelled && is_cancelled()) { throw_preparation_cancelled(); }
    if (Clock::now() >= deadline) {
        throw_request_error(ninfer::RequestError(RequestErrorKind::QueueTimeout,
                                                 "inference request expired during preparation"));
    }
}

class ServiceOutputSink final : public ninfer::OutputSink {
public:
    ServiceOutputSink(const StreamSink& sink, bool filter_tool_calls, bool trace_stream)
        : sink_(&sink), filter_tool_calls_(filter_tool_calls), trace_stream_(trace_stream) {}

    void publish(ninfer::OutputDelta delta) override {
        committed_tokens_ = delta.committed_tokens;
        if (delta.text.empty()) { return; }
        if (delta.channel == ninfer::OutputChannel::Reasoning) {
            if (sink_->on_reasoning) { sink_->on_reasoning(delta.text); }
        } else {
            std::string visible =
                filter_tool_calls_ ? tool_filter_.feed(delta.text) : std::move(delta.text);
            publish_content(visible);
        }
    }

    std::vector<StreamEmission> take_trace() { return std::move(stream_trace_); }

    std::size_t finish(bool is_tool_call_response) {
        if (filter_tool_calls_) { publish_content(tool_filter_.finish(is_tool_call_response)); }
        return content_bytes_;
    }

private:
    void publish_content(const std::string& text) {
        if (text.empty() || !sink_->on_content) { return; }
        sink_->on_content(text);
        content_bytes_ += text.size();
        if (trace_stream_) {
            stream_trace_.push_back(StreamEmission{content_bytes_, committed_tokens_});
        }
    }

    const StreamSink* sink_ = nullptr;
    bool filter_tool_calls_ = false;
    bool trace_stream_ = false;
    std::uint32_t committed_tokens_ = 0;
    std::vector<StreamEmission> stream_trace_;
    ToolCallStreamFilter tool_filter_;
    std::size_t content_bytes_ = 0;
};

} // namespace

GenerationService::GenerationService(ServeOptions options, LoadProgress load_progress)
    : options_(std::move(options)) {
    ninfer::EngineOptions engine_options;
    engine_options.artifact_path        = options_.artifact_path;
    engine_options.device               = options_.device;
    engine_options.max_context          = options_.max_context;
    engine_options.kv_capacity          = options_.kv_capacity;
    engine_options.max_concurrency      = options_.max_concurrency;
    engine_options.max_pending_requests = options_.max_pending_requests;
    engine_options.pending_timeout_ms   = options_.pending_timeout_ms;
    engine_options.prefill_chunk        = options_.prefill_chunk;
    engine_options.prefill_live_chunk   = options_.prefill_live_chunk;
    engine_options.max_prefill_lanes    = options_.max_prefill_lanes;
    engine_options.pack_prefill = options_.pack_prefill;
    engine_options.prefill_decode_rounds = options_.prefill_decode_rounds;
    engine_options.prefill_target_decode_tps = options_.prefill_target_decode_tps;
    engine_options.kv_cache             = options_.kv_cache;
    engine_options.enable_vision        = options_.enable_vision;
    engine_options.use_cuda_graph       = options_.use_cuda_graph;
    engine_options.speculative          = options_.speculative;
    engine_options.load_progress        = std::move(load_progress);
    const std::vector<int> devices = options_.devices.empty()
        ? std::vector<int>{options_.device} : options_.devices;
    for (int device : devices) {
        engine_options.device = device;
        engines_.push_back(std::make_unique<ninfer::Engine>(engine_options));
        media_input_capacities_.push_back(std::make_shared<MediaInputCapacity>());
    }
    prompt_capabilities_ = engines_.front()->prompt_capabilities();
    request_capacity_ = std::make_shared<RequestCapacity>(
        options_.total_concurrency() + options_.max_pending_requests);
    replica_router_ = std::make_unique<ReplicaRouter>(devices.size(),
        static_cast<std::size_t>(options_.max_concurrency) + options_.max_pending_requests,
        options_.max_concurrency);
}

std::shared_ptr<RequestLifetime> GenerationService::acquire_request_lifetime() const {
    const auto started = Clock::now();
    {
        std::lock_guard lock(request_capacity_->mutex);
        if (request_capacity_->active >= request_capacity_->maximum) {
            throw_request_error(ninfer::RequestError(RequestErrorKind::Overloaded,
                                                     "inference request queue is full"));
        }
        ++request_capacity_->active;
    }
    try {
        return std::make_shared<RequestLifetime>(
            request_capacity_, started,
            started + std::chrono::milliseconds(options_.pending_timeout_ms));
    } catch (...) {
        std::lock_guard lock(request_capacity_->mutex);
        --request_capacity_->active;
        throw;
    }
}

HostInputLease
GenerationService::acquire_media_input(Clock::time_point deadline,
                                       const std::function<bool()>& is_cancelled,
                                       std::size_t replica) const {
    const auto& media_input_capacity_ = media_input_capacities_.at(replica);
    std::unique_lock lock(media_input_capacity_->mutex);
    while (media_input_capacity_->occupied) {
        if (is_cancelled && is_cancelled()) { throw_preparation_cancelled(); }
        const Clock::time_point now = Clock::now();
        if (now >= deadline) {
            throw_request_error(ninfer::RequestError(
                RequestErrorKind::QueueTimeout,
                "inference request expired while waiting for media preparation"));
        }
        media_input_capacity_->cv.wait_until(
            lock, std::min(deadline, now + std::chrono::milliseconds(10)));
    }
    if (is_cancelled && is_cancelled()) { throw_preparation_cancelled(); }
    if (Clock::now() >= deadline) {
        throw_request_error(
            ninfer::RequestError(RequestErrorKind::QueueTimeout,
                                 "inference request expired while waiting for media preparation"));
    }

    media_input_capacity_->occupied = true;
    lock.unlock();
    try {
        auto permit = std::make_shared<MediaInputPermit>(media_input_capacity_);
        return HostInputLease(std::static_pointer_cast<void>(std::move(permit)));
    } catch (...) {
        {
            std::lock_guard capacity_lock(media_input_capacity_->mutex);
            media_input_capacity_->occupied = false;
        }
        media_input_capacity_->cv.notify_one();
        throw;
    }
}

PreparedRequest GenerationService::prepare(const GenerationRequest& request,
                                           std::function<bool()> is_cancelled) const {
    PreparedRequest prepared;
    ninfer::RequestOptions request_options = to_request_options(request, options_);
    prepared.include_usage                 = request.include_usage;
    prepared.tool_capable                  = request.uses_tools() || request.has_tool_history();
    prepared.tool_name_max_length          = request.tool_name_max_length;
    const ResolvedPromptSemantics semantics =
        resolve_prompt_semantics(request, options_, prompt_capabilities_);
    prepared.enable_thinking                   = semantics.enable_thinking;
    prepared.preserve_thinking                 = semantics.preserve_thinking;
    prepared.preserve_thinking_semantic_change = request.preserve_thinking_semantic_change;
    const std::size_t media_items              = media_item_count(request);
    const bool request_has_media               = media_items != 0;
    if (request_has_media && !options_.enable_vision) {
        const std::invalid_argument error("Vision is disabled for this server");
        throw_invalid_input(error, "vision_disabled");
    }
    if (media_items > kMaximumMediaItems) {
        throw_request_error(ninfer::RequestError(RequestErrorKind::MediaBudgetExceeded,
                                                 "request exceeds the 16-item media limit"));
    }
    prepared.lifetime = acquire_request_lifetime();
    ReplicaAffinity affinity;
    if (options_.allow_prefix_reuse) { affinity = request_replica_affinity(request); }
    const std::optional<std::uint64_t> current_key = affinity.count != 0
        ? std::optional<std::uint64_t>(affinity.turn_keys[affinity.count - 1]) : std::nullopt;
    prepared.replica_reservation = replica_router_->acquire(current_key, affinity.prefixes());
    if (!prepared.replica_reservation) {
        throw_request_error(ninfer::RequestError(RequestErrorKind::Overloaded,
                                                 "all inference replicas are full"));
    }
    prepared.replica_index = prepared.replica_reservation->replica();
    ninfer::Engine* engine = engines_[prepared.replica_index].get();
    prepared.device = engine->options().device;
    HostInputLease host_input;
    if (request_has_media) {
        host_input = acquire_media_input(prepared.lifetime->deadline, is_cancelled,
                                         prepared.replica_index);
    }

    try {
        std::size_t remaining_media_bytes = options_.max_request_bytes;
        ninfer::PromptInput input =
            to_prompt_input(request, semantics, [&](const ContentPart& part) {
                return acquire_media(part, prepared.lifetime->deadline, is_cancelled,
                                     remaining_media_bytes);
            });
        check_preparation_control(prepared.lifetime->deadline, is_cancelled);
        ninfer::PreparedPrompt prompt = engine->prepare(std::move(input));
        check_preparation_control(prepared.lifetime->deadline, is_cancelled);
        prepared.prompt_tokens = static_cast<int>(prompt.summary().prompt_tokens);
        prepared.prepare_seconds =
            std::chrono::duration<double>(Clock::now() - prepared.lifetime->started).count();
        prepared.generation = engine->submit(std::move(prompt), std::move(request_options),
                                              prepared.lifetime->deadline, std::move(host_input),
                                              prepared.lifetime->started);
        prepared.sampling   = prepared.generation.resolved_sampling();
    } catch (const ApiException&) { throw; } catch (const ninfer::RequestError& exception) {
        throw_request_error(exception);
    } catch (const std::invalid_argument& exception) { throw_invalid_input(exception); }
    return prepared;
}

int GenerationService::count_prompt_tokens(const GenerationRequest& request,
                                           std::function<bool()> is_cancelled) const {
    const std::size_t media_items = media_item_count(request);
    const bool request_has_media  = media_items != 0;
    if (request_has_media && !options_.enable_vision) {
        const std::invalid_argument error("Vision is disabled for this server");
        throw_invalid_input(error, "vision_disabled");
    }
    if (media_items > kMaximumMediaItems) {
        throw_request_error(ninfer::RequestError(RequestErrorKind::MediaBudgetExceeded,
                                                 "request exceeds the 16-item media limit"));
    }
    const Clock::time_point deadline =
        Clock::now() + std::chrono::milliseconds(options_.pending_timeout_ms);
    const ResolvedPromptSemantics semantics =
        resolve_prompt_semantics(request, options_, prompt_capabilities_);
    HostInputLease host_input;
    if (request_has_media) { host_input = acquire_media_input(deadline, is_cancelled); }
    try {
        std::size_t remaining_media_bytes = options_.max_request_bytes;
        ninfer::PromptInput input =
            to_prompt_input(request, semantics, [&](const ContentPart& part) {
                return acquire_media(part, deadline, is_cancelled, remaining_media_bytes);
            });
        check_preparation_control(deadline, is_cancelled);
        const int prompt_tokens = static_cast<int>(engines_.front()->count_tokens(std::move(input)));
        check_preparation_control(deadline, is_cancelled);
        return prompt_tokens;
    } catch (const ApiException&) { throw; } catch (const ninfer::RequestError& exception) {
        throw_request_error(exception);
    } catch (const std::invalid_argument& exception) { throw_invalid_input(exception); }
}

GenerationOutcome GenerationService::run(PreparedRequest& prepared, const StreamSink* sink,
                                         std::function<bool()> is_cancelled) {
    std::unique_ptr<ServiceOutputSink> output_sink;
    if (sink != nullptr) {
        output_sink = std::make_unique<ServiceOutputSink>(*sink, prepared.tool_capable,
                                                          options_.trace_stream);
    }
    ninfer::OutputSink* public_sink = output_sink.get();
    ninfer::CancellationView cancellation;
    if (is_cancelled || (sink != nullptr && sink->is_cancelled)) {
        cancellation = ninfer::CancellationView([external = std::move(is_cancelled), sink]() {
            return (external && external()) ||
                   (sink != nullptr && sink->is_cancelled && sink->is_cancelled());
        });
    }

    // Release GPU routing capacity at model completion, not at slow HTTP body release.
    auto reservation = std::move(prepared.replica_reservation);
    ninfer::GenerationResult result;
    try {
        result = prepared.generation.wait(public_sink, cancellation);
    } catch (const ninfer::RequestError& exception) { throw_request_error(exception); }
    if (reservation && options_.allow_prefix_reuse) { reservation->mark_reusable(); }
    reservation.reset();
    GenerationOutcome outcome;
    outcome.text              = std::move(result.content);
    outcome.reasoning         = std::move(result.reasoning);
    outcome.prompt_tokens     = static_cast<int>(result.prompt.prompt_tokens);
    outcome.completion_tokens = static_cast<int>(result.generated_token_ids.size());
    outcome.reasoning_tokens  = static_cast<int>(result.reasoning_tokens);
    outcome.finish_reason     = result.finish_reason;

    outcome.metrics.prepare_seconds = prepared.prepare_seconds;
    outcome.metrics.ttft_seconds =
        prepared.prepare_seconds +
        std::max(0.0, result.timings.first_token_seconds - result.timings.prepare_seconds);
    outcome.metrics.vision_seconds  = result.timings.vision_seconds;
    outcome.metrics.prefill_seconds = result.timings.prefill_seconds;
    outcome.metrics.decode_seconds  = result.timings.decode_seconds;
    outcome.metrics.total_seconds =
        prepared.prepare_seconds +
        std::max(0.0, result.timings.total_seconds - result.timings.prepare_seconds);
    outcome.metrics.prefix_cache_hit_tokens     = result.reused_prompt_tokens;
    outcome.metrics.prefix_reuse_path           = result.prefix_reuse_path;
    outcome.metrics.speculative_backend         = result.speculative.backend;
    outcome.metrics.speculative_draft_window    = result.speculative.draft_window;
    outcome.metrics.speculative_rounds          = result.speculative.rounds;
    outcome.metrics.speculative_draft_tokens    = result.speculative.drafted_tokens;
    outcome.metrics.speculative_accepted_tokens = result.speculative.accepted_tokens;
    outcome.metrics.speculative_fallback_steps  = result.speculative.fallback_steps;
    outcome.metrics.session_copy_rounds=result.speculative.session_copy_rounds;
    outcome.metrics.session_copy_drafted_tokens=result.speculative.session_copy_drafted_tokens;
    outcome.metrics.session_copy_accepted_tokens=result.speculative.session_copy_accepted_tokens;
    outcome.metrics.session_copy_factory_resumes=result.speculative.session_copy_factory_resumes;
    outcome.metrics.speculative_accepted_per_position =
        std::move(result.speculative.accepted_per_position);

    bool is_tool_call_response = false;
    if (prepared.tool_capable) {
        ParsedToolCallOutput parsed =
            parse_qwen_tool_call_output(outcome.text, prepared.tool_name_max_length);
        outcome.text          = std::move(parsed.content);
        is_tool_call_response = parsed.is_tool_call_response;
        if (is_tool_call_response) { outcome.tool_calls = std::move(parsed.tool_calls); }
    }
    if (output_sink) {
        outcome.streamed_content_bytes = output_sink->finish(is_tool_call_response);
        outcome.metrics.stream_trace = output_sink->take_trace();
    }
    return outcome;
}

ninfer::RuntimeStats GenerationService::runtime_stats() const {
    ninfer::RuntimeStats total;
    for (const auto& engine : engines_) {
        const auto current = engine->runtime_stats();
        total.computed_prefill_tokens += current.computed_prefill_tokens;
        total.packed_prefill_rounds += current.packed_prefill_rounds;
        total.packed_prefill_tokens += current.packed_prefill_tokens;
        total.committed_decode_tokens += current.committed_decode_tokens;
        total.decode_rounds += current.decode_rounds;
        total.decode_row_rounds += current.decode_row_rounds;
        total.running_requests += current.running_requests;
        total.prefilling_requests += current.prefilling_requests;
        total.decode_ready_requests += current.decode_ready_requests;
        total.waiting_requests += current.waiting_requests;
    }
    return total;
}

void GenerationService::warmup() {
    try {
        GenerationRequest request;
        ChatTurn turn;
        turn.role = "user";
        ContentPart content;
        content.kind     = ContentKind::Text;
        content.text     = "hi";
        content.type_raw = "text";
        turn.content.push_back(std::move(content));
        request.messages.push_back(std::move(turn));
        request.max_tokens       = 4;
        request.max_tokens_set   = true;
        std::vector<PreparedRequest> requests;
        for (std::size_t i = 0; i < engines_.size(); ++i) { requests.push_back(prepare(request)); }
        for (auto& prepared : requests) { run(prepared, nullptr); }
    } catch (const std::exception& exception) {
        write_console_log(ConsoleLogLevel::Warning,
                          std::string("warmup failed (continuing): ") + exception.what());
    }
}

} // namespace ninfer::serve
