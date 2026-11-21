#include "ninfer/engine.h"

#include <chrono>
#include <memory>

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <exception>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
void require(bool value, const char* message) {
    if (!value) { throw std::runtime_error(message); }
}

ninfer::PromptInput prompt(std::string text, int image = -1) {
    ninfer::PromptInput input;
    input.options.enable_thinking = false;
    ninfer::ChatMessage message;
    message.role = "user";
    if (image >= 0) {
        ninfer::MessagePart part;
        part.kind = ninfer::MessagePartKind::Media;
        part.media.kind = ninfer::MediaKind::Image;
        part.media.media_type = "image/x-portable-pixmap";
        part.media.source_name = "identity.ppm";
        const std::string header = "P6\n64 64\n255\n";
        part.media.bytes.assign(header.begin(), header.end());
        for (int pixel = 0; pixel < 64 * 64; ++pixel) {
            part.media.bytes.push_back(image == 0 ? 240 : 10);
            part.media.bytes.push_back(20);
            part.media.bytes.push_back(image == 0 ? 10 : 240);
        }
        message.parts.push_back(std::move(part));
    }
    message.parts.push_back(ninfer::MessagePart{.kind = ninfer::MessagePartKind::Text,
                                               .text = std::move(text), .media = {}});
    input.messages.push_back(std::move(message));
    return input;
}

ninfer::RequestOptions options(bool reuse, std::uint32_t count = 32) {
    ninfer::RequestOptions out;
    out.execution.requested_output_tokens = count;
    out.execution.sampling.temperature = 0.0F;
    out.execution.allow_prefix_reuse = reuse;
    out.stop.include_model_defaults = false;
    return out;
}

struct CancelSink final : ninfer::OutputSink {
    explicit CancelSink(std::uint32_t boundary) : boundary(boundary) {}
    void publish(ninfer::OutputDelta delta) override {
        seen = std::max(seen, delta.committed_tokens);
    }
    std::uint32_t boundary;
    std::uint32_t seen = 0;
};

ninfer::GenerationResult cancel(ninfer::Engine& engine, const ninfer::PromptInput& input,
                                 std::uint32_t boundary) {
    CancelSink sink(boundary);
    const auto result = engine.generate(engine.prepare(input), options(true, 128), &sink,
        ninfer::CancellationView([&] { return sink.seen >= sink.boundary; }));
    require(result.finish_reason == ninfer::FinishReason::Cancelled,
            "request failed to cancel during real generation");
    require(sink.seen >= boundary && result.generated_token_ids.size() < 128,
            "cancellation did not observe the intended publication boundary");
    return result;
}

void exercise(ninfer::Engine& engine, const ninfer::PromptInput& input, const char* label) {
    const auto cold = engine.generate(engine.prepare(input), options(false));
    const auto warm = engine.generate(engine.prepare(input), options(true));
    require(warm.reused_prompt_tokens > 0, "control did not reuse a real prompt prefix");
    if (cold.generated_token_ids != warm.generated_token_ids) {
        std::cerr << "CONTROL_DIFF " << label << " cold=" << cold.content
                  << " warm=" << warm.content << " reused=" << warm.reused_prompt_tokens << '\n';
        std::cerr << "cold_ids:";
        for (const auto id : cold.generated_token_ids) { std::cerr << ' ' << id; }
        std::cerr << "\nwarm_ids:";
        for (const auto id : warm.generated_token_ids) { std::cerr << ' ' << id; }
        std::cerr << '\n';
        // Cold and cached MTP paths can have different deterministic proposal
        // trajectories. Preserve that evidence, but judge this cancellation
        // mechanism against the same normal cached checkpoint it restores.
    }
    const auto warm_repeat = engine.generate(engine.prepare(input), options(true));
    require(warm_repeat.generated_token_ids == warm.generated_token_ids,
            "normal warm checkpoint is not a repeatable cancellation control");
    for (const std::uint32_t boundary : {1U, 7U, 13U}) {
        (void)cancel(engine, input, boundary);
        const auto reused = engine.generate(engine.prepare(input), options(true));
        require(reused.reused_prompt_tokens == warm.reused_prompt_tokens,
                "clean cancellation lost or extended the validated prompt checkpoint");
        require(reused.generated_token_ids == warm.generated_token_ids,
                "post-cancellation retained checkpoint changed exact greedy output");
        require(reused.timings.vision_seconds == 0.0,
                "post-cancellation exact media prefix was encoded again");
        std::cout << "PASS " << label << " boundary=" << boundary
                  << " reused=" << reused.reused_prompt_tokens
                  << " exact_tokens=" << reused.generated_token_ids.size() << '\n';
    }
    // Opting out of reuse is still authoritative after clean cancellation.
    (void)cancel(engine, input, 3);
    const auto uncached = engine.generate(engine.prepare(input), options(false));
    require(uncached.reused_prompt_tokens == 0, "allow_prefix_reuse=false was ignored");
    require(uncached.generated_token_ids == cold.generated_token_ids,
            "fresh generation changed after cancelled cache retirement");
}
} // namespace

int main(int argc, char** argv) {
    const char* artifact = std::getenv("NINFER_CANCEL_PREFIX_ARTIFACT");
    if (artifact == nullptr || *artifact == '\0') {
        std::cout << "skip: NINFER_CANCEL_PREFIX_ARTIFACT is not set\n";
        return 77;
    }
    try {
        ninfer::EngineOptions config;
        config.artifact_path = artifact;
        config.max_context = 4096;
        config.kv_capacity = ninfer::KvCapacityPolicy::automatic();
        config.max_concurrency = 4;
        config.prefill_chunk = argc > 2 ? std::stoul(argv[2]) : 128;
        require(config.prefill_chunk <= 4096, "cancellation test chunk exceeds qualified prefill widths");
        // A partial-prefill test must have room for more than one actual work unit.
        config.max_context = std::max(4096U, 2U * config.prefill_chunk);
        config.prefill_live_chunk = argc > 3 ? std::stoul(argv[3]) : 0;
        config.kv_cache = ninfer::KvCacheStorage::Int8Group64;
        config.speculative.backend = ninfer::SpeculativeBackend::Mtp;
        config.speculative.draft_tokens = argc > 1 ? std::stoul(argv[1]) : 1;
        config.speculative.proposal_head = ninfer::ProposalHead::Full;
        if (argc > 4) {
            const std::string profile(argv[4]);
            require(profile == "full-target-draft" || profile == "full-vocab-draft",
                    "unknown head-test profile");
            config.speculative.force_full_target_head = true;
            if (profile == "full-target-draft") {
                config.speculative.proposal_head = ninfer::ProposalHead::Optimized;
            } else {
                config.speculative.fast_full_vocab_draft = true;
                config.max_concurrency = 1;
            }
        }

        if (std::getenv("NINFER_CANCEL_RETAINED_C1") != nullptr) {
            config.max_concurrency = 1;
            config.speculative.draft_tokens = 3;
            config.speculative.proposal_head = ninfer::ProposalHead::Optimized;
            config.speculative.greedy_margin_delta = 7.0F;
            config.speculative.retained_mtp_packed_prefix = true;
        }
        if (std::getenv("NINFER_CANCEL_RADIX_C1") != nullptr) {
            config.max_concurrency = 1;
            config.speculative.draft_tokens = 3;
            config.speculative.proposal_head = ninfer::ProposalHead::Optimized;
            config.speculative.retained_mtp_packed_prefix = true;
            config.speculative.greedy_margin_delta = -1.0F;
            config.speculative.sm75_radix_projections = true;
        }
        if (std::getenv("NINFER_CANCEL_COMPOSED_C1") != nullptr) {
            config.max_concurrency=1;
            config.speculative.backend=ninfer::SpeculativeBackend::Mtp;
            config.speculative.draft_tokens=3;
            config.speculative.proposal_head=ninfer::ProposalHead::Optimized;
            config.speculative.force_full_target_head=false;
            config.speculative.greedy_margin_delta=-1.0F;
            config.speculative.retained_mtp_packed_prefix=true;
            config.speculative.sm75_radix_projections=true;
            config.speculative.fast_full_vocab_draft=true;
            config.speculative.retained_target_full_vocab_draft=true;
        }
        if(std::getenv("NINFER_TEST_TIMED_PREFILL") != nullptr) {
            config.prefill_target_decode_tps=24.0;
            config.prefill_decode_rounds=24;
        }
        if(std::getenv("NINFER_TEST_PACKED_PREFILL") != nullptr) {
            config.pack_prefill=true;config.max_prefill_lanes=config.max_concurrency;
        }
        if(std::getenv("NINFER_CANCEL_SESSION_COPY")!=nullptr) {
            config.max_concurrency=1;
            config.speculative.backend=ninfer::SpeculativeBackend::Mtp;
            config.speculative.draft_tokens=3;
            config.speculative.proposal_head=ninfer::ProposalHead::Full;
            config.speculative.force_full_target_head=true;
            config.speculative.fast_full_vocab_draft=true;
            config.speculative.retained_mtp_packed_prefix=true;
            config.speculative.sm75_radix_projections=true;
            config.speculative.session_copy_draft=true;
        }
        if(std::getenv("NINFER_CANCEL_NU3_C1")!=nullptr) {
            config.max_concurrency=1;
            config.speculative.backend=ninfer::SpeculativeBackend::Mtp;
            config.speculative.draft_tokens=3;
            config.speculative.proposal_head=ninfer::ProposalHead::Optimized;
            config.speculative.force_full_target_head=false;
            config.speculative.greedy_margin_delta=-1.0F;
            config.speculative.retained_mtp_packed_prefix=true;
            config.speculative.sm75_radix_projections=true;
            config.speculative.fast_full_vocab_draft=true;
            config.speculative.retained_target_full_vocab_draft=true;
            config.speculative.sm75_nu3_mlp=true;
        }
        if(std::getenv("NINFER_CANCEL_TARGET_UNION")!=nullptr) {
            config.max_concurrency=1;
            config.speculative.backend=ninfer::SpeculativeBackend::Mtp;
            config.speculative.draft_tokens=3;
            config.speculative.proposal_head=ninfer::ProposalHead::Optimized;
            config.speculative.force_full_target_head=false;
            config.speculative.greedy_margin_delta=-1.0F;
            config.speculative.retained_mtp_packed_prefix=true;
            config.speculative.sm75_radix_projections=true;
            config.speculative.sm75_nu3_mlp=true;
            config.speculative.fast_full_vocab_draft=false;
            config.speculative.retained_target_full_vocab_draft=false;
            config.speculative.target_draft_union=true;
        }
        if (const char* wide_slots = std::getenv("NINFER_TEST_WIDE_RADIX")) {
            // Value1 retains the original two-slot test selection.
            const std::string selected(wide_slots);
            config.max_concurrency = selected == "3" ? 3 : selected == "4" ? 4 : 2;
            if (config.max_concurrency > 2) {
                // Match the explicitly sized 4K-per-slot wide-D3 experiment.
                // Automatic sizing keeps its normal extra headroom and may reject
                // this optional high-memory profile; do not relax that guard.
                config.kv_capacity = ninfer::KvCapacityPolicy::explicit_capacity(
                    config.max_context * config.max_concurrency);
            }
            config.speculative = {};
            config.speculative.backend = ninfer::SpeculativeBackend::Mtp;
            config.speculative.draft_tokens = 3;
            config.speculative.proposal_head = ninfer::ProposalHead::Full;
            config.speculative.force_full_target_head = true;
            // The retained packed-MTP layout is C1-only, independently of
            // ordinary request-prefix reuse, which remains under test.
            config.speculative.retained_mtp_packed_prefix = false;
            config.speculative.sm75_radix_projections = true;
        }
        if (std::getenv("NINFER_TEST_RESIDENT_RADIX") != nullptr) {
            config.max_context = 16384;
            config.kv_capacity = ninfer::KvCapacityPolicy::explicit_capacity(65536);
            config.max_concurrency = 4;
            config.speculative = {};
            config.speculative.backend = ninfer::SpeculativeBackend::Mtp;
            config.speculative.draft_tokens = 1;
            config.speculative.force_full_target_head = true;
            config.speculative.sm75_radix_projections = true;
            config.speculative.sm75_resident_radix = true;
        }
        config.enable_vision = true;
        if (config.speculative.retained_target_full_vocab_draft) {
            auto invalid=config;
            invalid.speculative.fast_full_vocab_draft=false;
            bool rejected=false;
            try {ninfer::Engine forbidden(invalid);}catch(const std::invalid_argument&){rejected=true;}
            require(rejected,"direct Engine construction admitted an inconsistent composed policy");
        }
        ninfer::Engine engine(config);
        require(engine.options().speculative.retained_target_full_vocab_draft ==
                    config.speculative.retained_target_full_vocab_draft &&
                    engine.options().speculative.greedy_margin_delta == config.speculative.greedy_margin_delta,
                "Engine lost the frozen proposal/acceptance composition");
        require(engine.options().speculative.force_full_target_head == config.speculative.force_full_target_head,
                "Engine lost explicit full-target policy");

        // A completed early request must not lose its reusable checkpoint while
        // a second unrelated request can occupy an unused lane.
        const auto retained_a = prompt("CACHE A: Explain why idempotency makes retries safe.");
        const auto unrelated_b = prompt("CACHE B: Explain why checksums help detect corruption.");
        (void)engine.generate(engine.prepare(retained_a), options(false));
        if (config.max_concurrency > 1) {
            (void)engine.generate(engine.prepare(unrelated_b), options(true));
        }
        const auto preserved_a = engine.generate(engine.prepare(retained_a), options(true));
        require(preserved_a.reused_prompt_tokens > 0,
                "unrelated admission overwrote retained prefix despite free empty lanes");
        std::cout << (config.max_concurrency > 1
            ? "PASS empty-lane admission preserves unrelated retained prefix\n"
            : "PASS C1 repeated input preserves its retained prefix\n");
        // Backdated ingress changes only scheduling; deadlines and token results
        // still obey the ordinary request contract. Rejection must release leases.
        using Clock = std::chrono::steady_clock;
        const auto ingress = Clock::now() - std::chrono::seconds(1);
        const auto ingress_replay = engine.submit(engine.prepare(retained_a), options(true), {}, {}, ingress).wait();
        require(ingress_replay.generated_token_ids == preserved_a.generated_token_ids &&
                    ingress_replay.reused_prompt_tokens == preserved_a.reused_prompt_tokens,
                "original ingress changed token output or cache identity");
        auto lifetime = std::make_shared<int>(1);
        std::weak_ptr<int> weak_lifetime = lifetime;
        ninfer::HostInputLease lease(lifetime);
        lifetime.reset();
        bool rejected_future = false;
        try {
            (void)engine.submit(engine.prepare(retained_a), options(true), {}, std::move(lease),
                                Clock::now() + std::chrono::hours(1));
        } catch (const std::invalid_argument&) { rejected_future = true; }
        require(rejected_future && weak_lifetime.expired(),
                "future ingress was accepted or leaked its host input lease");
        bool expired_deadline = false;
        try {
            (void)engine.submit(engine.prepare(retained_a), options(true),
                                Clock::now() - std::chrono::seconds(1), {}, ingress);
        } catch (const ninfer::RequestError& error) {
            expired_deadline = error.kind() == ninfer::RequestErrorKind::QueueTimeout;
        }
        require(expired_deadline, "original ingress bypassed the queue deadline");
        std::cout << "PASS original ingress preserves output/reuse; future ingress releases lease; deadlines unchanged\n";

        const auto text = prompt("List twelve practical checks for a reliable software deployment. "
                                 "Explain each check in a full sentence.");
        const auto red = prompt("Describe the dominant color and appearance of this image.", 0);
        const auto blue = prompt("Describe the dominant color and appearance of this image.", 1);
        exercise(engine, text, "text");
        exercise(engine, red, "image");
        (void)cancel(engine, red, 3);
        const auto changed = engine.generate(engine.prepare(blue), options(true));
        require(changed.reused_prompt_tokens == 0,
                "changed media bytes incorrectly reused a cancelled visual checkpoint");
        const auto fresh = engine.generate(engine.prepare(blue), options(false));
        require(changed.generated_token_ids == fresh.generated_token_ids,
                "changed-media invalidation changed exact greedy output");
        std::cout << "PASS changed-media invalidation; no stale visual prefix\n";
        const auto changed_text = prompt("Different request identity. Return a list of five prime numbers.");
        (void)cancel(engine, text, 3);
        const auto distinct = engine.generate(engine.prepare(changed_text), options(true));
        require(distinct.reused_prompt_tokens == 0,
                "changed prompt reused an incompatible cancelled checkpoint");
        std::cout << "PASS changed-text invalidation\n";
        // Completion before the first output token is still a valid checkpoint.
        const auto completed_input = prompt("Completed prefill fixture. Explain why checksums detect changed bytes.");
        const auto completed_tokens = engine.prepare(completed_input).summary().prompt_tokens;
        require(completed_tokens <= config.prefill_chunk,
                "completed-prefill fixture does not fit one work unit");
        (void)engine.generate(engine.prepare(completed_input), options(false));
        const auto completed_control = engine.generate(engine.prepare(completed_input), options(true));
        const auto before_completed = engine.runtime_stats().computed_prefill_tokens;
        const auto completed_cancel = engine.generate(engine.prepare(completed_input), options(false), nullptr,
            ninfer::CancellationView([&] {
                return engine.runtime_stats().computed_prefill_tokens > before_completed;
            }));
        require(completed_cancel.finish_reason == ninfer::FinishReason::Cancelled &&
                    completed_cancel.generated_token_ids.empty(),
                "completed-prefill cancellation must precede licensed output");
        const auto completed_processed = engine.runtime_stats().computed_prefill_tokens - before_completed;
        require(completed_processed == completed_tokens,
                "completed-prefill fixture was not actually fully ingested");
        const auto completed_retry = engine.generate(engine.prepare(completed_input), options(true));
        require(completed_retry.reused_prompt_tokens > 0 &&
                    completed_retry.reused_prompt_tokens == completed_control.reused_prompt_tokens &&
                    completed_retry.generated_token_ids == completed_control.generated_token_ids,
                "cancelling after complete prefill lost or corrupted its valid checkpoint");
        std::cout << "PASS completed prefill cancelled before output prompt=" << completed_tokens
                  << " processed=" << completed_processed << " reused="
                  << completed_retry.reused_prompt_tokens << '\n';

        std::string long_text;
        for (int i = 0; i < 128; ++i) {
            long_text += "Distinct cold prefill cancellation fixture without a retained checkpoint. ";
        }
        auto long_input = prompt(long_text);
        auto long_prepared = engine.prepare(long_input);
        while (long_prepared.summary().prompt_tokens <= config.prefill_chunk) {
            long_text += long_text;
            long_input = prompt(long_text);
            long_prepared = engine.prepare(long_input);
        }
        const auto long_tokens = long_prepared.summary().prompt_tokens;
        require(long_tokens > config.prefill_chunk && long_tokens + 32 < config.max_context,
                "incomplete-prefill fixture does not cross a work-unit boundary with output headroom");
        const auto before_prefill = engine.runtime_stats().computed_prefill_tokens;
        const auto interrupted = engine.generate(std::move(long_prepared), options(true, 32), nullptr,
            ninfer::CancellationView([&] {
                return engine.runtime_stats().computed_prefill_tokens > before_prefill;
            }));
        require(interrupted.finish_reason == ninfer::FinishReason::Cancelled &&
                    interrupted.generated_token_ids.empty(),
                "fresh prefill was not cancelled before its first licensed token");
        const auto partial_processed = engine.runtime_stats().computed_prefill_tokens - before_prefill;
        require(partial_processed > 0 && partial_processed < long_tokens,
                "incomplete-prefill fixture finished ingestion before cancellation");
        const auto after_interrupted = engine.generate(engine.prepare(long_input), options(true));
        require(after_interrupted.reused_prompt_tokens == 0,
                "incomplete fresh prefill was incorrectly marked as a validated checkpoint");
        const auto fresh_long = engine.generate(engine.prepare(long_input), options(false));
        require(after_interrupted.generated_token_ids == fresh_long.generated_token_ids,
                "interrupted prefill left stale recurrent or KV state");
        std::cout << "PASS fresh prefill cancellation discards incomplete state prompt=" << long_tokens
                  << " processed=" << partial_processed << " chunk=" << config.prefill_chunk << '\n';
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "cancel_prefix_real: " << error.what() << '\n';
        return 1;
    }
}
