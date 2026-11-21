#include "ninfer/engine.h"
#include "core/device.h"

#include <cstddef>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

int main(int argc, char** argv) {
    const char* artifact = std::getenv("NINFER_RELOAD_ARTIFACT");
    if (artifact == nullptr || *artifact == '\0') { return 77; }
    try {
        CUDA_CHECK(cudaSetDevice(0));
        std::vector<ninfer::TokenId> expected;
        std::size_t baseline = 0;
        for (int cycle=0; cycle<3; ++cycle) {
            {
                ninfer::EngineOptions config;
                config.artifact_path = artifact;
                config.max_context = 1024;
                config.max_concurrency = argc > 1 ? std::stoul(argv[1]) : 1;
                config.kv_capacity = ninfer::KvCapacityPolicy::explicit_capacity(1024);
                config.prefill_chunk = 128;
                config.kv_cache = ninfer::KvCacheStorage::Int8Group64;
                config.speculative.backend = ninfer::SpeculativeBackend::Mtp;
                config.speculative.draft_tokens = 1;
                config.speculative.proposal_head = ninfer::ProposalHead::Full;
                if (argc > 2) {
                    if (std::string(argv[2]) != "full-vocab-draft" || config.max_concurrency != 1) {
                        throw std::invalid_argument("reload profile requires C1 full-vocab-draft");
                    }
                    config.speculative.draft_tokens = 3;
                    config.speculative.force_full_target_head = true;
                    config.speculative.fast_full_vocab_draft = true;
                }
                if (std::getenv("NINFER_RELOAD_RETAINED_C1") != nullptr) {
                    config.max_concurrency = 1;
                    config.speculative.draft_tokens = 3;
                    config.speculative.proposal_head = ninfer::ProposalHead::Optimized;
                    config.speculative.greedy_margin_delta = 7.0F;
                    config.speculative.retained_mtp_packed_prefix = true;
                }
                if (std::getenv("NINFER_RELOAD_RADIX_C1") != nullptr) {
                    config.max_concurrency = 1;
                    config.speculative.draft_tokens = 3;
                    config.speculative.proposal_head = ninfer::ProposalHead::Optimized;
                    config.speculative.retained_mtp_packed_prefix = true;
                    config.speculative.greedy_margin_delta = -1.0F;
                    config.speculative.sm75_radix_projections = true;
                }
                if (std::getenv("NINFER_RELOAD_COMPOSED_C1") != nullptr) {
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
                if(std::getenv("NINFER_RELOAD_SESSION_COPY")!=nullptr) {
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
                if(std::getenv("NINFER_RELOAD_NU3_C1")!=nullptr) {
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
                if(std::getenv("NINFER_RELOAD_TARGET_UNION")!=nullptr) {
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
                ninfer::Engine engine(config);
                ninfer::PromptInput prompt;
                prompt.options.enable_thinking = false;
                ninfer::ChatMessage message;
                message.role = "user";
                message.parts.push_back({.kind=ninfer::MessagePartKind::Text,
                                         .text="Reply with exactly the word READY.", .media={}});
                prompt.messages.push_back(std::move(message));
                ninfer::RequestOptions options;
                options.execution.requested_output_tokens = 32;
                options.execution.sampling.temperature = 0.0F;
                options.execution.allow_prefix_reuse = false;
                const auto memory = engine.memory_summary();
                std::cout << "cycle=" << cycle << " lanes=" << config.max_concurrency
                          << " graph_observed=" << memory.cuda_graph_observed_bytes
                          << " graph_allowance=" << memory.cuda_graph_allowance_bytes << '\n';
                const auto result = engine.generate(engine.prepare(std::move(prompt)), options);
                if (result.generated_token_ids.empty()) { throw std::runtime_error("reload test generated no tokens"); }
                if (cycle == 0) { expected = result.generated_token_ids; }
                else if (expected != result.generated_token_ids) {
                    throw std::runtime_error("fresh Engine reload changed deterministic output");
                }
            }
            CUDA_CHECK(cudaDeviceSynchronize());
            std::size_t available = 0, total = 0;
            CUDA_CHECK(cudaMemGetInfo(&available, &total));
            if (cycle == 0) { baseline = available; }
            std::cout << "reload=" << cycle << " free_bytes=" << available
                      << " warm_baseline=" << baseline << " exact_tokens=" << expected.size() << '\n';
            if (available < baseline) {
                throw std::runtime_error("Engine reload retained GPU allocation after destruction");
            }
        }
        std::cout << "PASS actual model load/generate/destroy/reload; exact output; no GPU bytes lost after warm-up\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "engine_reload_real: " << error.what() << '\n';
        return 1;
    }
}
