#include "ninfer/engine.h"
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
std::uint64_t digest(const std::vector<ninfer::TokenId>& ids) {
    std::uint64_t hash=14695981039346656037ULL;
    for(const auto id:ids) {
        std::uint32_t value=static_cast<std::uint32_t>(id);
        for(int byte=0;byte<4;++byte){hash^=value&255U;hash*=1099511628211ULL;value>>=8;}
    }
    return hash;
}
ninfer::PromptInput prompt(std::string text) {
    ninfer::PromptInput input;input.options.enable_thinking=false;
    ninfer::ChatMessage message;message.role="user";
    message.parts.push_back({.kind=ninfer::MessagePartKind::Text,.text=std::move(text),.media={}});
    input.messages.push_back(std::move(message));return input;
}
}

int main() {
    const char* artifact=std::getenv("NINFER_RETAINED_C1_ARTIFACT");
    if(!artifact || !*artifact){std::cout<<"skip: NINFER_RETAINED_C1_ARTIFACT not set\n";return 77;}
    try {
        struct RetainedProfile {
            const char* name;
            float margin;
            std::uint64_t token_digest;
            std::uint64_t rounds, drafted, accepted;
            std::vector<std::uint64_t> by_position;
        };
        const RetainedProfile profiles[] = {
            {"strict79", -1.0F, 0x14660c413b4cf185ULL, 43, 129, 83, {35,27,21}},
            {"margin7", 7.0F, 0xef4dc9c008396373ULL, 33, 99, 94, {33,32,29}},
        };
        for (const auto& profile : profiles) {
        ninfer::EngineOptions config;config.artifact_path=artifact;
        config.max_context=8192;config.kv_capacity=ninfer::KvCapacityPolicy::explicit_capacity(8192);
        config.max_concurrency=1;config.prefill_chunk=128;
        config.kv_cache=ninfer::KvCacheStorage::Int8Group64;
        config.speculative.backend=ninfer::SpeculativeBackend::Mtp;
        config.speculative.draft_tokens=3;
        config.speculative.proposal_head=ninfer::ProposalHead::Optimized;
        config.speculative.greedy_margin_delta=profile.margin;
        config.speculative.retained_mtp_packed_prefix=true;
        ninfer::Engine engine(config);
        ninfer::RequestOptions request;request.execution.sampling.temperature=0.0F;
        request.execution.allow_prefix_reuse=false;request.output.raw=true;
        request.execution.requested_output_tokens=8;
        (void)engine.generate(engine.prepare(prompt("Reply with exactly READY.")),request);
        request.execution.requested_output_tokens=128;
        std::vector<ninfer::TokenId> expected;
        for(int pass=0;pass<2;++pass) {
            const auto result=engine.generate(engine.prepare(prompt(
                "Write a concise technical explanation of CUDA memory bandwidth, quantized transformer inference, and speculative decoding.")),request);
            const auto& spec=result.speculative;
            if(result.generated_token_ids.size()!=128 || digest(result.generated_token_ids)!=profile.token_digest ||
               spec.rounds!=profile.rounds || spec.drafted_tokens!=profile.drafted ||
               spec.accepted_tokens!=profile.accepted || spec.accepted_per_position!=profile.by_position) {
                throw std::runtime_error("retained C1 golden output or speculative trajectory changed");
            }
            if(pass==0)expected=result.generated_token_ids;
            else if(expected!=result.generated_token_ids)throw std::runtime_error("repeated C1 request changed bytes");
            std::cout<<"PASS retained C1 "<<profile.name<<" after warm-up pass="<<pass
                     <<" tokens=128 rounds="<<spec.rounds<<" drafted="<<spec.drafted_tokens
                     <<" accepted="<<spec.accepted_tokens<<" decode_tps="
                     <<127.0/result.timings.decode_seconds<<'\n';
        }
        }
        return 0;
    } catch(const std::exception& error){std::cerr<<"retained_c1_real: "<<error.what()<<'\n';return 1;}
}
