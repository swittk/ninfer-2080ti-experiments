#include "ninfer/engine.h"
#include <algorithm>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <iterator>
#include <stdexcept>
#include <string>
#include <vector>
namespace {
void require(bool ok,const char* message){if(!ok)throw std::runtime_error(message);}
std::string read(const char* path){std::ifstream f(path);if(!f)throw std::runtime_error("missing copy fixture");return {std::istreambuf_iterator<char>(f),{}};}
std::string normal(std::string text){
    while(!text.empty()&&text.back()=='\n')text.pop_back();
    const std::string end="<|im_end|>";
    if(text.ends_with(end))text.resize(text.size()-end.size());
    while(!text.empty()&&text.back()=='\n')text.pop_back();
    return text;
}
ninfer::PromptInput prompt(const std::string& source){
    ninfer::PromptInput p;p.options.enable_thinking=false;
    ninfer::ChatMessage m;m.role="user";
    m.parts.push_back({.kind=ninfer::MessagePartKind::Text,
        .text="Edit the Python file below. Change only the default ceiling_seconds argument of retry_delay from 300 to 120. Return the complete file, retaining every other byte except a possible final newline. Do not use Markdown fences or explanations.\n\n"+source,.media={}});
    p.messages.push_back(std::move(m));return p;
}
std::string expected(std::string source){
    const std::string old="def retry_delay(attempt, base_seconds=2, ceiling_seconds=300):";
    auto at=source.find(old);require(at!=std::string::npos,"fixture has no intended edit");
    source.replace(at,old.size(),"def retry_delay(attempt, base_seconds=2, ceiling_seconds=120):");return source;
}
ninfer::RequestOptions request(bool reuse,std::uint32_t max=3072){
    ninfer::RequestOptions r;r.execution.requested_output_tokens=max;
    r.execution.allow_prefix_reuse=reuse;r.execution.sampling.temperature=0.f;return r;
}
struct Cancel final:ninfer::OutputSink{
    std::uint32_t seen=0;
    void publish(ninfer::OutputDelta delta)override{seen=std::max(seen,delta.committed_tokens);}
};
}
int main(){
    const char* artifact=std::getenv("NINFER_SESSION_COPY_ARTIFACT");
    const char* fixture=std::getenv("NINFER_SESSION_COPY_FIXTURE");
    if(!artifact||!fixture){std::cout<<"SKIP session-copy real inputs absent\n";return 77;}
    try{
        const auto source=read(fixture);const auto want=normal(expected(source));
        ninfer::EngineOptions o;o.artifact_path=artifact;o.max_context=8192;o.max_concurrency=1;
        o.kv_capacity=ninfer::KvCapacityPolicy::explicit_capacity(8192);o.prefill_chunk=128;
        o.kv_cache=ninfer::KvCacheStorage::Int8Group64;
        o.speculative.backend=ninfer::SpeculativeBackend::Mtp;o.speculative.draft_tokens=3;
        o.speculative.force_full_target_head=true;o.speculative.fast_full_vocab_draft=true;
        o.speculative.retained_mtp_packed_prefix=true;o.speculative.sm75_radix_projections=true;
        o.speculative.session_copy_draft=true;
        ninfer::Engine engine(o);
        const auto cold=engine.generate(engine.prepare(prompt(source)),request(false));
        std::cout<<"cold_tokens="<<cold.generated_token_ids.size()<<" copy_rounds="<<cold.speculative.session_copy_rounds
                 <<" resumes="<<cold.speculative.session_copy_factory_resumes<<std::endl;
        require(normal(cold.content)==want,"cold copy did not return the exact complete edit");
        require(cold.speculative.session_copy_rounds>0 && cold.speculative.session_copy_factory_resumes>0,
                "test did not exercise copy verification and native proposal resumption");
        const auto warm=engine.generate(engine.prepare(prompt(source)),request(true));
        require(warm.reused_prompt_tokens>0 && warm.generated_token_ids==cold.generated_token_ids,
                "warm checkpoint changed complete edited output");
        for(unsigned boundary:{128U,600U}){
            Cancel sink;
            const auto interrupted=engine.generate(engine.prepare(prompt(source)),request(true),&sink,
                ninfer::CancellationView([&]{return sink.seen>=boundary;}));
            require(interrupted.finish_reason==ninfer::FinishReason::Cancelled && sink.seen>=boundary,
                    "copy request failed to cancel at a real output boundary");
            require(interrupted.speculative.session_copy_rounds>0,"cancellation did not interrupt real copy work");
            require(interrupted.generated_token_ids.size()<cold.generated_token_ids.size(),"cancel arrived after completion");
            const auto retry=engine.generate(engine.prepare(prompt(source)),request(true));
            require(retry.generated_token_ids==warm.generated_token_ids && retry.reused_prompt_tokens==warm.reused_prompt_tokens,
                    "copy cancellation corrupted the validated prompt checkpoint");
            std::cout<<"PASS cancellation boundary="<<boundary<<" copied_rounds="
                     <<interrupted.speculative.session_copy_rounds<<" reused="<<retry.reused_prompt_tokens<<std::endl;
        }
        const auto bounded=engine.generate(engine.prepare(prompt(source)),request(true,128));
        require(bounded.generated_token_ids.size()==128 &&
                std::equal(bounded.generated_token_ids.begin(),bounded.generated_token_ids.end(),cold.generated_token_ids.begin()),
                "copy speculation exceeded or corrupted the output budget");
        // A later request owns a new index; no source tokens from the previous
        // model answer may supply this unrelated short prompt.
        auto simple=prompt("");simple.messages[0].parts[0].text="Reply only with READY.";
        const auto plain=engine.generate(engine.prepare(simple),request(false,32));
        require(normal(plain.content)=="READY" && plain.speculative.session_copy_rounds==0,
                "unrelated request inherited stale copy history");
        const auto final=engine.generate(engine.prepare(prompt(source)),request(false));
        require(final.reused_prompt_tokens==0 && final.generated_token_ids==cold.generated_token_ids,
                "new cold request inherited stale proposal state");
        const auto memory=engine.memory_summary();
        std::cout<<"graph_observed="<<memory.cuda_graph_observed_bytes
                 <<" graph_allowance="<<memory.cuda_graph_allowance_bytes<<std::endl;
        std::cout<<"PASS exact copy/mismatch/resume, warm reuse, cancellation, output bound and request isolation\n";
        return 0;
    }catch(const std::exception& e){std::cerr<<"session_copy_real: "<<e.what()<<'\n';return 1;}
}
