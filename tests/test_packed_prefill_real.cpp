#include "ninfer/engine.h"
#include <array>
#include <chrono>
#include <cctype>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>
namespace {
using Clock=std::chrono::steady_clock;
void require(bool value,const char* message){if(!value)throw std::runtime_error(message);}
const std::array<std::string,4> names{"ALDER","BIRCH","CEDAR","MAPLE"};
const std::array<int,4> numbers{173,286,491,628};
const std::array<std::string,4> rooms{"EAST","WEST","NORTH","SOUTH"};
ninfer::PromptInput input(std::size_t which){
    ninfer::PromptInput p;p.options.enable_thinking=false;
    std::string text="This request has a private register. Key: "+names[which]+". Serial: "+
        std::to_string(numbers[which])+". Room: "+rooms[which]+". These are the only register values.\n";
    for(int i=0;i<360;++i)text+="Unrelated background: records must stay isolated, and missing information must never be invented. ";
    text+="\nRead the register at the start. Return only its key, serial and room separated by |. No quotes, spaces, Markdown, or explanations.";
    ninfer::ChatMessage m;m.role="user";
    m.parts.push_back({.kind=ninfer::MessagePartKind::Text,.text=std::move(text),.media={}});
    p.messages.push_back(std::move(m));return p;
}
std::string trimmed(std::string s){
    while(!s.empty()&&std::isspace(static_cast<unsigned char>(s.back())))s.pop_back();
    while(!s.empty()&&std::isspace(static_cast<unsigned char>(s.front())))s.erase(s.begin());
    return s;
}
std::string expected(std::size_t i){return names[i]+"|"+std::to_string(numbers[i])+"|"+rooms[i];}
}
int main(){
 const char* artifact=std::getenv("NINFER_PACKED_PREFILL_ARTIFACT");
 if(!artifact){std::cout<<"SKIP NINFER_PACKED_PREFILL_ARTIFACT\n";return 77;}
 try{
    ninfer::EngineOptions o;o.artifact_path=artifact;o.max_context=8192;o.max_concurrency=4;
    o.kv_capacity=ninfer::KvCapacityPolicy::explicit_capacity(32768);
    o.prefill_chunk=1024;o.prefill_live_chunk=128;o.max_prefill_lanes=4;o.pack_prefill=true;
    o.prefill_target_decode_tps=24.0;o.prefill_decode_rounds=24;
    o.kv_cache=ninfer::KvCacheStorage::Int8Group64;o.enable_vision=true;
    o.speculative.backend=ninfer::SpeculativeBackend::Mtp;o.speculative.draft_tokens=1;
    o.speculative.force_full_target_head=true;
    ninfer::Engine engine(o);
    ninfer::RequestOptions request;request.execution.requested_output_tokens=64;
    request.execution.sampling.temperature=0.0F;request.execution.allow_prefix_reuse=false;
    std::vector<ninfer::PreparedPrompt> prepared;
    for(std::size_t i=0;i<4;++i){
        prepared.push_back(engine.prepare(input(i)));
        std::cout<<"input="<<i<<" tokens="<<prepared.back().summary().prompt_tokens<<std::endl;
        require(prepared.back().summary().prompt_tokens>3*o.prefill_chunk &&
                prepared.back().summary().prompt_tokens+64<o.max_context,"fixture geometry invalid");
    }
    std::vector<ninfer::GenerationHandle> handles;
    for(auto& p:prepared)handles.push_back(engine.submit(std::move(p),request));
    std::vector<std::vector<ninfer::TokenId>> controls;
    for(std::size_t i=0;i<4;++i){
        const auto result=handles[i].wait();std::cout<<"initial="<<i<<" answer="<<result.content<<std::endl;
        require(trimmed(result.content)==expected(i),"packed request mixed another owner's facts");
        controls.push_back(result.generated_token_ids);
    }
    auto stats=engine.runtime_stats();
    require(stats.packed_prefill_rounds==0,"all-cold reference was unnecessarily split into small work");
    handles.clear();prepared.clear();
    // Keep two real streams decoding while two long new text requests ingest.
    ninfer::PromptInput long_answer;long_answer.options.enable_thinking=false;
    ninfer::ChatMessage background;background.role="user";
    background.parts.push_back({.kind=ninfer::MessagePartKind::Text,
        .text="Write 200 numbered sections explaining software testing techniques. Give three detailed sentences per section and continue until all 200 are complete.",.media={}});
    long_answer.messages.push_back(std::move(background));
    auto stream_request=request;stream_request.execution.requested_output_tokens=4096;
    auto live0=engine.submit(engine.prepare(long_answer),stream_request);
    auto live1=engine.submit(engine.prepare(long_answer),stream_request);
    auto deadline=Clock::now()+std::chrono::seconds(90);
    while(Clock::now()<deadline && engine.runtime_stats().decode_ready_requests<2)
        std::this_thread::sleep_for(std::chrono::milliseconds(2));
    require(engine.runtime_stats().decode_ready_requests>=2,"established streams did not begin");
    prepared.push_back(engine.prepare(input(0)));prepared.push_back(engine.prepare(input(1)));
    const auto before=engine.runtime_stats().packed_prefill_rounds;
    for(auto& p:prepared)handles.push_back(engine.submit(std::move(p),request));
    deadline=Clock::now()+std::chrono::seconds(90);bool observed=false;
    while(Clock::now()<deadline){
        stats=engine.runtime_stats();
        if(stats.packed_prefill_rounds>before && stats.prefilling_requests>=2){observed=true;break;}
        std::this_thread::sleep_for(std::chrono::milliseconds(2));
    }
    require(observed,"no active packed boundary available for cancellation");
    handles[0]={};
    std::cout<<"cancelled_owner=0 at_packed_round="<<stats.packed_prefill_rounds
             <<" packed_tokens="<<stats.packed_prefill_tokens<<std::endl;
    const auto survivor=handles[1].wait();
    std::cout<<"survivor=1 answer="<<survivor.content<<std::endl;
    require(trimmed(survivor.content)==expected(1),"cancelled owner corrupted a surviving request");
    live0={};live1={};
    request.execution.allow_prefix_reuse=true;
    const auto retry=engine.generate(engine.prepare(input(0)),request);
    require(retry.reused_prompt_tokens==0 && trimmed(retry.content)==expected(0),
            "cancelled partial prefix survived or retry state was corrupted");
    const auto warm=engine.generate(engine.prepare(input(0)),request);
    require(warm.reused_prompt_tokens>0 && trimmed(warm.content)==expected(0),"finished prefix no longer reusable");
    const auto memory=engine.memory_summary();
    require(memory.request_transient.used_bytes==0,"packed requests leaked media transient storage");
    std::cout<<"PASS unchanged all-cold path, packed live-join owners, mid-prefill cancellation, clean retry and prefix reuse\n";
    return 0;
 }catch(const std::exception& e){std::cerr<<"packed_prefill_real: "<<e.what()<<'\n';return 1;}
}
