#include "ninfer/engine.h"
#include <array>
#include <chrono>
#include <cstdlib>
#include <iostream>
#include <string>
#include <stdexcept>
#include <vector>
namespace {
void require(bool ok,const char* why){if(!ok)throw std::runtime_error(why);}
const std::array<std::string,4> names{"ALDER","BIRCH","CEDAR","MAPLE"};
const std::array<unsigned,4> serials{173,286,491,628};
const std::array<std::string,4> rooms{"EAST","WEST","NORTH","SOUTH"};
std::string expected(unsigned i,bool changed=false){return names[i]+"|"+std::to_string(serials[i]+(changed?100:0))+"|"+rooms[i];}
ninfer::PromptInput prompt(unsigned i,bool changed=false){
    ninfer::PromptInput p;p.options.enable_thinking=false;
    std::string text="Private register: key "+names[i]+", serial "+std::to_string(serials[i]+(changed?100:0))+", room "+rooms[i]+".\n";
    if(i%2)text+="Be exact. ";
    text+="\nReturn only key|serial|room. No explanation.";
    ninfer::ChatMessage m;m.role="user";m.parts.push_back({.kind=ninfer::MessagePartKind::Text,.text=std::move(text),.media={}});p.messages.push_back(std::move(m));return p;
}
std::string trim(std::string text){
    auto a=text.find_first_not_of(" \t\r\n"),b=text.find_last_not_of(" \t\r\n");return a==std::string::npos?"":text.substr(a,b-a+1);
}
}
int main(){
 const char* artifact=std::getenv("NINFER_PACKED_PREFILL_ARTIFACT");
 if(!artifact){std::cout<<"SKIP NINFER_PACKED_PREFILL_ARTIFACT\n";return 77;}
 try {
    ninfer::EngineOptions options;options.artifact_path=artifact;options.max_context=8192;
    options.kv_capacity=ninfer::KvCapacityPolicy::explicit_capacity(32768);
    options.max_concurrency=4;options.max_prefill_lanes=4;options.pack_prefill=true;
    options.prefill_chunk=2048;options.prefill_live_chunk=128;
    options.prefill_decode_rounds=24;options.prefill_target_decode_tps=24;
    options.kv_cache=ninfer::KvCacheStorage::Int8Group64;options.enable_vision=true;
    options.speculative.backend=ninfer::SpeculativeBackend::Mtp;options.speculative.draft_tokens=1;
    options.speculative.force_full_target_head=true;
    options.speculative.sm75_radix_projections=true;options.speculative.sm75_resident_radix=true;
    ninfer::Engine engine(options);
    ninfer::RequestOptions request;request.execution.requested_output_tokens=64;
    request.execution.sampling.temperature=0;request.execution.allow_prefix_reuse=false;
    std::array<std::uint32_t,4> lengths{};
    auto inputs=[&](bool changed){
        std::vector<ninfer::PreparedPrompt> prepared;
        for(unsigned i=0;i<4;++i){prepared.push_back(engine.prepare(prompt(i,changed)));lengths[i]=prepared.back().summary().prompt_tokens;
            require(lengths[i]>1&&lengths[i]<64,"short ragged fixture outside bounds");}
        return prepared;
    };
    auto prepared=inputs(false);
    require(lengths[0]!=lengths[1]&&lengths[1]!=lengths[2],"fixture is not ragged");
    std::vector<ninfer::GenerationHandle> handles;
    for(auto& p:prepared)handles.push_back(engine.submit(std::move(p),request));
    for(unsigned i=0;i<4;++i){const auto r=handles[i].wait();
        std::cout<<"owner="<<i<<" tokens="<<lengths[i]<<" answer="<<r.content<<std::endl;
        require(trim(r.content)==expected(i),"ragged prefill mixed owner facts");
        require(r.finish_reason==ninfer::FinishReason::StopToken,"private-register answer did not finish naturally");}
    require(engine.runtime_stats().packed_prefill_rounds>0,"ready ragged burst never packed");
    // Above the backend boundary the old immediate path remains intact; these
    // prompts must not be delayed to build a bigger cohort.
    const auto packed_before_medium=engine.runtime_stats().packed_prefill_rounds;
    handles.clear();prepared.clear();
    for(unsigned i=0;i<4;++i){auto medium=prompt(i);medium.messages[0].parts[0].text +=
        " Background notes do not change the register. Ignore this unrelated descriptive text."
        " Background notes do not change the register. Ignore this unrelated descriptive text."
        " Background notes do not change the register. Ignore this unrelated descriptive text.";
        prepared.push_back(engine.prepare(std::move(medium)));
        require(prepared.back().summary().prompt_tokens>=64,"medium boundary fixture too short");}
    for(auto& input:prepared)handles.push_back(engine.submit(std::move(input),request));
    for(unsigned i=0;i<4;++i)require(trim(handles[i].wait().content)==expected(i),"medium immediate path corrupted");
    require(engine.runtime_stats().packed_prefill_rounds==packed_before_medium,"medium cold request was inappropriately batched");
    handles.clear();prepared=inputs(true);
    // Drop one ownership handle while the other submitted requests remain live.
    for(auto& p:prepared)handles.push_back(engine.submit(std::move(p),request));
    handles[0]={};
    for(unsigned i=1;i<4;++i){const auto r=handles[i].wait();require(trim(r.content)==expected(i,true),"cancelled lane corrupted survivor");}
    request.execution.allow_prefix_reuse=true;
    const auto changed=engine.generate(engine.prepare(prompt(0,true)),request);
    require(trim(changed.content)==expected(0,true),"cancelled or retained state corrupted retry");
    const auto warm=engine.generate(engine.prepare(prompt(0,true)),request);
    require(warm.reused_prompt_tokens>0&&trim(warm.content)==expected(0,true),"exact prefix no longer reusable");
    request.execution.allow_prefix_reuse=false;request.execution.requested_output_tokens=1;
    prepared=inputs(false);handles.clear();
    for(auto& p:prepared)handles.push_back(engine.submit(std::move(p),request));
    for(auto& handle:handles)require(handle.wait().generated_token_ids.size()==1,"staged finalization exceeded one-token budget");
    require(engine.memory_summary().request_transient.used_bytes==0,"transient owner leaked");
    std::cout<<"PASS ready ragged ownership, cancellation, changed source, prefix reuse and one-token service budget\n";
    return 0;
 }catch(const std::exception& e){std::cerr<<"ragged prefill: "<<e.what()<<std::endl;return 1;}
}
