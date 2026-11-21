#include "ninfer/engine.h"
#include <algorithm>
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
void require(bool value,const char* text){if(!value)throw std::runtime_error(text);}
ninfer::PromptInput input(int color,bool long_input){
    ninfer::PromptInput p;p.options.enable_thinking=false;
    ninfer::ChatMessage m;m.role="user";
    if(color>=0){
        ninfer::MessagePart image;image.kind=ninfer::MessagePartKind::Media;
        image.media.kind=ninfer::MediaKind::Image;image.media.media_type="image/x-portable-pixmap";
        image.media.source_name=color==0?"red.ppm":"blue.ppm";
        const std::string header="P6\n64 64\n255\n";image.media.bytes.assign(header.begin(),header.end());
        for(int pixel=0;pixel<64*64;++pixel){image.media.bytes.push_back(color==0?240:10);image.media.bytes.push_back(20);image.media.bytes.push_back(color==0?10:240);}
        m.parts.push_back(std::move(image));
    }
    std::string text;
    if(long_input){for(int i=0;i<150;++i)text+="This background paragraph is unrelated to the final question. Ignore it when answering. ";}
    text+=color>=0?"Name the dominant color of the image. Reply with only one English color word.":"Reply with exactly the word READY.";
    m.parts.push_back({.kind=ninfer::MessagePartKind::Text,.text=std::move(text),.media={}});
    p.messages.push_back(std::move(m));return p;
}
bool contains(std::string s,std::string word){for(auto&c:s)c=static_cast<char>(std::tolower(static_cast<unsigned char>(c)));return s.find(word)!=std::string::npos;}
}
int main(){
 const auto* artifact=std::getenv("NINFER_MULTI_PREFILL_ARTIFACT");if(!artifact){std::cout<<"SKIP NINFER_MULTI_PREFILL_ARTIFACT\n";return 77;}
 try{
    ninfer::EngineOptions config;config.artifact_path=artifact;config.max_context=4096;
    config.kv_capacity=ninfer::KvCapacityPolicy::explicit_capacity(16384);config.max_concurrency=4;
    config.max_prefill_lanes=4;config.prefill_chunk=128;config.prefill_live_chunk=128;
    config.kv_cache=ninfer::KvCacheStorage::Int8Group64;config.speculative.backend=ninfer::SpeculativeBackend::Mtp;
    config.speculative.draft_tokens=1;config.enable_vision=true;
    if(std::getenv("NINFER_TEST_TIMED_PREFILL") != nullptr) {
        config.prefill_target_decode_tps=24.0;
        config.prefill_decode_rounds=24;
    }
    if(std::getenv("NINFER_TEST_PACKED_PREFILL") != nullptr) {
        config.pack_prefill=true;config.prefill_chunk=1024;
    }
    ninfer::Engine engine(config);
    const auto reserved=engine.memory_summary().request_transient.capacity_bytes;
    ninfer::RequestOptions request;request.execution.requested_output_tokens=32;request.execution.sampling.temperature=0.0F;
    request.execution.allow_prefix_reuse=false;
    // Establish one real image's transient cost while idle. Reading memory_summary
    // takes the execution mutex, so it must not delay cancellation of a live owner.
    const auto blue_alone=engine.generate(engine.prepare(input(1,true)),request);
    std::cout<<"isolated_blue="<<blue_alone.content<<" tokens="<<blue_alone.generated_token_ids.size()<<std::endl;
    // Use the blue control for the single-image peak, so no earlier completed
    // red prompt could legitimately supply a reusable prefix after cancellation.
    const auto one_image_peak=engine.memory_summary().request_transient.peak_used_bytes;
    require(one_image_peak>0,"single image allocated no measurable transient slice");
    engine.reset_memory_peaks();
    auto red=engine.prepare(input(0,true));auto blue=engine.prepare(input(1,true));auto text=engine.prepare(input(-1,true));
    auto h0=engine.submit(std::move(red),request);auto h1=engine.submit(std::move(blue),request);auto h2=engine.submit(std::move(text),request);
    const auto deadline=Clock::now()+std::chrono::seconds(60);bool concurrent=false;
    while(Clock::now()<deadline){if(engine.runtime_stats().prefilling_requests>=3){concurrent=true;break;}std::this_thread::sleep_for(std::chrono::milliseconds(5));}
    require(concurrent,"did not observe three real staged owners");
    // runtime_stats is a published boundary snapshot, not a GPU execution lock.
    // Cancel before any blocking memory query can let the prompt finish.
    const auto observed_count=engine.runtime_stats().prefilling_requests;
    h0={};
    std::cout<<"observed_prefills="<<observed_count<<" cancelled red during staged work"<<std::endl;
    const auto b=h1.wait();const auto t=h2.wait();
    std::cout<<"blue_output="<<b.content<<" blue_tokens="<<b.generated_token_ids.size()
             <<" text_output="<<t.content<<std::endl;
    require(contains(b.content,"blue")&&!contains(b.content,"red"),"cancelled red owner corrupted blue media");
    require(b.generated_token_ids==blue_alone.generated_token_ids,
            "surviving staged image differs from the identical isolated control");
    require(contains(t.content,"ready"),"concurrent text state was corrupted");
    const auto joint=engine.memory_summary().request_transient;
    require(joint.capacity_bytes==reserved && joint.used_bytes==0 && joint.peak_used_bytes<=reserved,
            "joint completion leaked or expanded transient storage");
    require(one_image_peak<=joint.peak_used_bytes/2,
            "peak accounting does not prove two overlapping media slices");
    std::cout<<"one_image_peak="<<one_image_peak<<" joint_peak="<<joint.peak_used_bytes
             <<" unchanged_capacity="<<reserved<<std::endl;
    request.execution.allow_prefix_reuse=true;
    const auto fresh_red=engine.generate(engine.prepare(input(0,true)),request);
    require(fresh_red.reused_prompt_tokens==0,"incomplete cancelled prefix was retained");
    require(contains(fresh_red.content,"red")&&!contains(fresh_red.content,"blue"),"released region retained another request's image");
    const auto cached_red=engine.generate(engine.prepare(input(0,true)),request);
    require(cached_red.reused_prompt_tokens>0&&contains(cached_red.content,"red"),"completed image prefix not reusable");
    const auto after=engine.memory_summary().request_transient;
    require(after.capacity_bytes==reserved&&after.used_bytes==0&&after.peak_used_bytes<=reserved,"completion leaked or expanded transient storage");
    std::cout<<"PASS three staged owners, cancellation/media isolation, safe region reuse, prefix invalidation and unchanged memory capacity\n";
    return 0;
 }catch(const std::exception&e){std::cerr<<"multi_prefill_real: "<<e.what()<<'\n';return 1;}
}
