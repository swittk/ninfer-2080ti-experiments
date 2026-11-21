#include "targets/qwen3_6_27b/impl/dual_stage.h"
#include "core/device.h"
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <vector>
using namespace ninfer;
using namespace ninfer::targets;
int main(){
 const char* artifact=std::getenv("NINFER_DUAL_DRAFT_RELOAD_ARTIFACT");
 if(!artifact)return 77;
 try{
  CUDA_CHECK(cudaSetDevice(0));std::size_t baseline=0;
  TokenId expected=0;std::vector<TokenId> drafts;
  for(int cycle=0;cycle<3;++cycle){
   {
    qwen3_6_27b::detail::TextShardStage stage(artifact,0,1024,{0,64,true,true},nullptr,128,
      KvCacheStorage::Int8Group64,1,4,true,{},false,qwen3_6::LongContextComputation::Complete,8192,true);
    const auto prompt=stage.prepare_text_prompt("What is 7 plus 5? Reply only with the number.",false);
    if(prompt.token_ids.empty()||prompt.token_ids.size()>128)throw std::runtime_error("reload fixture width");
    stage.prefill_tokens(prompt.token_ids,0,0);
    const auto generated=stage.project_token();
    stage.mtp_prefill_prompt_chunk(prompt.token_ids,0,prompt.token_ids.size(),true,generated);
    const auto proposed=stage.mtp_initial_drafts();
    if(cycle==0){expected=generated;drafts=proposed;}
    else if(generated!=expected||proposed!=drafts)throw std::runtime_error("reloaded draft policy changed output");
   }
   CUDA_CHECK(cudaDeviceSynchronize());std::size_t available=0,total=0;CUDA_CHECK(cudaMemGetInfo(&available,&total));
   std::cout<<"cycle="<<cycle<<" free_bytes="<<available<<" target_token="<<expected<<" proposal_count="<<drafts.size()<<std::endl;
   if(cycle==0)baseline=available;
   else if(available<baseline)throw std::runtime_error("dual draft model retained GPU memory after teardown");
  }
  std::cout<<"PASS three complete-target full-vocabulary-draft load/prefill/propose/destroy cycles\n";return 0;
 }catch(const std::exception& e){std::cerr<<"dual_draft_reload: "<<e.what()<<'\n';return 1;}
}
