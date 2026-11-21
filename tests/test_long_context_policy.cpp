#include "targets/qwen3_6/impl/runtime/long_context_policy.h"
#include <iostream>
using namespace ninfer::targets::qwen3_6;
int main(){
 for(auto mode:{LongContextComputation::RetainedThroughput,LongContextComputation::Complete}) {
  for(unsigned keys:{0U,32767U,32768U,65536U,262144U}) {
   const bool expected=mode==LongContextComputation::RetainedThroughput && keys>=32768;
   if(use_retained_long_context(mode,keys)!=expected)return 1;
   if(long_context_split_subdivision(mode,keys)!=(expected?8U:1U))return 4;
  }
  for(unsigned width=1;width<=6;++width)for(unsigned extent=0;extent<width;++extent) {
   const bool expected=mode==LongContextComputation::RetainedThroughput&&width==5&&extent==4;
   if(license_retained_d4(mode,width,extent)!=expected)return 2;
  }
 }
 if(static_cast<unsigned>(LongContextComputation::RetainedThroughput)!=0 ||
    static_cast<unsigned>(LongContextComputation::Complete)!=1)return 3;
 for(auto mode:{LongContextComputation::RetainedThroughput,LongContextComputation::Complete})
  for(bool media:{false,true}) {
   if(stop_at_message_end(mode,media)!=(media||mode==LongContextComputation::Complete))return 5;
  }
 for(auto mode:{LongContextComputation::RetainedThroughput,LongContextComputation::Complete})
  for(bool mtp:{false,true})for(unsigned sessions:{1U,2U})for(unsigned depth:{1U,3U,4U})
   for(bool sidecar:{false,true})for(unsigned width:{0U,64U,128U,192U,8192U,8193U,262144U,262208U}) {
    const bool expected=width==0 || (mode==LongContextComputation::Complete && mtp &&
        sessions==1 && depth==3 && !sidecar && width>=128 && width<=262144 && width%64==0);
    if(complete_proposal_window_allowed(mode,mtp,sessions,depth,sidecar,width)!=expected)return 6;
   }
 std::cout<<"PASS complete target/no-licensed-drafts boundaries, scoped proposal windows and completed-answer stopping; historical policy unchanged\n";
}
