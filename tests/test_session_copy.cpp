#include "runtime/engine/session_copy.h"
#include <array>
#include <cstdint>
#include <iostream>
#include <numeric>
#include <random>
#include <vector>
using ninfer::runtime::SessionCopyIndex;
int main(){
    std::array<std::int32_t,3> out{-1,-1,-1};
    SessionCopyIndex empty;
    if(empty.propose({},out)||empty.storage_bytes()!=0)return 1;
    std::vector<std::int32_t> source(128);std::iota(source.begin(),source.end(),1000);
    auto history=source;
    history.insert(history.end(),source.begin(),source.begin()+32);
    SessionCopyIndex index;
    if(!index.propose(history,out)||out!=std::array<std::int32_t,3>{1032,1033,1034})return 2;
    history.insert(history.end(),out.begin(),out.end());
    if(!index.propose(history,out)||out!=std::array<std::int32_t,3>{1035,1036,1037})return 3;
    auto wrong=source;wrong.insert(wrong.end(),source.begin(),source.begin()+32);wrong.back()=-12;
    SessionCopyIndex miss;if(miss.propose(wrong,out))return 4;
    SessionCopyIndex other;if(other.propose(source,out))return 5;
    // A hash hit can never manufacture unobserved data beyond the source bound.
    std::vector<std::int32_t> one(32,17);if(other.propose(one,out))return 6;
    index.reset();if(index.propose(source,out))return 7;
    // The same index may observe bounded appended history over many rounds.
    std::mt19937 rng(91357);
    std::vector<std::int32_t> long_history(70000);
    for(auto& token:long_history)token=static_cast<std::int32_t>(rng()%200000);
    long_history.insert(long_history.end(),source.begin(),source.end());
    long_history.insert(long_history.end(),source.begin(),source.begin()+32);
    SessionCopyIndex large;
    if(!large.propose(long_history,out)||out!=std::array<std::int32_t,3>{1032,1033,1034})return 8;
    if(large.storage_bytes()>1024*1024)return 9;
    const auto allocated=large.storage_bytes();
    for(int round=0;round<200;++round) {
        std::array<std::int32_t,3> proposal{};
        const bool found=large.propose(long_history,proposal);
        if(found) {
            bool justified=false;
            const auto suffix=long_history.end()-32;
            const auto first=long_history.size()>SessionCopyIndex::history_limit?
                long_history.size()-SessionCopyIndex::history_limit:32;
            for(std::size_t end=std::max<std::size_t>(32,first);end+3<=long_history.size()-32;++end){
                if(std::equal(suffix,long_history.end(),long_history.begin()+end-32)&&
                   std::equal(proposal.begin(),proposal.end(),long_history.begin()+end)){
                    justified=true;break;
                }
            }
            if(!justified)return 10;
            long_history.insert(long_history.end(),proposal.begin(),proposal.end());
        } else {
            for(int j=0;j<3;++j)long_history.push_back(static_cast<std::int32_t>(rng()%200000));
        }
    }
    if(large.storage_bytes()!=allocated)return 11;
    std::array<std::int32_t,8> too_wide{};if(large.propose(long_history,too_wide))return 12;
    // Shrinking resets append state; two separate instances retain no shared tokens.
    if(large.propose(source,out))return 13;
    std::cout<<"PASS exact32 match, observed-only proposals, incremental history, misses, reset, isolation and bounded1MiB storage\n";
}
