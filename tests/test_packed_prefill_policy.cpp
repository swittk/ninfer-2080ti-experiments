#include "runtime/engine/packed_prefill_policy.h"
#include <array>
#include <iostream>
#include <limits>
using namespace ninfer::runtime;
int main() {
    for(unsigned capacity=128;capacity<=4096;capacity+=128) {
        for(unsigned unit=128;unit<=4096;unit+=128) {
            for(unsigned owners=2;owners<=8;++owners) {
                std::array<std::uint32_t,8> widths{}; widths.fill(unit);
                const auto expected=unit*owners<=capacity?unit*owners:0;
                if(packed_prefill_ragged_total({widths.data(),owners},capacity)!=expected)return 1;
            }
        }
        const std::array<std::uint32_t,2> large{capacity,capacity};
        if(packed_prefill_ragged_total(large,capacity)!=0)return 2;
    }
    const std::array<std::uint32_t,4> ragged{46,71,138,271};
    if(packed_prefill_ragged_total(ragged,1024)!=526)return 3;
    if(packed_prefill_ragged_total(ragged,512)!=0)return 4;
    if(packed_prefill_fragment(0,128)||packed_prefill_fragment(1,128))return 5;
    if(packed_prefill_fragment(47,2048)!=46 || packed_prefill_fragment(5000,2048)!=2048)return 6;
    const std::array<std::uint32_t,2> huge{std::numeric_limits<unsigned>::max(),1};
    if(packed_prefill_ragged_total(huge,std::numeric_limits<unsigned>::max())!=0)return 7;
    const std::array<std::uint32_t,2> empty{0,128},tiny{31,32};
    if(packed_prefill_ragged_total(empty,1024)||packed_prefill_ragged_total(tiny,1024)!=63)return 8;
    if(packed_prefill_ragged_total({ragged.data(),1},1024))return 9;
    std::cout<<"PASS real nonfinal widths, whole large quanta, ragged capacity and overflow protection\n";
}
