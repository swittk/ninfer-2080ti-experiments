#include "../apps/dual/tp2_packet_layout.h"
#include <array>
#include <climits>
#include <iostream>
using ninfer::apps::dual::detail::tp2_input_packet_layout;
static_assert(tp2_input_packet_layout(4).code_bytes==20480);
static_assert(tp2_input_packet_layout(4).scale_offset==20480);
static_assert(tp2_input_packet_layout(4).scale_bytes==640);
static_assert(tp2_input_packet_layout(4).packet_bytes==21120);
static_assert(tp2_input_packet_layout(5).code_bytes==25600);
static_assert(tp2_input_packet_layout(5).scale_offset==25600);
static_assert(tp2_input_packet_layout(5).scale_bytes==800);
static_assert(tp2_input_packet_layout(5).packet_bytes==26400);
int main(){
    for(int width:std::array{INT_MIN,-1,0,1,2,3,6,INT_MAX}) {
        bool rejected=false;
        try{(void)tp2_input_packet_layout(width);}catch(const std::invalid_argument&){rejected=true;}
        if(!rejected)return 1;
    }
    for(int width:std::array{4,5,4,5,5,4}) {
        const auto layout=tp2_input_packet_layout(width);
        if(layout.scale_offset%16 || layout.code_bytes>layout.scale_offset ||
           layout.packet_bytes!=layout.scale_offset+layout.scale_bytes || layout.packet_bytes>26400)return 2;
    }
    std::cout<<"PASS exact T4/T5 layout, allocation bounds, alternating widths and unsupported-width rejection\n";
}
