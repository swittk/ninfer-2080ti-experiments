#include "runtime/engine/transient_slots.h"
#include <array>
#include <cstdint>
#include <iostream>
#include <limits>
#include <stdexcept>
using ninfer::runtime::TransientSlots;
void require(bool v,const char*message){if(!v)throw std::runtime_error(message);}
int main(){try{
    TransientSlots<8> memory(1024);
    require(memory.find(0,513,256)==0,"first aligned reservation");
    auto a=memory.activate(0,513,256);
    auto b=memory.activate(1,256,256);
    require(b.offset==768,"disjoint aligned slice");
    require(!memory.find(2,1,256),"capacity including alignment exhausted");
    auto z=memory.activate(2,0,1);require(z.bytes==0&&memory.used()==769,"zero-byte request changed other slices");
    memory.release(2);require(memory.slice(0).active&&memory.slice(1).active,"empty release clobbered another owner");
    memory.release(0);auto c=memory.activate(3,512,256);require(c.offset==0,"freed hole not reused");
    require(memory.slice(1).offset==b.offset,"live slice moved");memory.release(0);require(memory.used()==768,"release not idempotent");
    require(memory.peak()==769,"peak tracks total simultaneous bytes");memory.clear();memory.reset_peak();require(!memory.used()&&!memory.peak(),"clear/reset leaked accounting");
    for(auto alignment:{0U,3U,512U}){bool threw=false;try{memory.activate(0,1,alignment);}catch(const std::invalid_argument&){threw=true;}require(threw,"invalid alignment allowed");}
    bool threw=false;try{memory.activate(8,0,1);}catch(const std::invalid_argument&){threw=true;}require(threw,"bad lane accepted");
    TransientSlots<8> large(std::numeric_limits<std::size_t>::max());
    large.activate(0,std::numeric_limits<std::size_t>::max()-255,1);
    require(!large.find(1,256,256),"overflow permitted");
    std::uint32_t rng=0x234abcdU;
    TransientSlots<8> randomized(8192);
    for(int n=0;n<30000;++n){rng^=rng<<13;rng^=rng>>17;rng^=rng<<5;auto lane=rng%8;
        if(randomized.slice(lane).active){randomized.release(lane);}else{
            auto bytes=(rng>>8)%3000;auto align=bytes?std::size_t(1)<<((rng>>22)%9):1;
            auto found=randomized.find(lane,bytes,align);if(found)randomized.activate(lane,bytes,align);
        }
        std::size_t used=0;
        for(int i=0;i<8;++i){auto x=randomized.slice(i);if(!x.active)continue;used+=x.bytes;
            require(x.offset<=8192&&x.bytes<=8192-x.offset&&x.offset%x.alignment==0,"slice escaped allocation");
            for(int j=i+1;j<8;++j){auto y=randomized.slice(j);if(y.active&&x.bytes&&y.bytes)require(x.offset+x.bytes<=y.offset||y.offset+y.bytes<=x.offset,"overlapping live request regions");}
        }require(used==randomized.used()&&used<=8192,"active memory accounting disagrees");
    }
    ninfer::runtime::PrefillWorkset<8> owners;owners.add(0);owners.add(3);owners.add(7);
    for(int i=0;i<4;++i){require(owners.take_next()==0&&owners.take_next()==3&&owners.take_next()==7,"round robin starved an owner");}
    owners.remove(3);owners.add(4);require(owners.take_next()==0&&owners.take_next()==4&&owners.take_next()==7,"cancel/admit disrupted other owners");
    owners.clear();require(owners.empty(),"workset clear failed");
    std::cout<<"PASS disjoint frozen memory, zero-byte coexistence, bounds, fragmentation, 30000 transitions and fair staged owners\n";return 0;
}catch(const std::exception&e){std::cerr<<e.what()<<'\n';return 1;}}
