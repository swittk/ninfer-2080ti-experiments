#include "core/layout.h"
#include "targets/qwen3_6_27b/impl/sm75_prefill_dense.h"
#include <algorithm>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string_view>

using namespace ninfer;
using targets::qwen3_6_27b::detail::Sm75PrefillDense;

// Independent pre-change live-allocation recipe. No CUDA calls or model values.
static std::size_t original_capacity(int t) {
    WorkspaceLayoutBuilder layout;
    (void)layout.alloc(DType::BF16,{17408,t});
    for(bool gate : {true,false}) {
        auto scope=layout.scope();
        const int k=gate?5120:17408,n=gate?34816:5120,rows=gate?8704:5120;
        (void)layout.alloc(DType::FP16,{k,rows});
        (void)layout.alloc(DType::FP16,{k,t});
        (void)layout.alloc(DType::FP32,{n,t});
        (void)layout.alloc_bytes(4ULL<<20);
    }
    return layout.peak_bytes(1);
}
int main(int argc,char** argv) {
    const bool streamed=argc==2 && std::string_view(argv[1])=="streamed";
    if(setenv("NINFER_SERVE_PREFILL_STREAMED",streamed?"1":"0",1)!=0) return 1;
    std::size_t prior=0;
    for(int t=32;t<=4096;++t) {
        const auto actual=Sm75PrefillDense::workspace_bytes(t);
        if(actual<prior){std::cerr<<"nonmonotone tail reservation at "<<t<<'\n';return 2;}
        if(!streamed || t<1024) {
            if(actual!=original_capacity(t)) return 3;
        } else {
            if(actual<original_capacity(1023) || actual>original_capacity(t)) return 4;
        }
        prior=actual;
    }
    std::size_t prior_projection=0;
    for(int t=64;t<=4096;++t) {
        const auto actual=Sm75PrefillDense::projection_workspace_bytes(12288,5120,t);
        WorkspaceLayoutBuilder old;
        (void)old.alloc(DType::FP16,{5120,12288});
        (void)old.alloc(DType::FP16,{5120,t});
        (void)old.alloc(DType::FP32,{12288,t});
        (void)old.alloc_bytes(4ULL<<20);
        if(actual<prior_projection || actual>old.peak_bytes(1))return 8;
        if((!streamed || t<1024) && actual!=old.peak_bytes(1))return 9;
        prior_projection=actual;
    }
    const auto capacity=Sm75PrefillDense::workspace_bytes(4096);
    if(streamed && original_capacity(4096)-capacity!=368ULL*1024*1024) return 5;
    for(int invalid : {0,31,4097}) {
        bool rejected=false;
        try{(void)Sm75PrefillDense::workspace_bytes(invalid);}catch(const std::invalid_argument&){rejected=true;}
        if(!rejected)return 6;
    }
    // Changing the environment after the first plan must not change execution
    // geometry midway through a Program's lifetime.
    setenv("NINFER_SERVE_PREFILL_STREAMED",streamed?"0":"1",1);
    if(Sm75PrefillDense::workspace_bytes(4096)!=capacity) return 7;
    std::cout<<"PASS mode="<<streamed<<" interval32..4096, old-tail bound, frozen planning; capacity="<<capacity<<'\n';
}
