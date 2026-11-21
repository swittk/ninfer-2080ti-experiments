#include "ops/linear/sm75_radix_projection.h"
#include "core/device.h"
#include <cuda_bf16.h>
#include <chrono>
#include "ninfer/ops/linear.h"
#include "ninfer/ops/linear_add.h"
#include "ninfer_bench_common.h"
#include "quantized_weight.cuh"
#include "t16_fixture.h"
#include <cstdio>
#include <cmath>
#include <vector>
#include <cstring>
using namespace ninfer;
namespace detail=ninfer::ops::detail;
struct PreparedOwner {
    const Weight& source;
    ~PreparedOwner() { detail::sm75_radix_projection_release(source); }
};

template<int Blocks,bool Q5>
void exercise(unsigned seed,cudaStream_t stream) {
    constexpr int T=Blocks*4;
    const int n=Q5?5120:4096, k=Q5?6144:5120, ld=n+8;
    auto weight=bench::make_row_split_weight(Q5?QType::Q5G64_F16S:QType::Q4G64_F16S,n,k,k);
    bench::t16_fill_weight(weight.weight,n,k,Q5,seed);
    CUDA_CHECK(cudaDeviceSynchronize()); // Fixture fill uses its own/default stream.
    const auto prepare_start=std::chrono::steady_clock::now();
    detail::sm75_radix_projection_prepare(weight.weight,stream,T);
    PreparedOwner owner{weight.weight};
    const auto preparation_ms=std::chrono::duration<double,std::milli>(
        std::chrono::steady_clock::now()-prepare_start).count();
    printf("PREPARATION Q%d T=%d reserved_bytes=%zu milliseconds=%.3f\n",Q5?5:4,T,
           detail::sm75_radix_projection_bytes(n,k,weight.weight.qtype,T),preparation_ms);
    DeviceBuffer input(k*T*2),initial(ld*T*2),reference(ld*T*2),candidate(ld*T*2),
                 fallback(n*T*2),flush(256ULL<<20);
    Tensor all_x(input.p,DType::BF16,{k,T});
    Tensor ref_storage(reference.p,DType::BF16,{ld,T}),out_storage(candidate.p,DType::BF16,{ld,T});
    Tensor ref_view=ref_storage.slice(0,0,n),out_view=out_storage.slice(0,0,n);
    bench::t16_fill_input(input.p,k*T,seed^54321u);
    bench::t16_fill_input(initial.p,ld*T,seed^98765u);
    CUDA_CHECK(cudaDeviceSynchronize()); // Establish fixture readiness before nonblocking stream copies.
    auto serial=[&](cudaStream_t s){
        CUDA_CHECK(cudaMemcpyAsync(reference.p,initial.p,initial.bytes,cudaMemcpyDeviceToDevice,s));
        for(int block=0;block<Blocks;++block) {
            Tensor x=all_x.slice(1,block*4,4),y=ref_view.slice(1,block*4,4);
            if(!detail::sm75_radix_projection_launch(x,weight.weight,y,Q5,s))
                throw std::runtime_error("prepared T4 operation was refused");
        }
    };
    auto shared=[&](cudaStream_t s){
        CUDA_CHECK(cudaMemcpyAsync(candidate.p,initial.p,initial.bytes,cudaMemcpyDeviceToDevice,s));
        if(!detail::sm75_radix_projection_launch(all_x,weight.weight,out_view,Q5,s))
            throw std::runtime_error("prepared wide operation was refused");
    };
    const auto bytes=Q5?ops::linear_add_workspace_capacity_bytes(weight.weight.qtype,n,k,ops::LinearPolicy::A16Only,T,T):ops::linear_workspace_capacity_bytes(weight.weight.qtype,n,k,ops::LinearPolicy::A16Only,T,T);
    WorkspaceArena workspace(std::max<std::size_t>(1, bytes));
    Tensor x(input.p,DType::BF16,{k,T}),f(fallback.p,DType::BF16,{n,T});
    auto canonical=[&](cudaStream_t s){
        if constexpr(Q5){
            CUDA_CHECK(cudaMemcpy2DAsync(fallback.p,n*2,initial.p,ld*2,n*2,T,cudaMemcpyDeviceToDevice,s));
            ops::linear_add(x,weight.weight,f,ops::LinearPolicy::A16Only,workspace,s);
        } else ops::linear(x,weight.weight,f,ops::LinearPolicy::A16Only,workspace,s);
    };
    serial(stream);shared(stream);canonical(stream);CUDA_CHECK(cudaStreamSynchronize(stream));
    std::vector<std::uint16_t>a(ld*T),b(ld*T),init(ld*T),c(n*T);
    reference.copy_to_host(a.data(),reference.bytes);candidate.copy_to_host(b.data(),candidate.bytes);initial.copy_to_host(init.data(),initial.bytes);fallback.copy_to_host(c.data(),fallback.bytes);
    std::size_t mismatches=0,guards=0;double error=0,norm=0;unsigned nonfinite=0;
    for(int t=0;t<T;++t){
        for(int r=0;r<n;++r){mismatches+=a[t*ld+r]!=b[t*ld+r];
            const double v=__bfloat162float(__ushort_as_bfloat16(b[t*ld+r]));
            const double expected=__bfloat162float(__ushort_as_bfloat16(c[t*n+r]));
            error+=(v-expected)*(v-expected);norm+=expected*expected;nonfinite+=!std::isfinite(v);}
        for(int r=n;r<ld;++r)guards+=b[t*ld+r]!=init[t*ld+r];
    }
    printf("PARITY Q%d N=%d K=%d T=%d seed=%u exact_T4_mismatches=%zu guard_errors=%zu nonfinite=%u canonical_relative_l2=%.9g\n",Q5?5:4,n,k,T,seed,mismatches,guards,nonfinite,std::sqrt(error/std::max(norm,1e-30)));
    if(mismatches||guards||nonfinite)throw std::runtime_error("wide exact-to-T4/guard test failed");
    cudaGraph_t graphs[3]{};cudaGraphExec_t execs[3]{};
    for(int i=0;i<3;++i){CUDA_CHECK(cudaStreamBeginCapture(stream,cudaStreamCaptureModeGlobal));if(i==0)canonical(stream);else if(i==1)serial(stream);else shared(stream);CUDA_CHECK(cudaStreamEndCapture(stream,&graphs[i]));CUDA_CHECK(cudaGraphInstantiate(&execs[i],graphs[i],nullptr,nullptr,0));}
    for(int pass=0;pass<2;++pass){
        double times[3]{};
        for(int j=0;j<3;++j){const int i=pass==0?j:2-j;auto time=bench::measure_cold_launch([&](cudaStream_t s){CUDA_CHECK(cudaGraphLaunch(execs[i],s));},flush,stream,4,24);times[i]=time.median_us;}
        printf("PERF Q%d T=%d seed=%u pass=%d canonical_us=%.3f serial_radix_us=%.3f shared_radix_us=%.3f versus_fallback=%.6f versus_serial=%.6f\n",Q5?5:4,T,seed,pass,times[0],times[1],times[2],times[0]/times[2],times[1]/times[2]);
    }
    std::vector<std::uint16_t> repeated(ld*T);candidate.copy_to_host(repeated.data(),candidate.bytes);if(repeated!=b)throw std::runtime_error("graph result instability");
    for(int i=0;i<3;++i){CUDA_CHECK(cudaGraphExecDestroy(execs[i]));CUDA_CHECK(cudaGraphDestroy(graphs[i]));}
}
int main(int argc,char**argv){try{
    const unsigned seed=argc>1?std::stoul(argv[1]):1234567;cudaStream_t stream{};CUDA_CHECK(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking));
    exercise<2,false>(seed,stream);exercise<2,true>(seed,stream);exercise<4,false>(seed,stream);exercise<4,true>(seed,stream);
    CUDA_CHECK(cudaStreamDestroy(stream));return 0;
}catch(const std::exception&e){fprintf(stderr,"wide radix: %s\n",e.what());return 1;}}
