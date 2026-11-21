#include "core/device.h"
#include "ops/linear/sm75_radix_projection.h"
#include "ninfer_bench_common.h"
#include "quantized_weight.cuh"
#include "t16_fixture.h"
// Instantiate the existing independent repack under test-only names so this
// benchmark has no dependency on ephemeral evidence-directory headers.
#define radix_projection resident_reference
#include "ops/linear/sm75_radix_projection.cuh"
#undef radix_projection
#include <cuda_bf16.h>
#include <algorithm>
#include <cstdio>
#include <vector>
#include <stdexcept>
using namespace ninfer;
namespace impl=ninfer::ops::detail;
void require(bool ok,const char* message){if(!ok)throw std::runtime_error(message);}
std::vector<unsigned char> bytes(const void* pointer,std::size_t size,cudaStream_t stream){
    std::vector<unsigned char> out(size);
    CUDA_CHECK(cudaMemcpyAsync(out.data(),pointer,size,cudaMemcpyDeviceToHost,stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));return out;
}
std::vector<unsigned char> payload(const Weight& w,cudaStream_t stream){
    const auto elements=std::size_t(w.n)*w.k;
    auto out=bytes(w.qdata,elements/2,stream);
    if(w.qtype==QType::Q5G64_F16S){auto v=bytes(w.qhigh,elements/8,stream);out.insert(out.end(),v.begin(),v.end());}
    auto scales=bytes(w.scales,elements/64*2,stream);out.insert(out.end(),scales.begin(),scales.end());return out;
}
void exercise(bool q5,unsigned seed,cudaStream_t stream){
    const int n=q5?5120:4096,k=q5?6144:5120,ld=n+64;
    const auto type=q5?QType::Q5G64_F16S:QType::Q4G64_F16S;
    auto reference=bench::make_row_split_weight(type,n,k,k);
    auto resident=bench::make_row_split_weight(type,n,k,k);
    bench::t16_fill_weight(reference.weight,n,k,q5,seed);
    bench::t16_fill_weight(resident.weight,n,k,q5,seed);
    // The fixture uploads on the default stream; source reads below are nonblocking.
    CUDA_CHECK(cudaDeviceSynchronize());
    const auto original=payload(resident.weight,stream);
    require(original==payload(reference.weight,stream),"fixture source mismatch");
    const void* address=resident.weight.qdata;
    impl::sm75_radix_projection_prepare(reference.weight,stream,16);
    impl::sm75_radix_resident_prepare(resident.weight,stream,16);
    require(resident.weight.qdata==address,"resident source address changed");
    require(resident.weight.layout==QuantLayout::RowSplitSm75Radix,"resident layout missing");
    impl::sm75_radix_resident_prepare(resident.weight,stream,16);
    bool rejected=false;
    try{impl::sm75_radix_resident_transform(resident.weight,QuantLayout::RowSplit,stream);}
    catch(const std::logic_error&){rejected=true;}
    require(rejected,"live prepared source was transformable");
    const auto packed=payload(resident.weight,stream);
    {
        const auto elements=std::size_t(n)*k;
        DeviceBuffer low(elements/2),high(q5?elements/8:256),scales(elements/64*2);
        const auto blocks=(n/16*(k/64)*32+255)/256;
        if(q5)impl::resident_reference::repack<true><<<blocks,256,0,stream>>>(
            (const unsigned char*)reference.weight.qdata,(const unsigned char*)reference.weight.qhigh,
            (const __half*)reference.weight.scales,(uint4*)low.p,(unsigned*)high.p,(__half*)scales.p,n,k);
        else impl::resident_reference::repack<false><<<blocks,256,0,stream>>>(
            (const unsigned char*)reference.weight.qdata,nullptr,(const __half*)reference.weight.scales,
            (uint4*)low.p,nullptr,(__half*)scales.p,n,k);
        CUDA_CHECK(cudaGetLastError());
        Weight expected=resident.weight;expected.qdata=low.p;expected.qhigh=q5?high.p:nullptr;expected.scales=scales.p;
        const auto ref=payload(expected,stream);
        std::size_t mismatches=0;
        for(std::size_t i=0;i<ref.size();++i)if(ref[i]!=packed[i]){
            if(mismatches<8)std::printf("PACK_DIFF byte=%zu expected=%u actual=%u\n",i,unsigned(ref[i]),unsigned(packed[i]));
            ++mismatches;
        }
        std::printf("PACK_IDENTITY Q%d bytes=%zu mismatches=%zu\n",q5?5:4,ref.size(),mismatches);
        require(mismatches==0,"in-place packed bytes differ from independent repack");
    }

    WorkspaceArena view_storage(impl::sm75_radix_canonical_workspace_bytes(n,k,type));
    {
        auto scope=view_storage.scope();const auto view=impl::sm75_radix_canonical_copy(resident.weight,view_storage,stream);
        require(payload(view,stream)==original,"canonical view changed source values");
        require(payload(resident.weight,stream)==packed,"canonical view mutated resident source");
    }
    DeviceBuffer input(k*16*2),seed_output(ld*16*2),a(ld*16*2),b(ld*16*2),flush(256ULL<<20);
    bench::t16_fill_input(input.p,k*16,seed^54321u);
    std::vector<std::uint16_t> initial(ld*16,0x4f25);
    if(q5){for(int t=0;t<16;++t)for(int j=0;j<n;++j)initial[t*ld+j]=__bfloat16_as_ushort(__float2bfloat16_rn(float((j*7+t*11)%41-20)*.01f));}
    seed_output.copy_from_host(initial.data(),seed_output.bytes);
    // Pageable H2D initialization uses the default stream. Establish completion
    // before this test starts work on its explicitly nonblocking stream.
    CUDA_CHECK(cudaDeviceSynchronize());
    for(int tokens:{4,8,16}){
        Tensor x(input.p,DType::BF16,{k,tokens});
        Tensor ao(a.p,DType::BF16,{ld,tokens});ao=ao.slice(0,0,n);
        Tensor bo(b.p,DType::BF16,{ld,tokens});bo=bo.slice(0,0,n);
        auto old=[&](cudaStream_t s){CUDA_CHECK(cudaMemcpyAsync(a.p,seed_output.p,a.bytes,cudaMemcpyDeviceToDevice,s));require(impl::sm75_radix_projection_launch(x,reference.weight,ao,q5,s),"old radix refused");};
        auto next=[&](cudaStream_t s){CUDA_CHECK(cudaMemcpyAsync(b.p,seed_output.p,b.bytes,cudaMemcpyDeviceToDevice,s));require(impl::sm75_radix_projection_launch(x,resident.weight,bo,q5,s),"resident radix refused");};
        old(stream);next(stream);CUDA_CHECK(cudaStreamSynchronize(stream));
        {
            const auto left=bytes(a.p,a.bytes,stream),right=bytes(b.p,b.bytes,stream);
            std::size_t differing=0;
            for(std::size_t j=0;j<left.size();++j)if(left[j]!=right[j]){
                if(differing<12)std::printf("OUT_DIFF T%d byte%zu token%zu row%zu old%u new%u\n",tokens,j,(j/2)/ld,(j/2)%ld,unsigned(left[j]),unsigned(right[j]));
                ++differing;
            }
            std::printf("OUTPUT_DIFF_BYTES=%zu\n",differing);
            require(differing==0,"resident output differs");
        }
        cudaGraph_t ga{},gb{};cudaGraphExec_t ea{},eb{};
        CUDA_CHECK(cudaStreamBeginCapture(stream,cudaStreamCaptureModeGlobal));old(stream);CUDA_CHECK(cudaStreamEndCapture(stream,&ga));CUDA_CHECK(cudaGraphInstantiate(&ea,ga,nullptr,nullptr,0));
        CUDA_CHECK(cudaStreamBeginCapture(stream,cudaStreamCaptureModeGlobal));next(stream);CUDA_CHECK(cudaStreamEndCapture(stream,&gb));CUDA_CHECK(cudaGraphInstantiate(&eb,gb,nullptr,nullptr,0));
        for(int pass=0;pass<2;++pass){
            auto ra=bench::measure_cold_launch([&](cudaStream_t s){CUDA_CHECK(cudaGraphLaunch(ea,s));},flush,stream,3,16);
            auto rb=bench::measure_cold_launch([&](cudaStream_t s){CUDA_CHECK(cudaGraphLaunch(eb,s));},flush,stream,3,16);
            std::printf("DECODE Q%d T%d pass%d copy_us=%.3f resident_us=%.3f ratio=%.5f\n",q5?5:4,tokens,pass,ra.median_us,rb.median_us,ra.median_us/rb.median_us);
        }
        // Simulate intervening canonical prefill use while executable pointers stay live.
        {auto scope=view_storage.scope();auto view=impl::sm75_radix_canonical_copy(resident.weight,view_storage,stream);require(payload(view,stream)==original,"interleaved canonical view failed");}
        CUDA_CHECK(cudaGraphLaunch(eb,stream));CUDA_CHECK(cudaStreamSynchronize(stream));
        require(bytes(a.p,a.bytes,stream)==bytes(b.p,b.bytes,stream),"graph replay after canonical use changed");
        CUDA_CHECK(cudaGraphExecDestroy(ea));CUDA_CHECK(cudaGraphExecDestroy(eb));CUDA_CHECK(cudaGraphDestroy(ga));CUDA_CHECK(cudaGraphDestroy(gb));
    }
    {
        // A second source uses the same stream-owned scratch. It must not retain
        // old-layer values, and releasing it must not free the first owner's data.
        auto other=bench::make_row_split_weight(type,n,k,k);
        bench::t16_fill_weight(other.weight,n,k,q5,seed+707);
        CUDA_CHECK(cudaDeviceSynchronize());
        const auto other_original=payload(other.weight,stream);
        impl::sm75_radix_resident_prepare(other.weight,stream,16);
        Tensor x(input.p,DType::BF16,{k,16});
        Tensor ao(a.p,DType::BF16,{ld,16});ao=ao.slice(0,0,n);
        Tensor bo(b.p,DType::BF16,{ld,16});bo=bo.slice(0,0,n);
        CUDA_CHECK(cudaMemcpyAsync(a.p,seed_output.p,a.bytes,cudaMemcpyDeviceToDevice,stream));
        require(impl::sm75_radix_projection_launch(x,reference.weight,ao,q5,stream),"interleaving reference refused");
        CUDA_CHECK(cudaMemcpyAsync(b.p,seed_output.p,b.bytes,cudaMemcpyDeviceToDevice,stream));
        require(impl::sm75_radix_projection_launch(x,other.weight,bo,q5,stream),"second source refused");
        impl::sm75_radix_projection_release(other.weight);
        impl::sm75_radix_resident_transform(other.weight,QuantLayout::RowSplit,stream);
        require(payload(other.weight,stream)==other_original,"second owner restore changed bytes");
        CUDA_CHECK(cudaMemcpyAsync(b.p,seed_output.p,b.bytes,cudaMemcpyDeviceToDevice,stream));
        require(impl::sm75_radix_projection_launch(x,resident.weight,bo,q5,stream),"remaining source refused");
        require(bytes(a.p,a.bytes,stream)==bytes(b.p,b.bytes,stream),"shared scratch leaked interleaved source values");
        std::printf("SHARED_SCRATCH Q%d interleaved_source_release_exact\n",q5?5:4);
    }
    for(int tokens=1;tokens<=16;++tokens) {
        const int padded=tokens<=4?4:tokens<=8?8:16;
        // Actual source allocation has only valid columns; inactive lanes must
        // not even read beyond it. The reference explicitly appends zero rows.
        DeviceBuffer tight_input(std::size_t(k)*tokens*2),padded_input(std::size_t(k)*padded*2);
        CUDA_CHECK(cudaMemcpyAsync(tight_input.p,input.p,tight_input.bytes,cudaMemcpyDeviceToDevice,stream));
        CUDA_CHECK(cudaMemsetAsync(padded_input.p,0,padded_input.bytes,stream));
        CUDA_CHECK(cudaMemcpyAsync(padded_input.p,input.p,tight_input.bytes,cudaMemcpyDeviceToDevice,stream));
        Tensor actual_x(tight_input.p,DType::BF16,{k,tokens});
        Tensor ref_x(padded_input.p,DType::BF16,{k,padded});
        Tensor actual_y(b.p,DType::BF16,{ld,tokens});actual_y=actual_y.slice(0,0,n);
        Tensor ref_y(a.p,DType::BF16,{ld,padded});ref_y=ref_y.slice(0,0,n);
        const auto old=[&](cudaStream_t st){
            CUDA_CHECK(cudaMemcpyAsync(a.p,seed_output.p,a.bytes,cudaMemcpyDeviceToDevice,st));
            require(impl::sm75_radix_projection_launch(ref_x,reference.weight,ref_y,q5,st),"padded reference refused");
        };
        const auto next=[&](cudaStream_t st){
            CUDA_CHECK(cudaMemcpyAsync(b.p,seed_output.p,b.bytes,cudaMemcpyDeviceToDevice,st));
            require(impl::sm75_radix_projection_launch(actual_x,resident.weight,actual_y,q5,st),"ragged resident refused");
        };
        old(stream);next(stream);CUDA_CHECK(cudaStreamSynchronize(stream));
        const auto expected=bytes(a.p,a.bytes,stream);
        const auto validate=[&]() {
            const auto observed=bytes(b.p,b.bytes,stream);
            for(int col=0;col<16;++col)for(int row=0;row<ld;++row) {
                const auto index=std::size_t(col)*ld+row;
                const unsigned short value=unsigned(observed[index*2])|(unsigned(observed[index*2+1])<<8);
                const unsigned short wanted=col<tokens&&row<n?
                    (unsigned(expected[index*2])|(unsigned(expected[index*2+1])<<8)):initial[index];
                require(value==wanted,"ragged output or inactive-column canary differs");
            }
        };
        validate();
        cudaGraph_t graph{};cudaGraphExec_t executable{};
        CUDA_CHECK(cudaStreamBeginCapture(stream,cudaStreamCaptureModeGlobal));next(stream);
        CUDA_CHECK(cudaStreamEndCapture(stream,&graph));CUDA_CHECK(cudaGraphInstantiate(&executable,graph,nullptr,nullptr,0));
        for(int pass=0;pass<3;++pass)CUDA_CHECK(cudaGraphLaunch(executable,stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));validate();
        CUDA_CHECK(cudaGraphExecDestroy(executable));CUDA_CHECK(cudaGraphDestroy(graph));
        std::printf("RAGGED Q%d T%d padded%d outputs_padding_inactive_exact graph_replay_exact\n",q5?5:4,tokens,padded);
    }
    impl::sm75_radix_projection_release(reference.weight);impl::sm75_radix_projection_release(resident.weight);
    impl::sm75_radix_resident_transform(resident.weight,QuantLayout::RowSplit,stream);
    require(payload(resident.weight,stream)==original,"resident roundtrip not byte exact");
    for(int repeat=0;repeat<3;++repeat){
        impl::sm75_radix_resident_transform(resident.weight,QuantLayout::RowSplitSm75Radix,stream);
        require(payload(resident.weight,stream)==packed,"repeated packed permutation changed");
        impl::sm75_radix_resident_transform(resident.weight,QuantLayout::RowSplit,stream);
        require(payload(resident.weight,stream)==original,"repeated canonical restore changed");
    }
    std::printf("PASS Q%d bytes=%zu source_address_unchanged canonical_roundtrip_exact graph_replay_exact old_T4_T8_T16_outputs_exact owned16=%zu resident16=%zu owned8=%zu resident8=%zu\n",q5?5:4,original.size(),impl::sm75_radix_projection_bytes(n,k,type,16),impl::sm75_radix_resident_bytes(n,k,type,16),impl::sm75_radix_projection_bytes(n,k,type,8),impl::sm75_radix_resident_bytes(n,k,type,8));
}
int main(int argc,char**argv){try{
    cudaStream_t stream{};CUDA_CHECK(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking));
    const unsigned seed=argc>1?std::stoul(argv[1]):20260916;
    exercise(false,seed,stream);exercise(true,seed+1,stream);
    CUDA_CHECK(cudaStreamDestroy(stream));return 0;
}catch(const std::exception&e){std::fprintf(stderr,"resident_radix: %s\n",e.what());return 1;}}
