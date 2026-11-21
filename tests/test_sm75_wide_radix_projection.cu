#include "core/device.h"
#include "ops/linear/sm75_radix_projection.h"
#include "quantized_weight.cuh"
#include "t16_fixture.h"
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <algorithm>
#include <cstring>
#include <iostream>
#include <vector>
using namespace ninfer;
namespace impl=ninfer::ops::detail;
void require(bool condition,const char* message){if(!condition)throw std::runtime_error(message);}
int main(){try{
    cudaStream_t stream{};CUDA_CHECK(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking));
    constexpr int n=128,k=5120,ld=n+8;
    int cases=0;std::uint64_t values=0;
    for(bool q5:{false,true}){
        auto storage=bench::make_row_split_weight(q5?QType::Q5G64_F16S:QType::Q4G64_F16S,n,k,k);
        const Weight& w=storage.weight;
        DeviceBuffer input(k*16*2),initial(ld*16*2),reference(ld*16*2),actual(ld*16*2);
        Tensor x(input.p,DType::BF16,{k,16}),whole(actual.p,DType::BF16,{ld,16});
        Tensor y=whole.slice(0,0,n);
        for(unsigned seed:{1234567u,9787654u,1357911u}){
            bench::t16_fill_weight(w,n,k,q5,seed);CUDA_CHECK(cudaDeviceSynchronize());
            impl::sm75_radix_projection_prepare(w,stream);
            Tensor eight=x.slice(1,0,8),eight_out=y.slice(1,0,8);
            require(!impl::sm75_radix_projection_launch(eight,w,eight_out,q5,stream),"narrow owner admitted wider scratch");
            bool rejected=false;try{impl::sm75_radix_projection_prepare(w,stream,16);}catch(const std::logic_error&){rejected=true;}
            require(rejected,"owner storage grew during lifetime");
            impl::sm75_radix_projection_release(w);
            impl::sm75_radix_projection_prepare(w,stream,16);
            std::size_t free_before,total,free_after;CUDA_CHECK(cudaMemGetInfo(&free_before,&total));
            impl::sm75_radix_projection_prepare(w,stream,16);CUDA_CHECK(cudaMemGetInfo(&free_after,&total));
            require(free_before==free_after,"duplicate wide allocation");
            for(int mode=0;mode<4;++mode){
                std::vector<std::uint16_t> bits(k*16);unsigned state=seed+mode;
                for(auto& b:bits){auto v=bench::t16_next(state);int e=mode==0?126:mode==1?120+int(v%13):mode==2?112+int(v%23):127;b=mode==3?0:((v&0x8000u)|(e<<7)|((v>>16)&127));}
                input.copy_from_host(bits.data(),input.bytes);bench::t16_fill_input(initial.p,ld*16,seed^81723u);CUDA_CHECK(cudaDeviceSynchronize());
                for(int tokens:{8,16,4,16,8}){
                    Tensor xt=x.slice(1,0,tokens),yt=y.slice(1,0,tokens);
                    CUDA_CHECK(cudaMemcpyAsync(reference.p,initial.p,initial.bytes,cudaMemcpyDeviceToDevice,stream));
                    CUDA_CHECK(cudaMemcpyAsync(actual.p,initial.p,initial.bytes,cudaMemcpyDeviceToDevice,stream));
                    Tensor refwhole(reference.p,DType::BF16,{ld,16});Tensor ref=refwhole.slice(0,0,n);
                    for(int off=0;off<tokens;off+=4){Tensor xs=x.slice(1,off,4),ys=ref.slice(1,off,4);require(impl::sm75_radix_projection_launch(xs,w,ys,q5,stream),"T4 reference refused");}
                    cudaGraph_t graph{};cudaGraphExec_t exec{};
                    CUDA_CHECK(cudaStreamBeginCapture(stream,cudaStreamCaptureModeGlobal));
                    CUDA_CHECK(cudaMemcpyAsync(actual.p,initial.p,initial.bytes,cudaMemcpyDeviceToDevice,stream));
                    require(impl::sm75_radix_projection_launch(xt,w,yt,q5,stream),"prepared wide shape refused");
                    CUDA_CHECK(cudaStreamEndCapture(stream,&graph));CUDA_CHECK(cudaGraphInstantiate(&exec,graph,nullptr,nullptr,0));
                    for(int replay=0;replay<3;++replay)CUDA_CHECK(cudaGraphLaunch(exec,stream));
                    CUDA_CHECK(cudaStreamSynchronize(stream));
                    std::vector<unsigned char> a(reference.bytes),b(actual.bytes);reference.copy_to_host(a.data(),a.size());actual.copy_to_host(b.data(),b.size());
                    require(a==b,"wide/T4 output, padding or inactive columns differ");
                    CUDA_CHECK(cudaGraphExecDestroy(exec));CUDA_CHECK(cudaGraphDestroy(graph));
                    ++cases;values+=n*tokens;
                }
            }
            Tensor unsupported=x.slice(1,0,12),unsupported_y=y.slice(1,0,12);
            require(!impl::sm75_radix_projection_launch(unsupported,w,unsupported_y,q5,stream),"ragged unsupported width skipped fallback");
            for(int field=0;field<4;++field){auto changed=w;if(field==0)changed.n/=2;if(field==1)changed.k+=64;if(field==2)changed.scales=nullptr;if(field==3)changed.layout=QuantLayout::RowSplitSm75Fragment;bool bad=false;try{impl::sm75_radix_projection_prepare(changed,stream,16);}catch(const std::exception&){bad=true;}require(bad,"mutated source identity accepted");}
            impl::sm75_radix_projection_release(w);impl::sm75_radix_projection_release(w);
            require(!impl::sm75_radix_projection_launch(x,w,y,q5,stream),"released wide owner remained active");
        }
    }
    for(int width:{0,1,3,5,12,17}){bool bad=false;try{(void)impl::sm75_radix_projection_bytes(128,5120,QType::Q4G64_F16S,width);}catch(const std::invalid_argument&){bad=true;}require(bad,"invalid capacity accepted");}
    CUDA_CHECK(cudaStreamDestroy(stream));
    std::cout<<"PASS cases="<<cases<<" exact_output_values="<<values<<" strided_padding inactive_columns graph_repeats source_identity fixed_capacity release_reprepare"<<std::endl;
    return 0;
}catch(const std::exception& e){std::cerr<<e.what()<<std::endl;return 1;}}
