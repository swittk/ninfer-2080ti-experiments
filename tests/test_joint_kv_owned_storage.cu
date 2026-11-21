#include "core/device.h"
#include "core/arena.h"
#include "../apps/dual/joint_kv/runtime_kernels.cuh"
#include <cuda_fp16.h>
#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <vector>
using namespace ninfer;
namespace jr = ninfer::apps::dual::joint_runtime;
__global__ void values(float* out,int count,unsigned seed) {
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i<count)out[i]=float(int((unsigned(i)*17U+seed)%129U)-64)/32.f;
}
std::vector<std::uint8_t> bytes(DeviceBuffer& source) {
    std::vector<std::uint8_t> out(source.bytes);
    source.copy_to_host(out.data(),out.size());return out;
}
void one_case(int begin,int end,int prefix,cudaStream_t stream) {
    constexpr int capacity=2048,blocks=capacity/32;
    const int rows=std::max(0,std::min(end,prefix)-begin);
    const int stored=std::max(1,rows/32),origin=begin/32;
    DeviceBuffer full_k(std::size_t(4)*blocks*4096),full_v(std::size_t(4)*blocks*2048);
    DeviceBuffer small_k(std::size_t(4)*stored*4096),small_v(std::size_t(4)*stored*2048);
    DeviceBuffer matrix(std::size_t(std::max(rows,1))*64*4);
    for(DeviceBuffer* p:{&full_k,&full_v,&small_k,&small_v})CUDA_CHECK(cudaMemsetAsync(p->p,0,p->bytes,stream));
    for(int h=0;h<4;++h) {
        if(rows==0)break;
        for(int rank:{64,32}) {
            unsigned seed=20260918U+unsigned(h)*31U+rank;
            values<<<(rows*rank+255)/256,256,0,stream>>>((float*)matrix.p,rows*rank,seed);
            if(rank==64) {
                jr::pack_projected<64,true><<<(std::size_t(rows/32)*1024+255)/256,256,0,stream>>>(
                    (const float*)matrix.p,begin,rows,h,blocks,(unsigned*)full_k.p);
                jr::pack_projected<64,true><<<(std::size_t(rows/32)*1024+255)/256,256,0,stream>>>(
                    (const float*)matrix.p,begin,rows,h,stored,(unsigned*)small_k.p,origin);
            } else {
                jr::pack_projected<32,false><<<(std::size_t(rows/32)*512+255)/256,256,0,stream>>>(
                    (const float*)matrix.p,begin,rows,h,blocks,(unsigned*)full_v.p);
                jr::pack_projected<32,false><<<(std::size_t(rows/32)*512+255)/256,256,0,stream>>>(
                    (const float*)matrix.p,begin,rows,h,stored,(unsigned*)small_v.p,origin);
            }
            CUDA_CHECK(cudaGetLastError());CUDA_CHECK(cudaStreamSynchronize(stream));
            auto& buffer=rank==64?small_k:small_v;
            std::vector<unsigned> packed(buffer.bytes/4);buffer.copy_to_host(packed.data(),buffer.bytes);
            // Independent host mapping from each logical row/feature pair to
            // its native matrix operand word, not another call to the packer.
            for(int block=0;block<rows/32;++block)for(int n=0;n<rank/8;++n)
                for(int nt=0;nt<4;++nt)for(int lane=0;lane<32;++lane) {
                    int row,feature,row2,feature2;
                    if(rank==64) { row=block*32+nt*8+lane/4;feature=n*8+2*(lane%4);row2=row;feature2=feature+1; }
                    else { row=block*32+nt*8+2*(lane%4);feature=n*8+lane/4;row2=row+1;feature2=feature; }
                    auto number=[&](int r,int f){return float(int((unsigned(r*rank+f)*17U+seed)%129U)-64)/32.f;};
                    unsigned expected=__half_as_ushort(__float2half_rn(number(row,feature))) |
                        (unsigned(__half_as_ushort(__float2half_rn(number(row2,feature2))))<<16);
                    auto at=((std::size_t(h)*stored+block)*(rank/8)+n)*128+nt*32+lane;
                    if(packed[at]!=expected)throw std::runtime_error("independent packed-coordinate mapping failed");
                }
        }
    }
    const int parts=(end+511)/512-begin/512;
    DeviceBuffer q(96*64*2),shift(96*4),positions(16),cut(96*4);
    DeviceBuffer old_masks(4*24*blocks*4),new_masks(old_masks.bytes),old_output(96*parts*32*4),new_output(old_output.bytes),old_states(96*parts*16),new_states(old_states.bytes);
    std::vector<__half> query(96*64);
    for(std::size_t i=0;i<query.size();++i)query[i]=__float2half_rn(float(int(i%17)-8)/128.f);
    std::vector<float> thresholds(96);
    for(int i=0;i<96;++i)thresholds[i]=float(i%3)-.75f;
    q.copy_from_host(query.data(),q.bytes);cut.copy_from_host(thresholds.data(),cut.bytes);
    CUDA_CHECK(cudaMemsetAsync(shift.p,0,shift.bytes,stream));CUDA_CHECK(cudaDeviceSynchronize());
    auto launch=[&](bool compact){
        auto& output=compact?new_output:old_output;auto& masks=compact?new_masks:old_masks;auto& states=compact?new_states:old_states;
        CUDA_CHECK(cudaMemsetAsync(output.p,0,output.bytes,stream));CUDA_CHECK(cudaMemsetAsync(masks.p,0,masks.bytes,stream));CUDA_CHECK(cudaMemsetAsync(states.p,0,states.bytes,stream));
        ninfer::ops::joint_partition_scan::allkey<false><<<(4*parts*2+3)/4,128,0,stream>>>(
            (const __half*)q.p,(const float*)shift.p,(const int*)positions.p,
            (const unsigned*)(compact?small_k.p:full_k.p),nullptr,nullptr,(const unsigned*)(compact?small_v.p:full_v.p),
            (const float*)cut.p,capacity,prefix,blocks,parts,(unsigned*)masks.p,(float*)output.p,(float4*)states.p,
            capacity,nullptr,nullptr,nullptr,begin,end,begin/512,compact?stored:0,compact?origin:0);
        CUDA_CHECK(cudaGetLastError());
    };
    cudaGraph_t graph;cudaGraphExec_t execution;
    CUDA_CHECK(cudaStreamBeginCapture(stream,cudaStreamCaptureModeGlobal));launch(true);
    CUDA_CHECK(cudaStreamEndCapture(stream,&graph));CUDA_CHECK(cudaGraphInstantiate(&execution,graph,nullptr,nullptr,0));
    for(int offset:{0,-64,0}) {
        std::array<int,4> pos{capacity-4+offset,capacity-3+offset,capacity-2+offset,capacity-1+offset};
        positions.copy_from_host(pos.data(),16);CUDA_CHECK(cudaDeviceSynchronize());
        launch(false);CUDA_CHECK(cudaGraphLaunch(execution,stream));CUDA_CHECK(cudaStreamSynchronize(stream));
        if(bytes(old_masks)!=bytes(new_masks)||bytes(old_output)!=bytes(new_output)||bytes(old_states)!=bytes(new_states))
            throw std::runtime_error("owned coordinates differ from full storage in mask/output/softmax state");
    }
    CUDA_CHECK(cudaGraphExecDestroy(execution));CUDA_CHECK(cudaGraphDestroy(graph));
    std::printf("PASS owned coordinates begin=%d end=%d prefix=%d stored_blocks=%d exact_pack_and_state\n",begin,end,prefix,stored);
}
int main(){try{
    int count=0;if(cudaGetDeviceCount(&count)!=cudaSuccess||count==0)return 77;
    cudaDeviceProp prop{};CUDA_CHECK(cudaGetDeviceProperties(&prop,0));if(prop.major!=7||prop.minor!=5)return 77;
    cudaStream_t stream;CUDA_CHECK(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking));
    try {one_case(320,1248,1024,stream);one_case(1024,2048,1024,stream);one_case(0,2048,2016,stream);one_case(0,1024,2016,stream);}
    catch(...){(void)cudaStreamDestroy(stream);throw;}
    CUDA_CHECK(cudaStreamDestroy(stream));return 0;
}catch(const std::exception& e){std::fprintf(stderr,"%s\n",e.what());return 1;}}
