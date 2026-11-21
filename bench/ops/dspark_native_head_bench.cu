#include "core/device.h"
#include "ninfer/ops/linear.h"
#include "ninfer/ops/argmax.h"
#include "ninfer_bench_common.h"
#include "quantized_weight.cuh"
#include "t16_fixture.h"
#include <cuda_bf16.h>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <stdexcept>
#include <vector>
using namespace ninfer;
constexpr int N=248320,K=5120,T=7;
__global__ void add_markov(const __nv_bfloat16* bias,__nv_bfloat16* logits){
 int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<T*N)logits[i]=__float2bfloat16_rn(__bfloat162float(logits[i])+__bfloat162float(bias[i]));
}
int main(int argc,char**argv){try{
 if(argc!=3)throw std::invalid_argument("artifact file and metadata-derived absolute head offset required");
 auto weight=bench::make_row_split_weight(QType::W8G32_F16S,N,K,K);
 std::ifstream file(argv[1],std::ios::binary);if(!file)throw std::runtime_error("cannot open source artifact");
 file.seekg(std::stoull(argv[2]));std::vector<std::uint8_t> bytes(weight.storage.bytes);
 file.read(reinterpret_cast<char*>(bytes.data()),bytes.size());if(!file)throw std::runtime_error("truncated head payload");
 CUDA_CHECK(cudaMemcpy(weight.storage.p,bytes.data(),bytes.size(),cudaMemcpyHostToDevice));
 bytes.clear();bytes.shrink_to_fit();
 cudaStream_t stream{};CUDA_CHECK(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking));{
 DeviceBuffer x(K*T*2),logits(N*T*2),bias(N*T*2),ids(T*4),flush(256ULL<<20);
 bench::t16_fill_input(x.p,K*T,20260915);bench::t16_fill_input(bias.p,N*T,20260916);
 Tensor input(x.p,DType::BF16,{K,T}),output(logits.p,DType::BF16,{N,T}),tokens(ids.p,DType::I32,{T});
 auto operation=[&](cudaStream_t s){ops::linear(input,weight.weight,output,s);add_markov<<<(T*N+255)/256,256,0,s>>>((const __nv_bfloat16*)bias.p,(__nv_bfloat16*)logits.p);ops::argmax(output,tokens,N,s);};
 operation(stream);CUDA_CHECK(cudaStreamSynchronize(stream));
 cudaGraph_t graph{};cudaGraphExec_t exe{};CUDA_CHECK(cudaStreamBeginCapture(stream,cudaStreamCaptureModeGlobal));operation(stream);CUDA_CHECK(cudaStreamEndCapture(stream,&graph));CUDA_CHECK(cudaGraphInstantiate(&exe,graph,nullptr,nullptr,0));
 for(int pass=0;pass<2;++pass){auto cost=bench::measure_cold_launch([&](cudaStream_t s){CUDA_CHECK(cudaGraphLaunch(exe,s));},flush,stream,3,15);printf("HEAD full_vocab=%d hidden=%d columns=%d pass=%d median_us=%.3f min_us=%.3f actual_target_W8=1 synthetic_hidden_bias=1\n",N,K,T,pass,cost.median_us,cost.min_us);}
 CUDA_CHECK(cudaGraphExecDestroy(exe));CUDA_CHECK(cudaGraphDestroy(graph));}
 CUDA_CHECK(cudaStreamDestroy(stream));return 0;
}catch(const std::exception&e){fprintf(stderr,"dspark_head_cost: %s\n",e.what());return 1;}}
