#include "core/device.h"
#include "ninfer/ops/linear.h"
#include "ops/linear/w8/w8_vocab_w4_a8_sm75.h"
#include "quantized_weight.cuh"
#include "ninfer_bench_common.h"
#include <cuda_bf16.h>
#include <fstream>
#include <vector>
#include <array>
#include <algorithm>
#include <cstdio>
using namespace ninfer;
void load(std::ifstream& file,std::uint64_t offset,void* target,std::size_t bytes){
 file.seekg(static_cast<std::streamoff>(offset));std::vector<char> block(std::min<std::size_t>(bytes,16ULL<<20));
 for(std::size_t done=0;done<bytes;){const auto n=std::min(block.size(),bytes-done);file.read(block.data(),n);
  if(!file)throw std::runtime_error("native head source is truncated");CUDA_CHECK(cudaMemcpy(static_cast<char*>(target)+done,block.data(),n,cudaMemcpyHostToDevice));done+=n;}
}
int main(int argc,char**argv){try{
 if(argc!=5)throw std::invalid_argument("expected artifact, code offset, scale offset, stopped hidden state");
 constexpr int N=248320,K=5120;cudaStream_t stream;CUDA_CHECK(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking));{
 auto head=bench::make_row_split_weight(QType::W8G32_F16S,N,K,K);std::ifstream artifact(argv[1],std::ios::binary);
 if(!artifact)throw std::runtime_error("cannot open source artifact");
 load(artifact,std::stoull(argv[2]),const_cast<void*>(head.weight.qdata),std::size_t(N)*K);
 load(artifact,std::stoull(argv[3]),const_cast<void*>(head.weight.scales),std::size_t(N)*(K/32)*2);
 DeviceBuffer hidden(K*4*2),full(N*4*2),metadata(N*4*2),tokens(4*4),drafts(3*4),extent(4),flush(256ULL<<20);
 std::ifstream input(argv[4],std::ios::binary);if(!input)throw std::runtime_error("missing stopped state");load(input,0,hidden.p,hidden.bytes);
 Tensor x(hidden.p,DType::BF16,{K,4}),all(full.p,DType::BF16,{N,4}),meta(metadata.p,DType::BF16,{N,4}),ids(tokens.p,DType::I32,{4}),d(drafts.p,DType::I32,{3}),e(extent.p,DType::I32,{1});
 const std::array<int,3> proposed{220,15,25};int active=3;drafts.copy_from_host(proposed.data(),12);extent.copy_from_host(&active,4);
 ops::detail::w8_vocab_w4_a8_prepare(head.weight,stream);
 struct Release {const Weight& w;~Release(){ops::detail::w8_vocab_w4_a8_release(w);}} owner{head.weight};
 ops::linear(x,head.weight,all,stream);CUDA_CHECK(cudaStreamSynchronize(stream));
 std::vector<__nv_bfloat16> values(N*4);full.copy_to_host(values.data(),full.bytes);std::array<int,4> reference{},before{},after{};
 for(int c=0;c<4;++c){float best=-1e30f;for(int id=0;id<N;++id){float value=__bfloat162float(values[c*N+id]);if(value>best){best=value;reference[c]=id;}}}
 ops::detail::w8_vocab_w4_a8_launch(x,head.weight,meta,ids,stream);CUDA_CHECK(cudaStreamSynchronize(stream));tokens.copy_to_host(before.data(),16);
 ops::detail::w8_vocab_w4_a8_include_drafts(x,head.weight,d,e,meta,ids,stream);CUDA_CHECK(cudaStreamSynchronize(stream));tokens.copy_to_host(after.data(),16);
 for(int i=0;i<4;++i)printf("NATIVE_HEAD column=%d previous=%d corrected=%d full_reference=%d\n",i,before[i],after[i],reference[i]);
 if(before[0]!=248046||reference[0]!=220||after!=reference)throw std::runtime_error("recorded false-stop correction does not match native head");
 active=0;extent.copy_from_host(&active,4);ops::detail::w8_vocab_w4_a8_launch(x,head.weight,meta,ids,stream);
 ops::detail::w8_vocab_w4_a8_include_drafts(x,head.weight,d,e,meta,ids,stream);CUDA_CHECK(cudaStreamSynchronize(stream));tokens.copy_to_host(after.data(),16);
 if(after!=before)throw std::runtime_error("zero-draft frame changed");
 active=3;extent.copy_from_host(&active,4);
 auto old=[&](cudaStream_t s){ops::detail::w8_vocab_w4_a8_launch(x,head.weight,meta,ids,s);};
 auto joined=[&](cudaStream_t s){old(s);ops::detail::w8_vocab_w4_a8_include_drafts(x,head.weight,d,e,meta,ids,s);};
 cudaGraph_t a,b;cudaGraphExec_t aa,bb;
 CUDA_CHECK(cudaStreamBeginCapture(stream,cudaStreamCaptureModeGlobal));old(stream);CUDA_CHECK(cudaStreamEndCapture(stream,&a));CUDA_CHECK(cudaGraphInstantiate(&aa,a,nullptr,nullptr,0));
 CUDA_CHECK(cudaStreamBeginCapture(stream,cudaStreamCaptureModeGlobal));joined(stream);CUDA_CHECK(cudaStreamEndCapture(stream,&b));CUDA_CHECK(cudaGraphInstantiate(&bb,b,nullptr,nullptr,0));
 for(int pass=0;pass<2;++pass){auto a=bench::measure_cold_launch([&](cudaStream_t s){CUDA_CHECK(cudaGraphLaunch(aa,s));},flush,stream,4,20);auto b=bench::measure_cold_launch([&](cudaStream_t s){CUDA_CHECK(cudaGraphLaunch(bb,s));},flush,stream,4,20);printf("HEAD_COST pass=%d previous_us=%.3f union_us=%.3f added_us=%.3f\n",pass,a.median_us,b.median_us,b.median_us-a.median_us);}
 CUDA_CHECK(cudaGraphExecDestroy(aa));CUDA_CHECK(cudaGraphExecDestroy(bb));CUDA_CHECK(cudaGraphDestroy(a));CUDA_CHECK(cudaGraphDestroy(b));
 }CUDA_CHECK(cudaStreamDestroy(stream));puts("PASS captured native false-stop reproducer and actual full-W8-head agreement, graph replay, zero extent");return 0;
}catch(const std::exception& e){fprintf(stderr,"native_head_reproducer: %s\n",e.what());return 1;}}
