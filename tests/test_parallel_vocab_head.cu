#include "dual/parallel_vocab_head.h"
#include "core/arena.h"
#include "core/device.h"
#include "ops/linear/w8/w8_small_t_mma.cuh"
#include <cuda_bf16.h>
#include <cstdint>
#include <cstdio>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <vector>
using namespace ninfer;
namespace {
constexpr int N=248320,K=5120,T=4,Guard=128;
using G=ops::detail::W8VocabularyProjectionGeometry;
using S=typename ops::detail::W8LinearSmallTProductionSchedule<G,T>::Type;
__device__ unsigned hash(unsigned x){x^=x>>16;x*=0x7feb352dU;x^=x>>15;x*=0x846ca68bU;return x^(x>>16);}
__global__ void initialize(unsigned* weights,std::uint16_t* scales,__nv_bfloat16* input,unsigned seed){
 auto i=std::size_t(blockIdx.x)*blockDim.x+threadIdx.x,stride=std::size_t(gridDim.x)*blockDim.x;
 for(auto j=i;j<std::size_t(N)*K/4;j+=stride)weights[j]=hash(unsigned(j)^seed);
 for(auto j=i;j<std::size_t(N)*(K/32);j+=stride)scales[j]=0x2000U+hash(unsigned(j)+seed)%0xc00U;
 for(auto j=i;j<T*K;j+=stride)input[j]=__float2bfloat16_rn(float(int(hash(unsigned(j)^seed)%2049)-1024)/1024.f);
}
std::vector<std::uint16_t> read(DeviceBuffer& buffer){std::vector<std::uint16_t>x(buffer.bytes/2);buffer.copy_to_host(x.data(),buffer.bytes);return x;}
template<class F> void rejects(F operation){bool failed=false;try{operation();}catch(const std::exception&){failed=true;}if(!failed)throw std::runtime_error("invalid API request was accepted");}
}
int main(int argc,char**argv){try{
 int count=0;if(cudaGetDeviceCount(&count)!=cudaSuccess||count<2)return 77;
 int owner=argc>1?std::stoi(argv[1]):1,helper=1-owner;
 for(int device:{owner,helper}){cudaDeviceProp p{};CUDA_CHECK(cudaGetDeviceProperties(&p,device));if(p.major!=7||p.minor!=5)return 77;}
 CUDA_CHECK(cudaSetDevice(owner));cudaStream_t stream;CUDA_CHECK(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking));
 {
 DeviceBuffer codes(std::size_t(N)*K),scales(std::size_t(N)*(K/32)*2),input(K*T*2),reference((std::size_t(N)*T+Guard)*2),candidate(reference.bytes),replacement(reference.bytes);
 Weight w{};w.qtype=QType::W8G32_F16S;w.layout=QuantLayout::RowSplit;w.n=N;w.k=K;w.group=32;w.qdata=codes.p;w.scales=scales.p;
 Tensor x(input.p,DType::BF16,{K,T}),y(candidate.p,DType::BF16,{N,T});
 DeviceBuffer greedy_ids((T+8)*sizeof(std::int32_t));
 Tensor ids(greedy_ids.p,DType::I32,{T});
 auto check_greedy=[&](apps::dual::ParallelVocabHead& head,const std::vector<std::uint16_t>& logits,int valid){
  CUDA_CHECK(cudaMemsetAsync(greedy_ids.p,0x55,greedy_ids.bytes,stream));
  if(!head.execute_greedy(x,w,ids,valid,stream))throw std::runtime_error("greedy result refused");
  CUDA_CHECK(cudaStreamSynchronize(stream));
  std::int32_t actual[T+8];greedy_ids.copy_to_host(actual,sizeof(actual));
  for(int t=0;t<T;++t){
   auto value=[&](int row){return __bfloat162float(__ushort_as_bfloat16(logits[t*N+row]));};
   int best=0;for(int row=1;row<valid;++row)if(value(row)>value(best))best=row;
   if(actual[t]!=best)throw std::runtime_error("greedy differs from independent full-domain CPU argmax");
  }
  for(int i=T;i<T+8;++i)if(actual[i]!=0x55555555)throw std::runtime_error("greedy output guard changed");
 };

 auto native=[&]{ops::detail::W8ContiguousOutput output{static_cast<__nv_bfloat16*>(reference.p),N};
  ops::detail::w8_small_t_mma_kernel<G,T,S><<<N/16,S::kThreads,0,stream>>>((const __nv_bfloat16*)input.p,(const std::uint8_t*)codes.p,(const std::uint8_t*)scales.p,output);CUDA_CHECK(cudaGetLastError());};
 std::vector<std::uint16_t> expected;
 {
 apps::dual::ParallelVocabHead head(helper);
 auto one=x.slice(1,0,1),one_output=y.slice(1,0,1);
 if(head.execute(one,w,one_output,stream))throw std::runtime_error("unsupported T1 intercepted");
 rejects([&]{(void)head.execute(x,w,y,stream);});
 {apps::dual::ParallelVocabHead invalid(owner);rejects([&]{invalid.prepare(w,stream);});}
 for(unsigned seed:{20260929U,916472U}){
  initialize<<<4096,256,0,stream>>>((unsigned*)codes.p,(std::uint16_t*)scales.p,(__nv_bfloat16*)input.p,seed);CUDA_CHECK(cudaGetLastError());CUDA_CHECK(cudaStreamSynchronize(stream));
  head.prepare(w,stream); // Deliberately changed source contents at the SAME addresses.
  if(head.stats().helper_weight_bytes!=675430400)throw std::runtime_error("head accounting changed");
  for(int pattern=0;pattern<3;++pattern){
   if(pattern==1)CUDA_CHECK(cudaMemsetAsync(input.p,0,input.bytes,stream));
   if(pattern==2){
    const std::uint16_t nan=0x7fc0;
    CUDA_CHECK(cudaMemcpyAsync(input.p,&nan,2,cudaMemcpyHostToDevice,stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
   }

   CUDA_CHECK(cudaMemsetAsync(reference.p,0x55,reference.bytes,stream));native();CUDA_CHECK(cudaStreamSynchronize(stream));expected=read(reference);
   for(int repeat=0;repeat<2;++repeat){
    CUDA_CHECK(cudaMemsetAsync(candidate.p,0x55,candidate.bytes,stream));
    if(!head.execute(x,w,y,stream))throw std::runtime_error("supported head refused");
    CUDA_CHECK(cudaStreamSynchronize(stream));if(read(candidate)!=expected)throw std::runtime_error("full native logits or guard differ");
   }
   for(int valid:{1,124161,248077}){check_greedy(head,expected,valid);check_greedy(head,expected,valid);}
   // Switching back from the compact result must fill all logits, not leave a
   // partial buffer or reuse a tokens-only graph.
   Tensor other(replacement.p,DType::BF16,{N,T});CUDA_CHECK(cudaMemsetAsync(replacement.p,0x55,replacement.bytes,stream));
   if(!head.execute(x,w,other,stream))throw std::runtime_error("rebound output refused");CUDA_CHECK(cudaStreamSynchronize(stream));
   if(read(replacement)!=expected)throw std::runtime_error("output rebinding reused stale graph");
   printf("PASS owner=%d seed=%u pattern=%d native_logits=%d reprepare/replay/rebind/guards\n",owner,seed,pattern,N*T);
  }
 }
 Tensor invalid=x;invalid.ne[2]=2;rejects([&]{(void)head.execute(invalid,w,y,stream);});
 invalid=x;invalid.data=nullptr;rejects([&]{(void)head.execute(invalid,w,y,stream);});
 rejects([&]{(void)head.execute_greedy(x,w,ids,0,stream);});
 rejects([&]{(void)head.execute_greedy(x,w,ids,N+1,stream);});
 Weight bad=w;bad.k-=32;rejects([&]{head.prepare(bad,stream);});
 CUDA_CHECK(cudaMemsetAsync(candidate.p,0x55,candidate.bytes,stream));
 if(!head.execute(x,w,y,stream))throw std::runtime_error("bad preparation destroyed valid old source");CUDA_CHECK(cudaStreamSynchronize(stream));
 if(read(candidate)!=expected)throw std::runtime_error("valid source changed after rejected prepare");
 // Outer graph capture must select native fallback without borrowing or mutating
 // the coordinator's internal graph; a fully graph-safe caller retains its route.
 cudaGraph_t graph{};cudaGraphExec_t executable{};
 CUDA_CHECK(cudaStreamBeginCapture(stream,cudaStreamCaptureModeGlobal));
 if(head.execute_greedy(x,w,ids,248077,stream))throw std::runtime_error("nested greedy graph consumed input");
 if(head.execute(x,w,y,stream))throw std::runtime_error("nested head graph unexpectedly consumed input");native();
 CUDA_CHECK(cudaStreamEndCapture(stream,&graph));CUDA_CHECK(cudaGraphInstantiate(&executable,graph,nullptr,nullptr,0));
 CUDA_CHECK(cudaGraphLaunch(executable,stream));CUDA_CHECK(cudaStreamSynchronize(stream));
 if(read(reference)!=expected)throw std::runtime_error("outer native graph changed");
 CUDA_CHECK(cudaGraphExecDestroy(executable));CUDA_CHECK(cudaGraphDestroy(graph));
 // Submit and destroy immediately: the destructor must join before freeing the
 // pinned reply/source representation. This is a real supported lifecycle edge.
 CUDA_CHECK(cudaMemsetAsync(candidate.p,0x55,candidate.bytes,stream));
 (void)head.execute(x,w,y,stream);
 }
 CUDA_CHECK(cudaSetDevice(owner));if(read(candidate)!=expected)throw std::runtime_error("destruction did not drain result");
 }
 CUDA_CHECK(cudaStreamDestroy(stream));printf("PARALLEL_HEAD_LIFECYCLE_PASS owner=%d\n",owner);return 0;
}catch(const std::exception&e){std::fprintf(stderr,"parallel head test: %s\n",e.what());return 1;}}
