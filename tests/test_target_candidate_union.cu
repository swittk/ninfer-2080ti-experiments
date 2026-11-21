#include "core/device.h"
#include "ops/linear/w8/w8_target_candidate_union.cuh"
#include <array>
#include <vector>
#include <random>
#include <cstring>
#include <cstdio>
#include <stdexcept>
using namespace ninfer;
int main(){try{
 constexpr int N=1024;cudaStream_t stream;CUDA_CHECK(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking));
 void *dm=nullptr,*de=nullptr,*dd=nullptr,*dx=nullptr,*dt=nullptr,*dr=nullptr;
 CUDA_CHECK(cudaMalloc(&dm,N*4*2));CUDA_CHECK(cudaMalloc(&de,16*4*2));CUDA_CHECK(cudaMalloc(&dd,3*4));
 CUDA_CHECK(cudaMalloc(&dx,4));CUDA_CHECK(cudaMalloc(&dt,4*4));CUDA_CHECK(cudaMalloc(&dr,16*4));
 std::mt19937 rng(20260916);int checks=0;
 for(int test=0;test<256;++test){
  std::vector<__nv_bfloat16> meta(N*4),exact(16*4);
  std::array<int,3> proposals{int(200+rng()%700),int(200+rng()%700),int(200+rng()%700)};
  if(test%11==0)proposals[1]=proposals[0];
  if(test%13==0)proposals[2]=-1;
  if(test%17==0)proposals[1]=N;
  const int extent=test%6-1,count=std::max(0,std::min(3,extent));
  std::array<int,4> initial{},expected{};std::array<float,4> best{};
  for(int c=0;c<4;++c){
   best[c]=-1e20f;initial[c]=0;
   for(int i=0;i<128;++i){
    const int id=i+1;const float score=float(int(rng()%101)-50)/4;
    std::memcpy(reinterpret_cast<char*>(meta.data()+c*N)+i*4,&id,4);
    meta[c*N+256+i]=__float2bfloat16_rn(score);
    if(score>best[c] || (score==best[c]&&id<initial[c])){best[c]=score;initial[c]=id;}
   }
   expected[c]=initial[c];
   for(int i=0;i<16;++i)exact[c*16+i]=__float2bfloat16_rn(float(int(rng()%161)-80)/4);
   // Force ties to exercise deterministic lowest-token-ID choice.
   if(test%7==0)exact[c*16]=__float2bfloat16_rn(best[c]);
   for(int i=0;i<count;++i){
    const int id=proposals[i];const float score=__bfloat162float(exact[c*16+i]);
    if(id>=0&&id<N&&(score>best[c]||(score==best[c]&&id<expected[c]))){best[c]=score;expected[c]=id;}
   }
  }
  const auto before=meta;
  CUDA_CHECK(cudaMemcpyAsync(dm,meta.data(),N*4*2,cudaMemcpyHostToDevice,stream));
  CUDA_CHECK(cudaMemcpyAsync(de,exact.data(),16*4*2,cudaMemcpyHostToDevice,stream));
  CUDA_CHECK(cudaMemcpyAsync(dd,proposals.data(),3*4,cudaMemcpyHostToDevice,stream));
  CUDA_CHECK(cudaMemcpyAsync(dx,&extent,4,cudaMemcpyHostToDevice,stream));
  CUDA_CHECK(cudaMemcpyAsync(dt,initial.data(),16,cudaMemcpyHostToDevice,stream));
  ops::detail::target_union::prepare<<<1,32,0,stream>>>((const int*)dd,(const int*)dx,(int*)dr,N);
  ops::detail::target_union::merge<<<1,32,0,stream>>>((const __nv_bfloat16*)de,(const int*)dd,(const int*)dx,(__nv_bfloat16*)dm,(int*)dt,N);
  CUDA_CHECK(cudaGetLastError());CUDA_CHECK(cudaStreamSynchronize(stream));
  std::array<int,4> actual;std::array<int,16> prepared;
  CUDA_CHECK(cudaMemcpy(actual.data(),dt,16,cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(prepared.data(),dr,64,cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(meta.data(),dm,N*4*2,cudaMemcpyDeviceToHost));
  if(actual!=expected)throw std::runtime_error("union does not match independent argmax");
  for(int id:prepared)if(id<0||id>=N)throw std::runtime_error("unsafe exact-head source row");
  if(count==0 && std::memcmp(before.data(),meta.data(),N*4*2)!=0)throw std::runtime_error("zero-extent metadata changed");
  for(int c=0;c<4;++c){
   bool found=false;
   for(int i=0;i<128;++i){int id;std::memcpy(&id,reinterpret_cast<const char*>(meta.data()+c*N)+i*4,4);
    if(id==actual[c] && __bfloat162float(meta[c*N+256+i])==best[c])found=true;
   }
   if(!found)throw std::runtime_error("winner missing from indexed metadata");
  }
  checks+=4;
 }
 for(void* p:{dm,de,dd,dx,dt,dr})CUDA_CHECK(cudaFree(p));CUDA_CHECK(cudaStreamDestroy(stream));
 std::printf("PASS target candidate union %d independent column argmax cases, ties, duplicate IDs, invalid IDs, clamped and zero extents, metadata winner\n",checks);return 0;
}catch(const std::exception& e){std::fprintf(stderr,"union_test: %s\n",e.what());return 1;}}
