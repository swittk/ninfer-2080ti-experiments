#include "core/device.h"
#include "ops/launcher/gqa_attention.h"
#include <cstdio>
#include <vector>
#include <stdexcept>
#include <cstring>
using namespace ninfer;
struct TestDeviceBuffer {
 void* p=nullptr;std::size_t bytes=0;
 explicit TestDeviceBuffer(std::size_t n):bytes(n){CUDA_CHECK(cudaMalloc(&p,n));}
 TestDeviceBuffer(const TestDeviceBuffer&)=delete;TestDeviceBuffer& operator=(const TestDeviceBuffer&)=delete;
 ~TestDeviceBuffer(){if(p)(void)cudaFree(p);}
};

int main(){try{
 int count=0;if(cudaGetDeviceCount(&count)!=cudaSuccess||count==0){std::puts("SKIP no CUDA device");return 77;}
 DeviceContext device(0);constexpr int SourcePages=4096,Rows=2,Window=8192,Sink=128,Pages=(Window+Sink)/64+2;
 TestDeviceBuffer source(SourcePages*Rows*4),row(4),pos(6*4),local(6*4),table(Pages*4),zero(4);
 std::vector<int> host(SourcePages*Rows);for(int i=0;i<(int)host.size();++i)host[i]=17*i+11;
 CUDA_CHECK(cudaMemcpy(source.p,host.data(),source.bytes,cudaMemcpyHostToDevice));
 for(int first:{0,1,63,64,127,128,8191,8192,8257,8320,65532,261899,262138})for(int width:{1,4,6}){
  if(first+width>262144)continue;
  std::vector<int> positions(width);for(int i=0;i<width;++i)positions[i]=first+i;
  int owner=1;CUDA_CHECK(cudaMemcpy(row.p,&owner,4,cudaMemcpyHostToDevice));CUDA_CHECK(cudaMemcpy(pos.p,positions.data(),width*4,cudaMemcpyHostToDevice));
  Tensor p(pos.p,DType::I32,{width}),src(source.p,DType::I32,{SourcePages,Rows}),r(row.p,DType::I32,{1});
  Tensor lp(local.p,DType::I32,{width}),tb(table.p,DType::I32,{Pages,1}),zr(zero.p,DType::I32,{1});
  ops::detail::gqa_recent_view_launch(p,src,r,Window,Sink,lp,tb,zr,device.stream);device.synchronize();
  std::vector<int> got(width),pages(Pages),unchanged(host.size());int localrow=-1;
  CUDA_CHECK(cudaMemcpy(got.data(),local.p,width*4,cudaMemcpyDeviceToHost));CUDA_CHECK(cudaMemcpy(pages.data(),table.p,Pages*4,cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(&localrow,zero.p,4,cudaMemcpyDeviceToHost));CUDA_CHECK(cudaMemcpy(unchanged.data(),source.p,source.bytes,cudaMemcpyDeviceToHost));
  if(localrow!=0||unchanged!=host)throw std::runtime_error("view changed source table or wrong destination row");
  const int tail=std::max(Sink/64,std::max(0,first+1-Window)/64);
  for(int i=0;i<width;++i){
   int g=positions[i],l=got[i];
   if(l<0||l>=Pages*64||pages[l/64]!=host[SourcePages+g/64]||l%64!=g%64)
    throw std::runtime_error("rebased append would reach a different physical slot");
  }
  for(int i=0;i<Pages;++i){int original=i<Sink/64?i:tail+i-Sink/64;
   if(pages[i]!=(original<SourcePages?host[SourcePages+original]:-1))throw std::runtime_error("sink/recent alias mismatch");}
 }
 std::puts("PASS 39 recent-view boundaries, physical-slot equality, nonzero owner, unchanged source table");return 0;
}catch(const std::exception& e){std::fprintf(stderr,"recent_view: %s\n",e.what());return 1;}}
