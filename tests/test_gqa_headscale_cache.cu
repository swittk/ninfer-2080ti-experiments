#include "core/device.h"
#include "core/paged_kv_cache.h"
#include "../apps/dual/headscale_gqa.cuh"
#include <vector>
#include <array>
#include <cmath>
#include <algorithm>
#include <cstdio>
#include <stdexcept>
using namespace ninfer;
struct Buffer{void* p=nullptr;std::size_t size;explicit Buffer(std::size_t n):size(n){CUDA_CHECK(cudaMalloc(&p,n));}~Buffer(){if(p)(void)cudaFree(p);}Buffer(const Buffer&)=delete;};
void require(bool value,const char* message){if(!value)throw std::runtime_error(message);}
int main(){try{
 int available=0;if(cudaGetDeviceCount(&available)!=cudaSuccess||!available)return 77;
 DeviceContext device(0);constexpr int Pages=3,Logical=8,Heads=4,Page=64,Dim=256;
 constexpr int CodeCount=Pages*Heads*Page*Dim,ScaleCount=CodeCount/64;
 Buffer k(CodeCount),v(CodeCount),ks(ScaleCount*2),vs(ScaleCount*2),table(Logical*4),positions(4*4);
 Tensor kt(k.p,DType::I8,{Dim,Heads,Page,Pages}),vt(v.p,DType::I8,{Dim,Heads,Page,Pages});
 Tensor kst(ks.p,DType::FP16,{Dim/64,Heads,Page,Pages}),vst(vs.p,DType::FP16,{Dim/64,Heads,Page,Pages});
 Tensor tb(table.p,DType::I32,{Logical}),pos(positions.p,DType::I32,{4});
 PagedKVLayerView cache{.k_pages=kt,.v_pages=vt,.k_scale_pages=kst,.v_scale_pages=vst,.block_table=tb,
                       .head_dim=Dim,.num_kv_heads=Heads,.dtype=DType::I8,.quant_group=64};
 const std::array<int,Logical> map{-1,2,0,-1,1,-1,-1,-1};
 CUDA_CHECK(cudaMemcpy(table.p,map.data(),table.size,cudaMemcpyHostToDevice));
 for(int seed:{1,7,41}){
  std::vector<std::int8_t> original(CodeCount);
  std::vector<__half> scales(ScaleCount);
  for(int i=0;i<CodeCount;++i)original[i]=static_cast<std::int8_t>((i*43+seed*19)%255-127);
  for(int i=0;i<ScaleCount;++i)scales[i]=__float2half_rn(std::ldexp(1.f,((i+seed)%4)-7));
  auto reset=[&]{CUDA_CHECK(cudaMemcpy(k.p,original.data(),k.size,cudaMemcpyHostToDevice));CUDA_CHECK(cudaMemcpy(ks.p,scales.data(),ks.size,cudaMemcpyHostToDevice));};
  auto reference=[&](std::vector<std::int8_t>& codes,std::vector<__half>& out,int token){
   if(token<0||token>=Logical*64||map[token/64]<0)return;
   const int page=map[token/64],off=token%64;
   for(int head=0;head<Heads;++head){
    const int sbase=((page*Heads+head)*Page+off)*4,cbase=sbase*64;
    float maximum=0;for(int g=0;g<4;++g)maximum=std::max(maximum,__half2float(out[sbase+g]));
    for(int d=0;d<Dim;++d){
     const float value=static_cast<float>(codes[cbase+d])*__half2float(out[sbase+d/64]);
     codes[cbase+d]=static_cast<std::int8_t>(std::clamp(static_cast<int>(std::nearbyint(value/maximum)),-127,127));
    }
    for(int g=0;g<4;++g)out[sbase+g]=__float2half_rn(maximum);
   }
  };
  auto check=[&](const std::vector<std::int8_t>& expected,const std::vector<__half>& exp_scales){
   device.synchronize();std::vector<std::int8_t> got(CodeCount);std::vector<__half> scale(ScaleCount);
   CUDA_CHECK(cudaMemcpy(got.data(),k.p,k.size,cudaMemcpyDeviceToHost));CUDA_CHECK(cudaMemcpy(scale.data(),ks.p,ks.size,cudaMemcpyDeviceToHost));
   require(got==expected,"converted codes disagree with independent reference or touched another physical slot");
   for(int i=0;i<ScaleCount;++i)require(__half2float(scale[i])==__half2float(exp_scales[i]),"scale ownership differs");
   std::array<int,Logical> after{};CUDA_CHECK(cudaMemcpy(after.data(),table.p,table.size,cudaMemcpyDeviceToHost));require(after==map,"cache page table changed");
  };
  for(int prefix:{0,1,63,64,127,128,129,319,320,321,512}){
   reset();auto expected=original;auto exp_scales=scales;
   for(int token=0;token<prefix;++token)reference(expected,exp_scales,token);
   ops::headscale_qk::prepare_headscale_keys(cache,nullptr,prefix,device.stream);check(expected,exp_scales);
   ops::headscale_qk::prepare_headscale_keys(cache,nullptr,prefix,device.stream);check(expected,exp_scales);
  }
  reset();const std::array<int,4> appended{127,128,319,320};
  CUDA_CHECK(cudaMemcpy(positions.p,appended.data(),positions.size,cudaMemcpyHostToDevice));
  auto expected=original;auto exp_scales=scales;for(int token:appended)reference(expected,exp_scales,token);
  ops::headscale_qk::prepare_headscale_keys(cache,&pos,0,device.stream);check(expected,exp_scales);
 }
 std::puts("PASS headscale conversion: 33 prefix boundaries, idempotence, appended positions, missing/reordered physical pages, unchanged table");return 0;
}catch(const std::exception& e){std::fprintf(stderr,"headscale_cache: %s\n",e.what());return 1;}}
