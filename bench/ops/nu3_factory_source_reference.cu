#include "core/device.h"
#include "ops/linear/nu3_mlp_sm75.h"
#include "quantized_weight.cuh"
#include "t16_fixture.h"
#include <cuda_fp16.h>
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <stdexcept>
#include <vector>
#include <fstream>
#include <filesystem>
#include <string>
#include <cuda_bf16.h>
using namespace ninfer;
struct Group {std::array<unsigned char,64> code;std::array<unsigned char,16> high;std::array<__half,2> scale;};
Group read(const Weight&w,std::size_t g,bool q5){Group x{};CUDA_CHECK(cudaMemcpy(x.code.data(),static_cast<const unsigned char*>(w.qdata)+g*64,64,cudaMemcpyDeviceToHost));if(q5)CUDA_CHECK(cudaMemcpy(x.high.data(),static_cast<const unsigned char*>(w.qhigh)+g*16,16,cudaMemcpyDeviceToHost));CUDA_CHECK(cudaMemcpy(x.scale.data(),static_cast<const __half*>(w.scales)+g*2,4,cudaMemcpyDeviceToHost));return x;}
int code(const Group&x,int i,bool q5){int q=(x.code[i/2]>>((i%2)*4))&15;if(q5)q-=((x.high[i/8]>>(i%8))&1)*16;else if(q>=8)q-=16;return q;}
int nearest(float value){const int book[8]={-7,-4,-2,0,1,3,4,7};int result=book[0];float distance=std::abs(value-result);for(int i=1;i<8;++i){float d=std::abs(value-book[i]);if(d<distance){distance=d;result=book[i];}}return result;}
int main(int argc,char**argv){try{
 if(argc!=2)throw std::invalid_argument("usage: native BF16 tensor directory");
 const std::filesystem::path root(argv[1]);
 cudaStream_t stream;CUDA_CHECK(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking));{
 for(bool q5:{false,true}){
  int n=q5?5120:34816,k=q5?17408:5120;
  auto weight=bench::make_row_split_weight(q5?QType::Q5G64_F16S:QType::Q4G64_F16S,n,k,k);
  const auto first=(root/(q5?"layer00_down_proj.bf16":"layer00_gate_proj.bf16")).string();
  const auto second=(root/"layer00_up_proj.bf16").string();
  ops::detail::nu3_mlp_prepare_bf16(weight.weight,first.c_str(),q5?nullptr:second.c_str(),stream);
  const std::size_t groups=std::size_t(n)*k/128,half=groups/2;
  std::vector<std::size_t> locations{0,1,half-1,half,half+1,groups-1};
  for(int i=0;i<64;++i)locations.push_back((std::size_t(i)*326791u+13u)%groups);
  std::ifstream file0(first,std::ios::binary),file1;
  if(!q5)file1.open(second,std::ios::binary);
  for(const auto at:locations){
   std::array<__nv_bfloat16,128> raw;
   auto& input=(!q5&&at>=half)?file1:file0;
   const auto group=(!q5&&at>=half)?at-half:at;
   input.seekg(group*128*2);input.read(reinterpret_cast<char*>(raw.data()),raw.size()*2);
   if(!input)throw std::runtime_error("independent source read failed");
   const auto after=read(weight.weight,at,q5);float scale=__half2float(after.scale[0]);
   if(scale!=__half2float(after.scale[1]))throw std::runtime_error("source scale pair diverged");
   std::array<float,128> value;float bound=0;
   for(int i=0;i<128;++i){value[i]=__bfloat162float(raw[i]);bound=std::max(bound,std::abs(value[i])/7.f);}
   double best=1e100;bool known=false;
   for(int candidate=0;candidate<9;++candidate){
    float s=__half2float(__float2half_rn(bound*(1.f-.05f*candidate))),inv=s?1.f/s:0;double error=0;
    known|=s==scale;
    for(float v:value){double delta=v-nearest(v*inv)*s;error+=delta*delta;}
    best=std::min(best,error);
   }
   double actual=0;float inv=scale?1.f/scale:0;
   for(int i=0;i<128;++i){int q=code(after,i,q5);if(q!=nearest(value[i]*inv))throw std::runtime_error("represented source coefficient mismatch");double d=value[i]-q*scale;actual+=d*d;}
   if(!known||actual>best*(1+1e-5)+1e-9)throw std::runtime_error("independent source MSE objective failed");
  }
  const auto before=read(weight.weight,0,q5);
  bool rejected=false;
  try{ops::detail::nu3_mlp_prepare_bf16(weight.weight,"/nonexistent-factory-source",nullptr,stream);}catch(const std::invalid_argument&){rejected=true;}
  const auto after=read(weight.weight,0,q5);
  if(!rejected||before.code!=after.code||before.high!=after.high)throw std::runtime_error("bad source admitted or changed weights before validation");
  printf("FACTORY_SOURCE_REFERENCE source=%s groups=%zu values=%zu includes_start_end_and_gate_up_boundary nearest_codes_scale_MSE_valid missing_source_rejected\n",q5?"Q5down":"Q4gate_up",locations.size(),locations.size()*128);
 }
 }CUDA_CHECK(cudaStreamDestroy(stream));return 0;
 }catch(const std::exception&e){fprintf(stderr,"factory_reference: %s\n",e.what());return 1;}}
