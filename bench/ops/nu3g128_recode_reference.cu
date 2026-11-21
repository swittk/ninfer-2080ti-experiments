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
using namespace ninfer;
struct Group {std::array<unsigned char,64> code;std::array<unsigned char,16> high;std::array<__half,2> scale;};
Group read(const Weight&w,std::size_t g,bool q5){Group x{};CUDA_CHECK(cudaMemcpy(x.code.data(),static_cast<const unsigned char*>(w.qdata)+g*64,64,cudaMemcpyDeviceToHost));if(q5)CUDA_CHECK(cudaMemcpy(x.high.data(),static_cast<const unsigned char*>(w.qhigh)+g*16,16,cudaMemcpyDeviceToHost));CUDA_CHECK(cudaMemcpy(x.scale.data(),static_cast<const __half*>(w.scales)+g*2,4,cudaMemcpyDeviceToHost));return x;}
int code(const Group&x,int i,bool q5){int q=(x.code[i/2]>>((i%2)*4))&15;if(q5)q-=((x.high[i/8]>>(i%8))&1)*16;else if(q>=8)q-=16;return q;}
int nearest(float value){const int book[8]={-7,-4,-2,0,1,3,4,7};int result=book[0];float distance=std::abs(value-result);for(int i=1;i<8;++i){float d=std::abs(value-book[i]);if(d<distance){distance=d;result=book[i];}}return result;}
int main(){try{cudaStream_t stream;CUDA_CHECK(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking));{
for(bool q5:{false,true}){
 int n=q5?5120:34816,k=q5?17408:5120;auto weight=bench::make_row_split_weight(q5?QType::Q5G64_F16S:QType::Q4G64_F16S,n,k,k);bench::t16_fill_weight(weight.weight,n,k,q5,q5?81351u:127431u);
 std::vector<std::size_t> locations;std::vector<Group> before;
 const std::size_t groups=static_cast<std::size_t>(n)*k/128;
 for(int i=0;i<64;++i){const auto at=(static_cast<std::size_t>(i)*326791u+13u)%groups;locations.push_back(at);before.push_back(read(weight.weight,at,q5));}
 ops::detail::nu3_mlp_prepare(weight.weight,stream);
 for(std::size_t j=0;j<locations.size();++j){const auto after=read(weight.weight,locations[j],q5);const auto&old=before[j];
  const float selected=__half2float(after.scale[0]);if(selected!=__half2float(after.scale[1]))throw std::runtime_error("two canonical scales differ");
  std::array<float,128> value;float bound=0;
  for(int i=0;i<128;++i){value[i]=code(old,i,q5)*__half2float(old.scale[i/64]);bound=std::max(bound,std::abs(value[i])/7.f);}
  double best_error=1e100;bool recognized=false;
  for(int variant=0;variant<9;++variant){const float scale=__half2float(__float2half_rn(bound*(1.f-.05f*variant)));const float inv=scale?1.f/scale:0.f;double error=0;
   recognized|=selected==scale;
   for(float v:value){int q=nearest(v*inv);double d=v-q*scale;error+=d*d;}
   best_error=std::min(best_error,error);
  }
  double actual_error=0;const float inv=selected?1.f/selected:0.f;
  for(int i=0;i<128;++i){int q=code(after,i,q5);if(q!=nearest(value[i]*inv))throw std::runtime_error("not nearest valid represented3bitcode");double d=value[i]-q*selected;actual_error+=d*d;}
  if(!recognized||actual_error>best_error*(1.+1e-5)+1e-7)throw std::runtime_error("reconstruction objective failed independent reference");
 }
 printf("NU3_RECODE_REFERENCE source=%s independent_groups=64 values=8192 duplicated_scale_valid nearest_codes_valid reconstruction_objective_valid\n",q5?"Q5down":"Q4gate");
}
}CUDA_CHECK(cudaStreamDestroy(stream));return 0;}catch(const std::exception&e){fprintf(stderr,"recode_reference: %s\n",e.what());return 1;}}
