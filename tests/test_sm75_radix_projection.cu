#include "core/device.h"
#include "ops/linear/sm75_radix_projection.h"
#include "ops/op_check.h"
#include "quantized_weight.cuh"
#include "t16_fixture.h"
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <vector>
#include <cmath>
#include <cstring>
#include <iostream>
#include <stdexcept>
using namespace ninfer;
namespace detail=ninfer::ops::detail;
static double bf(std::uint16_t b){return __bfloat162float(__ushort_as_bfloat16(b));}
int main(){try{
 const auto old_q4_reservation=detail::sm75_radix_projection_bytes(4096,5120);
 const auto new_q4_reservation=detail::sm75_radix_projection_bytes(4096,5120,QType::Q4G64_F16S);
 if(old_q4_reservation-new_q4_reservation!=(2ULL<<20))throw std::runtime_error("Q4 sign-plane reservation did not shrink exactly");
 if(detail::sm75_radix_projection_bytes(5120,6144)!=detail::sm75_radix_projection_bytes(5120,6144,QType::Q5G64_F16S))throw std::runtime_error("Q5 reservation changed");

 cudaStream_t s{};CUDA_CHECK(cudaStreamCreateWithFlags(&s,cudaStreamNonBlocking));
 constexpr int n=128,k=5120,t=4;int cases=0;
 for(bool q5:{false,true}){
  auto packed=bench::make_row_split_weight(q5?QType::Q5G64_F16S:QType::Q4G64_F16S,n,k,k);
  const auto&w=packed.weight;
  DeviceBuffer input(k*t*2),initial((n+8)*t*2),output((n+8)*t*2);
  Tensor x(input.p,DType::BF16,{k,t});Tensor whole(output.p,DType::BF16,{n+8,t});Tensor y=whole.slice(0,0,n);
  std::vector<std::uint16_t> init((n+8)*t,0x3e80),bits(k*t),result((n+8)*t);
  initial.copy_from_host(init.data(),initial.bytes);
  bool warmed=false;std::size_t free_after_warmup=0;
  for(unsigned seed:{1234567u,9787654u,1357911u}){
   bench::t16_fill_weight(w,n,k,q5,seed);
   std::vector<unsigned char> lo(n*k/2),hi(n*k/8);std::vector<__half> sc(n*k/64);
   CUDA_CHECK(cudaMemcpy(lo.data(),w.qdata,lo.size(),cudaMemcpyDeviceToHost));
   if(q5)CUDA_CHECK(cudaMemcpy(hi.data(),w.qhigh,hi.size(),cudaMemcpyDeviceToHost));
   CUDA_CHECK(cudaMemcpy(sc.data(),w.scales,sc.size()*2,cudaMemcpyDeviceToHost));
   std::size_t before,total;CUDA_CHECK(cudaMemGetInfo(&before,&total));
   detail::sm75_radix_projection_prepare(w,s);std::size_t prepared;CUDA_CHECK(cudaMemGetInfo(&prepared,&total));
   detail::sm75_radix_projection_prepare(w,s);std::size_t duplicate;CUDA_CHECK(cudaMemGetInfo(&duplicate,&total));
   if(prepared!=duplicate)throw std::runtime_error("duplicate prepared-owner allocation");
   for(int mode=0;mode<4;++mode){
    unsigned state=seed+mode;
    for(auto&b:bits){unsigned v=bench::t16_next(state);int e=mode==0?126:mode==1?120+int(v%13):mode==2?112+int(v%23):127;b=mode==3?0:((v&0x8000u)|(e<<7)|((v>>16)&127));}
    input.copy_from_host(bits.data(),input.bytes);CUDA_CHECK(cudaMemcpyAsync(output.p,initial.p,initial.bytes,cudaMemcpyDeviceToDevice,s));
    if(!detail::sm75_radix_projection_launch(x,w,y,q5,s))throw std::runtime_error("qualified T4 refused");
    CUDA_CHECK(cudaStreamSynchronize(s));output.copy_to_host(result.data(),output.bytes);
    std::vector<double> actual(n*t),expected(n*t);
    for(int col=0;col<t;++col){
     for(int row=0;row<n;++row){
      double sum=q5?bf(init[col*(n+8)+row]):0.;
      for(int j=0;j<k;++j){auto ix=static_cast<std::size_t>(row)*k+j;
       int code=(lo[ix/2]>>((j&1)*4))&15;
       if(q5){code|=((hi[ix/8]>>(j&7))&1)<<4;code=(code^16)-16;}else code=(code^8)-8;
       sum+=double(code)*double(__half2float(sc[row*(k/64)+j/64]))*bf(bits[col*k+j]);
      }
      expected[col*n+row]=sum;actual[col*n+row]=bf(result[col*(n+8)+row]);
     }
     for(int row=n;row<n+8;++row)if(result[col*(n+8)+row]!=init[col*(n+8)+row])throw std::runtime_error("projection overwrote strided guard");
    }
    // Same complete-A16 reduction criterion as the existing LinearAdd suite.
    const test::ReductionCriterion criterion{1.0/256,1.0/256,2.0/256};
    const auto stats=test::compute_reduction_stats(actual.data(),expected.data(),n*t);
    std::cout<<"q="<<(q5?5:4)<<" mode="<<mode<<" seed="<<seed<<" rel_l2="<<stats.relative_l2<<" max_abs="<<stats.maximum_absolute_error<<'\n';
    if(!test::reduction_passes(stats,n*t,criterion))throw std::runtime_error("independent FP64 projection criterion failed");
    ++cases;
   }
   for(int bad=0;bad<4;++bad){auto altered=w;if(bad==0)altered.n/=2;if(bad==1)altered.k+=64;if(bad==2)altered.scales=nullptr;if(bad==3)altered.layout=QuantLayout::RowSplitSm75Fragment;
    bool rejected=false;try{detail::sm75_radix_projection_prepare(altered,s);}catch(const std::exception&){rejected=true;}
    if(!rejected)throw std::runtime_error("prepared owner mutation accepted");
   }
   detail::sm75_radix_projection_release(w);detail::sm75_radix_projection_release(w);
   if(detail::sm75_radix_projection_launch(x,w,y,q5,s))throw std::runtime_error("released prepared owner remained active");
   CUDA_CHECK(cudaDeviceSynchronize());std::size_t after;CUDA_CHECK(cudaMemGetInfo(&after,&total));
   std::cout<<"release_free="<<after<<" before_first_launch="<<before<<" warmed="<<warmed<<'\n';
   // CUDA may lazily materialize module state on the first real launch. Check
   // subsequent complete owner cycles against the released warm context.
   if(warmed && after<free_after_warmup)throw std::runtime_error("prepared-owner reload retained GPU bytes after warm-up");
   if(!warmed){free_after_warmup=after;warmed=true;}
  }
 }
 CUDA_CHECK(cudaStreamDestroy(s));std::cout<<"PASS "<<cases<<" independent FP64 cases, strided guards, source identity, release/reprepare and no memory leak\n";return 0;
}catch(const std::exception&e){std::cerr<<"radix_projection: "<<e.what()<<'\n';return 1;}}
