#include "../../.chatgpt/production_serve_20260916/whole_mlp_nu3/kernel.cuh"
#include "../../.chatgpt/production_serve_20260916/whole_mlp_nu3/layout.cuh"
#include "ops/linear/nu3_mlp_sm75.h"
#include "ops/linear_add/q5/q5_linear_add_a8_sm75.h"
__global__ void compare(const std::uint8_t* a,const std::uint8_t* b,std::size_t count,unsigned* errors){
 const auto i=static_cast<std::size_t>(blockIdx.x)*blockDim.x+threadIdx.x;if(i<count&&a[i]!=b[i])atomicAdd(errors,1u);
}
int main(int argc,char**argv){try{
 const unsigned seed=argc>1?std::stoul(argv[1]):314159u;cudaStream_t stream{};CUDA_CHECK(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking));{
 auto gate=bench::make_row_split_weight(QType::Q4G64_F16S,34816,5120,5120),down=bench::make_row_split_weight(QType::Q5G64_F16S,5120,17408,17408);
 DeviceBuffer scratch(std::max(ops::detail::q4_linear_swiglu_sm75_transform_scratch_bytes(),ops::detail::q5_linear_add_sm75_transform_scratch_bytes())),errors(4);
 for(Weight* w:{&gate.weight,&down.weight}){
  bool isgate=w->n==34816;bench::t16_fill_weight(*w,w->n,w->k,!isgate,seed);
  ops::detail::nu3_mlp_prepare(*w,stream);
  const auto bytes=static_cast<std::size_t>(w->n)*w->k/2,high=!isgate?bytes/4:0,scales=bytes/16;
  DeviceBuffer original(bytes+high+scales);
  CUDA_CHECK(cudaMemcpyAsync(original.p,w->qdata,bytes,cudaMemcpyDeviceToDevice,stream));
  if(high)CUDA_CHECK(cudaMemcpyAsync(static_cast<std::uint8_t*>(original.p)+bytes,w->qhigh,high,cudaMemcpyDeviceToDevice,stream));
  CUDA_CHECK(cudaMemcpyAsync(static_cast<std::uint8_t*>(original.p)+bytes+high,w->scales,scales,cudaMemcpyDeviceToDevice,stream));
  for(int pass=0;pass<3;++pass){
   nu3_layout::transform(*w,scratch.p,scratch.bytes,true,stream);
   nu3_layout::transform(*w,scratch.p,scratch.bytes,false,stream);
   CUDA_CHECK(cudaMemsetAsync(errors.p,0,4,stream));
   compare<<<(bytes+255)/256,256,0,stream>>>((const std::uint8_t*)w->qdata,(const std::uint8_t*)original.p,bytes,(unsigned*)errors.p);
   if(high)compare<<<(high+255)/256,256,0,stream>>>((const std::uint8_t*)w->qhigh,(const std::uint8_t*)original.p+bytes,high,(unsigned*)errors.p);
   compare<<<(scales+255)/256,256,0,stream>>>((const std::uint8_t*)w->scales,(const std::uint8_t*)original.p+bytes+high,scales,(unsigned*)errors.p);
   CUDA_CHECK(cudaGetLastError());CUDA_CHECK(cudaStreamSynchronize(stream));unsigned error=0;errors.copy_to_host(&error,4);
   printf("ROUNDTRIP %s pass=%d compared_bytes=%zu mismatches=%u scratch=%zu\n",isgate?"gate":"down",pass,bytes+high+scales,error,scratch.bytes);if(error)return 2;
  }
 }
 // Exercise the exact packed full MLP with the real converted source matrices.
 nu3_layout::transform(gate.weight,scratch.p,scratch.bytes,true,stream);nu3_layout::transform(down.weight,scratch.p,scratch.bytes,true,stream);
 DeviceBuffer input(5120*4*2),iq(5120*4),is(40*4*2),cq(17408*4),cs(136*4*2),initial(5120*4*2),output(initial.bytes),reference(128*2);
 bench::t16_fill_input(input.p,5120*4,seed^99u);bench::t16_fill_input(initial.p,5120*4,seed^177u);CUDA_CHECK(cudaMemcpyAsync(output.p,initial.p,initial.bytes,cudaMemcpyDeviceToDevice,stream));
 auto gl=(const uint4*)gate.weight.qdata;auto gh=(const uint2*)((const std::uint8_t*)gate.weight.qdata+nu3_layout::low_bytes(gate.weight));
 auto dl=(const uint4*)down.weight.qdata;auto dh=(const uint2*)((const std::uint8_t*)down.weight.qdata+nu3_layout::low_bytes(down.weight));
 quantize_a8_128<<<(4*kGroups+7)/8,256,0,stream>>>((const __nv_bfloat16*)input.p,(std::int8_t*)iq.p,(__half*)is.p);
 gate_w3g128<<<kM/128,512,10240,stream>>>(gl,gh,(const unsigned*)gate.weight.scales,(const std::int8_t*)iq.p,(const __half*)is.p,(std::int8_t*)cq.p,(__half*)cs.p,nullptr);
 down_w3g128<<<DownN/16,128,0,stream>>>(dl,dh,(const __half*)down.weight.scales,(const std::int8_t*)cq.p,(const __half*)cs.p,(__nv_bfloat16*)output.p);
 reference_down_rows<<<1,128,0,stream>>>(dl,dh,(const __half*)down.weight.scales,(const std::int8_t*)cq.p,(const __half*)cs.p,(const __nv_bfloat16*)initial.p,(__nv_bfloat16*)reference.p);
 CUDA_CHECK(cudaGetLastError());CUDA_CHECK(cudaStreamSynchronize(stream));
 std::vector<__nv_bfloat16>a(5120*4),b(128);output.copy_to_host(a.data(),output.bytes);reference.copy_to_host(b.data(),reference.bytes);unsigned wrong=0;
 for(int i=0;i<128;++i)wrong+=__bfloat162float(a[(i/32)*5120+((i%32)*157+11)%5120])!=__bfloat162float(b[i]);
 printf("PACKED_CONVERTED_MLP rows=128 mismatches=%u\n",wrong);if(wrong)return 3;
 }CUDA_CHECK(cudaStreamDestroy(stream));return 0;
}catch(const std::exception&e){fprintf(stderr,"nu3_layout_test: %s\n",e.what());return 1;}}
