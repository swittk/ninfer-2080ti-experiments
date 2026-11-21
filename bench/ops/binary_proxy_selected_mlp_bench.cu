#include "../../.chatgpt/production_serve_20261002/intermediate_channel_quality_bound/full_input_binary_proxy/whole_mlp_cost/selected_gate.cuh"
#include "ops/linear_swiglu/q4/q4_linear_swiglu_a8_sm75.h"
#include "ops/linear_add/q5/q5_linear_add_a8_sm75.h"
#include "ninfer_bench_common.h"
#include "quantized_weight.cuh"
#include "t16_fixture.h"
#include <cub/device/device_radix_sort.cuh>
#include <vector>
#include <algorithm>
#include <stdexcept>
#include <cstring>
#include <numeric>
using namespace ninfer;namespace b=binary_mask;
template<class T>std::vector<T> read(DeviceBuffer&x){std::vector<T>v(x.bytes/sizeof(T));x.copy_to_host(v.data(),x.bytes);return v;}
float hf(std::uint16_t x){return __half2float(__ushort_as_half(x));}
float bf(std::uint16_t x){unsigned bits=unsigned(x)<<16;float value;std::memcpy(&value,&bits,4);return value;}
void binary_reference(const std::vector<unsigned char>&w,const std::vector<std::uint16_t>&scales,
 DeviceBuffer&iq,DeviceBuffer&is,DeviceBuffer&proxy,DeviceBuffer&dots){
 auto q=read<signed char>(iq);auto sx=read<std::uint16_t>(is);auto output=read<float>(proxy);auto actual=read<int>(dots);
 double error=0,norm=0;std::size_t exact=0;
 for(int t=0;t<4;++t)for(int row=0;row<16;++row){double sum[2]={0,0};
  for(int g=0;g<80;++g)for(int part=0;part<2;++part){int dot=0,absolute=0;const int r=row+part*b::M;
   for(int k=0;k<64;++k){const auto at=std::size_t(r)*b::K+g*64+k;int code=(w[at/2]>>((at%2)*4))&15;code=(code^8)-8;absolute+=abs(code);dot+=(code>=0?1:-1)*int(q[t*b::K+g*64+k]);}
   if(dot!=actual[((t*16+row)*80+g)*2+part])throw std::runtime_error("binary integer dot differs from original-code scalar reference");++exact;
   const float mu=__half2float(__float2half_rn(hf(scales[r*80+g])*(absolute/64.f)));
   sum[part]+=double(dot)*mu*hf(sx[t*80+g]);
  }
  const double expected=sum[0]/(1.+std::exp(-sum[0]))*sum[1],got=output[t*b::M+row];
  if(!std::isfinite(expected)||!std::isfinite(got))throw std::runtime_error("nonfinite binary proxy");error+=(expected-got)*(expected-got);norm+=expected*expected;
 }
 printf("BINARY_REFERENCE exact_group_dots=%zu full_proxy_relative_L2=%.9g\n",exact,std::sqrt(error/std::max(norm,1e-30)));
 if(std::sqrt(error/std::max(norm,1e-30))>.0002)throw std::runtime_error("binary proxy scaling/nonlinearity mismatch");
}
void output_reference(const std::vector<unsigned char>&low,const std::vector<unsigned char>&high,const std::vector<std::uint16_t>&scales,
 DeviceBuffer&input,DeviceBuffer&q,DeviceBuffer&s,DeviceBuffer&output){
 auto residual=read<std::uint16_t>(input);auto codes=read<signed char>(q);auto act=read<std::uint16_t>(s);auto actual=read<std::uint16_t>(output);double error=0,norm=0;
 for(int sample=0;sample<128;++sample){int t=sample%4,row=(sample*137+sample/13*421)%5120;double sum=bf(residual[t*5120+row]);
  for(int g=0;g<272;++g){long long dot=0;for(int k=0;k<64;++k){auto i=std::size_t(row)*17408+g*64+k;int w=((low[i/2]>>((i%2)*4))&15)|(((high[i/8]>>(i%8))&1)<<4);w=(w^16)-16;dot+=static_cast<long long>(w)*int(codes[t*17408+g*64+k]);}
   sum+=double(dot)*hf(scales[row*272+g])*hf(act[t*272+g]);}
  double got=bf(actual[t*5120+row]);if(!std::isfinite(got)||!std::isfinite(sum))throw std::runtime_error("nonfinite down output");error+=(sum-got)*(sum-got);norm+=sum*sum;
 }
 double relative=std::sqrt(error/std::max(norm,1e-30));printf("DOWN_REFERENCE samples=128 final_BF16_relative_L2=%.9g\n",relative);if(relative>.004)throw std::runtime_error("down reference failed");
}
int main(int argc,char**argv){try{
 unsigned seed=argc>1?std::stoul(argv[1]):20261002U;cudaStream_t stream;CUDA_CHECK(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking));{
 auto gate=bench::make_row_split_weight(QType::Q4G64_F16S,2*b::M,b::K,b::K),down=bench::make_row_split_weight(QType::Q5G64_F16S,b::K,b::M,b::M);
 bench::t16_fill_weight(gate.weight,2*b::M,b::K,false,seed);bench::t16_fill_weight(down.weight,b::K,b::M,true,seed^91387);CUDA_CHECK(cudaDeviceSynchronize());
 std::vector<unsigned char> gw(std::size_t(b::M)*b::K),dl(std::size_t(b::K)*b::M/2),dh(std::size_t(b::K)*b::M/8);
 std::vector<std::uint16_t> gs(2*b::M*80),ds(b::K*272);
 CUDA_CHECK(cudaMemcpy(gw.data(),gate.weight.qdata,gw.size(),cudaMemcpyDeviceToHost));CUDA_CHECK(cudaMemcpy(gs.data(),gate.weight.scales,gs.size()*2,cudaMemcpyDeviceToHost));
 CUDA_CHECK(cudaMemcpy(dl.data(),down.weight.qdata,dl.size(),cudaMemcpyDeviceToHost));CUDA_CHECK(cudaMemcpy(dh.data(),down.weight.qhigh,dh.size(),cudaMemcpyDeviceToHost));CUDA_CHECK(cudaMemcpy(ds.data(),down.weight.scales,ds.size()*2,cudaMemcpyDeviceToHost));
 DeviceBuffer tmp(std::max(ops::detail::q4_linear_swiglu_sm75_transform_scratch_bytes(),ops::detail::q5_linear_add_sm75_transform_scratch_bytes()));
 ops::detail::q4_linear_swiglu_sm75_transform_layout(gate.weight,QuantLayout::RowSplitSm75Fragment,tmp.p,tmp.bytes,stream);ops::detail::q5_linear_add_sm75_transform_layout(down.weight,QuantLayout::RowSplitSm75Fragment,tmp.p,tmp.bytes,stream);
 DeviceBuffer binary(std::size_t(b::M)*80*16),binary_scales(std::size_t(b::M)*80*4),input(5120*4*2),iq(5120*4),is(80*4*2),planes(80*4*16*4),proxy(17408*4*4),norms(4*8),keys(17408*8),keys_out(keys.bytes),ids(17408*4),order(ids.bytes),
 mid(17408*4*2),full_mid(mid.bytes),mq(17408*4),ms(272*4*2),rq(mq.bytes),rs(ms.bytes),candidate(input.bytes+32),baseline(candidate.bytes),dots(4*16*80*2*4),flush(256ULL<<20);
 b::prepare_weight<<<(b::M*80+255)/256,256,0,stream>>>((const uint4*)gate.weight.qdata,(const unsigned*)gate.weight.scales,(uint4*)binary.p,(unsigned*)binary_scales.p);CUDA_CHECK(cudaGetLastError());CUDA_CHECK(cudaStreamSynchronize(stream));
 std::size_t sort_bytes=0;CUDA_CHECK(cub::DeviceRadixSort::SortPairsDescending(nullptr,sort_bytes,(unsigned long long*)keys.p,(unsigned long long*)keys_out.p,(int*)ids.p,(int*)order.p,b::M,0,64,stream));DeviceBuffer sort_tmp(sort_bytes);
 printf("STORAGE proxy_weight_bytes=%zu sort_scratch_bytes=%zu\n",binary.bytes+binary_scales.bytes,sort_bytes);
 WorkspaceArena workspace(ops::detail::q4_linear_swiglu_sm75_a8_workspace_bytes(4));Tensor x(input.p,DType::BF16,{5120,4}),rqt(rq.p,DType::I8,{17408,4}),rst(rs.p,DType::FP16,{272,4}),out0(baseline.p,DType::BF16,{5120,4}),mqt(mq.p,DType::I8,{17408,4}),mst(ms.p,DType::FP16,{272,4}),out1(candidate.p,DType::BF16,{5120,4});
 auto original=[&](cudaStream_t st){CUDA_CHECK(cudaMemcpyAsync(baseline.p,input.p,input.bytes,cudaMemcpyDeviceToDevice,st));ops::detail::q4_linear_swiglu_sm75_a8_to_a8_launch(x,gate.weight,rqt,rst,workspace,st);ops::detail::q5_linear_add_sm75_a8_prequant_launch(rqt,rst,down.weight,out0,st);};
 auto revised=[&](cudaStream_t st,bool debug){
  CUDA_CHECK(cudaMemcpyAsync(candidate.p,input.p,input.bytes,cudaMemcpyDeviceToDevice,st));
  b::quantize<5120><<<40,256,0,st>>>((const __nv_bfloat16*)input.p,(signed char*)iq.p,(__half*)is.p);b::prepare_planes<<<40,256,0,st>>>((const signed char*)iq.p,(unsigned*)planes.p);
  b::proxy<<<136,512,0,st>>>((const uint4*)binary.p,(const unsigned*)binary_scales.p,(const unsigned*)planes.p,(const __half*)is.p,(float*)proxy.p,debug?(int*)dots.p:nullptr);
  b::energy_norms<<<4,512,0,st>>>((const float*)proxy.p,(double*)norms.p);b::score<<<68,256,0,st>>>((const float*)proxy.p,(const double*)norms.p,(unsigned long long*)keys.p,(int*)ids.p);
  CUDA_CHECK(cub::DeviceRadixSort::SortPairsDescending(sort_tmp.p,sort_bytes,(unsigned long long*)keys.p,(unsigned long long*)keys_out.p,(int*)ids.p,(int*)order.p,b::M,0,64,st));
  CUDA_CHECK(cudaMemsetAsync(mid.p,0,mid.bytes,st));b::selected_gate<<<68,512,4*40*64,st>>>((const std::uint8_t*)gate.weight.qdata,(const std::uint8_t*)gate.weight.scales,(const signed char*)iq.p,(const __half*)is.p,(const int*)order.p,(__nv_bfloat16*)mid.p);
  b::quantize<17408><<<136,256,0,st>>>((const __nv_bfloat16*)mid.p,(signed char*)mq.p,(__half*)ms.p);CUDA_CHECK(cudaGetLastError());
  ops::detail::q5_linear_add_sm75_a8_prequant_launch(mqt,mst,down.weight,out1,st);
 };
 for(int pattern=0;pattern<3;++pattern){
  if(pattern==2)CUDA_CHECK(cudaMemset(input.p,0,input.bytes));else bench::t16_fill_input(input.p,4*5120,seed+pattern*71271);CUDA_CHECK(cudaDeviceSynchronize());
  CUDA_CHECK(cudaMemsetAsync(candidate.p,0x55,candidate.bytes,stream));CUDA_CHECK(cudaMemsetAsync(baseline.p,0x55,baseline.bytes,stream));original(stream);revised(stream,true);CUDA_CHECK(cudaStreamSynchronize(stream));
  binary_reference(gw,gs,iq,is,proxy,dots);output_reference(dl,dh,ds,input,mq,ms,candidate);
  auto row_order=read<int>(order);auto kk=read<unsigned long long>(keys);std::vector<int> expected(b::M);std::iota(expected.begin(),expected.end(),0);std::sort(expected.begin(),expected.end(),[&](int a,int b){return kk[a]>kk[b];});if(expected!=row_order)throw std::runtime_error("GPU rank does not match independent CPU ordering");
  // Compare selected per-row equations with the full original kernel extraction.
  b::selected_gate<<<136,512,4*40*64,stream>>>((const std::uint8_t*)gate.weight.qdata,(const std::uint8_t*)gate.weight.scales,(const signed char*)iq.p,(const __half*)is.p,nullptr,(__nv_bfloat16*)full_mid.p);CUDA_CHECK(cudaStreamSynchronize(stream));
  auto full=read<std::uint16_t>(full_mid),selected=read<std::uint16_t>(mid);std::vector<bool> included(b::M,false);for(int i=0;i<b::Keep;++i)included[row_order[i]]=true;
  for(int t=0;t<4;++t)for(int i=0;i<b::M;++i)if(selected[t*b::M+i]!=(included[i]?full[t*b::M+i]:0))throw std::runtime_error("selected original Q4 row differs");
  b::quantize<17408><<<136,256,0,stream>>>((const __nv_bfloat16*)full_mid.p,(signed char*)mq.p,(__half*)ms.p);CUDA_CHECK(cudaStreamSynchronize(stream));if(read<signed char>(mq)!=read<signed char>(rq)||read<unsigned short>(ms)!=read<unsigned short>(rs))throw std::runtime_error("full kernel extraction changed original A8 output");
  auto result=read<unsigned short>(candidate);for(std::size_t i=input.bytes/2;i<result.size();++i)if(result[i]!=0x5555)throw std::runtime_error("output guard overwritten");
  printf("WHOLE_PATH_CHECK pattern=%d selected_rows=8704 original_mid_values=69632 gpu_sort_exact=1 guards=1\n",pattern);
 }
 bench::t16_fill_input(input.p,4*5120,seed);CUDA_CHECK(cudaDeviceSynchronize());revised(stream,false);CUDA_CHECK(cudaStreamSynchronize(stream));auto expected_output=read<unsigned short>(candidate);
 cudaGraph_t g0{},g1{};cudaGraphExec_t e0{},e1{};
 CUDA_CHECK(cudaStreamBeginCapture(stream,cudaStreamCaptureModeGlobal));original(stream);CUDA_CHECK(cudaStreamEndCapture(stream,&g0));CUDA_CHECK(cudaGraphInstantiate(&e0,g0,nullptr,nullptr,0));
 CUDA_CHECK(cudaStreamBeginCapture(stream,cudaStreamCaptureModeGlobal));revised(stream,false);CUDA_CHECK(cudaStreamEndCapture(stream,&g1));CUDA_CHECK(cudaGraphInstantiate(&e1,g1,nullptr,nullptr,0));
 for(int pass=0;pass<2;++pass){auto base=[&](cudaStream_t st){CUDA_CHECK(cudaGraphLaunch(e0,st));};auto next=[&](cudaStream_t st){CUDA_CHECK(cudaGraphLaunch(e1,st));};double a,b;
  if(pass==0){a=bench::measure_cold_launch(base,flush,stream,4,24).median_us;b=bench::measure_cold_launch(next,flush,stream,4,24).median_us;}else{b=bench::measure_cold_launch(next,flush,stream,4,24).median_us;a=bench::measure_cold_launch(base,flush,stream,4,24).median_us;}
  printf("BINARY_SELECTED_WHOLE_MLP seed=%u pass=%d native_us=%.3f proxy_selected_us=%.3f ratio=%.6f\n",seed,pass,a,b,a/b);fflush(stdout);
 }
 if(read<unsigned short>(candidate)!=expected_output)throw std::runtime_error("graph replay changed output");
 bench::t16_fill_input(input.p,4*5120,seed^73178);CUDA_CHECK(cudaDeviceSynchronize());revised(stream,false);CUDA_CHECK(cudaStreamSynchronize(stream));expected_output=read<unsigned short>(candidate);CUDA_CHECK(cudaGraphLaunch(e1,stream));CUDA_CHECK(cudaStreamSynchronize(stream));if(read<unsigned short>(candidate)!=expected_output)throw std::runtime_error("changed source graph failed");
 printf("CHANGED_SOURCE_DYNAMIC_SELECTION_GRAPH_EXACT\n");CUDA_CHECK(cudaGraphExecDestroy(e0));CUDA_CHECK(cudaGraphExecDestroy(e1));CUDA_CHECK(cudaGraphDestroy(g0));CUDA_CHECK(cudaGraphDestroy(g1));
 }CUDA_CHECK(cudaStreamDestroy(stream));return 0;
}catch(const std::exception&error){fprintf(stderr,"binary-selected MLP: %s\n",error.what());return 1;}}
