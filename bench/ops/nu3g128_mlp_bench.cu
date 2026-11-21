#include "../../.chatgpt/production_serve_20260916/whole_mlp_nu3/kernel.cuh"
#include "ops/linear_add/q5/q5_linear_add_a8_sm75.h"
__global__ void reference_rows(const uint4* low,const uint2* high,const unsigned* scales,
 const std::int8_t* aq,const __half* as,float* output){
 int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=128)return;
 int token=i/32,row=((i%32)*541+13)%kM,rb=row/8,gid=row%8;float gate=0,up=0;
 for(int g=0;g<kGroups;++g){int a=0,b=0;
  for(int k=0;k<128;++k){
   int step=k/16,lid=(k%16)/4,j=k%4;auto at=(static_cast<long long>(rb)*kGroups+g)*32+gid*4+lid;
   uint4 l=low[at];uint2 h=high[at];
   unsigned gl=step<4?l.x:l.y,ul=step<4?l.z:l.w;
   const int wg=nu3_level(((gl>>(8*(step%4)+j*2))&3)|(((h.x>>(step*4+j))&1)<<2));
   const int wu=nu3_level(((ul>>(8*(step%4)+j*2))&3)|(((h.y>>(step*4+j))&1)<<2));
   int x=aq[token*kK+g*128+k];a+=wg*x;b+=wu*x;
  }
  const unsigned scale=scales[(static_cast<long long>(rb)*kGroups+g)*8+gid];
  const float sa=__half2float(as[token*kGroups+g]);
  gate=__fmaf_rn(float(a),__half2float(__ushort_as_half(scale&65535))*sa,gate);
  up=__fmaf_rn(float(b),__half2float(__ushort_as_half(scale>>16))*sa,up);
 }
 output[i]=__bfloat162float(__float2bfloat16_rn(gate/(1.f+expf(-gate))*up));
}
int main(int argc,char**argv){try{
 const unsigned seed=argc>1?std::stoul(argv[1]):1234567;
 DeviceBuffer errors(4);CUDA_CHECK(cudaMemset(errors.p,0,4));codec_test<<<16,256>>>((unsigned*)errors.p);CUDA_CHECK(cudaGetLastError());CUDA_CHECK(cudaDeviceSynchronize());unsigned bad=0;errors.copy_to_host(&bad,4);printf("CODEC 4096 combinations mismatches=%u\n",bad);if(bad)return 2;
cudaStream_t stream;CUDA_CHECK(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking));{
 auto gate=bench::make_row_split_weight(QType::Q4G64_F16S,2*kM,kK,kK),down=bench::make_row_split_weight(QType::Q5G64_F16S,DownN,DownK,DownK);
 bench::t16_fill_weight(gate.weight,2*kM,kK,false,seed);bench::t16_fill_weight(down.weight,DownN,DownK,true,seed^831u);
 DeviceBuffer trans(std::max(ops::detail::q4_linear_swiglu_sm75_transform_scratch_bytes(),ops::detail::q5_linear_add_sm75_transform_scratch_bytes()));
 ops::detail::q4_linear_swiglu_sm75_transform_layout(gate.weight,QuantLayout::RowSplitSm75Fragment,trans.p,trans.bytes,stream);
 ops::detail::q5_linear_add_sm75_transform_layout(down.weight,QuantLayout::RowSplitSm75Fragment,trans.p,trans.bytes,stream);
 const std::size_t gc=static_cast<std::size_t>(kM/8)*kGroups*32,dc=static_cast<std::size_t>(DownN/16)*DownGroups*32;
 DeviceBuffer gl(gc*16),gh(gc*8),gs(gc),dl(dc*16),dh(dc*8),ds(dc),input(kK*4*2),iq(kK*4),is(kGroups*4*2),rq(kM*4),cq(kM*4),rs(kM/64*4*2),cs(kOutputGroups*4*2),initial(DownN*4*2),reference(initial.bytes),candidate(initial.bytes),selected(128*2),flush(256ULL<<20);
 prepare_random<<<(gc+255)/256,256,0,stream>>>((uint4*)gl.p,(uint2*)gh.p,(unsigned*)gs.p,(const unsigned*)gate.weight.scales,seed);
 prepare_down_random<<<(dc+255)/256,256,0,stream>>>((uint4*)dl.p,(uint2*)dh.p,(__half*)ds.p,(const __half*)down.weight.scales,seed^471u);
 bench::t16_fill_input(input.p,kK*4,seed^831u);bench::t16_fill_input(initial.p,DownN*4,seed^167u);
 WorkspaceArena workspace(ops::detail::q4_linear_swiglu_sm75_a8_workspace_bytes(4));
 Tensor x(input.p,DType::BF16,{kK,4}),q(rq.p,DType::I8,{kM,4}),s(rs.p,DType::FP16,{kM/64,4}),r(reference.p,DType::BF16,{DownN,4});
 auto baseline=[&](cudaStream_t stream){CUDA_CHECK(cudaMemcpyAsync(reference.p,initial.p,initial.bytes,cudaMemcpyDeviceToDevice,stream));ops::detail::q4_linear_swiglu_sm75_a8_to_a8_launch(x,gate.weight,q,s,workspace,stream);ops::detail::q5_linear_add_sm75_a8_prequant_launch(q,s,down.weight,r,stream);};
 auto newgate=[&](cudaStream_t stream){quantize_a8_128<<<(4*kGroups+7)/8,256,0,stream>>>((const __nv_bfloat16*)input.p,(std::int8_t*)iq.p,(__half*)is.p);gate_w3g128<<<kM/128,512,10240,stream>>>((const uint4*)gl.p,(const uint2*)gh.p,(const unsigned*)gs.p,(const std::int8_t*)iq.p,(const __half*)is.p,(std::int8_t*)cq.p,(__half*)cs.p,nullptr);};
 auto newdown=[&](cudaStream_t stream){down_w3g128<<<DownN/16,128,0,stream>>>((const uint4*)dl.p,(const uint2*)dh.p,(const __half*)ds.p,(const std::int8_t*)cq.p,(const __half*)cs.p,(__nv_bfloat16*)candidate.p);};
 auto compact=[&](cudaStream_t stream){CUDA_CHECK(cudaMemcpyAsync(candidate.p,initial.p,initial.bytes,cudaMemcpyDeviceToDevice,stream));newgate(stream);newdown(stream);CUDA_CHECK(cudaGetLastError());};
 compact(stream);reference_down_rows<<<1,128,0,stream>>>((const uint4*)dl.p,(const uint2*)dh.p,(const __half*)ds.p,(const std::int8_t*)cq.p,(const __half*)cs.p,(const __nv_bfloat16*)initial.p,(__nv_bfloat16*)selected.p);CUDA_CHECK(cudaGetLastError());CUDA_CHECK(cudaStreamSynchronize(stream));
 std::vector<__nv_bfloat16>a(DownN*4),b(128);candidate.copy_to_host(a.data(),candidate.bytes);selected.copy_to_host(b.data(),selected.bytes);unsigned wrong=0,nonfinite=0;
 for(auto v:a)nonfinite+=!std::isfinite(__bfloat162float(v));for(int i=0;i<128;++i)wrong+=__bfloat162float(a[(i/32)*DownN+((i%32)*157+11)%DownN])!=__bfloat162float(b[i]);
 printf("COMPLETE_MLP_CHECK rows=128 mismatches=%u nonfinite=%u compressed_weight_bytes=%zu\n",wrong,nonfinite,gl.bytes+gh.bytes+dl.bytes+dh.bytes);if(wrong||nonfinite)return 3;
 cudaGraph_t gr,gcgraph;cudaGraphExec_t er,ec;
 CUDA_CHECK(cudaStreamBeginCapture(stream,cudaStreamCaptureModeGlobal));baseline(stream);CUDA_CHECK(cudaStreamEndCapture(stream,&gr));CUDA_CHECK(cudaGraphInstantiate(&er,gr,nullptr,nullptr,0));
 CUDA_CHECK(cudaStreamBeginCapture(stream,cudaStreamCaptureModeGlobal));compact(stream);CUDA_CHECK(cudaStreamEndCapture(stream,&gcgraph));CUDA_CHECK(cudaGraphInstantiate(&ec,gcgraph,nullptr,nullptr,0));
 for(int pass=0;pass<2;++pass){auto old=bench::measure_cold_launch([&](cudaStream_t stream){CUDA_CHECK(cudaGraphLaunch(er,stream));},flush,stream,4,24);auto next=bench::measure_cold_launch([&](cudaStream_t stream){CUDA_CHECK(cudaGraphLaunch(ec,stream));},flush,stream,4,24);printf("WHOLE_MLP seed=%u pass=%d original_us=%.3f W3G128_us=%.3f ratio=%.6f\n",seed,pass,old.median_us,next.median_us,old.median_us/next.median_us);}
 CUDA_CHECK(cudaGraphExecDestroy(er));CUDA_CHECK(cudaGraphExecDestroy(ec));CUDA_CHECK(cudaGraphDestroy(gr));CUDA_CHECK(cudaGraphDestroy(gcgraph));
 }CUDA_CHECK(cudaStreamDestroy(stream));return 0;
 }catch(const std::exception&e){fprintf(stderr,"w3g128_mlp: %s\n",e.what());return 1;}}
