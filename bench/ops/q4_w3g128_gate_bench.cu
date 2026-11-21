#include "../../.chatgpt/production_serve_20260915/whole_mlp_w3g128/kernel.cuh"
__global__ void reference_rows(const uint4* low,const uint2* high,const unsigned* scales,
 const std::int8_t* aq,const __half* as,float* output){
 int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=128)return;
 int token=i/32,row=((i%32)*541+13)%kM,rb=row/8,gid=row%8;float gate=0,up=0;
 for(int g=0;g<kGroups;++g){int a=0,b=0;
  for(int k=0;k<128;++k){
   int step=k/16,lid=(k%16)/4,j=k%4;auto at=(static_cast<long long>(rb)*kGroups+g)*32+gid*4+lid;
   uint4 l=low[at];uint2 h=high[at];
   unsigned gl=step<4?l.x:l.y,ul=step<4?l.z:l.w;
   const int wg=int((gl>>(8*(step%4)+j*2))&3)-4*int((h.x>>(step*4+j))&1);
   const int wu=int((ul>>(8*(step%4)+j*2))&3)-4*int((h.y>>(step*4+j))&1);
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
 const unsigned seed=argc>1?std::stoul(argv[1]):1234567;cudaStream_t stream;CUDA_CHECK(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking));{
 DeviceBuffer errors(4);CUDA_CHECK(cudaMemsetAsync(errors.p,0,4,stream));codec_test<<<16,256,0,stream>>>((unsigned*)errors.p);CUDA_CHECK(cudaStreamSynchronize(stream));unsigned errors_host;errors.copy_to_host(&errors_host,4);printf("CODEC_TEST combinations=4096 components=16384 errors=%u\n",errors_host);if(errors_host)return 2;
 auto weight=bench::make_row_split_weight(QType::Q4G64_F16S,2*kM,kK,kK);bench::t16_fill_weight(weight.weight,2*kM,kK,false,seed);
 DeviceBuffer trans(ops::detail::q4_linear_swiglu_sm75_transform_scratch_bytes());ops::detail::q4_linear_swiglu_sm75_transform_layout(weight.weight,QuantLayout::RowSplitSm75Fragment,trans.p,trans.bytes,stream);
 const std::size_t cells=static_cast<std::size_t>(kM/8)*kGroups*32;
 DeviceBuffer low(cells*16),high(cells*8),scales(cells/4*4),input(kK*4*2),aq(kK*4),as(kGroups*4*2),refq(kM*4),candq(kM*4),refs(kM/64*4*2),cands(kOutputGroups*4*2),raw(kM*4*4),selected(128*4),flush(256ULL<<20);
 prepare_random<<<(cells+255)/256,256,0,stream>>>((uint4*)low.p,(uint2*)high.p,(unsigned*)scales.p,(const unsigned*)weight.weight.scales,seed);bench::t16_fill_input(input.p,kK*4,seed^51357u);
 WorkspaceArena ws(ops::detail::q4_linear_swiglu_sm75_a8_workspace_bytes(4));Tensor x(input.p,DType::BF16,{kK,4}),q(refq.p,DType::I8,{kM,4}),qs(refs.p,DType::FP16,{kM/64,4});
 auto baseline=[&](cudaStream_t s){ops::detail::q4_linear_swiglu_sm75_a8_to_a8_launch(x,weight.weight,q,qs,ws,s);};
 auto run=[&](cudaStream_t s,float* output){quantize_a8_128<<<(4*kGroups+7)/8,256,0,s>>>((const __nv_bfloat16*)input.p,(std::int8_t*)aq.p,(__half*)as.p);gate_w3g128<<<kM/kOutputRowsPerCta,kThreads,4*20*128,s>>>((const uint4*)low.p,(const uint2*)high.p,(const unsigned*)scales.p,(const std::int8_t*)aq.p,(const __half*)as.p,(std::int8_t*)candq.p,(__half*)cands.p,output);CUDA_CHECK(cudaGetLastError());};
 auto candidate=[&](cudaStream_t s){run(s,nullptr);};
 run(stream,(float*)raw.p);reference_rows<<<1,128,0,stream>>>((const uint4*)low.p,(const uint2*)high.p,(const unsigned*)scales.p,(const std::int8_t*)aq.p,(const __half*)as.p,(float*)selected.p);CUDA_CHECK(cudaGetLastError());CUDA_CHECK(cudaStreamSynchronize(stream));
 std::vector<float>actual(kM*4),expected(128);raw.copy_to_host(actual.data(),raw.bytes);selected.copy_to_host(expected.data(),selected.bytes);unsigned wrong=0,nonfinite=0;
 for(float v:actual)nonfinite+=!std::isfinite(v);
 for(int i=0;i<128;++i)wrong+=actual[(i/32)*kM+((i%32)*541+13)%kM]!=expected[i];
 printf("COMPLETE_REFERENCE independent_rows=128 unequal=%u nonfinite=%u compressed_payload=%zu original_payload=%zu\n",wrong,nonfinite,low.bytes+high.bytes,std::size_t(kM)*kK);if(wrong||nonfinite)return 3;
 cudaGraph_t gr,gc;cudaGraphExec_t er,ec;
 CUDA_CHECK(cudaStreamBeginCapture(stream,cudaStreamCaptureModeGlobal));baseline(stream);CUDA_CHECK(cudaStreamEndCapture(stream,&gr));CUDA_CHECK(cudaGraphInstantiate(&er,gr,nullptr,nullptr,0));
 CUDA_CHECK(cudaStreamBeginCapture(stream,cudaStreamCaptureModeGlobal));candidate(stream);CUDA_CHECK(cudaStreamEndCapture(stream,&gc));CUDA_CHECK(cudaGraphInstantiate(&ec,gc,nullptr,nullptr,0));
 for(int pass=0;pass<2;++pass){auto r=bench::measure_cold_launch([&](cudaStream_t s){CUDA_CHECK(cudaGraphLaunch(er,s));},flush,stream,4,24);auto c=bench::measure_cold_launch([&](cudaStream_t s){CUDA_CHECK(cudaGraphLaunch(ec,s));},flush,stream,4,24);printf("WHOLE_GATE seed=%u pass=%d original_us=%.3f w3g128_us=%.3f ratio=%.6f\n",seed,pass,r.median_us,c.median_us,r.median_us/c.median_us);}
 CUDA_CHECK(cudaGraphExecDestroy(er));CUDA_CHECK(cudaGraphExecDestroy(ec));CUDA_CHECK(cudaGraphDestroy(gr));CUDA_CHECK(cudaGraphDestroy(gc));
 }CUDA_CHECK(cudaStreamDestroy(stream));return 0;
 }catch(const std::exception&e){fprintf(stderr,"w3g128: %s\n",e.what());return 1;}}
