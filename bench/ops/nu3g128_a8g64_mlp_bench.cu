#include "../../.chatgpt/production_serve_20260916/whole_mlp_nu3/retained_a8/kernel.cuh"
#include "ops/linear/nu3_mlp_sm75.h"
#include "ops/linear_add/q5/q5_linear_add_a8_sm75.h"
#include "../../.chatgpt/production_serve_20260916/whole_mlp_nu3/layout.cuh"
int main(int argc,char**argv){try{
 const unsigned seed=argc>1?std::stoul(argv[1]):20260916;cudaStream_t stream{};CUDA_CHECK(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking));{
 auto gate=bench::make_row_split_weight(QType::Q4G64_F16S,34816,5120,5120),down=bench::make_row_split_weight(QType::Q5G64_F16S,5120,17408,17408);
 auto cg=bench::make_row_split_weight(QType::Q4G64_F16S,34816,5120,5120),cd=bench::make_row_split_weight(QType::Q5G64_F16S,5120,17408,17408);
 for(auto* w:{&gate.weight,&cg.weight}){bench::t16_fill_weight(*w,34816,5120,false,seed);ops::detail::nu3_mlp_prepare(*w,stream);}
 for(auto* w:{&down.weight,&cd.weight}){bench::t16_fill_weight(*w,5120,17408,true,seed^471u);ops::detail::nu3_mlp_prepare(*w,stream);}
 DeviceBuffer scratch(std::max(ops::detail::q4_linear_swiglu_sm75_transform_scratch_bytes(),ops::detail::q5_linear_add_sm75_transform_scratch_bytes()));
 ops::detail::q4_linear_swiglu_sm75_transform_layout(gate.weight,QuantLayout::RowSplitSm75Fragment,scratch.p,scratch.bytes,stream);
 ops::detail::q5_linear_add_sm75_transform_layout(down.weight,QuantLayout::RowSplitSm75Fragment,scratch.p,scratch.bytes,stream);
 ops::detail::nu3_mlp_transform(cg.weight,QuantLayout::RowSplitSm75Nu3,scratch.p,scratch.bytes,stream);
 ops::detail::nu3_mlp_transform(cd.weight,QuantLayout::RowSplitSm75Nu3,scratch.p,scratch.bytes,stream);
 DeviceBuffer input(5120*4*2),iq(5120*4),is(80*4*2),rq(17408*4),rs(272*4*2),cq(17408*4),cs(272*4*2),initial(5120*4*2),reference(initial.bytes),output(initial.bytes),flush(256ULL<<20);
 bench::t16_fill_input(input.p,5120*4,seed^5431u);bench::t16_fill_input(initial.p,5120*4,seed^373u);
 WorkspaceArena workspace(ops::detail::q4_linear_swiglu_sm75_a8_workspace_bytes(4));
 Tensor x(input.p,DType::BF16,{5120,4}),q(rq.p,DType::I8,{17408,4}),sc(rs.p,DType::FP16,{272,4}),r(reference.p,DType::BF16,{5120,4});
 auto baseline=[&](cudaStream_t s){CUDA_CHECK(cudaMemcpyAsync(reference.p,initial.p,initial.bytes,cudaMemcpyDeviceToDevice,s));ops::detail::q4_linear_swiglu_sm75_a8_to_a8_launch(x,gate.weight,q,sc,workspace,s);ops::detail::q5_linear_add_sm75_a8_prequant_launch(q,sc,down.weight,r,s);};
 auto candidate=[&](cudaStream_t s){
  CUDA_CHECK(cudaMemcpyAsync(output.p,initial.p,initial.bytes,cudaMemcpyDeviceToDevice,s));
  quantize_a8<4><<<(4*80+7)/8,256,0,s>>>((const __nv_bfloat16*)input.p,(std::int8_t*)iq.p,(__half*)is.p);
  gate_w3g128<<<kM/128,512,10240,s>>>((const uint4*)cg.weight.qdata,(const uint2*)((const std::uint8_t*)cg.weight.qdata+nu3_layout::low_bytes(cg.weight)),(const unsigned*)cg.weight.scales,(const std::int8_t*)iq.p,(const __half*)is.p,(std::int8_t*)cq.p,(__half*)cs.p,nullptr);
  down_w3g128<<<DownN/16,128,0,s>>>((const uint4*)cd.weight.qdata,(const uint2*)((const std::uint8_t*)cd.weight.qdata+nu3_layout::low_bytes(cd.weight)),(const __half*)cd.weight.scales,(const std::int8_t*)cq.p,(const __half*)cs.p,(__nv_bfloat16*)output.p);CUDA_CHECK(cudaGetLastError());
 };
 baseline(stream);candidate(stream);CUDA_CHECK(cudaStreamSynchronize(stream));
 std::vector<unsigned char>a(output.bytes),b(output.bytes),g0(rq.bytes),g1(cq.bytes),s0(rs.bytes),s1(cs.bytes);
 reference.copy_to_host(a.data(),a.size());output.copy_to_host(b.data(),b.size());rq.copy_to_host(g0.data(),g0.size());cq.copy_to_host(g1.data(),g1.size());rs.copy_to_host(s0.data(),s0.size());cs.copy_to_host(s1.data(),s1.size());
 auto mismatch=[](const auto&x,const auto&y){std::size_t n=0;for(std::size_t i=0;i<x.size();++i)n+=x[i]!=y[i];return n;};
 const auto out=mismatch(a,b),gate_codes=mismatch(g0,g1),gate_scales=mismatch(s0,s1);
 printf("CONVERTED_MODEL_EXACT output_bytes=%zu output_mismatches=%zu gate_code_mismatches=%zu gate_scale_mismatches=%zu\n",a.size(),out,gate_codes,gate_scales);if(out||gate_codes||gate_scales)return 2;
 cudaGraph_t gr{},gc{};cudaGraphExec_t er{},ec{};
 CUDA_CHECK(cudaStreamBeginCapture(stream,cudaStreamCaptureModeGlobal));baseline(stream);CUDA_CHECK(cudaStreamEndCapture(stream,&gr));CUDA_CHECK(cudaGraphInstantiate(&er,gr,nullptr,nullptr,0));
 CUDA_CHECK(cudaStreamBeginCapture(stream,cudaStreamCaptureModeGlobal));candidate(stream);CUDA_CHECK(cudaStreamEndCapture(stream,&gc));CUDA_CHECK(cudaGraphInstantiate(&ec,gc,nullptr,nullptr,0));
 for(int pass=0;pass<2;++pass){auto a=bench::measure_cold_launch([&](cudaStream_t s){CUDA_CHECK(cudaGraphLaunch(er,s));},flush,stream,4,24);auto b=bench::measure_cold_launch([&](cudaStream_t s){CUDA_CHECK(cudaGraphLaunch(ec,s));},flush,stream,4,24);printf("WHOLE_MLP_A8G64 seed=%u pass=%d reference_us=%.3f candidate_us=%.3f speedup=%.6f\n",seed,pass,a.median_us,b.median_us,a.median_us/b.median_us);}
 CUDA_CHECK(cudaGraphExecDestroy(er));CUDA_CHECK(cudaGraphExecDestroy(ec));CUDA_CHECK(cudaGraphDestroy(gr));CUDA_CHECK(cudaGraphDestroy(gc));
 }CUDA_CHECK(cudaStreamDestroy(stream));return 0;
 }catch(const std::exception&e){fprintf(stderr,"nu3_a8g64: %s\n",e.what());return 1;}}
