#include "../../.chatgpt/production_serve_20260928/monolithic_vector_decode/kernel.cuh"
#include "../../.chatgpt/production_serve_20260929/bank_owned_e8_dataflow/kernel.cuh"
__global__ void scalar_reference(const uint2* codes,const std::int8_t* full,
 const __half* scales,const std::int8_t* aq,const __half* as,float* output){
 int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=128)return;
 int t=i/32,row=((i%32)*541+13)%kM,rb=row/8,gid=row%8;float gate=0,up=0;
 for(int group=0;group<kGroups;++group){
  int gs=0,us=0;
  for(int k=0;k<64;++k){
   int half=k/32,lid=(k%16)/4,within=((k%32)/16)*4+k%4;
   auto ix=(static_cast<long long>(rb)*kGroups+group)*32+gid*4+lid;
   uint2 pair=codes[ix];unsigned base=half?pair.y:pair.x;
   int wgate=int(full[(base&65535)*8+within]);
   int wup=int(full[(base>>16)*8+within]);
   int a=aq[t*kK+group*64+k];gs+=wgate*a;us+=wup*a;
  }
  auto si=((static_cast<long long>(rb)*kGroups+group)*8+gid)*2;float sa=__half2float(as[t*kGroups+group]);
  float wg=.25f*__half2float(scales[si]),wu=.25f*__half2float(scales[si+1]);
  gate=__fmaf_rn(float(gs),wg*sa,gate);up=__fmaf_rn(float(us),wu*sa,up);
 }
 output[i]=__bfloat162float(__float2bfloat16_rn((gate/(1.f+expf(-gate)))*up));
}
void read_file(const char* path,DeviceBuffer& buffer){std::ifstream f(path,std::ios::binary);std::vector<char> v(buffer.bytes);if(!f.read(v.data(),v.size()))throw std::runtime_error(path);buffer.copy_from_host(v.data(),v.size());}

template<class T>std::vector<T> read(DeviceBuffer& b){std::vector<T>v(b.bytes/sizeof(T));b.copy_to_host(v.data(),b.bytes);return v;}
void validate_math(const std::vector<uint2>& codes,const std::vector<signed char>& table,const std::vector<std::uint16_t>& scales,DeviceBuffer& aq,DeviceBuffer& as,DeviceBuffer& raw){
 auto input=read<signed char>(aq);auto input_scales=read<std::uint16_t>(as);auto actual=read<float>(raw);double err=0,norm=0;unsigned nonfinite=0;
 for(int i=0;i<256;++i){const int t=i%4,row=(i*113+i/7*373)%kM;double out[2]={0,0};
  for(int group=0;group<kGroups;++group){int dot[2]={0,0};
   for(int k=0;k<64;++k){const int lane=(row%8)*4+(k%16)/4,within=((k%32)/16)*4+k%4;
    const uint2 pair=codes[((std::size_t)(row/8)*kGroups+group)*32+lane];const unsigned pack=k<32?pair.x:pair.y;
    for(int u=0;u<2;++u)dot[u]+=int(table[((pack>>(16*u))&65535)*8+within])*int(input[t*kK+group*64+k]);
   }
   for(int u=0;u<2;++u)out[u]+=double(dot[u])*.25*__half2float(__ushort_as_half(scales[(((std::size_t)(row/8)*kGroups+group)*8+row%8)*2+u]))*__half2float(__ushort_as_half(input_scales[t*kGroups+group]));
  }
  const double reference=out[0]/(1.+std::exp(-out[0]))*out[1],got=actual[t*kM+row];err+=(reference-got)*(reference-got);norm+=reference*reference;nonfinite+=!std::isfinite(got);
 }
 const double relative=std::sqrt(err/std::max(norm,1e-30));printf("FP64_E8_GATE samples=256 relative_L2=%.9g nonfinite=%u\n",relative,nonfinite);
 if(nonfinite||relative>.004)throw std::runtime_error("independent FP64 E8 gate failure");
}
int main(int argc,char**argv){try{
 unsigned seed=argc>1?std::stoul(argv[1]):20260929;cudaStream_t stream;CUDA_CHECK(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking));{
 DeviceBuffer full(65536*8),packed(256*4),decoded(65536ULL*32*8);
 read_file(".chatgpt/production_serve_20260929/bank_owned_e8_dataflow/full_reference_i8.bin",full);
 read_file(".chatgpt/production_serve_20260929/bank_owned_e8_dataflow/packed_nibbles.bin",packed);CUDA_CHECK(cudaDeviceSynchronize());
 check_exact_e8_decoder<<<4096,512,0,stream>>>((const unsigned*)packed.p,(uint2*)decoded.p);CUDA_CHECK(cudaGetLastError());CUDA_CHECK(cudaStreamSynchronize(stream));
 const auto reference_table=read<signed char>(full),decoded_table=read<signed char>(decoded);
 for(std::size_t code=0;code<65536;++code)for(int lane=0;lane<32;++lane)if(std::memcmp(reference_table.data()+code*8,decoded_table.data()+(code*32+lane)*8,8))throw std::runtime_error("bank-owned E8decoder differs from source mathematical codebook");
 printf("EXACT_E8_CODEC code_lane_pairs=2097152 coefficients=16777216\n");
 auto weight=bench::make_row_split_weight(QType::Q4G64_F16S,34816,5120,5120);bench::t16_fill_weight(weight.weight,34816,5120,false,seed);CUDA_CHECK(cudaDeviceSynchronize());
 DeviceBuffer scratch(ops::detail::q4_linear_swiglu_sm75_transform_scratch_bytes());ops::detail::q4_linear_swiglu_sm75_transform_layout(weight.weight,QuantLayout::RowSplitSm75Fragment,scratch.p,scratch.bytes,stream);CUDA_CHECK(cudaStreamSynchronize(stream));
 const std::size_t entries=std::size_t(kM)*kK/16;
 DeviceBuffer codes(entries*8),input(kK*4*2),aq(kK*4),as(kGroups*4*2),rq(kM*4),rs(kOutputGroups*4*2),cq(rq.bytes),cs(rs.bytes),raw(kM*4*4),flush(256ULL<<20);
 fill_indices<<<(entries+255)/256,256,0,stream>>>((uint2*)codes.p,entries,seed);CUDA_CHECK(cudaGetLastError());CUDA_CHECK(cudaStreamSynchronize(stream));
 const auto host_codes=read<uint2>(codes);std::vector<std::uint16_t> host_scales(34816*kGroups);CUDA_CHECK(cudaMemcpy(host_scales.data(),weight.weight.scales,host_scales.size()*2,cudaMemcpyDeviceToHost));
 WorkspaceArena ws(ops::detail::q4_linear_swiglu_sm75_a8_workspace_bytes(4));Tensor x(input.p,DType::BF16,{kK,4}),rqt(rq.p,DType::I8,{kM,4}),rst(rs.p,DType::FP16,{kOutputGroups,4});
 auto native=[&](cudaStream_t st){ops::detail::q4_linear_swiglu_sm75_a8_to_a8_launch(x,weight.weight,rqt,rst,ws,st);};
 auto vector=[&](cudaStream_t st,bool bank,bool observed){
  quantize_a8<4><<<40,256,0,st>>>((const __nv_bfloat16*)input.p,(signed char*)aq.p,(__half*)as.p);
  if(bank)bank_owned_e8_gate<<<136,512,0,st>>>((const uint2*)codes.p,(const std::uint8_t*)weight.weight.scales,(const signed char*)aq.p,(const __half*)as.p,(const unsigned*)packed.p,(signed char*)cq.p,(__half*)cs.p,observed?(float*)raw.p:nullptr);
  else lattice_gate<<<136,512,4*40*64,st>>>((const uint2*)codes.p,(const std::uint8_t*)weight.weight.scales,(const signed char*)aq.p,(const __half*)as.p,(const uint2*)full.p,(signed char*)cq.p,(__half*)cs.p,observed?(float*)raw.p:nullptr);
  CUDA_CHECK(cudaGetLastError());
 };
 for(int pattern=0;pattern<3;++pattern){
  if(pattern==2)CUDA_CHECK(cudaMemset(input.p,0,input.bytes));else bench::t16_fill_input(input.p,4*kK,seed+pattern*389);
  CUDA_CHECK(cudaDeviceSynchronize());vector(stream,false,true);CUDA_CHECK(cudaStreamSynchronize(stream));
  const auto before_q=read<signed char>(cq),before_s=read<signed char>(cs);const auto before_raw=read<unsigned>(raw);
  vector(stream,true,true);CUDA_CHECK(cudaStreamSynchronize(stream));
  if(before_q!=read<signed char>(cq)||before_s!=read<signed char>(cs)||before_raw!=read<unsigned>(raw))throw std::runtime_error("bank-owned E8 completeoutput differs from originalE8");
  for(auto scale:read<std::uint16_t>(cs))if(!std::isfinite(__half2float(__ushort_as_half(scale))))throw std::runtime_error("represented E8 output scale overflow");
  validate_math(host_codes,reference_table,host_scales,aq,as,raw);
  printf("EXACT_OLD_E8_OPERATION pattern=%d raw=69632 output_codes=69632 output_scales=1088\n",pattern);
 }
 bench::t16_fill_input(input.p,4*kK,seed);CUDA_CHECK(cudaDeviceSynchronize());vector(stream,true,false);CUDA_CHECK(cudaStreamSynchronize(stream));
 const auto expected_q=read<signed char>(cq),expected_s=read<signed char>(cs);
 cudaGraph_t graph[3]{};cudaGraphExec_t exe[3]{};
 for(int mode=0;mode<3;++mode){CUDA_CHECK(cudaStreamBeginCapture(stream,cudaStreamCaptureModeGlobal));if(mode==0)native(stream);else vector(stream,mode==2,false);CUDA_CHECK(cudaStreamEndCapture(stream,&graph[mode]));CUDA_CHECK(cudaGraphInstantiate(&exe[mode],graph[mode],nullptr,nullptr,0));}
 cudaFuncAttributes a{};CUDA_CHECK(cudaFuncGetAttributes(&a,bank_owned_e8_gate));printf("RESOURCES registers=%d shared=%zu code_bytes=%zu logical_base_bytes=1024\n",a.numRegs,a.sharedSizeBytes,codes.bytes);
 for(int pass=0;pass<2;++pass){double cost[3]{};
  for(int order=0;order<3;++order){int mode=pass?2-order:order;auto f=[&](cudaStream_t st){CUDA_CHECK(cudaGraphLaunch(exe[mode],st));};cost[mode]=bench::measure_cold_launch(f,flush,stream,4,24).median_us;}
  printf("BANK_OWNED_E8_COMPLETE_GATE seed=%u pass=%d native_us=%.3f global_table_us=%.3f bank_owned_us=%.3f native_over_bank=%.6f\n",seed,pass,cost[0],cost[1],cost[2],cost[0]/cost[2]);fflush(stdout);
 }
 if(expected_q!=read<signed char>(cq)||expected_s!=read<signed char>(cs))throw std::runtime_error("E8graphrepeat changedoutput");
 printf("E8_GRAPH_REPEAT_EXACT\n");for(int mode=0;mode<3;++mode){CUDA_CHECK(cudaGraphExecDestroy(exe[mode]));CUDA_CHECK(cudaGraphDestroy(graph[mode]));}
 }CUDA_CHECK(cudaStreamDestroy(stream));return 0;
}catch(const std::exception&e){std::fprintf(stderr,"bank-owned E8: %s\n",e.what());return 1;}}
