#include "core/arena.h"
#include "core/device.h"
#include "ninfer/ops/linear_swiglu.h"
#include "ninfer/ops/linear_add.h"
#include "ops/linear_swiglu/q4/q4_linear_swiglu_a8_sm75.h"
#include "ops/linear_add/q5/q5_linear_add_a8_sm75.h"
#include "targets/qwen3_6_27b/impl/sm75_prefill_dense.h"
#include "ninfer_bench_common.h"
#include "quantized_weight.cuh"
#include "t16_fixture.h"
#include <cuda_fp16.h>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>
using namespace ninfer;
using ninfer::targets::qwen3_6_27b::detail::Sm75PrefillDense;
__global__ void scale_weights(__half* values,std::size_t count) {
    const auto i=static_cast<std::size_t>(blockIdx.x)*blockDim.x+threadIdx.x;
    if(i<count) values[i]=__float2half_rn(__half2float(values[i])/64.f);
}
float value(std::uint16_t x){union{std::uint32_t u;float f;} v{static_cast<std::uint32_t>(x)<<16};return v.f;}
int main(int argc,char** argv) {
 try {
    const int width=argc>1?std::stoi(argv[1]):128;const int t=width*4;const unsigned seed=argc>2?std::stoul(argv[2]):1234567u;
    cudaStream_t stream{};CUDA_CHECK(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking));
    {
    DeviceBuffer flush(256ULL<<20);
    auto gate=bench::make_row_split_weight(QType::Q4G64_F16S,34816,5120,5120);
    auto down=bench::make_row_split_weight(QType::Q5G64_F16S,5120,17408,17408);
    bench::t16_fill_weight(gate.weight,34816,5120,false,seed);
    bench::t16_fill_weight(down.weight,5120,17408,true,seed^987654u);
    for(auto* p:{&gate,&down}) scale_weights<<<(p->scale_bytes/2+255)/256,256,0,stream>>>(const_cast<__half*>(static_cast<const __half*>(p->weight.scales)),p->scale_bytes/2);
    CUDA_CHECK(cudaStreamSynchronize(stream));
    DeviceBuffer input(static_cast<std::size_t>(5120)*t*2),initial(input.bytes),reference(input.bytes),candidate(input.bytes);
    bench::t16_fill_input(input.p,input.bytes/2,seed^0x1234u);bench::t16_fill_input(initial.p,initial.bytes/2,seed^0x5678u);
    DeviceBuffer activation(static_cast<std::size_t>(17408)*t*2);
    WorkspaceArena rw(Sm75PrefillDense::workspace_bytes(width)),cw(Sm75PrefillDense::workspace_bytes(t));
    Sm75PrefillDense serial(rw,stream),fast(cw,stream);
    Tensor x(input.p,DType::BF16,{5120,t}),r(reference.p,DType::BF16,{5120,t}),c(candidate.p,DType::BF16,{5120,t}),a(activation.p,DType::BF16,{17408,t});
    DeviceBuffer scratch(std::max(ops::detail::q4_linear_swiglu_sm75_transform_scratch_bytes(),ops::detail::q5_linear_add_sm75_transform_scratch_bytes()));
    ops::detail::q4_linear_swiglu_sm75_transform_layout(gate.weight,QuantLayout::RowSplitSm75Fragment,scratch.p,scratch.bytes,stream);
    ops::detail::q5_linear_add_sm75_transform_layout(down.weight,QuantLayout::RowSplitSm75Fragment,scratch.p,scratch.bytes,stream);
    const auto ref=[&](cudaStream_t s){
      CUDA_CHECK(cudaMemcpyAsync(reference.p,initial.p,initial.bytes,cudaMemcpyDeviceToDevice,s));
      for(int lane=0;lane<4;++lane) {
        Tensor xi=x.slice(1,lane*width,width),ri=r.slice(1,lane*width,width);
        if(!serial.execute(0,xi,gate.weight,down.weight,ri,s))throw std::runtime_error("serial rejected");
      }
    };
    const auto cand=[&](cudaStream_t s){
      CUDA_CHECK(cudaMemcpyAsync(candidate.p,initial.p,initial.bytes,cudaMemcpyDeviceToDevice,s));
      if(!fast.execute(0,x,gate.weight,down.weight,c,s))throw std::runtime_error("packed rejected");
    };
    ref(stream);cand(stream);CUDA_CHECK(cudaStreamSynchronize(stream));
    std::vector<std::uint16_t> hr(reference.bytes/2),hc(candidate.bytes/2);
    reference.copy_to_host(hr.data(),reference.bytes);candidate.copy_to_host(hc.data(),candidate.bytes);
    double err=0,norm=0;std::size_t nonfinite=0,mismatch=0;
    for(std::size_t i=0;i<hr.size();++i){
      double d=static_cast<double>(value(hc[i]))-value(hr[i]);err+=d*d;norm+=static_cast<double>(value(hr[i]))*value(hr[i]);
      nonfinite+=!std::isfinite(value(hc[i]));mismatch+=hc[i]!=hr[i];
    }
    if(nonfinite)throw std::runtime_error("nonfinite packed output");
    for(int pass=0;pass<2;++pass){
      const auto rt=bench::measure_cold_launch(ref,flush,stream,3,12);
      const auto ct=bench::measure_cold_launch(cand,flush,stream,3,12);
      std::printf("PACKED per_request=%d requests=4 seed=%u pass=%d serial_us=%.3f packed_us=%.3f speedup=%.6f rel_l2=%.9g mismatches=%zu serial_ws=%zu packed_ws=%zu\n",width,seed,pass,rt.median_us,ct.median_us,rt.median_us/ct.median_us,std::sqrt(err/std::max(norm,1e-30)),mismatch,rw.capacity(),cw.capacity());
    }
    }
    CUDA_CHECK(cudaStreamDestroy(stream));return 0;
 }catch(const std::exception&e){std::fprintf(stderr,"packed_prefill: %s\n",e.what());return 1;}
}
