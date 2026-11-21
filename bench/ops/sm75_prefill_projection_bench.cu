#include "core/arena.h"
#include "core/device.h"
#include "ninfer/ops/gdn_input_proj.h"
#include "ninfer/ops/attn_input_proj.h"
#include "ninfer/ops/linear_add.h"
#include "ops/gdn_input_proj/q4_q5/q5_a8_sm75.h"
#include "targets/qwen3_6_27b/impl/sm75_prefill_dense.h"
#include "ninfer_bench_common.h"
#include "quantized_weight.cuh"
#include "t16_fixture.h"
#include <cmath>
#include <cstring>
#include <cuda_fp16.h>
#include "../../tests/ops/op_check.h"
#include <cstdio>
#include <vector>
using namespace ninfer;
using ninfer::targets::qwen3_6_27b::detail::Sm75PrefillDense;
float value(std::uint16_t x){union{std::uint32_t u;float f;} v{static_cast<std::uint32_t>(x)<<16};return v.f;}
struct Diff {double rel;std::size_t mismatches,nonfinite;};
Diff compare(const DeviceBuffer& a,const DeviceBuffer& b){
 std::vector<std::uint16_t>x(a.bytes/2),y(b.bytes/2);a.copy_to_host(x.data(),a.bytes);b.copy_to_host(y.data(),b.bytes);
 double err=0,norm=0;std::size_t count=0,bad=0;
 for(std::size_t i=0;i<x.size();++i){double av=value(x[i]),bv=value(y[i]);err+=(bv-av)*(bv-av);norm+=av*av;count+=x[i]!=y[i];bad+=!std::isfinite(bv);}
 return{std::sqrt(err/std::max(norm,1e-30)),count,bad};
}
bool independent_output_oracle(const bench::PackedQuantizedWeight& weight, const DeviceBuffer& x,
                               const DeviceBuffer& initial, const DeviceBuffer& original,
                               const DeviceBuffer& candidate, int t) {
    // Use the existing complete LinearAdd A16 criterion, not a candidate-specific tolerance.
    constexpr test::ReductionCriterion criterion{1.0/256.0,1.0/256.0,2.0/256.0};
    std::vector<std::uint8_t> weights(weight.storage.bytes);
    weight.storage.copy_to_host(weights.data(),weights.size());
    std::vector<std::uint16_t> hx(x.bytes/2),hi(initial.bytes/2),hr(original.bytes/2),hc(candidate.bytes/2);
    x.copy_to_host(hx.data(),x.bytes);initial.copy_to_host(hi.data(),initial.bytes);
    original.copy_to_host(hr.data(),original.bytes);candidate.copy_to_host(hc.data(),candidate.bytes);
    std::vector<double> expected,old_actual,new_actual;
    for(const int token:{0,1,t/4,t/2,3*t/4,t-2,t-1}) {
        for(const int row:{0,1,1280,2560,3840,5118,5119}) {
            double sum=0;
            for(int k=0;k<6144;++k) {
                const std::size_t group=static_cast<std::size_t>(row)*96+k/64;
                const int j=k%64;
                const auto packed=weights[group*32+j/2];
                const auto high=weights[weight.high_offset+group*8+j/8];
                const int q=(((packed>>((j&1)*4))&15)|(((high>>(j&7))&1)<<4));
                std::uint16_t bits;std::memcpy(&bits,weights.data()+weight.scale_offset+group*2,2);
                const double scale=__half2float(__ushort_as_half(bits));
                sum+=static_cast<double>((q^16)-16)*scale*value(hx[static_cast<std::size_t>(token)*6144+k]);
            }
            const auto index=static_cast<std::size_t>(token)*5120+row;
            expected.push_back(sum+value(hi[index]));old_actual.push_back(value(hr[index]));new_actual.push_back(value(hc[index]));
        }
    }
    const auto old_stats=test::compute_reduction_stats(old_actual.data(),expected.data(),expected.size());
    const auto new_stats=test::compute_reduction_stats(new_actual.data(),expected.data(),expected.size());
    const bool old_pass=test::reduction_passes(old_stats,expected.size(),criterion);
    const bool new_pass=test::reduction_passes(new_stats,expected.size(),criterion);
    std::printf("FP64_ORACLE samples=%zu baseline_rel_l2=%.8g candidate_rel_l2=%.8g candidate_max_abs=%.8g gross_limit=%.8g baseline_pass=%d candidate_pass=%d\n",expected.size(),old_stats.relative_l2,new_stats.relative_l2,new_stats.maximum_absolute_error,test::gross_error_limit(new_stats,criterion),old_pass,new_pass);std::fflush(stdout);
    return new_pass;
}
int main(int argc,char** argv){try{
 const int t=argc>1?std::stoi(argv[1]):128;unsigned seed=argc>2?std::stoul(argv[2]):1234567u;
 cudaStream_t stream{};CUDA_CHECK(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking));
 {
  DeviceBuffer flush(256ULL<<20);WorkspaceArena work(Sm75PrefillDense::workspace_bytes(t));Sm75PrefillDense candidate(work,stream);
  {
    auto qk=bench::make_row_split_weight(QType::Q4G64_F16S,7168,5120,5120);
    auto gv=bench::make_row_split_weight(QType::Q5G64_F16S,7168,5120,5120);
    bench::t16_fill_weight(qk.weight,7168,5120,false,seed);
    bench::t16_fill_weight(gv.weight,7168,5120,true,seed^654321u);
    DeviceBuffer input(static_cast<std::size_t>(5120)*t*2);
    bench::t16_fill_input(input.p,input.bytes/2,seed^98765u);
    DeviceBuffer rq(static_cast<std::size_t>(6144)*t*2),rg(rq.bytes),cq(rq.bytes),cg(rq.bytes);
    DeviceBuffer rk(static_cast<std::size_t>(1024)*t*2),rv(rk.bytes),ck(rk.bytes),cv(rk.bytes);
    Tensor x(input.p,DType::BF16,{5120,t}),q(rq.p,DType::BF16,{6144,t}),g(rg.p,DType::BF16,{6144,t});
    Tensor k(rk.p,DType::BF16,{1024,t}),v(rv.p,DType::BF16,{1024,t});
    Tensor nq(cq.p,DType::BF16,{6144,t}),ng(cg.p,DType::BF16,{6144,t});
    Tensor nk(ck.p,DType::BF16,{1024,t}),nv(cv.p,DType::BF16,{1024,t});
    auto reference=[&](cudaStream_t s){ops::attn_input_proj(x,qk.weight,gv.weight,q,g,k,v,s);};
    auto run=[&](cudaStream_t s){if(!candidate.attention_input(x,qk.weight,gv.weight,nq,ng,nk,nv,s))throw std::runtime_error("attention refused");};
    reference(stream);run(stream);CUDA_CHECK(cudaStreamSynchronize(stream));
    const auto dq=compare(rq,cq),dg=compare(rg,cg),dk=compare(rk,ck),dv=compare(rv,cv);
    const auto rt=bench::measure_cold_launch(reference,flush,stream,3,12);
    const auto ct=bench::measure_cold_launch(run,flush,stream,3,12);
    std::printf("ATTENTION t=%d seed=%u reference_us=%.3f dense_us=%.3f speedup=%.5f rel_l2_q=%.8g gate=%.8g k=%.8g v=%.8g nonfinite=%zu\n",t,seed,rt.median_us,ct.median_us,rt.median_us/ct.median_us,dq.rel,dg.rel,dk.rel,dv.rel,dq.nonfinite+dg.nonfinite+dk.nonfinite+dv.nonfinite);
    std::fflush(stdout);
    if(dq.rel>.003||dg.rel>.003||dk.rel>.003||dv.rel>.003||dq.nonfinite||dg.nonfinite||dk.nonfinite||dv.nonfinite)return 2;
    work.reset_peak();
  }
  auto qk=bench::make_row_split_weight(QType::Q4G64_F16S,4096,5120,5120);
  auto vz=bench::make_row_split_weight(QType::Q5G64_F16S,12288,5120,5120);
  bench::t16_fill_weight(qk.weight,4096,5120,false,seed);bench::t16_fill_weight(vz.weight,12288,5120,true,seed^654321u);
  DeviceBuffer input(static_cast<std::size_t>(5120)*t*2);bench::t16_fill_input(input.p,input.bytes/2,seed^8765u);
  DeviceBuffer rq(static_cast<std::size_t>(10240)*t*2),rz(static_cast<std::size_t>(6144)*t*2),cq(rq.bytes),cz(rz.bytes),canonicalq(rq.bytes),canonicalz(rz.bytes);
  Tensor x(input.p,DType::BF16,{5120,t}),rqt(rq.p,DType::BF16,{10240,t}),rzt(rz.p,DType::BF16,{6144,t}),cqt(cq.p,DType::BF16,{10240,t}),czt(cz.p,DType::BF16,{6144,t});
  auto reference=[&](cudaStream_t s){ops::gdn_input_proj(x,qk.weight,vz.weight,rqt,rzt,s);};
  auto run=[&](cudaStream_t s){if(!candidate.gdn_input(x,qk.weight,vz.weight,cqt,czt,s))throw std::runtime_error("GDN refused");};
  reference(stream);run(stream);CUDA_CHECK(cudaStreamSynchronize(stream));
  Diff dq=compare(rq,cq),dz=compare(rz,cz);
  const auto rt=bench::measure_cold_launch(reference,flush,stream,3,12);
  CUDA_CHECK(cudaMemcpyAsync(canonicalq.p,cq.p,cq.bytes,cudaMemcpyDeviceToDevice,stream));CUDA_CHECK(cudaMemcpyAsync(canonicalz.p,cz.p,cz.bytes,cudaMemcpyDeviceToDevice,stream));
  DeviceBuffer scratch(ops::detail::gdn_q5_sm75_transform_scratch_bytes());
  ops::detail::gdn_q5_sm75_transform_layout(vz.weight,QuantLayout::RowSplitSm75Fragment,scratch.p,stream);
  run(stream);CUDA_CHECK(cudaStreamSynchronize(stream));
  Diff fq=compare(canonicalq,cq),fz=compare(canonicalz,cz);
  const auto ct=bench::measure_cold_launch(run,flush,stream,3,12);
  std::printf("GDN t=%d seed=%u reference_us=%.3f fragment_blas_us=%.3f speedup=%.5f qkv_rel_l2=%.8g z_rel_l2=%.8g fragment_mismatches=%zu nonfinite=%zu peak=%zu\n",t,seed,rt.median_us,ct.median_us,rt.median_us/ct.median_us,dq.rel,dz.rel,fq.mismatches+fz.mismatches,dq.nonfinite+dz.nonfinite,work.peak_used());std::fflush(stdout);
  if(dq.rel>.003||dz.rel>.003||fq.mismatches||fz.mismatches||dq.nonfinite||dz.nonfinite)return 2;
  auto output=bench::make_row_split_weight(QType::Q5G64_F16S,5120,6144,6144);bench::t16_fill_weight(output.weight,5120,6144,true,seed^4321u);
  DeviceBuffer oi(static_cast<std::size_t>(6144)*t*2),initial(static_cast<std::size_t>(5120)*t*2),rr(initial.bytes),cc(initial.bytes);
  bench::t16_fill_input(oi.p,oi.bytes/2,seed^100u);bench::t16_fill_input(initial.p,initial.bytes/2,seed^200u);
  Tensor ox(oi.p,DType::BF16,{6144,t}),orr(rr.p,DType::BF16,{5120,t}),occ(cc.p,DType::BF16,{5120,t});
  auto ref_out=[&](cudaStream_t s){CUDA_CHECK(cudaMemcpyAsync(rr.p,initial.p,rr.bytes,cudaMemcpyDeviceToDevice,s));ops::linear_add(ox,output.weight,orr,work,s);};
  auto cand_out=[&](cudaStream_t s){CUDA_CHECK(cudaMemcpyAsync(cc.p,initial.p,cc.bytes,cudaMemcpyDeviceToDevice,s));if(!candidate.residual(ox,output.weight,occ,s))throw std::runtime_error("output refused");};
  ref_out(stream);cand_out(stream);CUDA_CHECK(cudaStreamSynchronize(stream));Diff od=compare(rr,cc);
  auto rtime=bench::measure_cold_launch(ref_out,flush,stream,3,12);auto ctime=bench::measure_cold_launch(cand_out,flush,stream,3,12);
  std::printf("OUTPUT t=%d seed=%u reference_us=%.3f blas_us=%.3f speedup=%.5f rel_l2=%.8g nonfinite=%zu\n",t,seed,rtime.median_us,ctime.median_us,rtime.median_us/ctime.median_us,od.rel,od.nonfinite);std::fflush(stdout);
  if(od.rel>.003||od.nonfinite||!independent_output_oracle(output,oi,initial,rr,cc,t))return 2;
 }
 CUDA_CHECK(cudaStreamDestroy(stream));return 0;
}catch(const std::exception& e){std::fprintf(stderr,"prefill_projection_bench: %s\n",e.what());return 1;}}
