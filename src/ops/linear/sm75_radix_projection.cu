#include "ops/linear/sm75_radix_projection.h"
#include "ops/linear/sm75_radix_projection.cuh"
#include "ops/linear/sm75_wide_radix_projection.cuh"
#include "ops/linear/sm75_resident_radix.cuh"
#include <array>
#include <memory>
#include <mutex>
#include <unordered_map>
#include <stdexcept>
#include <vector>
namespace ninfer::ops::detail {
namespace {
struct ResidentScratch {
 int device=-1;cudaStream_t stream{};int max_tokens=4;void* block=nullptr;
 signed char* activation=nullptr;float* scales=nullptr;float* partial=nullptr;
 ~ResidentScratch() noexcept {
  int previous=-1;cudaGetDevice(&previous);
  if(device>=0&&previous!=device)cudaSetDevice(device);
  if(stream)cudaStreamSynchronize(stream);
  if(block)cudaFree(block);
  if(previous>=0&&previous!=device)cudaSetDevice(previous);
 }
};
struct Prepared {
 Weight source;cudaStream_t stream{};int device=-1;int max_tokens=4;void* block=nullptr;
 std::shared_ptr<ResidentScratch> resident;
 uint4* packed=nullptr;unsigned* high=nullptr;__half* scales=nullptr;
 signed char* activation=nullptr;float* activation_scales=nullptr;float* partial=nullptr;
 ~Prepared() noexcept {
  int old=-1;cudaGetDevice(&old);if(device>=0&&old!=device)cudaSetDevice(device);
  if(stream)cudaStreamSynchronize(stream);if(block)cudaFree(block);
  if(old>=0&&old!=device)cudaSetDevice(old);
 }
};
std::mutex& mutex(){static std::mutex m;return m;}
auto& entries(){static auto* values=new std::unordered_map<const void*,std::unique_ptr<Prepared>>;return *values;}
bool identity(const Weight&a,const Weight&b){return a.qtype==b.qtype&&a.layout==b.layout&&a.qdata==b.qdata&&a.qhigh==b.qhigh&&a.scales==b.scales&&a.n==b.n&&a.k==b.k;}
std::size_t align(std::size_t x){return (x+255)&~std::size_t(255);}
std::array<std::size_t,6> sizes(int n,int k,bool high_plane,int max_tokens){
 if(max_tokens!=4&&max_tokens!=8&&max_tokens!=16)throw std::invalid_argument("radix capacity must be four, eight or sixteen columns");
 if(n<=0||k<=0||n%64||k%64||n>34816||k>17408)throw std::invalid_argument("radix projection shape outside qualified bounds");
 return {align(std::size_t(n)*k/2),high_plane?align(std::size_t(n)*k/8):0,align(std::size_t(n)*(k/64)*2),align(std::size_t(k)*2*max_tokens),align(std::size_t(k/64)*max_tokens*4),align(std::size_t(n)*2*max_tokens*4*4)};
}
}
std::size_t sm75_radix_projection_bytes(int n,int k,QType type,int max_tokens) {
 if(type!=QType::Q4G64_F16S&&type!=QType::Q5G64_F16S)throw std::invalid_argument("radix reservation requires Q4 or Q5");
 std::size_t total=0;for(auto v:sizes(n,k,type==QType::Q5G64_F16S,max_tokens))total+=v;
 // Request and charge whole 2-MiB allocation units for this SM75 profile.
 // Otherwise many per-weight allocations consume more backing than payload sums.
 constexpr std::size_t unit=2ULL<<20;
 return (total+unit-1)/unit*unit;
}
void sm75_radix_projection_prepare(const Weight&w,cudaStream_t stream,int max_tokens){
 if((w.qtype!=QType::Q4G64_F16S&&w.qtype!=QType::Q5G64_F16S)||w.layout!=QuantLayout::RowSplit||w.scale_dtype!=DType::FP16||w.group_size!=64||w.padded_shape[1]!=w.k||!w.qdata||!w.scales||(w.qtype==QType::Q5G64_F16S&&!w.qhigh))throw std::invalid_argument("radix projection requires canonical immutable Q4/Q5 G64 source");
 auto amounts=sizes(w.n,w.k,w.qtype==QType::Q5G64_F16S,max_tokens);int device;CUDA_CHECK(cudaGetDevice(&device));
 cudaDeviceProp props{};CUDA_CHECK(cudaGetDeviceProperties(&props,device));if(props.major!=7||props.minor!=5)throw std::invalid_argument("radix projection requires SM75");
 std::lock_guard lock(mutex());auto it=entries().find(w.qdata);
 if(it!=entries().end()){if(!identity(it->second->source,w)||it->second->device!=device||it->second->stream!=stream||max_tokens>it->second->max_tokens)throw std::logic_error("radix source owner changed before release");return;}
 auto p=std::make_unique<Prepared>();p->source=w;p->device=device;p->stream=stream;p->max_tokens=max_tokens;
 CUDA_CHECK(cudaMalloc(&p->block,sm75_radix_projection_bytes(w.n,w.k,w.qtype,max_tokens)));
 auto* at=static_cast<char*>(p->block);p->packed=reinterpret_cast<uint4*>(at);at+=amounts[0];p->high=amounts[1]?reinterpret_cast<unsigned*>(at):nullptr;at+=amounts[1];p->scales=reinterpret_cast<__half*>(at);at+=amounts[2];p->activation=reinterpret_cast<signed char*>(at);at+=amounts[3];p->activation_scales=reinterpret_cast<float*>(at);at+=amounts[4];p->partial=reinterpret_cast<float*>(at);
 auto blocks=(w.n/16*(w.k/64)*32+255)/256;
 if(w.qtype==QType::Q5G64_F16S)radix_projection::repack<true><<<blocks,256,0,stream>>>((const unsigned char*)w.qdata,(const unsigned char*)w.qhigh,(const __half*)w.scales,p->packed,p->high,p->scales,w.n,w.k);
 else radix_projection::repack<false><<<blocks,256,0,stream>>>((const unsigned char*)w.qdata,nullptr,(const __half*)w.scales,p->packed,p->high,p->scales,w.n,w.k);
 CUDA_CHECK(cudaGetLastError());CUDA_CHECK(cudaStreamSynchronize(stream));entries().emplace(w.qdata,std::move(p));
}
void sm75_radix_projection_release(const Weight&w) noexcept {std::lock_guard lock(mutex());auto it=entries().find(w.qdata);if(it!=entries().end()&&identity(it->second->source,w))entries().erase(it);}
namespace {
// Layers sharing a stream execute serially. Their temporary projection workspaces
// can share addresses, while different engine streams retain separate ownership.
std::shared_ptr<ResidentScratch> acquire_resident_scratch(int device,cudaStream_t stream,int max_tokens) {
    static auto* owners=new std::vector<std::weak_ptr<ResidentScratch>>;
    for(auto it=owners->begin();it!=owners->end();) {
        auto owner=it->lock();
        if(!owner){it=owners->erase(it);continue;}
        if(owner->device==device && owner->stream==stream) {
            if(max_tokens>owner->max_tokens)
                throw std::logic_error("resident radix scratch capacity cannot grow while owners are live");
            return owner;
        }
        ++it;
    }
    auto owner=std::make_shared<ResidentScratch>();
    owner->device=device;owner->stream=stream;owner->max_tokens=max_tokens;
    const auto maximum=sizes(5120,6144,true,max_tokens);
    CUDA_CHECK(cudaMalloc(&owner->block,sm75_radix_resident_bytes(5120,6144,QType::Q5G64_F16S,max_tokens)));
    auto* at=static_cast<char*>(owner->block);
    owner->activation=reinterpret_cast<signed char*>(at);at+=maximum[3];
    owner->scales=reinterpret_cast<float*>(at);at+=maximum[4];
    owner->partial=reinterpret_cast<float*>(at);
    owners->push_back(owner);return owner;
}
void validate_resident_source(const Weight& w) {
    if ((w.qtype != QType::Q4G64_F16S && w.qtype != QType::Q5G64_F16S) ||
        (w.layout != QuantLayout::RowSplit && w.layout != QuantLayout::RowSplitSm75Radix) ||
        w.scale_dtype != DType::FP16 || w.group_size != 64 || w.padded_shape[1] != w.k ||
        w.n <= 0 || w.n > 5120 || w.n % 64 || w.k <= 0 || w.k % 64 || w.k > 6144 ||
        !w.qdata || !w.scales || (w.qtype == QType::Q5G64_F16S && !w.qhigh)) {
        throw std::invalid_argument("resident radix requires exact Q4/Q5 G64 planes and K <= 6144");
    }
}
void transform_resident_planes(Weight& w, QuantLayout target, cudaStream_t stream) {
    cudaStreamCaptureStatus capture;
    CUDA_CHECK(cudaStreamIsCapturing(stream,&capture));
    if(capture!=cudaStreamCaptureStatusNone)
        throw std::logic_error("resident source transforms are preparation-only, not graph operations");
    if (w.layout == target) return;
    const int groups=w.k/64;
    auto* low=const_cast<std::uint16_t*>(static_cast<const std::uint16_t*>(w.qdata));
    auto* high=const_cast<std::uint8_t*>(static_cast<const std::uint8_t*>(w.qhigh));
    auto* scales=const_cast<std::uint16_t*>(static_cast<const std::uint16_t*>(w.scales));
    const auto launch=[&]<bool Packed>() {
        resident_radix::permute_low<Packed><<<w.n/16,256,16*groups*32,stream>>>(low,groups);
        if(w.qtype==QType::Q5G64_F16S)
            resident_radix::permute_high<Packed><<<w.n/16,256,16*groups*8,stream>>>(high,groups);
        resident_radix::permute_scales<Packed><<<w.n/16,256,16*groups*2,stream>>>(scales,groups);
    };
    if(target==QuantLayout::RowSplitSm75Radix)launch.template operator()<true>();
    else launch.template operator()<false>();
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(stream));
    w.layout=target;
}
}
std::size_t sm75_radix_resident_bytes(int n,int k,QType type,int max_tokens) {
    if(type!=QType::Q4G64_F16S&&type!=QType::Q5G64_F16S)
        throw std::invalid_argument("resident radix requires Q4/Q5");
    const auto v=sizes(n,k,type==QType::Q5G64_F16S,max_tokens);
    constexpr std::size_t unit=2ULL<<20;
    return (v[3]+v[4]+v[5]+unit-1)/unit*unit;
}
void sm75_radix_resident_transform(Weight& w,QuantLayout target,cudaStream_t stream) {
    validate_resident_source(w);
    if(target!=QuantLayout::RowSplit && target!=QuantLayout::RowSplitSm75Radix)
        throw std::invalid_argument("resident radix transform target invalid");
    std::lock_guard lock(mutex());
    if(entries().contains(w.qdata))
        throw std::logic_error("release prepared radix owner before transforming its source");
    transform_resident_planes(w,target,stream);
}
void sm75_radix_resident_prepare(Weight& w,cudaStream_t stream,int max_tokens) {
    validate_resident_source(w);
    cudaStreamCaptureStatus capture;
    CUDA_CHECK(cudaStreamIsCapturing(stream,&capture));
    if(capture!=cudaStreamCaptureStatusNone)
        throw std::logic_error("resident radix preparation is forbidden during capture");
    const auto v=sizes(w.n,w.k,w.qtype==QType::Q5G64_F16S,max_tokens);
    int device;CUDA_CHECK(cudaGetDevice(&device));
    cudaDeviceProp props{};CUDA_CHECK(cudaGetDeviceProperties(&props,device));
    if(props.major!=7 || props.minor!=5)throw std::invalid_argument("resident radix requires SM75");
    std::lock_guard lock(mutex());
    const auto it=entries().find(w.qdata);
    if(it!=entries().end()) {
        if(w.layout!=QuantLayout::RowSplitSm75Radix || !identity(it->second->source,w) ||
           it->second->device!=device || it->second->stream!=stream || max_tokens>it->second->max_tokens)
            throw std::logic_error("resident radix owner changed before release");
        return;
    }
    auto p=std::make_unique<Prepared>();p->device=device;p->stream=stream;p->max_tokens=max_tokens;
    p->resident=acquire_resident_scratch(device,stream,max_tokens);
    p->activation=p->resident->activation;
    p->activation_scales=p->resident->scales;
    p->partial=p->resident->partial;
    transform_resident_planes(w,QuantLayout::RowSplitSm75Radix,stream);
    p->source=w;
    p->packed=const_cast<uint4*>(static_cast<const uint4*>(w.qdata));
    p->high=w.qtype==QType::Q5G64_F16S?const_cast<unsigned*>(static_cast<const unsigned*>(w.qhigh)):nullptr;
    p->scales=const_cast<__half*>(static_cast<const __half*>(w.scales));
    entries().emplace(w.qdata,std::move(p));
}
std::size_t sm75_radix_canonical_workspace_bytes(int n,int k,QType type) {
    if(n<=0||n%64||k<=0||k%64||k>6144||n>34816 ||
       (type!=QType::Q4G64_F16S && type!=QType::Q5G64_F16S))
        throw std::invalid_argument("canonical radix view outside supported geometry");
    const auto elements=std::size_t(n)*k;
    return elements/2+(type==QType::Q5G64_F16S?elements/8:0)+elements/64*2;
}
Weight sm75_radix_canonical_copy(const Weight& w,WorkspaceArena& workspace,cudaStream_t stream) {
    if(w.layout!=QuantLayout::RowSplitSm75Radix)return w;
    validate_resident_source(w);
    const auto bytes=sm75_radix_canonical_workspace_bytes(w.n,w.k,w.qtype);
    const auto elements=std::size_t(w.n)*w.k,high_bytes=w.qtype==QType::Q5G64_F16S?elements/8:0;
    auto storage=workspace.alloc_bytes(bytes);auto* p=static_cast<std::uint8_t*>(storage.data);
    Weight out=w;out.payload=p;out.payload_bytes=bytes;out.high_plane_bytes=high_bytes;
    out.qdata=p;out.qhigh=high_bytes?p+elements/2:nullptr;out.scales=p+elements/2+high_bytes;
    out.layout=QuantLayout::RowSplit;
    resident_radix::canonical_copy<<<(elements/4+255)/256,256,0,stream>>>(
        static_cast<const std::uint16_t*>(w.qdata),static_cast<const std::uint32_t*>(w.qhigh),
        static_cast<const std::uint16_t*>(w.scales),reinterpret_cast<std::uint16_t*>(p),
        high_bytes?p+elements/2:nullptr,reinterpret_cast<std::uint16_t*>(p+elements/2+high_bytes),w.n,w.k/64);
    CUDA_CHECK(cudaGetLastError());
    return out;
}
bool sm75_radix_projection_launch(const Tensor&x,const Weight&w,Tensor&y,bool residual,cudaStream_t stream){
 const bool ragged=x.ne[1]!=4&&x.ne[1]!=8&&x.ne[1]!=16;
 if((ragged&&w.layout!=QuantLayout::RowSplitSm75Radix)||x.ne[1]<1||x.ne[1]>16||
    x.ne[2]!=1||x.dtype!=DType::BF16||x.ne[0]!=w.k||!x.is_contiguous())return false;
 Prepared*p;
 {std::lock_guard lock(mutex());auto it=entries().find(w.qdata);if(it==entries().end())return false;p=it->second.get();if(!identity(p->source,w)||p->stream!=stream)throw std::logic_error("radix projection source identity mismatch");}
 if(x.ne[1]>p->max_tokens)return false;
 if(y.dtype!=DType::BF16||y.ne[0]!=w.n||y.ne[1]!=x.ne[1]||y.nb[0]!=2||y.nb[1]%2!=0||!y.data)throw std::invalid_argument("radix projection output geometry rejected");
 const int n=w.n,k=w.k,ld=y.nb[1]/2;
 if(ragged) {
  const auto run=[&]<int Tokens,bool IsQ5,bool Residual>() {
   resident_radix::quantize_active<Tokens><<<(Tokens*(k/64)*32+255)/256,256,0,stream>>>(
       (const __nv_bfloat16*)x.data,p->activation,p->activation_scales,k,x.ne[1]);
   if constexpr(Tokens==4)radix_projection::matmul<4,IsQ5><<<dim3(n/64,4),128,0,stream>>>(
       p->packed,p->high,p->scales,p->activation,p->activation_scales,p->partial,n,k);
   else wide_radix::matmul<Tokens/4,4,IsQ5><<<dim3(n/64,4),128,0,stream>>>(
       p->packed,p->high,p->scales,p->activation,p->activation_scales,p->partial,n,k);
   resident_radix::finish_active<Tokens/4,Residual><<<(n*x.ne[1]+255)/256,256,0,stream>>>(
       p->partial,Residual?(const __nv_bfloat16*)y.data:nullptr,(__nv_bfloat16*)y.data,n,ld,x.ne[1]);
  };
  const auto dispatch=[&]<int Tokens>() {
   if(w.qtype==QType::Q5G64_F16S) {
    if(residual)run.template operator()<Tokens,true,true>();
    else run.template operator()<Tokens,true,false>();
   } else {
    if(residual)run.template operator()<Tokens,false,true>();
    else run.template operator()<Tokens,false,false>();
   }
  };
  if(x.ne[1]<4)dispatch.template operator()<4>();
  else if(x.ne[1]<8)dispatch.template operator()<8>();
  else dispatch.template operator()<16>();
  CUDA_CHECK(cudaGetLastError());return true;
 }

 if(x.ne[1]!=4) {
  const auto run=[&]<int Tokens,bool IsQ5,bool Residual>() {
   wide_radix::quantize<Tokens><<<(Tokens*(k/64)*32+255)/256,256,0,stream>>>(
       (const __nv_bfloat16*)x.data,p->activation,p->activation_scales,k);
   wide_radix::matmul<Tokens/4,4,IsQ5><<<dim3(n/64,4),128,0,stream>>>(
       p->packed,p->high,p->scales,p->activation,p->activation_scales,p->partial,n,k);
   wide_radix::finish<Tokens/4,4,Residual><<<(n*Tokens+255)/256,256,0,stream>>>(
       p->partial,Residual?(const __nv_bfloat16*)y.data:nullptr,(__nv_bfloat16*)y.data,n,ld);
  };
  const auto dispatch=[&]<int Tokens>() {
   if(w.qtype==QType::Q5G64_F16S) {
    if(residual)run.template operator()<Tokens,true,true>();
    else run.template operator()<Tokens,true,false>();
   } else {
    if(residual)run.template operator()<Tokens,false,true>();
    else run.template operator()<Tokens,false,false>();
   }
  };
  if(x.ne[1]==8)dispatch.template operator()<8>();
  else dispatch.template operator()<16>();
  CUDA_CHECK(cudaGetLastError());return true;
 }

 radix_projection::quantize<<<(4*(k/64)*32+255)/256,256,0,stream>>>((const __nv_bfloat16*)x.data,p->activation,p->activation_scales,k);
 if(w.qtype==QType::Q5G64_F16S)radix_projection::matmul<4,true><<<dim3(n/64,4),128,0,stream>>>(p->packed,p->high,p->scales,p->activation,p->activation_scales,p->partial,n,k);
 else radix_projection::matmul<4,false><<<dim3(n/64,4),128,0,stream>>>(p->packed,nullptr,p->scales,p->activation,p->activation_scales,p->partial,n,k);
 if(residual)radix_projection::finish<4,true><<<(n*4+255)/256,256,0,stream>>>(p->partial,(const __nv_bfloat16*)y.data,(__nv_bfloat16*)y.data,n,ld);
 else radix_projection::finish<4,false><<<(n*4+255)/256,256,0,stream>>>(p->partial,nullptr,(__nv_bfloat16*)y.data,n,ld);
 CUDA_CHECK(cudaGetLastError());return true;
}
}
