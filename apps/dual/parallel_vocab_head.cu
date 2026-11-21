#include "dual/parallel_vocab_head.h"
#include "core/arena.h"
#include "core/device.h"
#include "ops/linear/w8/w8_small_t_mma.cuh"
#include "ops/common/argmax_reduce.cuh"
#include <cuda_bf16.h>
#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <utility>

namespace ninfer::apps::dual {
namespace {
constexpr int Rows = 248320, Hidden = 5120, Columns = 4, HalfRows = Rows / 2;
constexpr std::size_t HiddenBytes = Columns * Hidden * sizeof(__nv_bfloat16);
constexpr std::size_t HalfLogitBytes = Columns * HalfRows * sizeof(__nv_bfloat16);
constexpr std::size_t HalfCodeBytes = std::size_t(HalfRows) * Hidden;
constexpr std::size_t HalfScaleBytes = std::size_t(HalfRows) * (Hidden / 32) * sizeof(__half);
using Geometry = ops::detail::W8VocabularyProjectionGeometry;
using Schedule = typename ops::detail::W8LinearSmallTProductionSchedule<Geometry,Columns>::Type;

class RestoreDevice {
    int original_ = -1;
public:
    RestoreDevice() { CUDA_CHECK(cudaGetDevice(&original_)); }
    ~RestoreDevice() { if (original_ >= 0) (void)cudaSetDevice(original_); }
};
void validate_weight(const Weight& w) {
    if (w.qtype != QType::W8G32_F16S || w.layout != QuantLayout::RowSplit ||
        w.n != Rows || w.k != Hidden || w.qdata == nullptr || w.scales == nullptr)
        throw std::invalid_argument("parallel vocabulary head requires canonical W8G32 248320x5120");
}
__global__ void copy_upper_logits(const __nv_bfloat16* partial, __nv_bfloat16* logits) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < Columns * HalfRows) logits[(i / HalfRows) * Rows + HalfRows + i % HalfRows] = partial[i];
}
void project_half(const Tensor& x, const void* codes, const void* scales,
                  void* logits, int stride, cudaStream_t stream) {
    ops::detail::W8ContiguousOutput output{static_cast<__nv_bfloat16*>(logits),stride};
    ops::detail::w8_small_t_mma_kernel<Geometry,Columns,Schedule>
        <<<HalfRows/16,Schedule::kThreads,0,stream>>>(
        static_cast<const __nv_bfloat16*>(x.data),static_cast<const std::uint8_t*>(codes),
        static_cast<const std::uint8_t*>(scales),output);
    CUDA_CHECK(cudaGetLastError());
}
constexpr int WinnerTiles = (HalfRows+511)/512;
__global__ void reduce_head_tiles(const __nv_bfloat16* logits,int start,int valid_rows,uint2* tiles) {
    const int local=blockIdx.x*512+threadIdx.x,token=blockIdx.y,global=start+local;
    float value=-CUDART_INF_F;int index=INT32_MAX;
    if(local<HalfRows && global<valid_rows) {
        const float v=__bfloat162float(logits[token*HalfRows+local]);
        if(!isnan(v)) {value=v;index=global;}
    }
    ops::argmax_block_reduce(value,index);
    if(threadIdx.x==0)tiles[token*WinnerTiles+blockIdx.x]=make_uint2(__float_as_uint(value),unsigned(index));
}
__global__ void reduce_head_winners(const __nv_bfloat16* logits,int start,const uint2* tiles,uint2* winners) {
    const int token=blockIdx.x,i=threadIdx.x;
    float value=-CUDART_INF_F;int index=INT32_MAX;
    if(i<WinnerTiles){const auto v=tiles[token*WinnerTiles+i];value=__uint_as_float(v.x);index=int(v.y);}
    ops::argmax_block_reduce(value,index);
    if(i==0){
        const float row_zero=__bfloat162float(logits[token*HalfRows]);
        if(start==0 && isnan(row_zero)){value=row_zero;index=0;}
        winners[token]=make_uint2(__float_as_uint(value),unsigned(index));
    }
}
__global__ void merge_head_winners(const uint2* lower,const uint2* upper,std::int32_t* tokens) {
    const int t=threadIdx.x;if(t>=Columns)return;
    const auto a=lower[t],b=upper[t];
    tokens[t]=ops::argmax_better(__uint_as_float(b.x),int(b.y),__uint_as_float(a.x),int(a.y))?int(b.y):int(a.y);
}
void reduce_head(const void* logits,int start,int valid_rows,DeviceBuffer& tiles,DeviceBuffer& winners,cudaStream_t stream) {
    reduce_head_tiles<<<dim3(WinnerTiles,Columns),512,0,stream>>>(static_cast<const __nv_bfloat16*>(logits),start,valid_rows,static_cast<uint2*>(tiles.p));
    reduce_head_winners<<<Columns,256,0,stream>>>(static_cast<const __nv_bfloat16*>(logits),start,static_cast<const uint2*>(tiles.p),static_cast<uint2*>(winners.p));
    CUDA_CHECK(cudaGetLastError());
}
// Each allocation and CUDA object has one explicit device owner. Destruction
// joins submitted work before releasing source copies or pinned transfer storage.
struct Resources {
    int owner = -1, helper = -1;
    cudaStream_t owner_stream{}, transfer_stream{}, helper_stream{};
    cudaEvent_t input_ready{}, input_sent{}, helper_done{};
    cudaGraph_t graph{};
    cudaGraphExec_t executable{};
    const void* graph_hidden = nullptr;
    void* graph_logits = nullptr;
    bool graph_greedy = false;
    int graph_valid_rows = 0;
    DeviceBuffer helper_codes, helper_scales, helper_hidden, helper_logits, owner_reply;
    DeviceBuffer owner_tiles, helper_tiles, owner_winner, helper_winner, remote_winner;
    std::unique_ptr<PinnedHostBuffer> input_packet, output_packet;
    Weight source{};
    std::uint64_t calls = 0;

    void drain() {
        if (owner >= 0) {
            CUDA_CHECK(cudaSetDevice(owner));
            if (owner_stream) CUDA_CHECK(cudaStreamSynchronize(owner_stream));
            if (transfer_stream) CUDA_CHECK(cudaStreamSynchronize(transfer_stream));
        }
        if (helper >= 0) {
            CUDA_CHECK(cudaSetDevice(helper));
            if (helper_stream) CUDA_CHECK(cudaStreamSynchronize(helper_stream));
        }
    }
    void clear_graph() {
        drain();
        CUDA_CHECK(cudaSetDevice(owner));
        if (executable) CUDA_CHECK(cudaGraphExecDestroy(std::exchange(executable,nullptr)));
        if (graph) CUDA_CHECK(cudaGraphDestroy(std::exchange(graph,nullptr)));
        graph_hidden = nullptr; graph_logits = nullptr;
    }
    ~Resources() {
        int previous = -1; (void)cudaGetDevice(&previous);
        if (owner >= 0) {
            (void)cudaSetDevice(owner);
            if (owner_stream) (void)cudaStreamSynchronize(owner_stream);
            if (transfer_stream) (void)cudaStreamSynchronize(transfer_stream);
        }
        if (helper >= 0) {
            (void)cudaSetDevice(helper);
            if (helper_stream) (void)cudaStreamSynchronize(helper_stream);
        }
        if (owner >= 0) {
            (void)cudaSetDevice(owner);
            if (executable) (void)cudaGraphExecDestroy(executable);
            if (graph) (void)cudaGraphDestroy(graph);
            if (input_ready) (void)cudaEventDestroy(input_ready);
            if (input_sent) (void)cudaEventDestroy(input_sent);
            if (transfer_stream) (void)cudaStreamDestroy(transfer_stream);
            owner_reply = DeviceBuffer{};
            owner_tiles = DeviceBuffer{}; owner_winner = DeviceBuffer{}; remote_winner = DeviceBuffer{};
        }
        if (helper >= 0) {
            (void)cudaSetDevice(helper);
            if (helper_done) (void)cudaEventDestroy(helper_done);
            if (helper_stream) (void)cudaStreamDestroy(helper_stream);
            helper_codes = DeviceBuffer{}; helper_scales = DeviceBuffer{};
            helper_hidden = DeviceBuffer{}; helper_logits = DeviceBuffer{};
            helper_tiles = DeviceBuffer{}; helper_winner = DeviceBuffer{};
        }
        input_packet.reset(); output_packet.reset();
        if (previous >= 0) (void)cudaSetDevice(previous);
    }
    void enqueue_greedy(const Tensor& hidden,Tensor& tokens,int valid_rows) {
        CUDA_CHECK(cudaSetDevice(owner));
        CUDA_CHECK(cudaEventRecord(input_ready,owner_stream));
        CUDA_CHECK(cudaStreamWaitEvent(transfer_stream,input_ready,0));
        CUDA_CHECK(cudaMemcpyAsync(input_packet->data(),hidden.data,HiddenBytes,cudaMemcpyDeviceToHost,transfer_stream));
        CUDA_CHECK(cudaEventRecord(input_sent,transfer_stream));
        CUDA_CHECK(cudaSetDevice(helper));
        CUDA_CHECK(cudaStreamWaitEvent(helper_stream,input_sent,0));
        CUDA_CHECK(cudaMemcpyAsync(helper_hidden.p,input_packet->data(),HiddenBytes,cudaMemcpyHostToDevice,helper_stream));
        Tensor input(helper_hidden.p,DType::BF16,{Hidden,Columns});
        project_half(input,helper_codes.p,helper_scales.p,helper_logits.p,HalfRows,helper_stream);
        reduce_head(helper_logits.p,HalfRows,valid_rows,helper_tiles,helper_winner,helper_stream);
        CUDA_CHECK(cudaMemcpyAsync(output_packet->data(),helper_winner.p,Columns*sizeof(uint2),cudaMemcpyDeviceToHost,helper_stream));
        CUDA_CHECK(cudaEventRecord(helper_done,helper_stream));
        CUDA_CHECK(cudaSetDevice(owner));
        // Reuse the full-logit path's reply workspace for this local half. Its
        // graph is drained before switching contracts or changing addresses.
        project_half(hidden,source.qdata,source.scales,owner_reply.p,HalfRows,owner_stream);
        reduce_head(owner_reply.p,0,valid_rows,owner_tiles,owner_winner,owner_stream);
        CUDA_CHECK(cudaStreamWaitEvent(owner_stream,helper_done,0));
        CUDA_CHECK(cudaMemcpyAsync(remote_winner.p,output_packet->data(),Columns*sizeof(uint2),cudaMemcpyHostToDevice,owner_stream));
        merge_head_winners<<<1,32,0,owner_stream>>>(static_cast<const uint2*>(owner_winner.p),static_cast<const uint2*>(remote_winner.p),static_cast<std::int32_t*>(tokens.data));
        CUDA_CHECK(cudaGetLastError());
    }
    void enqueue(const Tensor& hidden, Tensor& logits) {
        CUDA_CHECK(cudaSetDevice(owner));
        CUDA_CHECK(cudaEventRecord(input_ready,owner_stream));
        CUDA_CHECK(cudaStreamWaitEvent(transfer_stream,input_ready,0));
        CUDA_CHECK(cudaMemcpyAsync(input_packet->data(),hidden.data,HiddenBytes,
                                  cudaMemcpyDeviceToHost,transfer_stream));
        CUDA_CHECK(cudaEventRecord(input_sent,transfer_stream));
        CUDA_CHECK(cudaSetDevice(helper));
        CUDA_CHECK(cudaStreamWaitEvent(helper_stream,input_sent,0));
        CUDA_CHECK(cudaMemcpyAsync(helper_hidden.p,input_packet->data(),HiddenBytes,
                                  cudaMemcpyHostToDevice,helper_stream));
        Tensor input(helper_hidden.p,DType::BF16,{Hidden,Columns});
        project_half(input,helper_codes.p,helper_scales.p,helper_logits.p,HalfRows,helper_stream);
        CUDA_CHECK(cudaMemcpyAsync(output_packet->data(),helper_logits.p,HalfLogitBytes,
                                  cudaMemcpyDeviceToHost,helper_stream));
        CUDA_CHECK(cudaEventRecord(helper_done,helper_stream));
        CUDA_CHECK(cudaSetDevice(owner));
        project_half(hidden,source.qdata,source.scales,logits.data,Rows,owner_stream);
        CUDA_CHECK(cudaStreamWaitEvent(owner_stream,helper_done,0));
        CUDA_CHECK(cudaMemcpyAsync(owner_reply.p,output_packet->data(),HalfLogitBytes,
                                  cudaMemcpyHostToDevice,owner_stream));
        copy_upper_logits<<<(Columns*HalfRows+255)/256,256,0,owner_stream>>>(
            static_cast<const __nv_bfloat16*>(owner_reply.p),static_cast<__nv_bfloat16*>(logits.data));
        CUDA_CHECK(cudaGetLastError());
    }
};
}
class ParallelVocabHead::Impl {
public:
    explicit Impl(int device) : helper(device) {}
    int helper;
    std::unique_ptr<Resources> resources;
};
ParallelVocabHead::ParallelVocabHead(int helper_device) : impl_(std::make_unique<Impl>(helper_device)) {}
ParallelVocabHead::~ParallelVocabHead() = default;

void ParallelVocabHead::prepare(const Weight& canonical, cudaStream_t owner_stream) {
    RestoreDevice restore;
    validate_weight(canonical);
    int owner = -1, devices = 0;
    CUDA_CHECK(cudaGetDevice(&owner)); CUDA_CHECK(cudaGetDeviceCount(&devices));
    if (owner_stream == nullptr || impl_->helper < 0 || impl_->helper >= devices || owner == impl_->helper)
        throw std::invalid_argument("parallel vocabulary head requires two devices and an owner stream");
    for (int d : {owner,impl_->helper}) {
        cudaDeviceProp properties{}; CUDA_CHECK(cudaGetDeviceProperties(&properties,d));
        if (properties.major != 7 || properties.minor != 5)
            throw std::invalid_argument("parallel vocabulary head currently requires two SM75 devices");
    }
    CUDA_CHECK(cudaStreamSynchronize(owner_stream));
    // Destroy the previous representation first. A source mutation at the same
    // addresses followed by prepare cannot accidentally reuse stale half weights.
    impl_->resources.reset();
    auto state = std::make_unique<Resources>();
    state->owner = owner; state->helper = impl_->helper; state->owner_stream = owner_stream;
    state->source = canonical;
    CUDA_CHECK(cudaSetDevice(owner));
    CUDA_CHECK(cudaStreamCreateWithFlags(&state->transfer_stream,cudaStreamNonBlocking));
    CUDA_CHECK(cudaEventCreateWithFlags(&state->input_ready,cudaEventDisableTiming));
    CUDA_CHECK(cudaEventCreateWithFlags(&state->input_sent,cudaEventDisableTiming));
    state->owner_reply = DeviceBuffer(HalfLogitBytes);
    state->owner_tiles=DeviceBuffer(Columns*WinnerTiles*sizeof(uint2));
    state->owner_winner=DeviceBuffer(Columns*sizeof(uint2));
    state->remote_winner=DeviceBuffer(Columns*sizeof(uint2));
    state->input_packet = std::make_unique<PinnedHostBuffer>(HiddenBytes);
    state->output_packet = std::make_unique<PinnedHostBuffer>(HalfLogitBytes);
    CUDA_CHECK(cudaSetDevice(state->helper));
    CUDA_CHECK(cudaStreamCreateWithFlags(&state->helper_stream,cudaStreamNonBlocking));
    CUDA_CHECK(cudaEventCreateWithFlags(&state->helper_done,cudaEventDisableTiming));
    state->helper_codes = DeviceBuffer(HalfCodeBytes);
    state->helper_scales = DeviceBuffer(HalfScaleBytes);
    state->helper_hidden = DeviceBuffer(HiddenBytes);
    state->helper_logits = DeviceBuffer(HalfLogitBytes);
    state->helper_tiles=DeviceBuffer(Columns*WinnerTiles*sizeof(uint2));
    state->helper_winner=DeviceBuffer(Columns*sizeof(uint2));
    PinnedHostBuffer staging(16ULL << 20);
    const auto copy = [&](const void* from, void* to, std::size_t bytes) {
        for (std::size_t offset = 0; offset < bytes; offset += staging.size()) {
            const auto count = std::min(staging.size(),bytes-offset);
            CUDA_CHECK(cudaSetDevice(owner));
            CUDA_CHECK(cudaMemcpyAsync(staging.data(),static_cast<const char*>(from)+bytes+offset,
                                      count,cudaMemcpyDeviceToHost,owner_stream));
            CUDA_CHECK(cudaStreamSynchronize(owner_stream));
            CUDA_CHECK(cudaSetDevice(state->helper));
            CUDA_CHECK(cudaMemcpyAsync(static_cast<char*>(to)+offset,staging.data(),count,
                                      cudaMemcpyHostToDevice,state->helper_stream));
            CUDA_CHECK(cudaStreamSynchronize(state->helper_stream));
        }
    };
    copy(canonical.qdata,state->helper_codes.p,HalfCodeBytes);
    copy(canonical.scales,state->helper_scales.p,HalfScaleBytes);
    impl_->resources = std::move(state);
}
bool ParallelVocabHead::execute(const Tensor& hidden, const Weight& canonical,
                                Tensor& logits, cudaStream_t owner_stream) {
    if (hidden.ne[1] != Columns) return false;
    validate_weight(canonical);
    if (hidden.data == nullptr || hidden.dtype != DType::BF16 || hidden.ne[0] != Hidden ||
        hidden.ne[2] != 1 || hidden.ne[3] != 1 || !hidden.is_contiguous() ||
        logits.data == nullptr || logits.dtype != DType::BF16 || logits.ne[0] != Rows ||
        logits.ne[1] != Columns || logits.ne[2] != 1 || logits.ne[3] != 1 || !logits.is_contiguous())
        throw std::invalid_argument("parallel vocabulary T4 tensor contract violated");
    RestoreDevice restore;
    auto* state = impl_->resources.get();
    int device = -1; CUDA_CHECK(cudaGetDevice(&device));
    if (!state || state->owner != device || state->owner_stream != owner_stream ||
        state->source.qdata != canonical.qdata || state->source.scales != canonical.scales)
        throw std::logic_error("parallel vocabulary head requires explicit preparation for this source/owner");
    cudaStreamCaptureStatus status{}; CUDA_CHECK(cudaStreamIsCapturing(owner_stream,&status));
    if (status != cudaStreamCaptureStatusNone) return false;
    if (!state->executable || state->graph_greedy || state->graph_hidden != hidden.data || state->graph_logits != logits.data) {
        state->clear_graph();
        CUDA_CHECK(cudaStreamBeginCapture(owner_stream,cudaStreamCaptureModeGlobal));
        try {
            state->enqueue(hidden,logits);
            CUDA_CHECK(cudaStreamEndCapture(owner_stream,&state->graph));
        } catch (...) {
            (void)cudaSetDevice(state->owner);
            cudaGraph_t invalid{}; (void)cudaStreamEndCapture(owner_stream,&invalid);
            if (invalid) (void)cudaGraphDestroy(invalid);
            throw;
        }
        CUDA_CHECK(cudaGraphInstantiate(&state->executable,state->graph,nullptr,nullptr,0));
        state->graph_hidden = hidden.data; state->graph_logits = logits.data;
        state->graph_greedy=false;
    }
    CUDA_CHECK(cudaGraphLaunch(state->executable,owner_stream));
    ++state->calls;
    return true;
}
bool ParallelVocabHead::execute_greedy(const Tensor& hidden,const Weight& canonical,
                                       Tensor& tokens,std::int32_t valid_rows,cudaStream_t owner_stream) {
    if(hidden.ne[1]!=Columns)return false;
    validate_weight(canonical);
    if(valid_rows<=0 || valid_rows>Rows || !hidden.data || hidden.dtype!=DType::BF16 ||
       hidden.ne[0]!=Hidden || hidden.ne[2]!=1 || hidden.ne[3]!=1 || !hidden.is_contiguous() ||
       !tokens.data || tokens.dtype!=DType::I32 || tokens.ne[0]!=Columns || tokens.ne[1]!=1 ||
       tokens.ne[2]!=1 || tokens.ne[3]!=1 || !tokens.is_contiguous())
        throw std::invalid_argument("parallel greedy head tensor/domain contract violated");
    RestoreDevice restore;
    auto* state=impl_->resources.get();int device=-1;CUDA_CHECK(cudaGetDevice(&device));
    if(!state || state->owner!=device || state->owner_stream!=owner_stream ||
       state->source.qdata!=canonical.qdata || state->source.scales!=canonical.scales)
        throw std::logic_error("parallel greedy head requires prepared source/owner");
    cudaStreamCaptureStatus status{};CUDA_CHECK(cudaStreamIsCapturing(owner_stream,&status));
    if(status!=cudaStreamCaptureStatusNone)return false;
    if(!state->executable || !state->graph_greedy || state->graph_hidden!=hidden.data ||
       state->graph_logits!=tokens.data || state->graph_valid_rows!=valid_rows){
        state->clear_graph();
        CUDA_CHECK(cudaStreamBeginCapture(owner_stream,cudaStreamCaptureModeGlobal));
        try{state->enqueue_greedy(hidden,tokens,valid_rows);CUDA_CHECK(cudaStreamEndCapture(owner_stream,&state->graph));}
        catch(...){(void)cudaSetDevice(state->owner);cudaGraph_t invalid{};(void)cudaStreamEndCapture(owner_stream,&invalid);if(invalid)(void)cudaGraphDestroy(invalid);throw;}
        CUDA_CHECK(cudaGraphInstantiate(&state->executable,state->graph,nullptr,nullptr,0));
        state->graph_hidden=hidden.data;state->graph_logits=tokens.data;
        state->graph_greedy=true;state->graph_valid_rows=valid_rows;
    }
    CUDA_CHECK(cudaGraphLaunch(state->executable,owner_stream));++state->calls;return true;
}
ParallelVocabHeadStats ParallelVocabHead::stats() const noexcept {
    if (!impl_->resources) return {};
    return {impl_->resources->calls,HalfCodeBytes+HalfScaleBytes,HalfLogitBytes+Columns*(WinnerTiles+2)*sizeof(uint2),
            HiddenBytes+HalfLogitBytes+Columns*(WinnerTiles+1)*sizeof(uint2),HiddenBytes+HalfLogitBytes};
}
}
