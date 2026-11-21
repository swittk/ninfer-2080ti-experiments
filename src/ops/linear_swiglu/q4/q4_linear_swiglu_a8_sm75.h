#pragma once
#include "core/arena.h"
#include "core/tensor.h"
#include <cuda_runtime.h>
#include <cstddef>
namespace ninfer::ops::detail {
std::size_t q4_linear_swiglu_sm75_a8_workspace_bytes(std::int32_t tokens);
void q4_linear_swiglu_sm75_a8_to_a8_launch(const Tensor& x,const Weight& w,Tensor& out_q,
                                           Tensor& out_scales,WorkspaceArena& ws,cudaStream_t stream);
void q4_linear_swiglu_sm75_rmsnorm_a8_to_a8_launch(
    const Tensor& residual,const Tensor& norm_weight,float eps,const Weight& w,Tensor& out_q,
    Tensor& out_scales,WorkspaceArena& ws,cudaStream_t stream);
void q4_linear_swiglu_sm75_a8_launch(const Tensor& x, const Weight& w, Tensor& out,
                                     WorkspaceArena& ws, cudaStream_t stream);
std::size_t q4_linear_swiglu_sm75_transform_scratch_bytes() noexcept;
void q4_linear_swiglu_sm75_transform_layout(Weight& w, QuantLayout target, void* scratch,
                                             std::size_t scratch_bytes,
                                             cudaStream_t stream);
} // namespace ninfer::ops::detail
