#pragma once
#include "core/arena.h"
#include "core/tensor.h"
#include <cuda_runtime.h>
#include <cstddef>
namespace ninfer::ops::detail {
std::size_t q5_linear_add_sm75_a8_workspace_bytes(std::int32_t tokens);
std::size_t q5_linear_add_sm75_transform_scratch_bytes();
void q5_linear_add_sm75_a8_prequant_launch(const Tensor& aq,const Tensor& as,const Weight& w,
                                            Tensor& residual,cudaStream_t stream);
void q5_linear_add_sm75_a8_launch(const Tensor& x, const Weight& w, Tensor& residual,
                                  WorkspaceArena& ws, cudaStream_t stream);
void q5_linear_add_sm75_a8_partition4_t8_launch(const Tensor& x, const Weight& w, Tensor& residual,
                                                 WorkspaceArena& ws, cudaStream_t stream);
void q5_linear_add_sm75_a8_native_row8_launch(const Tensor& x, const Weight& w, Tensor& residual,
                                                 WorkspaceArena& ws, cudaStream_t stream);
void q5_linear_add_sm75_transform_layout(Weight& w, QuantLayout target, void* scratch,
                                         std::size_t scratch_bytes, cudaStream_t stream);
} // namespace ninfer::ops::detail
