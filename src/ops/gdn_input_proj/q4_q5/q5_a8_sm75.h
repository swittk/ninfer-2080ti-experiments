#pragma once
#include "core/arena.h"
#include "core/tensor.h"
#include <cuda_runtime.h>
#include <cstddef>
namespace ninfer::ops::detail {
std::size_t gdn_q5_sm75_a8_workspace_bytes(std::int32_t tokens);
void gdn_q5_sm75_a8_launch(const Tensor& x,const Weight& qk,const Weight& value_z,
                           Tensor& qkv,Tensor& z,WorkspaceArena& ws,cudaStream_t stream);
std::size_t gdn_q5_sm75_transform_scratch_bytes() noexcept;
void gdn_q5_sm75_transform_layout(Weight& w,QuantLayout target,void* scratch,cudaStream_t stream);
}
