#pragma once
#include "core/tensor.h"
#include "core/arena.h"
#include <cuda_runtime.h>
namespace ninfer::ops::detail {
// Explicit numerical target-MLP representation. Preparation acts on the loaded
// source only, before derived caches. Packed storage reuses original allocations.
void nu3_mlp_prepare(Weight&,cudaStream_t);
// Offline factory-source precision test. Sources must already be authenticated
// by the caller; conversion writes only the loaded canonical device weights.
void nu3_mlp_prepare_bf16(Weight&,const char*,const char*,cudaStream_t);
void nu3_mlp_transform(Weight&,QuantLayout,void*,std::size_t,cudaStream_t);
void nu3_mlp_launch(const Tensor&,const Weight&,const Weight&,Tensor&,WorkspaceArena&,cudaStream_t);
}
