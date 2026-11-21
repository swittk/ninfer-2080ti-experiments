#pragma once
#include "core/tensor.h"
#include "core/arena.h"
#include <cuda_runtime.h>
#include <cstddef>
namespace ninfer::ops::detail {
// Resident mode borrows the original model planes after an exact permutation.
// Call release before transform back; never transform a graph-owned live source.
std::size_t sm75_radix_resident_bytes(int rows, int columns, QType type, int max_tokens = 4);
void sm75_radix_resident_transform(Weight& source, QuantLayout target, cudaStream_t stream);
void sm75_radix_resident_prepare(Weight& source, cudaStream_t stream, int max_tokens = 4);
std::size_t sm75_radix_canonical_workspace_bytes(int rows, int columns, QType type);
Weight sm75_radix_canonical_copy(const Weight& source, WorkspaceArena& workspace, cudaStream_t stream);

// The default retains the prior conservative Q5 reservation for existing callers.
std::size_t sm75_radix_projection_bytes(int n,int k,QType type=QType::Q5G64_F16S,int max_tokens=4);
// Prepared storage is fixed at construction and charged for the largest admitted width.
// Supported capacities are 4, 8 and 16; smaller ragged widths use the caller's fallback.
void sm75_radix_projection_prepare(const Weight&,cudaStream_t,int max_tokens=4);
void sm75_radix_projection_release(const Weight&) noexcept;
bool sm75_radix_projection_launch(const Tensor&,const Weight&,Tensor&,bool add_residual,cudaStream_t);
}
