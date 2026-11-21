#pragma once

#include "core/arena.h"
#include "core/tensor.h"
#include <cuda_runtime.h>
#include <cstddef>

namespace ninfer::ops::detail {
// A non-owning canonical view in caller-owned workspace. The resident fragment
// source is read-only, so already-captured decode graphs keep valid pointers.
std::size_t q5_sm75_canonical_workspace_bytes(std::int32_t rows, std::int32_t columns);
Weight q5_sm75_canonical_copy(const Weight& source, WorkspaceArena& workspace,
                               cudaStream_t stream);
} // namespace ninfer::ops::detail
