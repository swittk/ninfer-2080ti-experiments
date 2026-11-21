#pragma once

#include "runtime/contract/types.h"

#include <cstddef>

namespace ninfer::runtime {

[[nodiscard]] KvCapacityResolution resolve_kv_capacity(const KvCapacityPolicy& policy,
                                                       const SequenceCapacityCurve& curve,
                                                       std::size_t available_runtime_bytes);

void validate_startup_memory(const KvCapacityResolution& resolution,
                             std::size_t available_after_startup_bytes);

} // namespace ninfer::runtime
