#pragma once

#include "ninfer/types.h"
#include "runtime/contract/transient_region.h"

#include <cstddef>
#include <memory>

namespace ninfer {

struct DeviceContext;

namespace runtime {

// Owns one startup-frozen allocation shared by disjoint per-lane slices. Live
// slices never move. No request-time device allocation or replacement is allowed.
class RequestMemory {
public:
    static constexpr std::size_t kDeviceAllocationAlignment = 256;

    RequestMemory(DeviceContext& device, std::size_t frozen_capacity_bytes);
    ~RequestMemory();

    RequestMemory(const RequestMemory&)            = delete;
    RequestMemory& operator=(const RequestMemory&) = delete;
    RequestMemory(RequestMemory&&)                 = delete;
    RequestMemory& operator=(RequestMemory&&)      = delete;

    [[nodiscard]] bool can_activate(std::size_t lane, std::size_t bytes, std::size_t alignment) const;
    void activate(std::size_t lane, std::size_t bytes, std::size_t alignment);
    void deactivate(std::size_t lane) noexcept;
    [[nodiscard]] TransientRegion region(std::size_t lane) const noexcept;
    void activate(std::size_t bytes, std::size_t alignment) { activate(0, bytes, alignment); }
    void deactivate() noexcept;

    [[nodiscard]] TransientRegion region() const noexcept { return region(0); }
    [[nodiscard]] ArenaMemorySummary summary() const noexcept;
    void reset_peak() noexcept;

private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace runtime
} // namespace ninfer
