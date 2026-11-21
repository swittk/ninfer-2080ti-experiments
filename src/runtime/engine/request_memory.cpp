#include "runtime/engine/request_memory.h"
#include "runtime/engine/transient_slots.h"

#include "core/arena.h"
#include "core/device.h"

#include <algorithm>
#include <cstddef>
#include <memory>
#include <stdexcept>

namespace ninfer::runtime {
class RequestMemory::Impl {
public:
    Impl(DeviceContext& context, std::size_t capacity) : device(context.device), slots(capacity) {
        if (capacity != 0) {
            CUDA_CHECK(cudaSetDevice(device));
            arena = std::make_unique<DeviceArena>(capacity);
        }
    }

    ~Impl() {
        if (arena != nullptr) {
            (void)cudaSetDevice(device);
            arena.reset();
        }
    }

    int device = 0;
    std::unique_ptr<DeviceArena> arena;
    TransientSlots<kMaximumConcurrency> slots;
};

RequestMemory::RequestMemory(DeviceContext& device, std::size_t frozen_capacity_bytes)
    : impl_(std::make_unique<Impl>(device, frozen_capacity_bytes)) {}

RequestMemory::~RequestMemory() = default;

bool RequestMemory::can_activate(std::size_t lane, std::size_t bytes, std::size_t alignment) const {
    return impl_->slots.find(lane, bytes, alignment).has_value();
}

void RequestMemory::activate(std::size_t lane, std::size_t bytes, std::size_t alignment) {
    (void)impl_->slots.activate(lane, bytes, alignment);
}

void RequestMemory::deactivate(std::size_t lane) noexcept { impl_->slots.release(lane); }
void RequestMemory::deactivate() noexcept { impl_->slots.clear(); }

TransientRegion RequestMemory::region(std::size_t lane) const noexcept {
    const auto slot = impl_->slots.slice(lane);
    if (!slot.active || slot.bytes == 0) { return {}; }
    return {static_cast<std::byte*>(impl_->arena->base()) + slot.offset,
            slot.bytes, slot.alignment};
}

ArenaMemorySummary RequestMemory::summary() const noexcept {
    return ArenaMemorySummary{
        impl_->arena != nullptr ? impl_->arena->capacity() : 0,
        impl_->slots.used(),
        impl_->slots.peak(),
    };
}

void RequestMemory::reset_peak() noexcept { impl_->slots.reset_peak(); }

} // namespace ninfer::runtime
