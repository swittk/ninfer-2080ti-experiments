#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <optional>
#include <span>
#include <stdexcept>
#include <vector>

namespace ninfer::serve {

// Admission only. Each Engine remains the sole owner of its scheduler and KV.
// Fingerprints are bounded locality hints, NEVER authorization to reuse state;
// the selected Engine independently validates the complete prefix identity.
class ReplicaRouter {
    struct Affinity {
        std::uint64_t key = 0;
        std::size_t replica = 0;
        bool valid = false;
    };
    struct State {
        State(std::size_t count, std::size_t maximum, std::size_t slots)
            : active(count), limit(maximum), active_slots(slots), affinities(64) {}
        std::mutex mutex;
        std::vector<std::size_t> active;
        const std::size_t limit;
        const std::size_t active_slots;
        std::vector<Affinity> affinities;
        std::size_t cursor = 0;
        std::size_t affinity_cursor = 0;
    };
public:
    class Reservation {
    public:
        ~Reservation() {
            std::lock_guard lock(state_->mutex);
            --state_->active[replica_];
            if (reusable_ && key_) {
                auto it = std::find_if(state_->affinities.begin(), state_->affinities.end(),
                    [&](const Affinity& a) { return a.valid && a.key == *key_; });
                if (it == state_->affinities.end()) {
                    it = state_->affinities.begin() + state_->affinity_cursor;
                    state_->affinity_cursor =
                        (state_->affinity_cursor + 1) % state_->affinities.size();
                }
                *it = Affinity{*key_, replica_, true};
            }
        }
        Reservation(const Reservation&) = delete;
        Reservation& operator=(const Reservation&) = delete;
        [[nodiscard]] std::size_t replica() const noexcept { return replica_; }
        void mark_reusable() noexcept { reusable_ = true; }
    private:
        friend class ReplicaRouter;
        Reservation(std::shared_ptr<State> state, std::size_t replica,
                    std::optional<std::uint64_t> key)
            : state_(std::move(state)), replica_(replica), key_(key) {}
        std::shared_ptr<State> state_;
        std::size_t replica_;
        std::optional<std::uint64_t> key_;
        bool reusable_ = false;
    };

    ReplicaRouter(std::size_t replicas, std::size_t maximum_per_replica,
                  std::size_t active_slots_per_replica = 0) {
        if (active_slots_per_replica == 0) { active_slots_per_replica = maximum_per_replica; }
        if (replicas == 0 || maximum_per_replica == 0 ||
            active_slots_per_replica > maximum_per_replica) {
            throw std::invalid_argument("replica admission capacity must be positive");
        }
        state_ = std::make_shared<State>(replicas, maximum_per_replica, active_slots_per_replica);
    }

    [[nodiscard]] std::unique_ptr<Reservation>
    acquire(std::optional<std::uint64_t> affinity = {},
            std::span<const std::uint64_t> prior_turns = {}) const {
        std::lock_guard lock(state_->mutex);
        const std::size_t count = state_->active.size();
        std::size_t selected = state_->cursor;
        for (std::size_t offset = 1; offset < count; ++offset) {
            const std::size_t candidate = (state_->cursor + offset) % count;
            if (state_->active[candidate] < state_->active[selected]) { selected = candidate; }
        }
        if (state_->active[selected] >= state_->limit) { return {}; }
        const auto prefer = [&](std::uint64_t key) {
            for (const Affinity& entry : state_->affinities) {
                // Prefer the longest available cached owner. Never queue behind
                // a full owner while a different replica has free active capacity.
                if (entry.valid && entry.key == key &&
                    state_->active[entry.replica] < state_->limit &&
                    (state_->active[entry.replica] < state_->active_slots ||
                     state_->active[entry.replica] == state_->active[selected])) {
                    selected = entry.replica;
                    return true;
                }
            }
            return false;
        };
        bool found = affinity && prefer(*affinity);
        for (auto it = prior_turns.rbegin(); !found && it != prior_turns.rend(); ++it) {
            if (!affinity || *it != *affinity) { found = prefer(*it); }
        }
        auto lease = std::unique_ptr<Reservation>(new Reservation(state_, selected, affinity));
        ++state_->active[selected];
        state_->cursor = (selected + 1) % count;
        return lease;
    }
private:
    std::shared_ptr<State> state_;
};

} // namespace ninfer::serve
