#include "serve/replica_router.h"
#include "serve/replica_affinity.h"
#include <array>
#include <atomic>
#include <iostream>
#include <thread>
#include <vector>

using ninfer::serve::ReplicaRouter;
int main() {
    int failures = 0;
    auto check = [&](bool ok, const char* reason) {
        if (!ok) { std::cerr << reason << '\n'; ++failures; }
    };
    ReplicaRouter router(2, 4);
    std::vector<std::unique_ptr<ReplicaRouter::Reservation>> live;
    for (std::size_t i = 0; i < 8; ++i) {
        live.push_back(router.acquire(i));
        check(live.back() && live.back()->replica() == i % 2,
              "cold admission did not balance four requests on each replica");
    }
    check(!router.acquire(), "overload admitted a ninth active reservation");
    live[3].reset();
    auto refill = router.acquire();
    check(refill && refill->replica() == 1, "retired slot was not immediately reusable");
    refill.reset(); live.clear();

    auto cached = router.acquire(42);
    const auto owner = cached->replica();
    cached->mark_reusable(); cached.reset();
    auto hit = router.acquire(42);
    check(hit->replica() == owner, "idle prefix affinity lost its owning replica");
    auto second_hit = router.acquire(42);
    check(second_hit->replica() == owner, "warm admission discarded a free cached slot");
    auto third_hit = router.acquire(42);
    auto fourth_hit = router.acquire(42);
    check(third_hit->replica() == owner && fourth_hit->replica() == owner,
          "warm burst moved away from available cached slots");
    auto spill = router.acquire(42);
    check(spill->replica() != owner, "affinity queued behind a full replica instead of spilling");
    hit.reset(); second_hit.reset(); third_hit.reset(); fourth_hit.reset(); spill.reset();

    {
        ReplicaRouter growing(2, 4);
        auto seed = growing.acquire(100);
        const auto first_owner = seed->replica();
        seed->mark_reusable(); seed.reset();
        const std::array<std::uint64_t, 3> boundaries{100, 200, 300};
        auto append = growing.acquire(300, boundaries);
        check(append->replica() == first_owner, "conversation append lost original owner");
        append->mark_reusable(); append.reset();
        auto blocker = growing.acquire(300); // Force a newer prefix onto the other GPU.
        auto second_owner = growing.acquire(400);
        check(second_owner->replica() != first_owner, "test did not establish distinct owners");
        second_owner->mark_reusable(); second_owner.reset(); blocker.reset();
        const std::array<std::uint64_t, 4> longer{100, 300, 400, 500};
        auto longest = growing.acquire(500, longer);
        check(longest->replica() != first_owner, "short ancestor overrode longer cached prefix");
        const auto preferred = longest->replica();
        std::vector<std::unique_ptr<ReplicaRouter::Reservation>> fill;
        for (int i = 0; i < 3; ++i) { fill.push_back(growing.acquire(400)); }
        auto fallback = growing.acquire(600, longer);
        check(fallback && fallback->replica() != preferred, "prefix history disabled full-owner spill");
    }
    {
        using namespace ninfer::serve;
        GenerationRequest request;
        ChatTurn user; user.role = "user";
        ContentPart part; part.kind = ContentKind::Text; part.text = "Original";
        user.content.push_back(part); request.messages.push_back(user);
        const auto original = request_replica_affinity(request);
        ChatTurn answer; answer.role = "assistant"; answer.content = {part};
        request.messages.push_back(answer); request.messages.push_back(user);
        const auto appended = request_replica_affinity(request);
        check(appended.count == 3 && appended.turn_keys[0] == original.turn_keys[0] &&
              appended.turn_keys[2] != original.turn_keys[0], "appended transcript has no original boundary");
        auto changed = request;
        changed.messages[0].role = "system";
        check(request_replica_affinity(changed).turn_keys[0] != original.turn_keys[0], "roles absent from affinity");
        changed = request; changed.messages[0].content[0].kind = ContentKind::Image;
        changed.messages[0].content[0].source.value = "image-one";
        auto media = request_replica_affinity(changed);
        changed.messages[0].content[0].source.value = "image-two";
        check(request_replica_affinity(changed).turn_keys[0] != media.turn_keys[0], "media absent from affinity");
        changed = request; changed.messages[1].tool_calls.push_back({"call-1", "lookup", "{}"});
        check(request_replica_affinity(changed).turn_keys[1] != appended.turn_keys[1], "tool history absent from affinity");
        for (int i = 0; i < 100; ++i) { request.messages.push_back(user); }
        const auto bounded = request_replica_affinity(request);
        request.messages.push_back(answer);
        const auto advanced = request_replica_affinity(request);
        check(bounded.count == 64 && advanced.count == 64 &&
              bounded.turn_keys[63] == advanced.turn_keys[62], "bounded history lost recent prefix");
    }

    std::array<std::atomic<int>, 2> active{};
    std::atomic<bool> overflow = false;
    std::vector<std::thread> threads;
    for (int worker = 0; worker < 16; ++worker) {
        threads.emplace_back([&] {
            for (int i = 0; i < 1000; ++i) {
                auto lease = router.acquire();
                if (!lease) { std::this_thread::yield(); continue; }
                const auto replica = lease->replica();
                if (++active[replica] > 4) { overflow = true; }
                std::this_thread::yield();
                --active[replica];
            }
        });
    }
    for (auto& thread : threads) { thread.join(); }
    check(!overflow, "concurrent admission exceeded a replica's capacity");
    check(active[0] == 0 && active[1] == 0, "churn leaked an active reservation");
    for (int i = 0; i < 8; ++i) { live.push_back(router.acquire()); }
    for (const auto& lease : live) { check(bool(lease), "churn lost capacity"); }
    check(!router.acquire(), "churn corrupted overload accounting");
    std::cout << (failures ? "FAILED" : "replica routing/lifecycle/affinity/churn passed") << '\n';
    return failures ? 1 : 0;
}
