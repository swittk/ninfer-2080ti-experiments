#include "targets/qwen3_6/impl/frontend/media_prepare_cache.h"
#include <array>
#include <iostream>
#include <thread>
#include <vector>
using namespace ninfer::targets::qwen3_6::frontend_internal;
int main() {
    ProcessorOptions options;
    auto value = std::make_shared<const PreparedMedia>(PreparedMedia{VisionItem{}, std::vector<std::uint8_t>(128, 128)});
    const std::vector<std::uint8_t> a{1,2,3}, b{1,2,4}, c{1,2,5};
    const std::array<std::uint8_t,32> digest{}; // Deliberate digest collision.
    MediaPreparationCache sizing(1U << 20);
    sizing.store(ChatPartKind::Image,a,options,value);
    const auto charge = sizing.retained_bytes();
    if (!charge || !sizing.find(ChatPartKind::Image,a,digest,options) ||
        sizing.find(ChatPartKind::Image,b,digest,options) ||
        sizing.find(ChatPartKind::Video,a,digest,options)) { return 1; }
    auto changed = options; changed.video_fps = 3;
    if (sizing.find(ChatPartKind::Image,a,digest,changed)) { return 2; }
    changed = options; changed.max_raw_patches = 4;
    if (sizing.find(ChatPartKind::Image,a,digest,changed)) { return 3; }
    MediaPreparationCache cache(charge * 2);
    cache.store(ChatPartKind::Image,a,options,value);
    cache.store(ChatPartKind::Image,b,options,value);
    auto retained = cache.find(ChatPartKind::Image,a,digest,options);
    cache.store(ChatPartKind::Image,c,options,value);
    if (cache.find(ChatPartKind::Image,b,digest,options) ||
        !cache.find(ChatPartKind::Image,a,digest,options) ||
        !cache.find(ChatPartKind::Image,c,digest,options) || cache.retained_bytes() > charge*2) { return 4; }
    MediaPreparationCache disabled(0), small(charge-1);
    disabled.store(ChatPartKind::Image,a,options,value);
    small.store(ChatPartKind::Image,a,options,value);
    if (disabled.retained_bytes() || small.retained_bytes()) { return 5; }
    auto oversized = std::make_shared<const PreparedMedia>(PreparedMedia{VisionItem{},std::vector<std::uint8_t>(charge*3,255)});
    cache.store(ChatPartKind::Image,b,options,oversized);
    if (cache.retained_bytes() > charge*2 || retained->pixels.front()!=128 || cache.find(ChatPartKind::Image,b,digest,options)) { return 6; }
    // 40 temporal segments of the real fixture have 35,200 patches. The same
    // 64-MiB cache now retains their RGB8 payload; FP32 would require four times it.
    MediaPreparationCache video_cache(64ULL << 20);
    auto large = std::make_shared<const PreparedMedia>(PreparedMedia{VisionItem{},
        std::vector<std::uint8_t>(35200ULL * 1536, 47)});
    video_cache.store(ChatPartKind::Video,a,options,large);
    if (!video_cache.find(ChatPartKind::Video,a,digest,options) ||
        video_cache.retained_bytes() > (64ULL << 20)) return 8;
    std::vector<std::thread> threads;
    for (int i=0;i<8;++i) threads.emplace_back([&] {
        for (int j=0;j<100;++j) {
            cache.store(ChatPartKind::Image,b,options,value);
            (void)cache.find(ChatPartKind::Image,a,digest,options);
            cache.store(ChatPartKind::Image,c,options,value);
        }
    });
    for (auto& thread:threads) thread.join();
    if (cache.retained_bytes() > charge*2) { return 7; }
    std::cout << "PASS exact bytes under digest collision, modality/options isolation, LRU, storage cap, oversized bypass and concurrent access\n";
    return 0;
}
