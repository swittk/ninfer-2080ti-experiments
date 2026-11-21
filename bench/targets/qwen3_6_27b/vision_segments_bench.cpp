#include "artifact/reader.h"
#include "artifact/binder.h"
#include "artifact/materializer.h"
#include "core/device.h"
#include "targets/qwen3_6/impl/frontend/test_access.h"
#include "targets/qwen3_6_27b/impl/variant.h"
#include <ninfer/targets/qwen3_6/prepared_prompt.h>
#include <ninfer/targets/qwen3_6/vision_control.h>
#include <ninfer/targets/qwen3_6/vision_limits.h>
#define NINFER_QWEN36_VARIANT ::ninfer::targets::qwen3_6_27b::detail::Variant
#define NINFER_QWEN36_RUNTIME_NS qwen3_6_27b_runtime
#include "targets/qwen3_6/impl/runtime/vision_context.h"
#include <algorithm>
#include <bit>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <iterator>
#include <stdexcept>
#include <string>
#include <vector>

namespace ninfer::targets::qwen3_6::detail::qwen3_6_27b_runtime::schedule {
struct VisionContextTestAccess {
    static std::size_t original_bytes(const qwen3_6::VisionItemControl& item) {
        return VisionContext::batch_workspace_bytes(item);
    }
    static void original(const VisionContext& context, const VisionItemView& item,
                         Tensor& output, WorkspaceArena& workspace) {
        context.encode_batch(item, output, workspace);
    }
};
}
using namespace ninfer;
namespace target = ninfer::targets::qwen3_6_27b;
namespace family = ninfer::targets::qwen3_6;
namespace schedule = ninfer::targets::qwen3_6::detail::qwen3_6_27b_runtime::schedule;

static float as_float(std::uint16_t bits) {
    return std::bit_cast<float>(static_cast<std::uint32_t>(bits) << 16);
}
static std::vector<std::uint8_t> read_bytes(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("cannot read media: " + path);
    return {std::istreambuf_iterator<char>(f), std::istreambuf_iterator<char>()};
}
static PromptInput media_input(const std::string& path, bool video) {
    MessagePart part; part.kind = MessagePartKind::Media;
    part.media.kind = video ? MediaKind::Video : MediaKind::Image;
    part.media.bytes = read_bytes(path); part.media.source_name = path;
    part.media.media_type = video ? "video/mp4" : "image/png";
    ChatMessage user; user.role = "user"; user.parts.push_back(std::move(part));
    PromptInput input; input.messages.push_back(std::move(user));
    input.options.enable_thinking = false;
    return input;
}
static family::PreparedPromptData extended_video(const family::PreparedPromptData& source,
                                                 int copies) {
    const auto& first = source.vision_items.at(0);
    family::PreparedPromptData out;
    family::VisionItem item = first;
    item.patch_begin = 0; item.patch_count *= copies;
    item.grid.temporal *= copies; item.token_spans.clear(); item.timestamps.clear();
    const std::size_t per_segment = static_cast<std::size_t>(item.grid.height) * item.grid.width / 4;
    for (int i = 0; i < item.grid.temporal; ++i) {
        item.token_spans.push_back({static_cast<std::size_t>(i) * per_segment, per_segment});
        item.timestamps.push_back(i);
    }
    out.token_ids.assign(item.patch_count / 4, 248057);
    out.token_types.assign(out.token_ids.size(), static_cast<std::uint8_t>(family::PromptModality::Video));
    out.patches.reserve(item.patch_count * 1536);
    const auto start = source.patches.begin() + first.patch_begin * 1536;
    for (int i = 0; i < copies; ++i)
        out.patches.insert(out.patches.end(), start, start + first.patch_count * 1536);
    out.prepare.media_items = 1;
    out.prepare.raw_patches = item.patch_count;
    out.prepare.vision_tokens = item.patch_count / 4;
    out.prepare.patch_bytes = out.patches.size() * sizeof(float);
    const auto spatial = static_cast<std::uint64_t>(item.grid.height) * item.grid.width;
    out.prepare.attention_pairs = spatial * spatial * item.grid.temporal;
    out.vision_items.push_back(std::move(item));
    return out;
}
int main(int argc, char** argv) {
    if (argc < 2) { std::fprintf(stderr,"usage: vision_segments_bench ARTIFACT [DEVICE]\n"); return 2; }
    try {
        DeviceContext device(argc > 2 ? std::stoi(argv[2]) : 0);
        artifact::Reader reader(argv[1]);
        artifact::Binder binder(reader);
        auto plan = target::detail::bind_artifact(binder, target::Package::resolve_weights(reader.identity()),
                                                 family::StartupFeatures{.vision = true});
        auto backing = artifact::materialize(reader, plan.materialization, device);
        target::detail::LoadedModelData model(std::move(plan.bindings), std::move(backing));
        auto frontend = family::make_frontend(model.frontend, true);
        if (argc > 3 && std::string(argv[3]) == "prepare") {
            auto input = media_input(".chatgpt/production_serve_20260908/cold_memory/temporal_events_40s.mp4",true);
            for (int run = 0; run < 3; ++run) {
                const auto start=std::chrono::steady_clock::now();
                auto prepared=frontend.prepare(input);
                const auto& data=family::PreparedPromptAccess::view(prepared);
                std::printf("PREPARE run=%d seconds=%.6f cached=%zu patches=%llu fp32_bytes=%zu retained=%zu\n",
                    run,std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count(),
                    data.prepare.cached_media_items,static_cast<unsigned long long>(data.prepare.raw_patches),
                    data.patches.size()*sizeof(float),family::FrontendTestAccess::retained_media_bytes(frontend));
                std::fflush(stdout);
            }
            return 0;
        }
        schedule::VisionContext encoder(device, model.runtime);
        auto image = frontend.prepare(media_input("examples/cli/media/visual_chart.png", false));
        auto video = frontend.prepare(media_input("examples/cli/media/temporal_events.mp4", true));
        auto long_video = extended_video(family::PreparedPromptAccess::view(video), 8);
        const std::pair<const char*, const family::PreparedPromptData*> cases[] = {
            {"image", &family::PreparedPromptAccess::view(image)},
            {"video", &family::PreparedPromptAccess::view(video)},
            {"extended_video", &long_video},
        };
        for (const auto& [name, prompt] : cases) {
            const auto controls = family::build_vision_control(*prompt);
            const auto& control = controls.items.at(0);
            const auto old_bytes = schedule::VisionContextTestAccess::original_bytes(control);
            const auto new_bytes = schedule::VisionContext::workspace_bytes(control);
            const auto cap = schedule::VisionContext::workspace_capacity_bytes(16384,384);
            if (new_bytes > cap) throw std::runtime_error("request exceeds planned workspace");
            DeviceBuffer old_out(5120 * control.merged_count * 2), new_out(old_out.bytes);
            Tensor a(old_out.p,DType::BF16,{5120,static_cast<int>(control.merged_count)});
            Tensor b(new_out.p,DType::BF16,{5120,static_cast<int>(control.merged_count)});
            WorkspaceArena original(old_bytes), candidate(new_bytes);
            const schedule::VisionItemView item{
                std::span<const float>(prompt->patches).subspan(control.patch_begin * 1536,
                                                               control.patch_count * 1536), &control};
            auto measure = [&](auto&& run) {
                device.synchronize(); const auto begin = std::chrono::steady_clock::now();
                run(); device.synchronize();
                return std::chrono::duration<double>(std::chrono::steady_clock::now()-begin).count();
            };
            schedule::VisionContextTestAccess::original(encoder,item,a,original);
            encoder.encode(item,b,candidate); device.synchronize();
            const double old_seconds = measure([&]{schedule::VisionContextTestAccess::original(encoder,item,a,original);});
            const double new_seconds = measure([&]{encoder.encode(item,b,candidate);});
            std::vector<std::uint16_t> reference(old_out.bytes/2), actual(reference.size());
            old_out.copy_to_host(reference.data(),old_out.bytes); new_out.copy_to_host(actual.data(),new_out.bytes);
            double err=0,norm=0,maxdiff=0,maxref=0; std::size_t mismatch=0,bad=0;
            for(std::size_t i=0;i<reference.size();++i) {
                const double x=as_float(reference[i]), y=as_float(actual[i]), d=y-x;
                err+=d*d; norm+=x*x; maxdiff=std::max(maxdiff,std::abs(d)); maxref=std::max(maxref,std::abs(x));
                mismatch+=reference[i]!=actual[i]; bad+=!std::isfinite(y);
            }
            const double rel=std::sqrt(err/std::max(norm,1e-30));
            std::printf("VISION name=%s segments=%d patches=%zu merged=%zu old_bytes=%zu new_bytes=%zu capacity_bytes=%zu old_s=%.6f new_s=%.6f rel_l2=%.9g max_abs=%.9g max_reference=%.9g mismatches=%zu nonfinite=%zu\n",
                name,control.segment_count,control.patch_count,control.merged_count,old_bytes,new_bytes,cap,
                old_seconds,new_seconds,rel,maxdiff,maxref,mismatch,bad); std::fflush(stdout);
            // These comparisons characterize the complete encoder, not an isolated
            // attention primitive. Exact paths must stay exact; changed batch shapes
            // receive the same BF16-scale numerical bound before behavioral checks.
            if (bad || rel > 1.0/256.0 || maxdiff > 1e-3 + maxref/128.0 ||
                (control.patch_count <= family::kVisionBatchPatchCapacity && mismatch)) return 3;
            auto invalid=control; invalid.cu_seqlens[1]-=1;
            bool rejected=false;
            try { encoder.encode({item.patches,&invalid},b,candidate); }
            catch(const std::invalid_argument&) { rejected=true; }
            if(!rejected) throw std::runtime_error("invalid segment boundary accepted");
        }
        return 0;
    } catch (const std::exception& e) { std::fprintf(stderr,"vision_segments_bench: %s\n",e.what()); return 1; }
}
