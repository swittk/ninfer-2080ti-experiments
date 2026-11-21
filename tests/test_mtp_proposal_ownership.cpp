#include "artifact/binder.h"
#include "artifact/reader.h"
#include "targets/qwen3_6_27b/impl/load/bindings.h"
#include <ninfer/targets/qwen3_6_27b/package.h>
#include <ninfer/targets/qwen3_6/startup_features.h>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <map>
#include <set>
#include <stdexcept>
#include <string>
#include <variant>
using namespace ninfer;
using namespace ninfer::targets::qwen3_6_27b::detail;
namespace {
std::string name(const artifact::ObjectDescriptor& object) {
    return std::visit([](const auto& value) { return value.name; }, object);
}
struct Summary {
    artifact::MaterializationPlan plan;
    std::map<std::string, std::uint64_t> device;
    std::set<std::string> host;
};
Summary summarize(const artifact::Reader& reader, targets::qwen3_6::StartupFeatures features) {
    artifact::Binder binder(reader);
    auto bound = bind_artifact(binder, targets::qwen3_6_27b::Package::resolve_weights(reader.identity()), features);
    Summary out{bound.materialization, {}, {}};
    for (const auto& object : out.plan.device_objects) {
        if (!out.device.emplace(name(binder.descriptor(object.object)), object.bytes).second)
            throw std::runtime_error("duplicate device binding");
    }
    for (const auto& object : out.plan.host_objects)
        out.host.insert(name(binder.descriptor(object.object)));
    return out;
}
}
int main() {
    try {
        const char* path = std::getenv("NINFER_MTP_OWNERSHIP_ARTIFACT");
        if (path == nullptr || !std::filesystem::is_regular_file(path)) {
            std::cerr << "skip: NINFER_MTP_OWNERSHIP_ARTIFACT must name the current model\n";
            return 77;
        }
        const artifact::Reader reader(path);
        targets::qwen3_6::StartupFeatures legacy;
        legacy.speculative = SpeculativeBackend::Mtp;
        legacy.proposal_head = ProposalHead::Optimized;
        auto dynamic = legacy;
        dynamic.fast_full_vocab_draft = true;
        dynamic.retained_target_full_vocab_draft = true;
        dynamic.target_draft_union = true;
        auto native = dynamic;
        native.proposal_head = ProposalHead::Full;
        native.force_full_target_head = true;
        native.retained_target_full_vocab_draft = false;
        native.target_draft_union = false;
        if (!legacy.requires_optimized_proposal_payload() ||
            dynamic.requires_optimized_proposal_payload() ||
            native.requires_optimized_proposal_payload() ||
            !dynamic.allow_optimized_target_head() || native.allow_optimized_target_head())
            throw std::runtime_error("independent target/proposal feature policy failed");
        const auto full = summarize(reader, native);
        const auto old = summarize(reader, legacy);
        const auto current = summarize(reader, dynamic);
        const std::set<std::string> omitted{"text/draft_head", "text/draft_head_token_ids"};
        auto expected = old.device;
        std::uint64_t saved = 0;
        for (const auto& key : omitted) {
            const auto it = expected.find(key);
            if (it == expected.end()) throw std::runtime_error("legacy proposal tensor missing");
            saved += it->second;
            expected.erase(it);
        }
        if (current.device != expected || current.device != full.device ||
            old.host != current.host || current.host != full.host ||
            old.plan.object_count != current.plan.object_count ||
            old.plan.device_capacity_bytes - current.plan.device_capacity_bytes != saved)
            throw std::runtime_error("full-vocabulary plan removed or added an unrelated tensor");
        if (!current.device.contains("text/output_head") || !current.device.contains("text/token_embedding"))
            throw std::runtime_error("native vocabulary sources were removed");
        std::cout << "PASS real materialization: exactly two unused tensors omitted; "
                  << "legacy retained; native W8/embedding and all target/MTP weights unchanged; "
                  << "saved_device_bytes=" << saved << '\n';
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
