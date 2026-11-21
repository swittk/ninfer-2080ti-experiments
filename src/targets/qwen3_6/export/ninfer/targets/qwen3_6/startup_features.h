#pragma once

#include "ninfer/types.h"

namespace ninfer::targets::qwen3_6 {

struct StartupFeatures {
    bool vision                    = false;
    SpeculativeBackend speculative = SpeculativeBackend::None;
    ProposalHead proposal_head     = ProposalHead::Full;
    bool force_full_target_head    = false;
    bool fast_full_vocab_draft     = false;
    bool retained_target_full_vocab_draft = false;
    float greedy_margin_delta     = -1.0F;
    bool retained_mtp_packed_prefix = false;
    // Opt-in source-owned two-limb D3 projections; does not relax acceptance.
    bool sm75_radix_projections = false;
    bool sm75_resident_radix = false;
    bool sm75_nu3_mlp = false;
    bool target_draft_union = false;
    bool session_copy_draft = false;

    bool operator==(const StartupFeatures&) const = default;

    [[nodiscard]] bool speculative_enabled() const noexcept {
        return speculative != SpeculativeBackend::None;
    }

    [[nodiscard]] bool mtp() const noexcept { return speculative == SpeculativeBackend::Mtp; }

    [[nodiscard]] bool dflash() const noexcept { return speculative == SpeculativeBackend::DFlash; }

    [[nodiscard]] bool optimized_proposal() const noexcept {
        return speculative_enabled() && proposal_head == ProposalHead::Optimized;
    }

    // The legacy optimized selector also enables target-head acceleration.
    // Full-vocabulary MTP ranks from the native output head instead, so it must
    // not load or prepare the optional corpus-selected proposal payload.
    [[nodiscard]] bool requires_optimized_proposal_payload() const noexcept {
        return optimized_proposal() && !(mtp() && fast_full_vocab_draft);
    }

    // Eligibility only: the exact variant and token width still own dispatch.
    [[nodiscard]] bool allow_optimized_target_head() const noexcept {
        return optimized_proposal() && !force_full_target_head;
    }
};

[[nodiscard]] inline StartupFeatures startup_features(const EngineOptions& options) noexcept {
    return StartupFeatures{
        .vision        = options.enable_vision,
        .speculative   = options.speculative.backend,
        .proposal_head = options.speculative.proposal_head,
        .force_full_target_head = options.speculative.force_full_target_head,
        .fast_full_vocab_draft = options.speculative.fast_full_vocab_draft,
        .retained_target_full_vocab_draft = options.speculative.retained_target_full_vocab_draft,
        .greedy_margin_delta = options.speculative.greedy_margin_delta,
        .retained_mtp_packed_prefix = options.speculative.retained_mtp_packed_prefix,
        .sm75_radix_projections = options.speculative.sm75_radix_projections,
        .sm75_resident_radix = options.speculative.sm75_resident_radix,
        .sm75_nu3_mlp = options.speculative.sm75_nu3_mlp,
        .target_draft_union = options.speculative.target_draft_union,
        .session_copy_draft = options.speculative.session_copy_draft,
    };
}

} // namespace ninfer::targets::qwen3_6
