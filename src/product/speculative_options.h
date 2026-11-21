#pragma once

#include "ninfer/types.h"

#include <stdexcept>
#include <string>
#include <string_view>

namespace ninfer::product {

[[nodiscard]] inline SpeculativeBackend parse_speculative_backend(std::string_view value) {
    if (value == "mtp") { return SpeculativeBackend::Mtp; }
    if (value == "dflash") { return SpeculativeBackend::DFlash; }
    throw std::invalid_argument("invalid speculative backend: " + std::string(value));
}

[[nodiscard]] inline const char* speculative_backend_name(SpeculativeBackend backend) noexcept {
    switch (backend) {
    case SpeculativeBackend::None:
        return "none";
    case SpeculativeBackend::Mtp:
        return "mtp";
    case SpeculativeBackend::DFlash:
        return "dflash";
    }
    return "unknown";
}

inline void validate_mtp_logit_margin(const SpeculativeOptions& options) {
    if (options.retained_mtp_packed_prefix &&
        (options.backend != SpeculativeBackend::Mtp || options.draft_tokens != 3)) {
        throw std::invalid_argument("retained packed-prefix drafting requires C1 MTP D3");
    }
    if (options.session_copy_draft &&
        (options.backend!=SpeculativeBackend::Mtp || options.draft_tokens!=3 ||
         options.greedy_margin_delta!=-1.0F || !options.force_full_target_head)) {
        throw std::invalid_argument("session-copy requires strict MTP D3 and explicit full-target verification");
    }
    if(options.target_draft_union && (options.backend!=SpeculativeBackend::Mtp ||
        options.draft_tokens!=3 || options.greedy_margin_delta!=-1.0F ||
        options.force_full_target_head || options.proposal_head!=ProposalHead::Optimized)) {
        throw std::invalid_argument("target draft union requires strict D3 and an explicitly optimized target head");
    }
    if (options.sm75_nu3_mlp && (options.backend!=SpeculativeBackend::Mtp ||
        options.draft_tokens!=3 || options.greedy_margin_delta!=-1.0F)) {
        throw std::invalid_argument("NU3 target MLP requires strict native MTP D3");
    }
    if (options.sm75_resident_radix &&
        (!options.sm75_radix_projections || options.sm75_nu3_mlp)) {
        throw std::invalid_argument("resident radix requires radix projections without NU3");
    }
    if (options.sm75_radix_projections &&
        (options.backend != SpeculativeBackend::Mtp ||
         (options.draft_tokens != 3 && !(options.sm75_resident_radix && options.draft_tokens == 1)) ||
         options.greedy_margin_delta != -1.0F)) {
        throw std::invalid_argument("radix projections require strict MTP D3 or resident C4/D1");
    }
    const float delta = options.greedy_margin_delta;
    if (delta != -1.0F && !(delta >= 0.0F && delta <= 128.0F)) {
        throw std::invalid_argument("MTP logit margin must be -1 (strict) or finite in [0,128]");
    }
    if (delta >= 0.0F &&
        (options.backend != SpeculativeBackend::Mtp || options.draft_tokens != 3)) {
        throw std::invalid_argument("--mtp-logit-margin requires --spec mtp --draft-tokens 3");
    }
}

inline void validate_radix_topology(const SpeculativeOptions& options, std::uint32_t slots) {
    if (options.sm75_radix_projections &&
        (slots == 0 || slots > 4 || (options.draft_tokens == 1 && slots != 4))) {
        throw std::invalid_argument("radix projections support C1-C4/D3 or resident C4/D1");
    }
}

inline void validate_speculative_cli_options(const SpeculativeOptions& options) {
    validate_mtp_logit_margin(options);
    if (options.retained_target_full_vocab_draft &&
        (!options.fast_full_vocab_draft || options.backend != SpeculativeBackend::Mtp ||
         options.draft_tokens != 3 || options.greedy_margin_delta != -1.0F ||
         options.proposal_head != ProposalHead::Optimized || options.force_full_target_head)) {
        throw std::invalid_argument("retained-target full-vocabulary drafts require explicit optimized target, strict MTP D3 and no full-target override");
    }
    if (options.fast_full_vocab_draft && !options.retained_target_full_vocab_draft &&
        (options.backend != SpeculativeBackend::Mtp ||
         options.proposal_head != ProposalHead::Full || !options.force_full_target_head)) {
        throw std::invalid_argument("--fast-full-vocab-draft requires --spec mtp --full-target-head and cannot combine with --lm-head-draft");
    }
    switch (options.backend) {
    case SpeculativeBackend::None:
        if (options.draft_tokens != 0 || options.proposal_head != ProposalHead::Full) {
            throw std::invalid_argument(
                "--draft-tokens and --lm-head-draft require --spec mtp|dflash");
        }
        return;
    case SpeculativeBackend::Mtp:
        if (options.draft_tokens == 0 || options.draft_tokens > 5) {
            throw std::invalid_argument("--spec mtp requires --draft-tokens in [1,5]");
        }
        return;
    case SpeculativeBackend::DFlash:
        if (options.draft_tokens == 0 || options.draft_tokens > 15) {
            throw std::invalid_argument("--spec dflash requires --draft-tokens in [1,15]");
        }
        return;
    }
    throw std::invalid_argument("invalid speculative backend");
}

} // namespace ninfer::product
