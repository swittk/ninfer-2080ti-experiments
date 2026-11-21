#include "ninfer/types.h"
#include <ninfer/targets/qwen3_6/startup_features.h>
#include <iostream>
#include <limits>
#include "product/speculative_options.h"

int main() {
    using namespace ninfer;
    using targets::qwen3_6::startup_features;
    EngineOptions options;
    if (options.speculative.force_full_target_head ||
        startup_features(options).allow_optimized_target_head()) return 1;
    options.speculative.backend = SpeculativeBackend::Mtp;
    options.speculative.draft_tokens = 3;
    options.speculative.proposal_head = ProposalHead::Optimized;
    const auto native_layout = startup_features(options);
    options.speculative.retained_mtp_packed_prefix=true;
    if(!startup_features(options).retained_mtp_packed_prefix || startup_features(options)==native_layout)return 9;
    options.speculative.retained_mtp_packed_prefix=false;
    const auto strict = startup_features(options);
    options.speculative.greedy_margin_delta=7.0F;
    if(startup_features(options).greedy_margin_delta!=7.0F || startup_features(options)==strict)return 7;
    for(float invalid : {std::numeric_limits<float>::quiet_NaN(),std::numeric_limits<float>::infinity(),-2.0F,129.0F}) {
        options.speculative.greedy_margin_delta=invalid;
        bool rejected=false;try{product::validate_mtp_logit_margin(options.speculative);}
        catch(const std::invalid_argument&){rejected=true;}if(!rejected)return 8;
    }
    options.speculative.greedy_margin_delta=-1.0F;
    const auto legacy = startup_features(options);
    if (!legacy.optimized_proposal() || !legacy.allow_optimized_target_head() ||
        !legacy.requires_optimized_proposal_payload()) return 2;
    options.speculative.force_full_target_head = true;
    const auto separated = startup_features(options);
    if (!separated.optimized_proposal() || separated.allow_optimized_target_head() ||
        !separated.force_full_target_head || separated == legacy) return 3;
    options.speculative.proposal_head = ProposalHead::Full;
    options.speculative.fast_full_vocab_draft = true;
    const auto full_draft = startup_features(options);
    if (!full_draft.fast_full_vocab_draft || !full_draft.force_full_target_head ||
        full_draft.optimized_proposal() || full_draft.allow_optimized_target_head() ||
        full_draft.requires_optimized_proposal_payload()) return 6;
    options.speculative.fast_full_vocab_draft = false;
    if (startup_features(options).optimized_proposal() ||
        startup_features(options).allow_optimized_target_head()) return 4;
    options.speculative.backend = SpeculativeBackend::None;
    options.speculative.proposal_head = ProposalHead::Optimized;
    options.speculative.force_full_target_head = false;
    if (startup_features(options).optimized_proposal() ||
        startup_features(options).allow_optimized_target_head()) return 5;
    options.speculative.backend = SpeculativeBackend::Mtp;
    options.speculative.draft_tokens = 3;
    options.speculative.greedy_margin_delta = -1.0F;
    const auto no_radix = startup_features(options);
    options.speculative.sm75_radix_projections = true;
    const auto radix = startup_features(options);
    if (!radix.sm75_radix_projections || radix == no_radix) return 20;
    ninfer::product::validate_mtp_logit_margin(options.speculative);
    for (int bad=0;bad<3;++bad) {
        auto invalid=options.speculative;
        if (bad==0) invalid.greedy_margin_delta=7;
        if (bad==1) invalid.draft_tokens=1;
        if (bad==2) invalid.backend=SpeculativeBackend::None;
        bool rejected=false;
        try { ninfer::product::validate_mtp_logit_margin(invalid); }
        catch (const std::invalid_argument&) { rejected=true; }
        if (!rejected) return 21+bad;
    }
    for (std::uint32_t slots=1;slots<=8;++slots) {
        for (std::uint32_t depth=1;depth<=5;++depth) {
            SpeculativeOptions resident;
            resident.backend=SpeculativeBackend::Mtp;
            resident.draft_tokens=depth;
            resident.sm75_radix_projections=true;
            resident.sm75_resident_radix=true;
            bool admitted=true;
            try {product::validate_mtp_logit_margin(resident);product::validate_radix_topology(resident,slots);}
            catch(const std::invalid_argument&){admitted=false;}
            const bool expected=(depth==3 && slots<=4)||(depth==1 && slots==4);
            if(admitted!=expected)return 68;
        }
    }
    options.speculative.fast_full_vocab_draft = true;
    options.speculative.retained_target_full_vocab_draft = true;
    const auto composed = startup_features(options);
    product::validate_speculative_cli_options(options.speculative);
    if (!composed.allow_optimized_target_head() || composed.greedy_margin_delta != -1.0F ||
        !composed.retained_target_full_vocab_draft || composed==radix ||
        composed.requires_optimized_proposal_payload()) return 30;
    for (int variant=0; variant<6; ++variant) {
        auto invalid=options.speculative;
        if (variant==0) invalid.fast_full_vocab_draft=false;
        if (variant==1) invalid.force_full_target_head=true;
        if (variant==2) invalid.proposal_head=ProposalHead::Full;
        if (variant==3) invalid.greedy_margin_delta=7.0F;
        if (variant==4) invalid.draft_tokens=1;
        if (variant==5) invalid.backend=SpeculativeBackend::None;
        bool rejected=false;
        try{product::validate_speculative_cli_options(invalid);}
        catch(const std::invalid_argument&){rejected=true;}
        if(!rejected)return 32+variant;
    }
    options.speculative.retained_target_full_vocab_draft=false;
    bool rejected=false;
    try{product::validate_speculative_cli_options(options.speculative);}
    catch(const std::invalid_argument&){rejected=true;}
    if(!rejected)return 31;
    EngineOptions copy_options;
    copy_options.speculative.backend=SpeculativeBackend::Mtp;
    copy_options.speculative.draft_tokens=3;
    copy_options.speculative.force_full_target_head=true;
    const auto without_copy=startup_features(copy_options);
    copy_options.speculative.session_copy_draft=true;
    const auto with_copy=startup_features(copy_options);
    if(!with_copy.session_copy_draft || with_copy==without_copy ||
       with_copy.greedy_margin_delta!=-1.f || with_copy.allow_optimized_target_head())return 60;
    product::validate_speculative_cli_options(copy_options.speculative);
    for(int i=0;i<4;++i) {
        auto bad=copy_options.speculative;
        if(i==0)bad.draft_tokens=1;
        if(i==1)bad.backend=SpeculativeBackend::None;
        if(i==2)bad.greedy_margin_delta=7.f;
        if(i==3)bad.force_full_target_head=false;
        bool rejected=false;
        try{product::validate_speculative_cli_options(bad);}catch(const std::invalid_argument&){rejected=true;}
        if(!rejected)return 61+i;
    }
    EngineOptions nu3_options;
    nu3_options.speculative.backend=SpeculativeBackend::Mtp;
    nu3_options.speculative.draft_tokens=3;
    const auto original_nu3=startup_features(nu3_options);
    nu3_options.speculative.sm75_nu3_mlp=true;
    const auto explicit_nu3=startup_features(nu3_options);
    if(!explicit_nu3.sm75_nu3_mlp || explicit_nu3==original_nu3 || explicit_nu3.greedy_margin_delta!=-1.0F)return 70;
    product::validate_mtp_logit_margin(nu3_options.speculative);
    for(int variant=0;variant<3;++variant){
        auto bad=nu3_options.speculative;
        if(variant==0)bad.draft_tokens=1;
        if(variant==1)bad.backend=SpeculativeBackend::None;
        if(variant==2)bad.greedy_margin_delta=7;
        bool rejected=false;try{product::validate_mtp_logit_margin(bad);}catch(const std::invalid_argument&){rejected=true;}
        if(!rejected)return 71+variant;
    }
    EngineOptions candidate_union;
    candidate_union.speculative.backend=SpeculativeBackend::Mtp;
    candidate_union.speculative.draft_tokens=3;
    candidate_union.speculative.proposal_head=ProposalHead::Optimized;
    const auto without_union=startup_features(candidate_union);
    candidate_union.speculative.target_draft_union=true;
    const auto with_union=startup_features(candidate_union);
    if(!with_union.target_draft_union || with_union==without_union || with_union.greedy_margin_delta!=-1.0F)return 80;
    product::validate_speculative_cli_options(candidate_union.speculative);
    for(int i=0;i<4;++i){auto invalid=candidate_union.speculative;
        if(i==0)invalid.draft_tokens=1;
        if(i==1)invalid.force_full_target_head=true;
        if(i==2)invalid.greedy_margin_delta=7;
        if(i==3)invalid.proposal_head=ProposalHead::Full;
        bool rejected=false;try{product::validate_speculative_cli_options(invalid);}catch(const std::invalid_argument&){rejected=true;}
        if(!rejected)return 81+i;
    }
    std::cout << "PASS independent proposal/target policies, frozen feature identity and legacy defaults\n";
}
