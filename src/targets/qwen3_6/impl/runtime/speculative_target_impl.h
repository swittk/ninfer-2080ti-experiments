#include "targets/qwen3_6/impl/runtime/instance.h"
#include "targets/qwen3_6/impl/runtime/schedule.h"

#include "ninfer/ops/scatter.h"
#include "ninfer/ops/speculative_round.h"

namespace ninfer::targets::qwen3_6::detail::NINFER_QWEN36_RUNTIME_NS::schedule {

void target_verify_accept(ExecutionCore& execution, Tensor& continuation_hidden_store,
                          TextContext& card, TargetVerifyFrameView frame,
                          ops::GqaExecutionEnvelope envelope) {
    if (frame.replay_records == nullptr) {
        throw std::logic_error("speculative target verify has no ReplaySSM record storage");
    }
    card.set_gdn_state_action(GdnStateAction::RecordForReplay, frame.replay_records);
    if (frame.feature_sink != nullptr) {
        card.target_verify_batch(frame.ids, frame.cache_positions, frame.rope_positions,
                                 frame.valid_columns, frame.kv_table_rows, frame.lanes, envelope,
                                 frame.target_hidden, frame.target_logits, frame.target_tokens,
                                 *frame.feature_sink);
    } else {
        card.target_verify_batch(frame.ids, frame.cache_positions, frame.rope_positions,
                                 frame.valid_columns, frame.kv_table_rows, frame.lanes, envelope,
                                 frame.target_hidden, frame.target_logits, frame.target_tokens);
    }
    if(execution.model.features.target_draft_union) {
        card.include_target_draft_candidates(frame.target_hidden,frame.drafts,frame.current_extents,
                                               frame.target_logits,frame.target_tokens);
    }
    const float margin = execution.model.features.greedy_margin_delta;
    const bool full_logits = !execution.model.features.allow_optimized_target_head();
    if (margin >= 0.0F) {
        if (frame.drafts.ne[0] != 3 || frame.drafts.ne[1] != 1) {
            throw std::logic_error("retained margin policy escaped its C1/D3 frame");
        }
        // The independently reproduced C1 binary did NOT score absent candidates
        // here (the separate dual route did). Preserve its shortlist acceptance
        // without reading an unwritten "mandatory draft" metadata slot.
    }
    ops::speculative_accept_greedy_drafts(frame.target_tokens, frame.target_logits, frame.drafts,
                                          frame.current_extents, frame.frontiers, frame.anchors,
                                          frame.licensed_tokens, frame.licensed_counts,
                                          frame.accepted_drafts, TextConfig::token_domain,
                                          frame.sampling, execution.work, execution.device.stream,
                                          margin, full_logits ? ops::GreedyMarginLogits::FullVocabulary
                                              : ops::GreedyMarginLogits::IndexedCandidates);
    ops::speculative_select_accepted_hidden(frame.target_hidden, frame.accepted_drafts,
                                            frame.selected_hidden, execution.device.stream);
    ops::scatter(frame.selected_hidden, frame.lanes, continuation_hidden_store,
                 execution.device.stream);
}

} // namespace ninfer::targets::qwen3_6::detail::NINFER_QWEN36_RUNTIME_NS::schedule
