# OWNER CORRECTION — GENERIC STRICT C1 MUST NOT DEPEND ON A CORPUS-FREQUENCY VOCABULARY SHORTLIST

Effective 2026-09-16. This sharpens the genericity requirement for the strict single-GPU C1 >=100 gate.

The current optimized proposal artifact uses `text/draft_head_token_ids`, a fixed 131072-token subset selected by the converter from a corpus frequency ranking (`ranking.train.counts.i64`) plus tokenizer special IDs, then materializes those LM-head rows. This is not a finetune-specific literal table, but it is corpus-dependent/static and therefore does **not** qualify as the foundation of the generic raw-power C1 gate.

## Hard rule

For **generic strict C1 >=100 credit**, candidate/proposal vocabulary coverage must be one of:

- full vocabulary; or
- dynamically/mechanically derived from the currently loaded model weights and current runtime hidden/state without external corpus statistics, prompt-specific tables, checkpoint IDs/hashes, learned local adapters, or behavior-specific calibration.

A corpus-frequency shortlist may remain as a separately labelled product/serving optimization, but its speed must not be presented as the generic single-GPU frontier.

The target-side SM75 shortlist that is recomputed from the loaded native W8 target head and current hidden state, followed by exact reranking with those same target weights, is acceptable as a mechanically derived runtime accelerator. The new target-draft-union mechanism is also acceptable in itself because it exact-scores the actual live D3 proposal IDs with the loaded native target head before merging; however, a run whose D3 proposal IDs ultimately come from the corpus-frequency `draft_head_token_ids` subset still does not earn generic strict-C1 credit.

## Immediate consequence

Do not count the current ~99 canonical target-union / 111-128 code numbers as the generic strict >=100 candidate while they use `--optimized-proposal` backed by the corpus-frequency 131072-token proposal head. Preserve the evidence as a serving/product lane.

Return the primary campaign to a full-vocabulary or weight/runtime-derived proposal path and retain the whole-round architecture discipline: large mechanisms first, small probes only to prune them. If the existing fast-full-vocabulary proposal path is too expensive, redesign its dataflow rather than substituting an external corpus shortlist.

All prior safety, no-training, no-subagent, strict-quality, multimodal, and full-context gates remain unchanged.
