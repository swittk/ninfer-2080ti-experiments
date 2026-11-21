# RESUME NOW — finish transfer, clean tree, build NInfer tree verifier, then train a real Qwen3.8 drafter

Read the 0343 report + latest `progress.md`. Do not reopen closed current-MTP/tree/LoRA/kernel routes.

## Precisely unfinished

Lucebox SM75 reference is accepted: HumanEval10 B22 mean 72.55 t/s, median 69.05, AL 6.86; draft 12.65 ms/round; tree verify/select 81.62 ms/round. This proves DFlash+DDTree is real on TU102 but is not an NInfer hard-goal result.

Qwen3.6-DFlash -> Qwen3.8 transfer already captured Qwen3.8 taps [1,16,31,46,61], target-owned embeddings, public-draft hidden columns, and exact Qwen3.8 W8 full-vocab ranks. Early top1 aggregate is only ~2.49% (general 5.21, Japanese 4.07, JSON 2.21, Arabic 1.71, Thai .93, code .79). Zero-shot checkpoint transfer is almost certainly CLOSED, not DFlash itself.

Do NOW:
1. Run `../ninfer-2080ti/.chatgpt/tmp/analyze_dflash_transfer.py`.
2. Record top1/4/8/16/32 recall, rank percentiles, linear prefix yield, DDTree budgets 12/16/22 per workload.
3. Write transfer CONCLUSION + `progress.md` classification.
4. Restore `src/targets/qwen3_6/impl/runtime/text_context_impl.h` from `.chatgpt/dual_c1_20260819/dflash_transfer/text_context_impl.before.h` unless deliberately promoting the capture seam with tests.
5. Rebuild + argmax/GQA/GDN-replay regressions. Do not commit a dirty temporary capture patch.

## Training research update — do not wait for Z-Lab training release

Official Z-Lab DFlash still lists Qwen3.6/3.5 targets but no Qwen3.8-27B draft.

Tencent **AngelSpec** is now open source: https://github.com/Tencent/AngelSpec . It supports DFlash, DFlare, DFly, DSpark, Eagle3 and MTP under one training pipeline, Qwen3 examples, disaggregated target-hidden extraction/training, online real acceptance evaluation, and acceptance-aware objectives. Its docs explicitly allow draft embeddings + lm_head to come from the TARGET and stay frozen.

DFlare (arXiv:2606.02091) is relevant if plain DFlash conditioning is insufficient: per-draft-layer target-feature fusion, training scaled from ~800K to 2.4M samples. The failed 1,719-row adapters only close those tiny adapter hypotheses.

After transfer closure, make a MINIMAL AngelSpec Qwen3.8 feasibility smoke before any huge dataset:
- map Qwen3.8 config/tokenizer/feature taps into nearest Qwen3 config;
- reuse frozen Qwen3.8 embedding/lm_head;
- attempt CPT/init from Qwen3.6 DFlash only where shape-compatible;
- use offline captured hidden-feature shards so 27B target need not co-reside with trainer;
- prove one train step -> checkpoint -> offline/online acceptance eval fits on available hardware.

Only then scale a diverse corpus to hundreds-of-thousands+ anchors (general/code/JSON/Thai/Japanese/Arabic/reasoning).

## Build the TARGET TREE VERIFIER now — independent of drafter quality

We need this regardless of whether DFlash/DFlare/DFly/JetSpec wins. Feed externally supplied deterministic trees first.

Phase A correctness oracle:
- `VerifyTreePlan {token,parent,depth}` with parents before children;
- exact physical-KVP ancestor-only visibility: committed prefix + ancestors + self, never siblings;
- parent-indexed conv history;
- parent-indexed GDN recurrence;
- target logits for all nodes;
- selected-path commit + next-round continuation.

Start tiny 2-branch depth2/3 ONLY to validate semantics against canonical branch-by-branch execution. This is not reopening the old current-MTP tiny-tree performance experiment.

Phase B target-side economics BEFORE expensive training:
- benchmark node budgets 8/12/16/22 at 65K using teacher/perfect trees and Luce-like topologies;
- report complete tree wall, physical-KVP attention, MLP/GDN/head, transient state bytes, commit wall;
- project useful TPS for AL 5/7/9/11;
- project 262K from measured node/KVP costs and only populate-run if >=100 remains plausible.

## Production GDN tree state: use compact replay factors, not a dense state image per node

FP32 per-node state is fine for the tiny correctness oracle only.

ReplaySSM / compact linear spec-cache work is now active in vLLM/SGLang for GDN spec verification. SGLang RFC #28511 describes compact per-token transition inputs instead of full intermediate recurrent states, then replaying the accepted state once. vLLM RFC #49232 tracks the same direction.

This maps naturally to NInfer `GdnReplayRecordLayer`, `gated_delta_net_replay_record`, `gdn_replay_fold`.

After correctness, make the tree store minimal parent-indexed transition/replay factors + conv ancestry; verify without mutating committed dense state; replay/fold only the accepted root-to-leaf path once. Likely a better 22GB production design than Luce full persistent state per node.

## JetSpec reference

JetSpec has no Qwen3.8-27B head today, but its official engine is useful for one-pass conditioned tree drafting, large tree budgets, paged tree KV/attention, CUDA graphs, and cumulative-logprob/adaptive fanout builders. Use its algorithmic ideas after NInfer tree verification works; do not port its whole target runtime.

## Single >=80 lane

Audit ReplaySSM-style ordinary GDN decode too. Existing speculative replay/fold does not prove normal decode avoids dense recurrent-state HBM writes. Only implement if measured state traffic gives >=5% E2E ceiling; ~72 -> 80 needs ~11% total.

## Order
1. Finish transfer analyzer/classification.
2. Restore + qualify capture seam.
3. Native parent-indexed tree correctness oracle.
4. Budget 8/12/16/22 target-cost matrix.
5. Parallel AngelSpec Qwen3.8 DFlash/DFlare training smoke.
6. Compact ReplaySSM-style tree records.
7. Marry trained block drafter to verifier.

Hard gates remain single >=80, dual one-request >=100, populated ~262K dual one-request >=100. Do not stop after another closure; move immediately to the next bounded discriminator.
