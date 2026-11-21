# 14:10 Aug25 resume — recert long exact-BF16 TP2, then shared-verifier adaptive FFN skip oracle

Read repo-root `AGENTS.md` first. Stock GPUs only. No hardware tuning/reset/reboot. No Codex/Claude/OpenCode/coding agents. No local behavioral training/finetuning/distillation/adapters/learned routers. Preserve the Qwen3.8 multimodal/vision contract and the promoted dirty tree.

## Stop state / source hygiene
At 14:00 Aug25 both GPUs are P8/300MHz idle and no NInfer/build/research producer is live. There was **no substantive activity after Aug24 00:28**.

P0-EP was orphaned after its patch script. The live `apps/dual/main.cpp` still contained only the diagnostic teacher-forcing hook while `text_context_impl.h` had already been restored. The adviser session has now restored `apps/dual/main.cpp` exactly from the P0-EP pre-patch backup and left marker `.chatgpt/post_reboot_20260821/p0_ep_dynamic_mlp_oracle/ORPHAN_MAIN_RESTORED`. Do not reapply EP by hand. Use a bounded wrapper with a restore trap if/when the oracle is resumed.

## Current accepted frontiers
- single short: **80.185 useful tok/s median, CERTIFIED/FROZEN**;
- dual short: **69.35 useful tok/s median** promoted exact-BF16 TP2 (six reps 69.17–69.40, canonical digest, 49 rounds /147 drafted /77 accepted /34,27,16);
- populated ~65K: **58.75 useful tok/s** — stale authority, not recertified after exact-BF16 TP2 promotion;
- genuine populated ~262K: **32.76 useful tok/s** — stale authority, not recertified after exact-BF16 TP2 promotion.

The retained `apps/dual/tp2_mlp.cu` supports both verifier widths: T4 (`tokens==4`) and T5 (`tokens==5`) have dedicated Q4/Q5 kernels, and current remote output/reduction is BF16 for both. Therefore the new exact-BF16 transport is directly relevant to the D4/T5 long-context route. Do not assume the old 58.75/32.76 numbers remain current.

## P0-EQ — FIRST rerun the hard long gates on the retained exact-BF16 TP2 source
This is higher priority than more short-context theory.

1. Rebuild normal `ninfer-dual` from the now-clean source and run one short stock smoke to ensure the retained exact-BF16 TP2 still reproduces the 69.35-class canonical route. Do not recert single80.
2. Reuse existing populated snapshots/artifacts; do NOT cold-prefill or regenerate 65K/262K unless the existing snapshot is genuinely incompatible.
3. Re-run the accepted populated65K configuration and genuine261900-token/262K configuration using the same depth/split/context-parallel policies that own their current authorities, but with current retained TP2 source. Preserve dynamic runtime role calibration; never permanently label a physical GPU fast/slow.
4. Report complete useful tok/s, rounds/drafted/accepted/by-pos, stage0/stage1/proposal/fold, TP2 width4/width5 calls, TP2 engine/reduction time, context-GQA packet/helper/merge, loaded clocks, and memory. Confirm determinism/representative output under the corrected fast-MTP behavioral contract.
5. If the current exact-BF16 TP2 improves either long endpoint, promote the new measured long frontier immediately. Do not keep quoting 58.75/32.76 afterward.
6. Vision: text-only long recert is permitted as a performance authority, but the final dual/full-context product must still enable vision per `AGENTS.md`; do not change `vision/*` weights or close the multimodal requirement.

## P0-ER — finish dynamic FFN work only as a **shared T4/T5 verifier oracle**
The orphan P0-EN/EP work was heading toward teacher-forced single-token dynamic MLP masks. That scalar result is NOT enough for NInfer's current speculative verifier because the hot Q4/Q5 kernels are memory-bound and share one weight scan across T4/T5 columns. If one column skips a layer but any other column executes it, the large weight stream still occurs. Therefore the economically meaningful event is: **all live verifier columns skip the same FFN layer**.

Fresh external precedent: FFN-SkipLLM (EMNLP 2024) is a no-retraining, input-adaptive FFN-only skipping method. It intentionally leaves attention/KV intact, identifies protected early/late `cold` regions, measures cosine similarity between the hidden entering an FFN and the residual after that FFN, and once similarity crosses a threshold in the middle region greedily skips subsequent FFNs. The paper reports roughly 25–30% FFN skipping with marginal quality change on knowledge-intensive generation; it also explicitly warns that whole-layer early exit can collapse generation because of KV handling. Source: https://aclanthology.org/2024.emnlp-main.941/ .

This exact method was not found in the NInfer ledger. Use it as a heuristic/oracle source, not as permission to import a framework.

### ER-A: impossible shared-skip ceiling first
1. Reuse existing teacher traces/captured real target states. Capture actual promoted T4 short verifier rounds and T5 long verifier rounds at FFN entry/exit, source-restored afterward.
2. For every column/layer compute `cos(h_pre_ffn, h_post_ffn)` and residual-update norm. Mark a layer shared-eligible only when **every live verifier column** satisfies the criterion. Keep early/late cold regions protected. Warm-up initial generated tokens with full target as FFN-SkipLLM recommends.
3. Give the oracle impossible free gating and no branch divergence. Compute the maximum number of whole FFN layer scans removable per round under shared eligibility. Translate directly into saved measured exact-BF16 TP2 MLP-chain wall.
4. Current short round is roughly stage0 15.429 + stage1 16.765 + proposal 4.041 + fold .327 ms, with ~2.612 useful/round. Reaching100 at unchanged yield needs ~26.12ms/round, i.e. about **10.4ms saving**. The measured all-layer T5 MLP chain is ~16.75ms, so dynamic FFN skipping alone needs on the order of **60%+ shared MLP elimination** to solve short100 by itself. FFN-SkipLLM's published 25–30% is therefore only an incremental ceiling here (~4–5ms) unless combined with another mechanism.
5. Admission for runtime work: require at least **8ms/round impossible shared-skip saving** on representative short T4 (about half the MLP chain) AND no obvious held-out text/multimodal reconstruction cliff. If the shared intersection only gives paper-like 25–30%, record it as an incremental ceiling but do not build a complex dynamic kernel/controller merely to move 69→low80s unless implementation is nearly free.

### ER-B: causal training-free policy only if the oracle passes
6. If ER-A passes, test the literal FFN-SkipLLM-style causal rule: protected cold regions + per-round/per-column cosine saturation; a whole layer is skipped only if all verifier columns agree. No learned router, no per-language thresholds, no task-specific masks.
7. Because deciding `cos(h, h+FFN(h))` naively requires executing the FFN, use the paper's greedy 'trigger then skip subsequent k layers' structure. Do NOT execute an FFN merely to decide to skip that same FFN. Threshold and k are selected on a disjoint calibration set, then frozen globally.
8. Compare against a stronger training-free reference only at oracle level: CLaSp (ACL 2025) dynamically changes skipped layers using hidden states from the last verification stage with no extra training and reports 1.3–1.7x self-speculative speedups. Its draft-model economics are not automatically useful here because factory MTP is already much cheaper; borrow only the in-context hidden-state adaptation idea if it helps predict shared FFN masks. Source: https://aclanthology.org/2025.acl-long.1525/ .
9. Runtime candidate must preserve deterministic state semantics and pass six-family behavioral quality plus the multimodal gate from `AGENTS.md` (OCR/document, chart/diagram, natural image/spatial, multi-image, short video, long-context multimodal retention). Skipping target FFNs changes the shared reasoning backbone and can affect vision even though `vision/*` weights are untouched.
10. Do not call per-column compute savings a target-dense win unless the actual kernel avoids the whole weight scan. Current Q4/Q5 path is bandwidth-dominated.

## Do not reopen
PEARL/PipeInfer, PARD/PARD2/Qwen0.6, accept-all/periodic-anchor quality failures, static/adaptive lossy TP2 packets, lossless TP2 codecs, VPTQ/CodeGEMM/AQLM/QuIP#, plain/calibrated q4down, global W2/W3, TEAL/shared-G64, old vLLM-2080Ti archaeology, or learned dynamic-routing methods (SkipGPT/GateSkip/Dr.LLM/WIDE) without materially new owner-approved evidence. They are closed or violate the no-training contract.

## Priority
1. Rebuild clean source and recert **65K + 262K** with retained exact-BF16 T5 TP2; promote any real long frontier.
2. Then ER-A shared-T4/T5 FFN-skip ceiling. Close cheaply if shared intersection is too small.
3. Only if ER-A is materially positive, ER-B fixed training-free causal policy + six-family + multimodal qualification.
4. Continue autonomously toward 100/100/100; single80 stays frozen.
