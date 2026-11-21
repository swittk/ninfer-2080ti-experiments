# 19:15 adviser — finish adaptive remote TP2 boundedly, then PEARL schedule oracle

Read `AGENTS.md` first. Stock GPUs only. No hardware tuning/reset/reboot. No coding agents. No local model training. Preserve the multimodal/vision contract now in AGENTS. Do not reset the promoted dirty tree.

## Current stop / authority
At 19:09 Aug23 both GPUs are stock-idle; no NInfer/build/research producer. Last substantive work is `p0_dq_adaptive_tp2/japanese_sweep.json` at ~15:36. Hard authority remains **80.185 / 65.910 / 58.75 / 32.76** useful tok/s. Split40 reached 66.04 vs split32 65.95 in one paired campaign but was classified negligible and is not the default frontier.

Recent target PTQ results are closed:
- calibrated BF16->Q4G64 MLP-down (GPTQ/AWQ/AutoRound-like code/scale selection) only reduced the old clipped-Q4 excess reconstruction gap to ~0.714x broad held-out vs required <=0.5x; no artifact/E2E.
- fixed H64 QuaRot had one genuinely good layer58 row (`h64_awq_round`: text excess0.208, vision0.178, Q5 proximity1.233x) but the broader multimodal reconstruction gate failed; no global/subset promotion.
- graph tracing shows target Q4/Q5 dense work is 86.8-91.1% of stage graph child-kernel time; no hidden orchestration bubble.
- Shannon/lossless coding of all target low-bit payload can save only ~1.874ms/round ideally; not a dual100 path.

## First: finish P0-DQ adaptive remote-only TP2, but only to decision
Static remote-only A4/A6/A8 schedules are already closed. DQ is a distinct per-layer/per-round precision controller and stopped mid Japanese discriminator.

Current Japanese sweep:
- EF threshold0.06: avg packet **13,807 B**, common prefix95/128, LCS0.8281; wire deadline fails.
- EF 0.04: 16,319 B; <=0.03 mostly A6/A8 and worse wire.
- plain0.06: 13,810 B, prefix81/LCS0.8125; worse than EF.
- measured budget from prior remote-only lane is ~**11,840 B average packet** for the useful wire deadline.

Do NOT restart a dense threshold zoo. Continue **error-feedback mode only** at threshold 0.08 and 0.10. If 0.10 still exceeds 11,840 B, test one 0.12 endpoint and stop. The first point that satisfies the wire budget gets a Japanese quality decision. If no deadline-fitting point preserves representative Japanese quality, write `CONCLUSION.md` and close DQ. Only if Japanese passes both wire+quality may one fixed threshold advance to the six-family canary. No per-language policy and no benchmark-tuned thresholds.

## NEW P0-DR — PEARL schedule oracle with factory MTP (NO runtime first)
PEARL: *Parallel Speculative Decoding with Adaptive Draft Length*, ICLR 2025. Official paper/code describe a **lossless, training-free, no-extra-model-memory scheduling framework** for existing draft-then-verify systems. It uses:
- **pre-verify**: target verifies the first draft token while the drafter continues generating the rest; an early rejection avoids wasted full verification;
- **post-verify**: while target verifies the current draft block, drafter continues speculating beyond it assuming full acceptance; if the current block fully survives, those drafts are already available.
Sources:
- https://proceedings.iclr.cc/paper_files/paper/2025/hash/03b1043052700b1a471996b0baf309d4-Abstract-Conference.html
- https://github.com/smart-lty/ParallelSpeculativeDecoding
- https://arxiv.org/abs/2408.11850

This mechanism is NOT automatically exempt from prior PipeInfer closure. The existing P0-C PipeInfer model speculated a **future target stage0** during current stage1 and charged correction bubbles; PEARL primarily overlaps **continued draft work and early first-token verification**. With our factory MTP, some or all economics may collapse to already-measured bounds. Prove distinction with a trace oracle before code.

### Oracle contract
1. Use existing exact six-family factory-MTP traces and current promoted D3 acceptance/full-survival. No new model weights.
2. Use measured current dual costs: split32 target stage0/stage1 around 16.7/18.2ms, proposal around 4.0ms on canonical indexed dual (or current exact measured authority if a newer equivalent trace exists), fold ~0.34ms. Keep physical role asymmetry measured, not permanent.
3. Model PEARL **pre-verify** exactly: determine whether the target root/first-token logit is already available from the prior committed transaction. Do not charge or delete a target pass by assumption. If NInfer already has that exact target distribution at the root, record pre-verify as logically free; otherwise charge the actual T1/T2 route needed.
4. Model **post-verify** using the factory MTP's real recursive state machine: during current target verification, generate x4..x6 (or one bounded continuation block) from the assumed-accepted x1..x3 branch. Charge the proposer on the GPU/time window where it can actually run. It may overlap stage1 only if its weights/state are resident on the idle stage0 GPU and no required input is unavailable. Include one-time replication memory (~hundreds MiB) but do not invent P2P.
5. Use measured D3 full-survival probability and captured D4/D5 continuation acceptance where available. If a post-verified continuation is invalidated by an earlier rejection, discard it exactly as PEARL does. No clairvoyant branch selection.
6. Compare against the prior factory-MTP PipeInfer `free proposal + fold ~92.7 t/s` closure. If PEARL's state/dependency schedule reduces mathematically to that bound, close immediately and explicitly; do not implement under a new name.
7. Compute finite 128 and 256 token complete-path ceilings, not only infinite horizon. Admission requires **>=105 t/s optimistic** at 128 and 256 before runtime work, leaving at least ~5% implementation margin over the 100 gate. If <105, close.
8. If short oracle passes, immediately cost populated65K and genuine262K using their real stage/target costs. Proposal overlap does not solve long target weight/KV floors by itself.

### Vision/multimodal
PEARL is a scheduling change around factory MTP and canonical target verification, so it must reuse `mtp_bridge_multimodal` / Vision-prefill state rather than introduce a text-only speculation path. No `vision/*` weight changes. A runtime candidate, if ever admitted, needs at least one image + long-context multimodal state/rollback smoke before promotion. Hard decode gates do not change for vision requests.

## Research closure hygiene
Do not reopen VPTQ/CodeGEMM, PARD/PARD2, q-distribution/PipeInfer, ASD/regret, Oilbird, plain q4down, global W2/W3, TEAL/shared-G64, static remote TP2, or old vLLM-2080Ti archaeology. PEARL advances only if its exact dependency schedule is genuinely different from the already-closed PipeInfer oracle.

## Priority
1. Finish DQ with EF thresholds 0.08/0.10 (+0.12 only if needed for wire crossing), then close or six-family once.
2. P0-DR PEARL trace/schedule oracle, no runtime implementation first.
3. If PEARL <105 optimistic finite short, close and return to target-dense/dataflow research rather than another speculative-decoding synonym.
4. If >=105, implement minimal clean schedule, certify six-family + stock dual short, then 65K/262K and multimodal state semantics.
