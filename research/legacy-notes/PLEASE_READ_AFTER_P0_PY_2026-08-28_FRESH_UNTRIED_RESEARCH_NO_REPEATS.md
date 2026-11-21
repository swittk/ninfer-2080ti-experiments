# READ AFTER P0-PY — FRESH UNTRIED RESEARCH ONLY / DO NOT REPEAT CLOSED ROUTES

Owner instruction: **do not reopen old experiments under new names.** The current representative262 exact control is ~86.46 tok/s, digest `cb7cda10a3883e76`, 48 rounds, 191 drafted / 188 accepted, positions 47,47,47,47. Acceptance is already saturated; the remaining problem is target-pass cost / dataflow.

## FIRST: finish P0-PY correctly before inventing another branch
P0-PY `stage0 proposal` did **not** fail performance or quality. It failed setup because the changed stage0 state layout made the existing prefix snapshot incompatible:

`prefix snapshot does not match current state layout`

Rebuild/re-capture a compatible representative262 prefix snapshot for the candidate state layout and run the real paired control/candidate. Preserve the exact PA/PW environment. Do not classify P0-PY until that E2E exists.

Why this is worth finishing: current 240-token control decode is ~2.78 s. 100 tok/s is 2.40 s, so the gap is only ~0.38 s. Proposal currently costs ~0.326 s over 48 rounds. Even perfect proposal hiding alone nearly closes the whole gap; one additional modest kernel win could cross 100.

Current PA control timing authority:
- target stage0 ~1.270 s
- target stage1/head ~1.028 s
- proposal ~0.326 s
- fold ~0.018 s
- context-GQA packet ~0.364 s
- context-GQA helper ~0.610 s
- TP2 MLP engine ~1.244 s

## FRESH MECHANISM A — native packed INT4 tensor-core dense Q4 on SM75
This is **not** the failed native-S4 KV/QK work. That route changed attention/KV arithmetic. This route keeps target weights/semantics and attacks the dense Q4 target kernels.

Current production SM75 Q4/Q5 target kernels still expand packed Q4/Q5 codes and call `mma_s8`; source inspection finds no production `.m8n8k32.s4/u4` dense target path.

Bounded preflight only:
1. Compile a tiny SM75 PTX canary for native `mma.sync.aligned.m8n8k32...s4/u4...` and verify the actual SASS opcode on TU102. Do not trust mnemonic assumptions without SASS.
2. Prototype only the hottest exact Q4 target shape first: verifier T5 gate/up (`K=5120, N=34816`), preserving existing group64 scales and output arithmetic.
3. Prefer decomposing A8 activations into signed/unsigned 4-bit planes and consuming packed Q4 weights directly, rather than expanding every Q4 nibble to S8. Recombine integer dot products exactly before existing scale/epilogue.
4. Microbench graph-warm against current `q4_linear_swiglu_a8_sm75` on both cards.
5. Kill immediately unless the complete Q4 op gains roughly >=8% and numerical output is exact/tolerance-equivalent under the normal operator contract.
6. Do **not** touch Q5 until Q4 shows a real kernel win.

Research basis: NVIDIA PTX/WMMA SM75 supports sub-byte INT4 tensor-core MMA. This is a hardware path we are not currently using for dense Q4 target execution.

## FRESH MECHANISM B — INT8 PV arithmetic while KEEPING CURRENT INT8-G64 KV STORAGE
Do **not** reopen K4/V4/K4V2, sidecars, eviction, sink/tail, or low-bit cache layouts. Keep the already-qualified INT8-G64 KV cache byte-for-byte.

Current GQA kernel already does native INT8 QK, but PV still dequantizes V to BF16 and runs BF16 P×V MMA. Test whether PV itself can become INT8:

For V group `g` and key `k`, fold the stored V scale into the probability:

`Pprime[k,g] = P[k] * Vscale[k,g]`

Then quantize the small `Pprime` tile/group to INT8 and compute native S8 tensor-core `Pprime_i8 × Vcode_i8`, followed by the probability quantization scale. This removes V->BF16 staging/dequant and BF16 PV MMA while leaving KV storage unchanged.

**Offline oracle first — no CUDA until it passes.**
- Capture actual representative262 softmax P tiles plus existing V codes/scales across all 16 attention layers.
- Quantize Pprime per row/group with several bounded scale choices.
- Compare attention output cosine/rel-L2 and downstream target/logit/token stability.
- Include the real T4/T5 verifier widths, not only T1.
- Reject if quality is not extremely clean or if an idealized latency/byte model cannot plausibly save several ms per round.

Research inspiration: SageAttention-family low-precision attention arithmetic. Do not port kernels literally; adapt the arithmetic to SM75 INT8 tensor cores and our existing INT8 V representation.

## AFTER 262K NATURAL FRONTIER — 8-user serving, fresh direction only
The CLI accepts up to 8 sessions, but optimized MTP batching is currently specialized around B2/B4. Do not extrapolate single-user 100+ and call B8 done.

Fresh serving work to investigate later:
- PersistentKV-style compact workqueue for mixed-length B4/B8 decode: schedule only nonempty `(request, kv-head, sequence-split)` tasks instead of static rectangular grids. This is different from the existing exact sequence splitting; the novelty is request-aware persistent work scheduling.
- Helix-style **batchwise** overlap: overlap communication / stage work across independent requests. Do NOT recreate the old thousands-of-fine-event per-layer DAG that already regressed.
- Measure 1/2/4/8 users at 2K, 8K, 16K/32K: per-user tok/s, aggregate tok/s, TTFT, p95 inter-token gap, fairness, and memory.

## Explicit DO-NOT-REOPEN list
Do not spend time on any of these without a materially new mechanism and an explicit explanation of why the previous closure does not apply:
- generic K4/V4/K4V2 KV compression, layerwise K8V4, key-channel sidecars, outlier-token sidecars
- sink+tail / StreamingLLM / prefix summaries / page top-k / semantic retrieval / SnapKV/PyramidKV/LServe/Quest-like eviction
- native S4 **QK/KV** routes already tested (P0-PI/PJ)
- LeanAttention / Flash-Decoding-style sequence splitting by itself: current GQA already splits sequence across CTAs and exactly reduces online-softmax states
- columnwise T5 draft/final KV ranges, serialized or fused
- canonical-T5 draft/final shortened ranges
- direct-global-V / compact-direct-V
- Wc16/Bc16 occupancy retunes already closed
- graph replay / self-launch / generic issue-surface cleanup
- ordinary D4/D5/tree/Bole/PEARL/fifth-proposal work
- static/shared FFN skip, final-column FFN skip, alternating FFN delta reuse
- full target replica / generic full-layer TP
- fine-grained per-layer event DAG overlap

## Promotion discipline
For any candidate, report:
- exact command/env and artifact
- loaded clocks / UUIDs
- representative262 useful tok/s
- target0 / target1 / proposal / fold
- context-GQA packet/helper and helper bytes
- TP2 calls/engine time where relevant
- rounds/drafted/accepted/by-position
- deterministic digest/stdout hash
- exact/Q6/REP/EXP classification without blurring them

Do not call an oracle or smoke test a frontier. Do not hardcode prompt/token rescue behavior. No training/finetuning/distillation/LoRA.
