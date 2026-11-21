# STOP — no workload-curated proposal/token hacks; return to portable grunt work

Owner decision: the prompt-derived 131K proposal-vocabulary rebuild is **diagnostic/research-only**. Do not promote it as the route to the product performance goals.

The multilingual coverage finding is useful evidence that the existing proposal shortlist is biased. Preserve the measurements. But STOP doing any of the following as production optimization:
- adding/removing proposal token IDs because they occurred in our train/qualification prompts;
- language/domain token banks selected from benchmark/workload fixtures;
- token rescue lists, prompt-specific rows, or benchmark-shaped candidate tables;
- claiming useful-tok/s gains from a proposal vocabulary fitted to our observed workload distribution;
- any target/checkpoint modification whose performance depends on knowing the qualification prompt distribution.

The target verifier may preserve target semantics, but this is still workload-shaped accelerator tuning and is not the owner's desired path. Qualification prompts must remain qualification, not optimization input.

Existing `.chatgpt/int8_tu102/proposal_rebuild/` artifacts may remain as evidence only. Do not spend more GPU/engineering time enlarging or polishing that corpus-driven shortlist unless the owner explicitly reopens it.

## Return to grunt work / portable mechanisms

Prioritize evidence-backed work that does not depend on handcrafted token IDs or workload fixtures.

### Single GPU >=80 useful tok/s on INT8 KV
1. Re-port/rebenchmark the archived **Q4/B2 proposal screen -> top-K -> exact W8 target-head rerank** on the current trusted single-GPU runtime, WITHOUT changing the existing proposal token map. This changes proposal mechanics, not workload vocabulary. Same fixed candidate map for baseline/candidate. Target output correctness remains canonical.
2. Re-open **attention-input A8** on the current INT8-KV base because the old experiments predated the present runtime. Require same target digest/acceptance for an exact-engine claim.
3. NCU/SASS grunt work on dominant Q4/Q5/W8 kernels on the normally boosting Core-X card: DRAM bytes, long-scoreboard, register pressure, issue-pipe balance. Try only changes justified by counters.
4. Turing-specific low-risk experiments: INT32/FP32 pipe overlap in Q4/Q5 unpack+scale loops; per-kernel L1/shared carveout A/B; T4/T8 geometry/weight reuse where applicable.
5. Drafter-only low-bit kernel work is allowed if it is **model-/prompt-agnostic** (e.g. native SM75 W4A4 MTP body using weights/format rules, not token tables). Report proposal cost + acceptance separately from target engine speed.
6. BF16-source checkpoint co-design may use the model weights themselves and generic quantization mathematics. Do not tune codebooks/bit allocation against the qualification prompt set. Any activation-calibration campaign must use a clearly separate broad calibration source and held-out qualification; do not use this as an excuse for workload fitting.

### Dual one-request >=100 useful tok/s at populated ~262K
1. Build the bounded **TP2 T4 MLP microbenchmark** first: two half-row gate/up shards + matching down partials, both GPUs concurrent, including real host-staged activation/reduction latency. Compare wall time against the production one-GPU layer. Account for the current physical-card clock asymmetry; do not treat a 50/50 split as universal.
2. Profile populated 8K/32K/64K/128K/~262K target timing before architecture: full-attention KV scan vs GDN/weights vs head vs proposal vs boundaries.
3. If long-context attention dominates, test exact **context/head parallelism** for the 16 full-attention layers. Keep GDN recurrent state separate.
4. Shallow tree/pseudo-tree verification is allowed as a generic algorithm/hardware-utilization experiment; do not derive branches from handcrafted token lists.
5. Placement/split/head-owner/proposal-owner values are calibrated policy, not model constants. Follow `PLEASE_READ_THIS_2026-08-18_PORTABLE_MECHANISMS_CALIBRATED_POLICY.md`.

## Benchmark discipline
- Use the normally boosting Core-X card for the single-GPU frontier and record UUID + loaded clocks.
- Same candidate/reference physical device assignment for software A/Bs.
- Keep execution efficiency separate from speculative acceptance/trajectory effects.
- No benchmark prompt may become a source of token IDs, rescue rows, codebooks, or placement rules.

The immediate intent is simple: **stop clever workload-fitting and do the hard portable engineering.**
