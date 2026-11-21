# OWNER ADVISER — 2026-08-21 09:55 +07 — unblock after short-C1 closures

Read repo-root `AGENTS.md` first. All hard gates and NO-agent / NO-training rules remain absolute.

## Current verified state

- Single-owner short Qwen3.8 factory-MTP depth3: ~76.6-76.7 useful t/s on the fixed short workload. Single >=80 is NOT met, but the gap is now only ~4.4%.
- Representative six-family short D3 mean: 68.25 t/s; code 81.0, JSON 88.2, general 71.2, Thai 47.7, Japanese 62.3, Arabic 59.1. Never quote code/JSON alone as frontier.
- Layer-sharded short dual: ~65.3 t/s. Full-owner + host-staged TP2 helper: 66.37 t/s and CLOSED; owner-only 76.63 is faster.
- Populated65K INT8 physical-KVP + T5 TP: 58.75 useful t/s. 65K >=100 remains unmet.
- 262K physical INT8 capacity works; 262K >=100 remains unmet.
- Device-resident accept/propose control: CLOSED as engine gain.
- Fixed MTP D4/D5, entropy/adaptive width, short one-branch dynamic tree: CLOSED for current factory proposal distribution/cost.
- Exact Qwen3.8 DFlash2: CLOSED for current NInfer integration economics.
- FlashQLA ReplaySSM T4/T5 adaptation: CLOSED; exact-ish savings are far below the ~3.6 ms/round required even for single80.

## IMPORTANT CORRECTION: previous `Marlin CLOSED` statement is too broad

The 06:41 experiment correctly proved **stock SpecMQuant GPTQ-Marlin / stock AWQ GEMM** are SM80-oriented and that old AWQ G64 GEMV is slower than NInfer. It did NOT test the actual maintained `weicj/vLLM-2080Ti-Definitive` dense SM75 path.

Current external real-hardware documentation explicitly says the weicj fork has a patched `sm_75` dense Marlin path that is genuinely fast; on dense AWQ, selecting plain AWQ instead of the fork's `awq_marlin` path was measured ~7.8x slower in one published Turing recipe. The same maintained fork is behind the documented dual-2080Ti Qwen3.6 AWQ + factory MTP K3 101.3 t/s 4K result.

Therefore DO NOT reopen local tile/warp experiments, but REOPEN **patched-SM75-dense-Marlin as an external executable/source oracle only**.

Next discriminator:
1. Pin/clone `weicj/vLLM-2080Ti-Definitive` (current branch/release; changelog v0.1.15 dated 2026-08-15 validates official Qwen3.8-27B-FP8 on dual SM75).
2. Locate the exact dense AWQ/Marlin path actually selected on SM75. Do not infer from upstream Marlin.
3. Benchmark or extract same-shape single-GPU dense-linear timings at Qwen27 shapes/T1/T4 where possible. If source build is needed, this is an oracle build, not a replacement-engine project.
4. Compare against NInfer complete buckets, not one microkernel. Require a modeled >=8-10% complete-round saving before porting any mechanism.
5. If the fork's `awq_marlin` is only naming/fallback and does not beat NInfer on real shapes, close it with direct evidence. If it materially wins, source-audit the Turing-specific dataflow and port the mechanism, not the framework.

## Stronger system oracle: run the maintained SM75 fork on OUR topology

The fork now claims official Qwen3.8-27B-FP8 end-to-end validation on dual RTX2080Ti (2026-08-15). Use it as a black-box hardware/system oracle with professionally published weights only:
- one GPU, noMTP and native MTP3 if supported;
- two GPUs on our actual internal+eGPU/no-P2P topology;
- short 4K-ish request first; 65K only if short is informative.

This does NOT change NInfer's groupwise-int target contract. It answers: how much of our ~76.6 single / ~65 dual deficit is NInfer versus topology versus checkpoint/kernel family? Record loaded clocks because boost asymmetry is runtime state.

Public reference to keep in mind: the NVLink dual2080Ti Qwen3.6 route reports 101.3 t/s peak at 4K and 55.3 t/s at64K. Another GPTQ-INT4 MTP3 sweep reports 85.5 t/s at ~65K with FP16 KV. These are mechanism/topology oracles, not our gates.

## NEW architecture oracle for the second GPU: async speculative parallelism

Our no-P2P topology makes per-layer TP coordination toxic. Do not keep bolting helpers onto every target layer. Research/code now exists for **Speculative Speculative Decoding (Saguaro/SSD, ICLR 2026)** and **SpecBranch (ICLR 2026)**:
- SSD runs drafting/predrafting concurrently with target verification and caches likely next verification outcomes; official code: `tanishqkumar/ssd`. Paper reports up to 2x over optimized speculative decoding.
- SpecBranch uses rollback-aware parallel speculative branches; official code: `Sylvan820/Specbranch`, reported 1.8-4.5x over autoregressive baselines while preserving the verifier distribution.

Do NOT implement either wholesale. First build a trace/ceiling model from our existing factory-MTP D3 traces and measured 38-40 ms short rounds:
- model a second full-target/speculation worker doing next-outcome/branch work concurrently rather than host-staged layer TP;
- include actual factory-MTP outcome frequencies and rollback/branch hit probability;
- account for our second card's *measured current* loaded clock, not a permanent slow-card assumption;
- hard implementation gate: the optimistic architecture must show >=1.31x over the 76.6 single-owner short baseline (enough to clear100) with realistic communication and memory.

Factory MTP's dependency on target hidden state means SSD cannot be copied naïvely. Treat Saguaro/SpecBranch as scheduling algorithms/oracles. NInfer's ReplaySSM/branch state machinery may make rollback/branch execution easier than in generic transformer engines.

## Professionally trained external drafter option — only if cheap discriminator

`RadixArk/Qwen3.8-27B-DSpark` is now public and SGLang has native DSpark support. It is an upstream professionally trained 1.36B, 5-layer Qwen3.8 speculator with a confidence head and block size7. This is ALLOWED by the owner rule (upstream published weights), unlike homemade training.

Do not integrate blindly. Only run a bounded SM75/source-cost discriminator if the Saguaro/SpecBranch or target-engine work needs a better independent draft source. DFlash2 already showed that a good external drafter can still lose when draft cost + target verify cost are too high.

## What remains CLOSED

- same-family Q4/Q5 tile/warp/cache/prefetch sweeps;
- stock SpecMQuant Marlin as an SM75 path;
- old AWQ G64 GEMV;
- per-layer host-staged TP2 helper/coordinator;
- generic event DAG/coarse scheduler already measured;
- local model/MTP/draft training of any kind;
- Codex/Claude/OpenCode/sub-agents.

## Immediate execution order

1. Read `AGENTS.md`; confirm no agent/training process.
2. Ensure source remains reverted/clean except owner docs.
3. External oracle A: actual `weicj/vLLM-2080Ti-Definitive` patched dense SM75 path + current Qwen3.8 runtime on our hardware. Classify with real timings.
4. In parallel/on paper: SSD/SpecBranch trace-only short-dual ceiling using existing D3 traces. No runtime code unless it can clear100 with margin.
5. If neither passes, inspect professionally published Qwen3.8 DSpark only as a bounded proposer-cost/yield oracle; no training.
6. Recompute single80 / dual100 / 65K100 / 262K100 after every material result and continue.

Do not stop because the previous local kernel families closed. They closed hypotheses, not the owner goals.
