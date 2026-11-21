# PLEASE READ THIS — SM75 PERF AUDIT / OWNER NOTE

**Dated:** 2026-08-16 14:09 +07:00 (Asia/Bangkok)

This is **not** an instruction to throw away current work or blindly implement every idea below.
It is a checkpoint/audit written because the owner cannot currently talk to the perf worker.
Please incorporate the considerations into the search strategy when convenient, especially before starting another broad Q4/Q5 parameter sweep.

## TL;DR

1. **Preserve the exact/correct ~70 tok/s lane.** A frozen `exact70-localtop2` checkpoint exists; keep it as the reference before risky experiments.
2. The archive shows that Q4/Q5 have already been tuned extremely aggressively. Many obvious ideas have been tried and failed. Avoid repeating them without new evidence.
3. **The largest conspicuous methodological gap is Nsight Compute (`ncu`) hardware-counter analysis of the dominant Q4/Q5 kernels.** `ncu` is installed. We have lots of `nsys` timing data, but no obvious `.ncu-rep` / hardware-counter report in the perf archive.
4. A second, genuinely different-looking avenue is a **true SM75 software producer/consumer pipeline** (double-buffered shared-memory staging with loader-ish warps overlapping compute), but only pursue it if counters indicate latency/scoreboard/shared-memory staging is actually limiting.
5. The BMMA vocabulary idea is useful but is no longer the main GPU-time consumer. Keep correctness calibration concerns separate from the main Q4/Q5 bottleneck work.

---

## Current evidence worth preserving

Recent exact/correct lane:

- `w2_bmma_top128_localtop2_validation/n1024_0`: ~70.21 tok/s, exact reference output hash.
- `w2_bmma_top128_localtop2_validation/n1024_1`: ~70.03 tok/s, exact reference output hash.
- `rmsnorm_exact_divisor_top128`: ~69.74–70.07 tok/s, exact reference output hash.
- `w2_bmma_top128_localtop1`: ~72.1–72.3 tok/s, **but changed trajectory**; useful as a performance upper signal, not an exact checkpoint.
- `w2_bmma_top112`: one exact-looking ~69.92 run followed by a trajectory-changing ~65.18 run; top-112 is not safe on even the current trajectory.

BMMA rank diagnostic on the inspected 1024-token trajectory:

- records: 1488
- median true-winner approximate rank: 1
- p90: 3
- p95: 4
- p99: 9
- max: 111
- top-64 missed one observed true winner
- top-128 contained all 1488 observed winners

**Do not interpret this as universal top-128 safety.** It is one trajectory. Fine-tunes / different prompts / different hidden-state distributions can change the tail. Eventually this should probably become a calibrated policy (precompiled 128/256/512/etc. variants + a rank-diagnostic/calibration tool), not a claim that 128 is universally exact.

---

## Latest exact-path profile: where GPU time is actually going

From the recent `w2_bmma_top128_localtop2_profile/kernsum.csv` inspection, roughly:

- **Q4 A8 SwiGLU** `q4_a8_swiglu_t4_interleaved_chunk40_to_a8`: ~24.2%
- **Q5 A8 add** `q5_a8_add_split4_interleave_t4`: ~15.4%
- **Q5 value / GDN-ish A8** `q5_value_z_a8<4>`: ~8.7%
- **Q5 rowsplit small-T** `q5_rowsplit_gemm_simt_split2_r2_sm75_kernel`: ~7.6%
- additional Q4 rowsplit kernels: several more percent
- BMMA vocab `project_b2_local_top2`: ~1.3%

So the vocab trick is now a small component. The profiler is again pointing hard at Q4/Q5.

---

## Q4/Q5 avenues already tried — please avoid rediscovering these blindly

The perf archive contains evidence of all/most of the following:

### Layout / data movement

- offline SM75 fragment/interleaved Q4 transforms
- scale transpose / interleaved scale layouts
- wide/coalesced `uint4` / 128-bit loading
- lane/fragment-oriented reshuffles
- cache-policy variants (`ca`, `cg`, `cs`, prefer-L1, other load-hint trials)
- noncoherent-ish / alternate load-policy trials

### Q4 geometry / scheduling

- chunk-size sweeps including 10/20/40/80 (40 was best in the inspected sweep; 80 materially worse)
- multiple row/warp geometries (4/8/16/32-ish families)
- multigroup geometry variants
- register epilogue variants
- fused post-mixer / fused A8 epilogues
- next-weight-fragment register prefetch
- deeper `group+2`-style prefetch (`prefetch2`) — measured worse in the archived sweep
- SASS dumps comparing Q4 generations exist (`q4_*_t4.sass`), so instruction-level inspection has already happened to some extent

### Q5

- split-2 / split-3 / split-4 variants
- two-rows-per-CTA (`r2`) variants
- Q5 scale pair loading (essentially neutral in the inspected full benchmark)
- multiple cache-hint variants
- occupancy / launch-bound experimentation; current source explicitly says deeper pipelines / occupancy caps were measured slower due spills or no gain
- WMMA/MMA output prototypes
- Q5 output shape/tile experiments

### Other related experiments

- fused RMSNorm -> A8 paths
- attention A8 overlays / Q4-only attention experiments
- two-stream GDN experiments (not a durable win)
- DP4A approximate/hybrid proposal experiments
- multiple approximate proposal/rank top-K sizes

In other words: **the easy CUTLASS/Marlin-style checklist has mostly already been independently rediscovered**: offline reshuffle, vector loads, interleaved dequant/math, fused epilogues, cache tuning, multi-warp work, prefetching, shape-specific specialization.

---

# HIGH-VALUE GAP #1: use `ncu` before inventing Q4 kernel #147

`/usr/local/cuda-12.5/bin/ncu` is installed.

I found many Nsight Systems captures (`.nsys-rep`, `.sqlite`, `kernsum.csv`) but no obvious Nsight Compute hardware-counter reports in the perf archive.

At this maturity level, please answer **why** each dominant kernel cannot go faster before another broad rewrite.

Suggested target kernels (exact70/localtop2 lane):

1. `q4_a8_swiglu_t4_interleaved_chunk40_to_a8`
2. `q5_a8_add_split4_interleave_t4`
3. `q5_value_z_a8<4>`
4. `q5_rowsplit_gemm_simt_split2_r2_sm75_kernel`

Questions worth answering with counters:

### Memory / cache

- DRAM throughput as % of peak during each kernel
- bytes actually moved vs theoretical payload
- L2 hit rate
- global load sector/request efficiency
- is the kernel truly bandwidth-saturated or merely latency/issue limited?

### Warp scheduler / stalls

- eligible warps per scheduler
- issued warps / scheduler utilization
- long scoreboard stalls
- short scoreboard / dependency stalls
- barrier stalls
- memory throttle / LG throttle if exposed
- not-selected / insufficient-warp symptoms

### On-chip resources

- registers/thread
- achieved occupancy
- spills / local-memory traffic
- shared-memory throughput
- shared bank conflicts

### Execution pipes

- Tensor Core / MMA pipe utilization
- INT / FP / load-store issue mix
- whether dequant/unpack instruction chains are dependency-bound
- whether SFU (e.g. sigmoid/SwiGLU) is visible enough to matter

The goal is not “collect every metric.” The goal is to classify each dominant kernel as something like:

- DRAM-bandwidth-bound
- DRAM-latency / insufficient-outstanding-load-bound
- shared-memory-bound
- register/occupancy-bound
- dependency-chain / instruction-latency-bound
- Tensor/INT issue-bound
- barrier/synchronization-bound

Then only write a new architecture that directly attacks the measured wall.

---

# HIGH-VALUE GAP #2: true SM75 producer/consumer software pipeline (conditional)

I do **not** see an obvious SM75 implementation where some warps load the *next* shared-memory activation tile while other warps compute the current tile, with double-buffered shared storage and producer/consumer synchronization.

The current Q4 T4 structure looks broadly like:

1. copy activation chunk to shared
2. `__syncthreads()`
3. compute chunk (with useful next-weight register prefetch)
4. `__syncthreads()`
5. load next activation chunk

That is good code, but it is not the same as overlapping the *shared-memory staging* of N+1 with MMA work on N.

A genuinely different SM75 dataflow would be roughly:

- shared buffer A = current activation chunk
- shared buffer B = next activation chunk
- loader-ish warp(s) / phase issue ordinary global loads + shared stores for B
- compute warps consume A
- synchronize producer/consumer safely
- swap A/B

This is the software analogue of the `cp.async` conveyor-belt idea, but it costs real Turing warp/register/issue resources.

**Do not implement this just because it sounds clever.** First use `ncu` to see whether the current Q4 kernel has enough memory-latency / scoreboard / barrier headroom for this to plausibly win. If Q4 is already saturating DRAM or another pipe, producer warps may only make it worse.

---

## Q5 bit-decomposition idea: already ACTIVE — do not duplicate

While this audit was being performed, fresh artifacts appeared:

- `q5_down_q4base_residual_trial.cu`
- `q5_down_q4base_biascorrect_trial.cu`

The conceptual idea is the right kind of structural experiment:

`Q5 ~= Q4-like base + extra high-bit correction`

Potentially use a faster base path plus an exact/cheap correction instead of treating Q5 entirely as scalar-ish 5-bit unpack math.

The first naive q4base residual trial produced roughly:

- 66.57 tok/s
- MTP acceptance ~50.25%
- changed trajectory / quality behavior

So the naive approximation is not acceptable. A bias-corrected follow-up was already being built/tested around 14:00.

**Please do not rediscover/repeat the naive trial.** Continue it only if the exact correction can be made both fast and behavior-preserving.

---

## Lower-priority / speculative considerations

### SM work distribution / striping

Marlin-style striped scheduling is a known technique for keeping SMs evenly occupied on awkward shapes. It does not look like the highest-probability gap here because these Q4/Q5 matrices already expose lots of output-row CTAs, but `ncu` scheduler/load-balance evidence could change that conclusion.

### Full cross-kernel fusion

Be skeptical of fusing Q4 SwiGLU output directly into Q5 down merely to save the A8 intermediate write/read. The intermediate activation is tiny compared with the weight streams; a giant fusion can easily lose occupancy/scheduling flexibility for negligible DRAM savings. Only pursue with measured evidence.

### Calibration / portability of BMMA vocab path

Longer-term, consider preserving the candidate count as a compile-time/template policy with multiple compiled variants (e.g. 128/256/512) and a calibration/diagnostic tool that measures exact-winner rank on a corpus. This would make the idea usable for fine-tunes or other compatible models without pretending one cutoff is universally safe.

---

# Suggested immediate workflow

If there is no currently-promising experiment in flight:

1. Preserve/freeze the current exact70/localtop2 state.
2. Run a **bounded `ncu` pass** on the 4 dominant kernels above, ideally on a short repeatable workload that still captures the real T4 decode path.
3. Write down one-line bottleneck classifications for each kernel.
4. Choose **one structural experiment** justified by those counters.
5. Validate exact trajectory / repeated determinism before accepting the speedup.
6. Keep the BMMA calibration concern separate from core Q4/Q5 tuning.

If a promising experiment **is** already in flight, finish/measure it first; this note should not interrupt a coherent hypothesis test.

---

## Why this note exists

The owner asked for an attention-catching note because they cannot communicate with the perf worker directly right now. The goal is to reduce duplicate search and encourage evidence-driven exploration now that the SM75 port is in the long-tail optimization phase.

The current work is already far past basic tuning. Reaching ~70 exact tok/s from the original state is strong evidence that the port is extracting real TU102 capability. The next gains are increasingly likely to require either:

- hardware-counter-guided identification of an unrecognized stall/resource wall, or
- a genuinely different dataflow/algorithmic trick,

rather than another blind parameter sweep.
