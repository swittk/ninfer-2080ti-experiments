# OWNER TRAJECTORY — KERNEL MAX FIRST, THEN FULL-CONTEXT DUAL GPU

**Owner direction — 2026-08-17 10:51 +07**

The owner is going to be unavailable for a while. Continue autonomously. This note is intentionally **not** a rigid implementation recipe. Use judgment, follow evidence, preserve good checkpoints, and do not stop merely because an experiment is not bit-for-bit identical to some older path.

Read the recent context notes first, especially:

- `PLEASE_READ_THIS_2026-08-17_0830_STOP_CHASING_MTP_T1_BITWISE_PARITY.md`
- `PLEASE_READ_THIS_2026-08-17_0845_UPSTREAM_CORRECTNESS_HEURISTICS_NOT_BITWISE_RELIGION.md`

This note supersedes any overly restrictive interpretation of earlier instructions.

---

# THE OWNER'S ACTUAL GOAL

The desired trajectory is:

1. **Get as much useful single-GPU performance as reasonably possible from the SM75 / RTX 2080 Ti implementation.**
2. Be willing to use aggressive kernel engineering and unusual low-bit / approximate techniques if they are **safe enough under a correctness standard comparable to the original NINFER fast path**, rather than demanding impossible bitwise identity everywhere.
3. Do not let MTP parity archaeology consume the project. MTP is worth opportunistic work if there is a clear payoff, but **kernel performance is the immediate priority**.
4. Once the meaningful single-GPU kernel opportunities have been harvested or convincingly exhausted, **move hard to dual GPU**.
5. Dual-GPU milestone #1 is **maximum usable context, ideally the complete Qwen3.8 native 262,144-token context**.
6. Only after full context works, optimize **combined inference / pipeline scheduling / multi-user aggregate throughput**.

Do not stop after merely proving that an experiment cannot be promoted. A rejected candidate is a result, not completion of the project.

---

# 1. SINGLE-GPU PERFORMANCE: GO HARD, BUT USE EVIDENCE

The established reliability baseline is:

`38369da — fix(sm75): harden q45g graph replay determinism`

The current empirically validated Qwen3.8 MTP3/B2 performance lane is around **~72 tok/s** on the canonical 1024-token workload. This remains a useful performance reference, not a religiously immutable output oracle.

The truly full-exact proposal + target-head diagnostic measured about **61.24–61.30 tok/s**, showing that the empirical B2 shortcuts are worth roughly 10–11 tok/s on this checkpoint.

## NCU has already identified the most useful kernel targets

Current measured classification:

- **Q4 A8 SwiGLU**: ~549 GB/s effective DRAM, ~67.5% long-scoreboard, near-full occupancy, very low L2 reuse.
- **Q5 A8 down/add**: ~535 GB/s, ~55.8% long-scoreboard.
- **GDN Q5 A8 value/z**: ~480 GB/s, ~42% long-scoreboard plus other latency.
- **Q5 6144 SIMT output**: ~340 GB/s, high L2 hit, only ~13.9% long-scoreboard; this is a different bottleneck class.

This means the next work should be driven by counters/SASS/source correlation rather than another giant blind shape sweep.

### Start with Q4 A8 SwiGLU, but do not artificially limit yourself to one preconceived implementation

Useful avenues include, but are not limited to:

- increase weight-load-to-use distance;
- register or shared-memory double buffering;
- software pipelining across groups/tiles;
- more independent outstanding global loads;
- producer/consumer organization if it actually maps well to SM75;
- different CTA/warp responsibility splits if NCU supports the hypothesis;
- load packing / vectorization / memory transaction improvements;
- reducing redundant loads, transforms, or dequant work;
- exploiting reuse across verification columns;
- fusion where it eliminates meaningful traffic rather than only launches;
- schedule changes that reduce long-scoreboard without exploding registers;
- architectural rewrites suggested by SASS/source correlation.

Turing does not have `cp.async` or TMA, so do not blindly transplant an Ampere/Hopper pattern. But **do not interpret that as a ban on producer/consumer or software-pipeline experimentation**. If a Turing-appropriate implementation overlaps real work and wins end to end, keep it.

If Q4 work demonstrates a real reusable pattern, try it on Q5-down. If Q4 fails, use the failure evidence to decide whether Q5 deserves an independent idea.

Treat Q5-6144 separately because its counters say it is not mainly the same DRAM/long-scoreboard problem. Profile its actual scheduler/math/dependency bottleneck before optimizing it.

## Do not judge success only by isolated kernel microseconds

For every serious candidate, measure:

- target kernel latency;
- relevant NCU counters before/after;
- full 256/1024 decode throughput on the intended lane;
- acceptance / round statistics where relevant;
- memory footprint / graph impact;
- determinism/reproducibility.

A kernel can become prettier while the model gets slower. End-to-end wins are what matter.

---

# 2. CORRECTNESS: USE AN UPSTREAM-LIKE CONTRACT, NOT BITWISE RELIGION

**Do not require every optimized kernel or batched-MTP path to be bit-for-bit identical to ordinary T1. Original NINFER itself did not ultimately require that.**

Historical upstream NINFER explicitly documented that batched target verify uses different small-T kernels/reduction order from T1 decode and can legitimately flip an argmax at a near-tie. The old strict-sequential path existed as an algorithm/state debugging oracle, not as the permanent arithmetic implementation for production.

The correctness hierarchy for this project should therefore be:

## Hard failures — these are real bugs

Examples:

- wrong causal mask or RoPE/cache position;
- wrong KV append/rewind/commit;
- wrong GDN recurrent or convolution state selection/rollback;
- invalid snapshot/slot addressing;
- cross-column contamination;
- stale/uninitialized memory dependence;
- out-of-bounds/tail/indexing error;
- nondeterminism for identical deterministic runs caused by races/stale state;
- operator output outside the established upstream-style numerical oracle/tolerance for its compute class;
- a fast-path argmax flip with a **healthy reference margin** that cannot plausibly be explained by the measured numerical perturbation;
- approximate shortlist dropping a clearly superior true winner without an explicit safe fallback/qualification contract.

Fix these.

## Acceptable numerical differences

These are **not automatically bugs**:

- BF16 low-bit differences caused by a different legal reduction order;
- tensor-core versus SIMT accumulation differences that remain within the established operator numerical envelope;
- width-dependent batched-kernel output bits when all state/index/mask invariants are correct;
- a different greedy token at the first sequence fork when the competing logits are genuinely near-tied and the measured numerical movement is sufficient to explain the order flip.

Do not spend hours forcing a fast T4/T5/T6 path through T1 arithmetic just to obtain the same SHA.

## Preferred automated divergence classification

When a fast path and a trusted reference first choose different tokens `A` and `B`, capture both paths' logits at that exact shared history.

Let:

`reference_margin = L_ref[A] - L_ref[B]`

and estimate the actual perturbation capable of swapping them from the measured movement of those logits, for example:

`flip_budget = |L_fast[A]-L_ref[A]| + |L_fast[B]-L_ref[B]|`

If the reference margin is comparable to or below the observed numerical flip budget, while the underlying operators/state invariants pass their normal numerical/state checks, classify the sequence fork as **numerically explainable near-tie drift**, not a correctness blocker.

If the reference margin is much larger than the observed perturbation, investigate a real bug.

Do not invent a magic universal logit threshold if measured perturbation can give a better case-specific answer.

## Canonical SHA policy

The canonical SHA/trajectory remains valuable as:

- a strong smoke test;
- a regression alarm;
- a way to detect unintended changes quickly.

It is **not the universal definition of mathematical correctness** for every future optimized kernel.

---

# 3. AGGRESSIVE LOW-BIT / APPROXIMATE IDEAS ARE ALLOWED

The owner explicitly does **not** want the project to become so conservative that all interesting low-bit ideas are forbidden.

You may explore and seriously evaluate:

- B1 / 1-bit ranking;
- B2 / 2-bit ranking;
- mixed B1→B2→exact cascades;
- approximate proposal heads;
- approximate target/vocabulary shortlists;
- hierarchical shortlist/rerank designs;
- activation/weight sketches;
- checkpoint-calibrated acceleration;
- other weird bit-level SM75/Tensor-Core tricks.

The requirement is **honest classification**, not universal mathematical perfection.

There can be multiple legitimate product lanes:

### A. General/default lane

Should be broadly portable across compatible checkpoints/fine-tunes and obey strong state/math correctness. Approximation that can silently discard a healthy-margin winner needs a fallback or some other robust guard.

### B. Checkpoint-qualified turbo lane

A checkpoint-specific optimization is allowed to exist if it is clearly labeled as such and has a real qualification story. It does **not** need to pretend to be universally safe for arbitrary future fine-tunes.

For a turbo lane, require evidence roughly analogous in spirit to upstream's fast-path acceptance rather than eternal byte identity:

- deterministic behavior on the qualified checkpoint;
- operator/state invariants remain valid;
- broad prompt corpus, not one benchmark;
- include code, JSON, reasoning, prose, Thai, Japanese, Arabic/another non-Latin workload;
- inspect first divergences against a trusted path and reject healthy-margin unexplained flips;
- no memory/state corruption;
- quality does not obviously collapse;
- performance gain is real enough to justify the special lane.

If a low-bit idea differs only at genuine near-ties, that may be perfectly acceptable for a qualified high-performance lane.

If a shortcut drops a clearly superior token because the true winner fell outside its shortlist, that is a different failure class. Do not wave that away as ordinary BF16 drift.

### Hardcoded token rescue sets

Treat literal token-ID rescue/pin sets skeptically because prior experiments clearly overfit. They may remain useful diagnostic/research tools. Do not silently make them the universal production default.

But do not let that prior failure discourage **model-derived, runtime-derived, checkpoint-qualified, or mathematically structured** low-bit ideas.

The point is: **be adventurous, but label the guarantee correctly.**

---

# 4. MTP: IMPORTANT, BUT DO NOT LET IT EAT THE PROJECT

MTP has large theoretical upside because it can reduce the number of expensive target weight sweeps, and the existing data are interesting:

- canonical/general prose prefers roughly MTP3;
- strict JSON showed MTP5 ~84.6 tok/s versus MTP3 ~79.8;
- long code showed MTP5 ~99 tok/s versus MTP3 ~94 tok/s;
- Thai has poor draft acceptance and likely benefits from shorter speculation rather than wider fixed depth.

This makes adaptive MTP a credible future optimization.

However, **kernel max is the immediate priority**.

Do not spend another large block of time trying to prove batched MTP == ordinary T1 forever. The upstream-like numerical rules above apply.

Reasonable MTP work while doing the kernel phase:

- fix an obvious state/mask/index/rollback bug if one is clearly demonstrated;
- collect useful width economics;
- preserve MTP5 code/JSON evidence;
- note cheap confidence signals that could eventually drive adaptive width;
- exploit wider-column reuse if it naturally falls out of kernel work.

Not reasonable right now:

- days of bitwise T1 parity archaeology;
- forcing all widths onto slow T1 kernels;
- building a huge adaptive scheduler before the kernel frontier is understood;
- tree speculation before simpler work is harvested.

After the kernel phase, if adaptive MTP looks like a quick high-payoff addition, pursue it. Otherwise preserve the evidence and move to dual GPU.

---

# 5. KNOW WHEN TO MOVE ON FROM SINGLE GPU

Do not declare the single-GPU frontier exhausted just because one Q4 pipeline attempt fails.

Use NCU to build a ranked view of remaining whole-model time and ask whether there are still credible opportunities in:

- Q4;
- Q5-down;
- GDN Q5;
- Q5-6144 SIMT;
- vocabulary/proposal heads;
- launch/fusion/state traffic;
- column-width reuse;
- low-bit/turbo paths.

At the same time, do not thrash indefinitely. A good stopping condition is:

- the major kernels have been characterized with counters;
- the best plausible structural ideas have been tried or ruled out with evidence;
- remaining expected wins are small compared with the value of dual-GPU full-context work;
- validated wins are checkpointed and dead ends are recorded.

Then move on decisively.

---

# 6. DUAL GPU — FIRST GET FULL NATIVE CONTEXT WORKING

Healthy hardware:

### Internal RTX 2080 Ti 22 GB

UUID:

`GPU-020906b4-0904-3471-e4ee-e41ec2350be3`

### Razer Core X RTX 2080 Ti 22 GB

UUID:

`GPU-2fffc240-5d58-fbc8-27d6-6b349448fda9`

The eGPU is healthy and performs essentially like the internal card when the model is fully resident locally.

CUDA P2P is unavailable in both directions. No NVLink.

Therefore the preferred full-context architecture remains **coarse layer/pipeline sharding**, not chatty tensor parallelism.

## First milestone: context capacity, not elegance or speed

Get one Qwen3.8-27B session running at the **largest feasible context, ideally the full native 262,144 tokens**.

Partition layers so each GPU permanently owns:

- its subset of weights;
- those layers' full-attention KV;
- those layers' GDN recurrent/conv state;
- relevant persistent scratch/state.

Move only stage-boundary hidden activations between devices, using pinned host staging because peer access is unavailable.

Do **not** repeatedly fetch remote KV or weights over Thunderbolt.

Do actual memory accounting for:

- per-layer weights;
- placement of the 16 full-attention layers;
- 48 GDN layers;
- KV at target context;
- GDN state;
- graph memory;
- runtime reservations;
- workspaces;
- embeddings/head/MTP asymmetry;
- safety headroom.

Do not blindly assume a 32/32 layer split.

If full 262144 fits, prove it with a real allocation/run and report headroom on both GPUs. If it does not, report the actual maximum and the limiting allocation rather than guessing.

---

# 7. AFTER FULL CONTEXT: MAKE BOTH GPUS WORK TOGETHER HARD

Once full-context sharding is functional:

1. balance stage execution time;
2. minimize/overlap boundary copies where possible;
3. preserve layer-local state;
4. pipeline multiple users/microbatches so one GPU can process request N+1 while the other works on request N;
5. benchmark 1 / 2 / 4 / 8 concurrent users;
6. report per-user speed, latency, aggregate tok/s, GPU utilization, TB3 traffic, and total context/KV capacity.

Remember the already-proven ordinary-context baseline:

- internal replica: ~68.15 tok/s in simultaneous test;
- eGPU replica: ~68.47 tok/s;
- aggregate ~136.6 tok/s with essentially no interference.

So eventual serving should probably have two complementary modes:

### Replica mode

Use a full model per GPU for ordinary/moderate contexts and maximum aggregate multi-user throughput.

### Sharded full-context mode

Use combined VRAM for huge contexts, then pipeline requests to recover aggregate utilization.

A future scheduler can choose between them based on context/KV demand and current load.

---

# 8. EXPERIMENTATION STYLE

The owner wants exploration, not bureaucracy.

### Do

- follow measurements;
- try creative SM75-specific ideas;
- use NCU/SASS when useful;
- make temporary experimental lanes/worktrees;
- compare end to end;
- preserve interesting rejected artifacts;
- checkpoint real wins;
- write concise conclusions so the next worker does not rerun dead ends;
- change direction when evidence changes.

### Do not

- spend hours proving a non-requirement;
- confuse output SHA inequality with an automatic bug;
- confuse a checkpoint-qualified turbo lane with a universally portable default;
- hide a real healthy-margin approximation error behind "floating point" language;
- let a single failed experiment terminate an entire promising category;
- run endless undirected parameter sweeps;
- make memory/clock overclocking the main optimization route;
- rely on numeric GPU IDs when UUIDs are available;
- hot-unplug the eGPU during active NVIDIA use.

When an experimental direction becomes clearly unproductive, close it cleanly and proceed.

---

# 9. AUTONOMOUS ORDER OF ATTACK

Use judgment, but the owner's preferred trajectory is approximately:

**A. Re-ground current source / preserve any useful diagnostics.**

Do not accidentally throw away a genuine unfinished fix, but do not let stale tracing scaffolding contaminate performance conclusions.

**B. Maximize single-GPU kernels.**

Start from the NCU evidence; Q4 is the obvious first target. Continue through other important kernels if evidence supports it. Low-bit/bit-level experiments are welcome if honestly qualified.

**C. Briefly harvest obvious MTP opportunities if they emerge.**

Do not make T1 bitwise parity the gate. Adaptive width is interesting but not mandatory before dual GPU.

**D. Declare a practical single-GPU frontier when evidence warrants it.**

Checkpoint the best useful lanes and summarize guarantees/performance.

**E. Build dual-GPU full-context mode.**

Maximum context first, ideally 262144.

**F. Then make sharded/combined inference fast and multi-user capable.**

Benchmark against the ~136.6 tok/s two-replica aggregate baseline.

**G. Only after that, return to larger speculative ideas such as adaptive MTP refinement, suffix/ngram drafting, or tree speculation if they remain attractive.**

---

# 10. WHAT TO REPORT WHEN THE OWNER RETURNS

Do not just say "done." Leave a concise status report answering:

1. What single-GPU performance is now achieved on the main qualified Qwen3.8 lane?
2. What kernel/algorithm changes produced it?
3. Which correctness guarantee does each lane have: general/default vs checkpoint-qualified turbo/research?
4. Which NCU bottlenecks were improved or proven irreducible?
5. What MTP conclusions are worth keeping without further rabbit-hole work?
6. Did full 262144 context work across both 22 GB GPUs? If not, what exact maximum did?
7. What is sharded single-user speed?
8. What is 2/4/8-user aggregate throughput?
9. Which commits/checkpoints should be preserved?
10. What are the next highest-value ideas?

---

# SHORT VERSION

**Be aggressive. Max the single-GPU kernels first. Weird 1/2-bit and checkpoint-qualified tricks are allowed if they are honestly validated under an upstream-like numerical/state acceptance standard; do not demand universal T1 bitwise identity. MTP is interesting but should not consume the project. Once the meaningful single-GPU frontier is reached, prioritize dual-GPU full native context first, then make combined/pipelined multi-user inference scream. Preserve wins, record dead ends, and keep moving.**
