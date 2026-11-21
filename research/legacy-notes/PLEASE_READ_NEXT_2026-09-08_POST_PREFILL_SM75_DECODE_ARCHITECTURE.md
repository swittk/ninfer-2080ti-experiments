# PLEASE READ NEXT — after prefill is blazing, reopen the real SM75 steady-decode architecture nut

Owner direction, 2026-09-08.

**Do not derail the current cold-prefill / vision / memory qualification work. Finish the current coherent tranche first.** The new prefill work is valuable and should be made trustworthy. But do **not** interpret a good cold-prefill result as the end of the performance program, and do **not** seal steady serving at the current ~230–250 physical C8 common-window decode range.

## Why this is still worth reopening

The current fast serving stack is materially different from the stack under which many old SM75 decode experiments were rejected: physical C4+C4 replicas, C4 A8, full-head D1, exact T16 fragment reuse, resident layouts, different admission/prefix behavior, and a much more realistic production HTTP harness now exist. Old negative results remain important evidence, but they do not prove that the *current* C4/D1/A8 dataflow is globally optimal.

The remaining prize is **steady decode throughput**, not TTFT: make already-decoding C1/C2/C4/C8 requests generate more useful tokens per second. This is distinct from the current dense-prefill work.

## MTP guidance

Treat native factory MTP as **largely exploited for the present serving topology unless fresh profiling proves otherwise**. D1 has already beaten deeper drafting at C4 serving concurrency because proposal/verification width competes with real cross-request batching. Do not spend another large search budget on draft-depth knobs, proposal-window cosmetics, or old MTP microvariants merely because they are easy to benchmark.

MTP can still receive a targeted fix if a fresh profile identifies a concrete dominant cost, and session-history copy/ngram remains a separately-labelled future agentic accelerator. But the next general serving-throughput breakthrough should not be assumed to come from “more speculation.”

## The architectural decode question to reopen

Freshly profile the **current production C1/C2/C4 paths** (and physical C8 as two C4 replicas) with NCU/Nsight after the prefill tranche stabilizes. Specifically re-measure the dominant Q4/Q5/GDN decode kernels and stalls rather than importing old percentages as current truth.

Historically the important kernels showed very large long-scoreboard / load-to-use stalls (roughly Q4 A8 SwiGLU ~67%, Q5-down ~56%, GDN Q5 ~42% in the old profile family). Earlier deeper lookahead / same-warp prefetch / producer-consumer / Q5 lookahead / barrier-removal attempts sometimes reduced those stall metrics but did **not** improve whole-model TPS; one larger Q4 pipeline even regressed. Do not repeat those exact geometries blindly.

Instead ask the more structural question:

> **Why are packed Q4/Q5 fragments being fetched/dequantized/lifetimed the way they are in today's C4/D1/A8 execution? Can one loaded fragment feed more useful live-token work before it dies, without register/shared-memory growth destroying occupancy?**

The Q95/Q96/T16 work is evidence that this class of idea is real, not hypothetical: exact isolated operator gains were roughly Q4 1.14–1.16x, Q5 1.51–1.53x, GDN ~1.13x. Q98 T32 showed ~1.20x Q4 and ~1.60x Q5 while T32 GDN failed. Those results say **weight-fragment lifetime / cross-token reuse is a real lever**, even though not every wide geometry translates.

Explore genuinely different dataflow if the fresh profile supports it: fragment lifetime across active token lanes, different producer/consumer partitioning tied to the current C1/C2/C4 geometry, staged packed-load/dequant/consume schedules, or another architecture that changes how often weights are fetched and converted. The goal is not to make an NCU stall percentage prettier; the goal is lower end-to-end layer time and higher real Engine tok/s.

## Hard experimental discipline

- Re-profile current source first. Old long-scoreboard numbers are hypotheses, not authority.
- Preserve C1 quality and the existing exact/qualified arithmetic contracts. Operator speedups must prove numerical/output correctness before integration.
- Measure **real production Engine serving**, not only microbench wins. A candidate that wins an isolated Q4/Q5 bench but loses C1/C4/C8 useful tok/s is rejected.
- Keep C1, C2, C4 and physical C8 distinct. A C4-specific batch-amortization architecture may be excellent even if it is irrelevant to C1, but it must not silently regress C1.
- Do not force T32 GDN or resurrect already-rejected geometries without a new reason.
- Do not treat current ~230–250 aggregate C8 as a ceiling. If the architecture is cracked, materially higher aggregate throughput (300+ research territory) remains worth pursuing, while preserving >=20 useful tok/s per user and fairness.
- No training, no implementation agents/subagents, no hardware-state changes, no benchmark cheating.

## Ordering

1. Finish and qualify the current prefill/vision/memory tranche; harvest real gains and reject bad numerical variants.
2. Produce/retain the clean serving matrix and realistic appended-continuation evidence.
3. Then run a **fresh decode profile of today's A8+D1+T16 production path** at C1/C2/C4 (physical C8 = two C4 replicas).
4. If Q4/Q5/GDN weight/dataflow stalls are still dominant, pursue one or two **architecturally different** fragment-lifetime/dataflow candidates. Do not reopen a giant old knob sweep.
5. Translate any operator win through the actual HTTP Engine and report whole-request + shared-decode-window throughput, per-user floor, p95/fairness, and quality.

The intent is: **prefill blazing first, then one serious attempt at the remaining steady-decode architecture nut.** Do not declare the nut cracked unless real serving throughput moves.

## OWNER CLARIFICATION — C1 RAW SINGLE-GPU DECODE IS THE NORTH STAR

This priority is stronger than the generic C1/C2/C4/C8 wording above. For **one stock RTX 2080 Ti**, the owner's rawest performance number is **one real user / C1 steady decode tok/s**. Do not let attractive aggregate C4/C8 numbers substitute for it.

- Historical short C1 around ~107 tok/s is a floor/reference, **not a ceiling and not a satisfactory end state**.
- The performance research objective is to push real one-GPU C1 as high as the hardware/current model architecture can support: 120, 150, 200+ if genuinely achievable. There is **no artificial target ceiling**.
- Any Q4/Q5/GDN architectural investigation must explicitly ask whether it can reduce the C1 weight/dataflow cost, not only amortize weights across multiple users. A C4-only win is still useful for serving, but it does not close the owner's single-GPU raw-performance objective.
- Freshly measure the current production C1 path with the full retained stack and a long enough decode window before declaring the kernel campaign mapped. Report client-visible decode and the exact internal/common decode interval separately.
- Keep C1 raw throughput distinct from prefill/TTFT, dual-GPU single-request sharding, and C4/C8 aggregate throughput. Do not report aggregate as a proxy for this number.
- Preserve quality/correctness and the no-training/no-hardware-state rules. Do not manufacture 200+ via semantic degradation, benchmark-only shortcuts, or a non-production path.

**Priority hierarchy for decode research:** (1) raw one-GPU C1 maximum useful tok/s; (2) one-GPU C2/C4 scaling and weight amortization; (3) physical C8=C4+C4 aggregate/fairness; (4) workload-specific accelerators under separate labels.
