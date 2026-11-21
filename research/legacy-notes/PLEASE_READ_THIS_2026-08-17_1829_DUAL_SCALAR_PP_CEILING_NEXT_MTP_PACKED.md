# PLEASE READ — dual throughput diagnosis: scalar PP is saturated; next step is MTP/packed serving, not more stream shuffling

Owner/adviser correction, 2026-08-17 18:29 +07. This supplements the 18:18 single-GPU note. `be6530d` is useful as a **capacity/correctness milestone**, but its 39-44 tok/s results are not the optimized dual performance frontier.

## 1. The current 44.49 tok/s is basically the ceiling of THIS scalar 2-stage design

One-session timing at split36 was:
- stage0: 2.532 s / 127 decode steps ~= 19.94 ms/token
- stage1: 2.490 s / 127 ~= 19.61 ms/token
- output head: 0.322 s / 127 ~= 2.54 ms/token

So a perfectly filled 2-stage scalar pipeline is limited by stage1+head ~=22.15 ms/job, i.e. ~45.1 jobs/s. The measured 2-session 44.49 aggregate is already essentially that ceiling. Do NOT spend another long sweep merely rearranging T1 stream synchronization hoping for 100+.

The 3-session split32 result (~39.85 aggregate) is also consistent with a more imbalanced stage1 (32 layers + full head) plus pipeline overhead. A third scalar session cannot create a third hardware stage.

## 2. Current dual path deliberately leaves the major single-GPU speed machinery OFF

The shard state currently sets `enable_mtp = false`; decode calls width-1 `forward_token` / `forward_hidden`; `project_shard_head` performs the full exact output-head linear+argmax. Therefore 39-44 aggregate MUST NOT be compared as though it were the dualized version of the 65.3 portable MTP3 lane or the ~72 checkpoint-qualified lane. It is scalar autoregressive PP.

Next performance architecture should reuse the existing target semantics/kernels and pipeline **verification rounds**, not individual scalar tokens. Preserve per-session KV/GDN state/rollback correctness while allowing each stage to process the actual MTP verify width (typically T4 for MTP3) and overlap independent sessions/rounds across stage0/stage1.

Back-of-envelope sanity: portable MTP3 currently gets ~2.69 useful tokens per target round at ~65.3 useful tok/s, so a whole-model round is ~41 ms. If a balanced 2-stage implementation can keep each stage around ~20-22 ms/round, the natural aggregate upper region is roughly 120-130 useful tok/s before overhead. This is NOT a promise, but it shows the owner >=100 dual target is not absurd and scalar PP simply cannot test it.

## 3. Revisit batching correctly; do not resurrect the rejected batch implementation unchanged

The rejected batch experiment measured only ~32.94 aggregate at B2 and ~44.83 at B3, so that implementation was not competitive and should remain rejected. However, output digest inequality by itself is not a valid rejection reason under the upstream-oriented correctness contract. If a future packed/batched path changes kernel geometry, qualify it with operator numerical tolerances + state correctness + first-fork margin/perturbation evidence, not SHA equality alone.

For serving throughput, use packed/in-flight/continuous batching concepts: combine currently ready decode sequences into width-N work so each weight read serves multiple users, rather than processing every user as an independent GEMV forever. The scheduler must tolerate different prompt lengths/frontiers and dynamically form ready batches.

## 4. Hardware topology means PP is a sensible starting point, but host boundary transfer is not the bottleneck

Current two internal 2080 Ti GPUs show `NODE` topology and CUDA P2P `CNS` (chipset not supported), i.e. no usable direct peer path/NVLink in this machine configuration. A 5120-wide BF16 stage boundary is only 10,240 bytes, so pinned-host boundary staging is reasonable for PP and should not dominate ~20 ms stage compute. Tensor parallelism would require repeated cross-GPU reduction/communication inside layers and is not automatically preferable on this topology.

## 5. Benchmark the actual product target before claiming full-context performance

Current 262144 tests mostly prove **allocation/capacity** (16-21 prompt tokens, often one generated token). They do not benchmark a populated 262K decode. Before calling dual performance qualified, measure at least:
- contexts populated to representative 8K / 32K / 64K / 128K / 262K frontiers;
- 1/2/3 simultaneous sessions, distinct prompts as well as identical controls;
- >=256-512 steady decode tokens where feasible;
- per-user tok/s, aggregate useful tok/s, TPOT, stage0/stage1/head time, bubbles/utilization, MTP rounds/acceptance/useful tokens per round;
- NSYS timeline for overlap and one bounded NCU pass on any newly dominant kernels.

Capacity success is good, but `max-context=262144` with a 16-token prompt is not "full populated context throughput".

## 6. Keep the code architecture coherent

The current dual work is mostly additive and target-local (`apps/dual`, `dual_stage`, shard entrypoints) and still calls the canonical `TextContext::run_layers`/Variant machinery. That is acceptable for a milestone; do NOT fork/copy the model kernels into a separate dual implementation.

The next phase should ideally make `TextShardStage` a thin transport/state adapter around existing `TextContext` batch/verify primitives and move scheduling policy upward. Reuse target-verify/MTP machinery rather than building a second independent inference runtime. Keep dual-specific state ownership explicit, but avoid growing `apps/dual/main.cpp` into a permanent hand-written serving engine.

## Immediate direction

1. Treat `be6530d` as scalar capacity/correctness baseline.
2. Stop trying to get 100+ from scalar T1 PP.
3. Design/measure sharded MTP3 target verification + per-session speculative state first.
4. Then add packed ready-session batching / dynamic scheduling where it actually improves stage arithmetic intensity.
5. Qualify numerical differences with the upstream-oriented gate, not byte/digest identity.
6. Re-run throughput at populated long contexts.

Relevant modern design references to consult: TensorRT-LLM in-flight/continuous batching and PP guidance; EasySpec (layer-parallel speculative decoding); 2026 Speculative Pipeline Decoding / distributed pipelined speculative-decoding work. Adapt concepts to this hybrid Qwen/GDN architecture; do not cargo-cult their kernels.
