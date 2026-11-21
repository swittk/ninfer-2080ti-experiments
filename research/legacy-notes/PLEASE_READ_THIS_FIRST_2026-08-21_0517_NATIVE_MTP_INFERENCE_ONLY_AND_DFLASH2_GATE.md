# OWNER ADVISER — 2026-08-21 05:17 +07 — native MTP inference-only trajectory + DFlash2 gate

Read after the 2026-08-20 2100 owner note. This supersedes any remaining trajectory toward locally trained/tuned MTP weights.

## Non-negotiable owner constraints

- NO Codex/Claude/OpenCode/sub-agents/agent loops. Primary ChatGPT performs inspection, edits, builds, benchmarks, profiling, and classification directly.
- NO homemade model training/tuning: no native-MTP finetune, CE/TTT/D-PACE training, distillation, LoRA/adapters, interpolation/blending, custom DFlash/DFlare training, or new behavioral checkpoints.
- Allowed model inputs: factory Qwen3.8 native MTP weights; upstream professionally trained exact-target drafts such as Qwen3.8 DFlash2; mechanical conversion/quantization/export only.
- GPU asymmetry is a runtime/driver/eGPU boost-state observation, NOT a permanent healthy/unhealthy hardware identity. Calibrate clocks/perf at runtime and assign roles dynamically.
- Hard gates: single >=80 useful decode t/s; dual one-request >=100; populated ~262K dual one-request >=100. Do not normalize ~59 as success.

## Current trusted numbers

- Single exact proposal-B2 anchor: ~71.8–72.1 useful t/s.
- Dual short representative TP2: ~65–66 useful t/s. split48 hit 68.55 but was later classified as mostly proposal/trajectory geometry, not a portable engine frontier.
- Dual populated65K, factory MTP depth4 + physical INT8 KVP + dedup T5 TP: 58.75 useful t/s, 67.56 ms/round, 3.90625 useful/round, deterministic, zero fallback.
- BF16 physical KVP CLOSED as speed route: 33.19 t/s matched65K vs 58.75 INT8. INT8 remains speed+capacity route.
- 262K physical INT8 KVP + dedup TP capacity works; >=100 throughput not proven.

Fixed depth4 at 67.56 ms/round cannot hit100 even at perfect 5 useful/round (~74 t/s). At current yield, 100 requires <=39.1 ms/round. 65K/262K >=100 therefore needs structural target-round reduction and/or richer proposal topology; tiny acceptance/scheduler tweaks cannot solve it.

## Current live DFlash2 sidecar — correct lane, but bounded

Exact upstream Qwen3.8 DFlash2 executes on SM75 in current llama.cpp (`draft-dflash`). Treat this as a proposer characterization oracle, NOT replacement engine and NOT NInfer frontier.

Observed n-max4 greedy 8K cohort:
- general: 28.68 sidecar t/s, mean emitted/round 2.65, 41.75% acceptance, ~11.41 ms draft/round
- code: 40.81, AL 4.29, 82.93%, ~12.01 ms/round
- JSON: 31.32, AL 2.98, 50.29%, ~11.69 ms/round
- Thai: 24.71, AL 2.32, 33.18%, ~11.76 ms/round
- Japanese: 30.03, AL 2.89, 47.75%, ~12.04 ms/round
- Arabic: 38.98, AL 3.88, 73.13%, ~11.84 ms/round
Sampled general: 23.23, AL 2.20, 30.21%.

General n-max7: AL only 2.83 vs 2.65 at n4, draft cost ~15.1 ms/round, sidecar decode ~28.1 -> ~21.6 t/s. Deeper is not automatically better.

Finish the already-running bounded matrix (n4/n7, greedy + canonical sampled, six families); then run the existing NInfer economics model. DO NOT tune DFlash2 itself or integrate it merely because code has a good row.

Integration gate: measured AL + draft cost must produce a material complete-round ceiling with NInfer target costs across representative workloads. If even an optimistic verifier floor cannot move toward 80/100, classify DFlash2 integration CLOSED AT CURRENT TARGET COST while preserving the upstream sidecar reference. One code prompt does not reopen it.

## P0 after DFlash2 matrix: exploit FACTORY native MTP better, inference-only

### A. Make the common native-MTP round device-resident

Audit/measure current `target verify -> accept -> select hidden -> next MTP proposal`. Current dual code copies acceptance metadata GPU->host, synchronizes, then copies much of it host->GPU for proposal continuation. Quantify on short and65K traces.

Build only if complete-round ceiling matters. If material, keep acceptance metadata and selected hidden on-device through proposal recursion, host egress only for optional telemetry/debug. Preserve exact target semantics/determinism. Aim for graphable/coarse execution, not another fine event DAG. This is distinct from the CLOSED generic scheduler lane because it removes an avoidable CPU algorithm-control round trip.

### B. MTP-Eagle one-model is an upstream semantic oracle; do not reimplement what we already have

TensorRT-LLM MTP-Eagle reuses the same native MTP module and MTP KV recursively for a one-module checkpoint. NInfer already largely does this: same StageTextContext, same MTP KV rows, previous token + previous MTP hidden recursively advance the factory module. Verify parity and steal only missing execution details.

Upstream also exposes EAGLE-style dynamic-tree drafting for one-model MTP. Do NOT blindly reopen the CLOSED 65K factory-MTP tree. Separately model whether a native-MTP dynamic tree can clear the SHORT-context dual >=100 gate, where prefix verification is cheaper. Build only if the short-context oracle ceiling is material.

### C. Windowed-MTP for long context — training-free, factory weights unchanged

Windowed-MTP (arXiv:2607.21535) windows only native MTP draft attention/KV via sliding window + attention sink; target verification remains full/exact. First measure factory MTP draft attention/KV contribution at65K and projected262K. If material, build an opt-in draft-only 2K/4K/8K window+sinks discriminator. Never window target KV. Measure proposal ms + accepted length across six families and pursue only if complete E2E improves.

### D. Entropy-guided MTP topology/depth — training-free, oracle first

EntMTP (arXiv:2606.27550) chooses among profiled MTP topologies from running local entropy. This is NOT permission to revive old heuristic adaptive-width work. Build a trace oracle using factory logits first; require a portable quantified ceiling over fixed K/depth before runtime implementation. Likely more relevant to short/moderate context than65K.

### E. Proposal vocabulary only after measuring remaining head cost

NInfer already has compact proposal-head/B2 machinery, so generic training-free vocab pruning overlaps existing work. Measure proposal-head fraction before doing anything. Do not spend days saving sub-ms head work while target verification costs tens of ms.

## P0 dual-GPU systems question: why short dual still loses to single

Do not let65K research obscure the short-context hard gate. Single exact is ~72; representative dual is ~65–66; dual must become >100.

Public executable evidence: dual modified 22GB RTX2080Ti + Qwen3.6-27B-AWQ + vLLM + factory MTP K=3 reaches 101.3 t/s median at4K with TP=2/NVLink/AWQ Marlin/FlashInfer/FlashQLA. Our topology lacks active P2P/NVLink, so communication mechanics do not transfer, but factory MTP on this model family is sufficient to cross100 on this silicon class.

Do a same-shape mechanism audit, not wholesale engine replacement:
- quantized target GEMM vs AWQ/Marlin mechanisms;
- GDN vs FlashQLA SM75;
- small-T/verify attention vs FlashInfer/FA2 mechanisms;
- native MTP K=3 control overhead;
- TP sync/transport.

For no-P2P short context, explicitly model an alternative role split: one GPU runs the full target chain, the other is proposal/speculation coprocessor, rather than host-staged TP on every target layer. Implement only if measured/optimistic ceiling beats the single-GPU anchor by a large margin.

## Cleanup / state hygiene

- `src/targets/qwen3_6_27b/impl/dual_stage.cpp` still has dirty teacher-dump instrumentation from the abandoned training/root-capture lane. It is NOT production and must not be committed. After the active DFlash2 matrix finishes, restore it unless some non-training correctness tool demonstrably requires it (currently none).
- INT8 must not regress. Preserve BF16/INT8 conclusions as artifacts, not experimental churn.
- No agent delegation, no new custom weights, no watcher state.

## Execution order

1. Finish bounded DFlash2 matrix; classify by NInfer complete-round economics.
2. Remove teacher-dump training instrumentation; restore reviewed clean source baseline.
3. Recompute hard-goal table: single, dual4K/short, dual65K, dual262K.
4. P0 factory-MTP inference audit: device-resident control + short-context dynamic-tree ceiling.
5. P0 short dual mechanism/architecture audit against public SM75 vLLM route; identify a >=100-capable plan.
6. Long-context factory-MTP draft-KV audit -> Windowed-MTP only if material.
7. EntMTP-style entropy topology oracle; implement only with large portable ceiling.
8. Combine only proven pieces; representative E2E; never one code workload as frontier.

After every material result: correctness/determinism -> useful E2E TPS -> ACCEPTED/REJECTED/CLOSED -> progress ledger -> recompute 80/100/262K economics -> continue directly.
