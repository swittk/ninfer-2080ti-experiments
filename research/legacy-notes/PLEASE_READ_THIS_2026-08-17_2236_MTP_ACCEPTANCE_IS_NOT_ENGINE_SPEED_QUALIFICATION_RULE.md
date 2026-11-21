# PLEASE READ — HARD QUALIFICATION RULE: do not count lucky MTP acceptance as engine-performance success

Owner/adviser correction, 2026-08-17 22:36 +07. This is a hard benchmark/qualification rule for all single- and dual-GPU performance claims. It does not forbid wider/adaptive MTP or better proposal models, but it prevents favorable-acceptance workloads from being mistaken for execution-engine speedups or product-gate qualification.

## Core distinction

Useful decode throughput is jointly determined by at least two independent things:

1. **Execution efficiency** — target/proposal round cost, critical-path latency, overlap, kernel speed, synchronization, memory movement, etc.
2. **Speculative efficiency** — useful committed tokens per round, driven heavily by MTP draft acceptance on that workload.

A workload that happens to have unusually high MTP acceptance can show high useful tok/s even if the execution engine has not improved. Conversely, a hard workload with low acceptance can show poor useful tok/s even if the engine itself is faster.

Therefore: **never use a cherry-picked high-acceptance prompt as proof that a >=80 or >=100 product gate is satisfied.**

## Current example

The recent dual B2 proposal-overlap results make this concrete:

- ~52.4% acceptance -> ~84 tok/s aggregate on the 128-token control;
- ~60.7% acceptance -> ~93.6 tok/s aggregate on JSON;
- ~38.5% acceptance -> ~71.3 tok/s aggregate on code.

The 93.6 number is useful as evidence that the pipeline has no obvious ~85 tok/s hard ceiling, but it is **NOT** qualification of the >=100 dual multi-user goal and must not be presented as "almost 100" without the acceptance context.

Historically this checkpoint's ordinary MTP3 acceptance on representative workloads is often around the ~50–55% region. Do not hardcode 50% as a model law; instead measure and preserve a fixed representative corpus and report its actual acceptance distribution.

## Required reporting for every serious performance comparison

For each run/corpus point, report together:

- useful committed decode tok/s;
- MTP draft window/backend;
- MTP rounds;
- drafted tokens;
- accepted tokens and acceptance rate;
- useful tokens/round;
- target stage / critical-path latency or rounds/s;
- proposal/alignment time;
- workload identity and output length;
- correctness/determinism status.

Do not headline tok/s without acceptance.

## Qualification corpus / anti-cherry-pick rule

Maintain a fixed representative corpus that includes at minimum general technical prose, code, structured/JSON, and multilingual/non-Latin text. Do not choose the qualification prompts after seeing their acceptance.

For product-gate claims:

- publish per-workload results plus corpus aggregate/median;
- preserve the same corpus across candidate vs reference;
- a single favorable workload cannot satisfy the gate;
- unusually high-acceptance workloads may be reported separately as conditional/high-acceptance ceilings;
- unusually low-acceptance workloads remain important robustness points, not excuses to discard the corpus.

The goal is >=100 aggregate useful tok/s for dual multi-user under representative normal model behavior, not >=100 only when the drafter happens to be unusually lucky. Likewise single-GPU >=80 must not depend on selecting a high-acceptance prompt.

## Separate execution optimization from speculation/model optimization

If a change leaves target model, proposal model, draft depth, and acceptance semantics unchanged, then improvements in round latency/rounds-per-second are clean execution-engine gains.

If a change alters draft width, adaptive-depth policy, proposal model, tree/suffix speculation, or other mechanism that changes accepted tokens/round, that may be a legitimate product optimization, but label it separately as **speculation/algorithmic gain**. It cannot be used to claim that a kernel/runtime path itself became faster unless the execution metrics also improved.

Wider/adaptive MTP is not "forbidden cheating" if it preserves exact target verification semantics and improves representative end-to-end product throughput. What is forbidden for qualification is **benchmark selection based on lucky acceptance** or silently conflating acceptance gain with engine gain.

## Current hard interpretation of the goals

1. **Single GPU / one user INT8 >=80 useful tok/s**: qualify on the fixed representative corpus / normal acceptance behavior, with engine metrics reported separately. A lucky 80+ short/high-acceptance prompt is diagnostic only.
2. **Dual GPU / one user >=100 useful tok/s + 262K-class context**: same anti-cherry-pick rule, plus populated long-context qualification.
3. **Dual GPU / multiple users >=100 aggregate useful tok/s**: same anti-cherry-pick rule. The recent ~93.6 JSON result is not the gate; current representative ~50%-acceptance evidence is still around the mid-80s.

Do not move these goalposts.
