# PLEASE READ: UPSTREAM CORRECTNESS HEURISTICS — NOT BITWISE RELIGION

**Owner/adviser correction — 2026-08-17 08:45 +07**

This note supplements and OVERRIDES any older automation or status logic that treats `MTP output != ordinary T1 output` as an automatic correctness failure.

Read this together with:

- `PLEASE_READ_THIS_2026-08-17_0830_STOP_CHASING_MTP_T1_BITWISE_PARITY.md`
- reliability baseline `38369da` (`fix(sm75): harden q45g graph replay determinism`)

The goal is to preserve the **original NINFER correctness philosophy** while still allowing aggressive SM75 performance work. Do not replace that philosophy with either extreme:

1. **too loose:** “different output is always fine”; or
2. **too strict:** “every optimized batched kernel must be bitwise-identical to ordinary T1 forever.”

Both are wrong.

---

## 1. WHAT ORIGINAL NINFER ACTUALLY REQUIRED

Historical source-controlled evidence before `b374d41 refactor(mtp): remove parity-only validation paths` explicitly separated two contracts.

### A. Hard algorithmic self-consistency

From the old MTP overview §8.1:

> committed output must exactly be the greedy output of the **verify path itself**.

Concretely:

- every accepted draft must equal the corresponding verifier greedy token for that position;
- the correction / continuation token must be the verifier's own greedy continuation after the accepted prefix;
- acceptance quality changes throughput only; a poor drafter may reduce acceptance but must not violate the verify/commit rule.

This is a **hard invariant**. If it fails, that is a real bug.

### B. MTP-on vs ordinary-T1 sequence equality was an acceptance target with controlled numerical deviation

The old overview explicitly said batched Verify T=2..6 and ordinary T1 decode use different kernels and different floating-point reduction orders. It documented that near-tie argmax flips can therefore cause sequence divergence.

The historical reference work had already observed a real K5 batched-target divergence around token index 37.

Therefore original NINFER did **not** define the production fast batched path as bitwise-identical to T1 forever.

The old two-level strategy was:

- strict-sequential mode: same T1 target path; must equal MTP-off token-for-token; this validates the round/state algorithm;
- fast batched mode: operators must satisfy numerical parity tolerances and any first argmax divergence from strict/T1 must be explainable as a near-tie numerical flip rather than a state/math bug.

Later `b374d41` deliberately removed parity-only runtime/debug machinery and relaxed production E2E tests further.

**Do not re-impose a stronger bitwise contract by accident during SM75 optimization.**

---

## 2. THE CORRECTNESS LADDER TO USE NOW

Use these exact categories in notes/automation where possible.

### `PASS_IDENTICAL`

Strongest smoke-test result:

- same tokens / output SHA as the chosen reference;
- same expected state/trajectory where the test is designed to require it;
- deterministic across repeats.

Great evidence, but **not the only acceptable outcome** for a mathematically equivalent batched kernel.

### `PASS_NUMERICAL_NEAR_TIE`

Output eventually differs from ordinary T1 / another width, but all of the following hold:

1. hard state/algorithm invariants are intact;
2. operator outputs stay inside the frozen numerical oracle/tolerance for their compute class;
3. the **first** argmax flip is quantitatively explainable by the measured numerical perturbation of a near-tied pair;
4. the fast path remains deterministic for the same binary/input/settings.

This is an **allowed production numerical divergence**, not an automatic bug and not a reason to force T4/T5/T6 through T1 arithmetic.

### `FAIL_ALGORITHM_OR_STATE`

Any hard semantic/state invariant fails, including examples below. This is a real bug and may block promotion.

### `FAIL_NUMERICAL_UNEXPLAINED`

The first token flip cannot plausibly be caused by allowed numerical error / near-tie behavior, or operator tensors fail their numerical oracle by a meaningful margin.

Treat as a bug until explained.

### `EMPIRICAL_APPROX_ACCELERATOR`

An approximation such as current B2 top-K reproduces the trusted fast verifier on the tested checkpoint/corpus, but lacks a universal mathematical guarantee for arbitrary fine-tunes.

This is **not** the same category as an ordinary batched floating-point reduction difference.

Current B2 belongs here for Qwen3.8 until/unless a proof/fallback mechanism changes that classification.

### `UNIVERSAL_SAFE_FALLBACK`

A path whose correctness does not depend on checkpoint-specific shortlist membership / token IDs / corpus behavior. It may still use normal accepted numerical kernel tolerances; “universal safe” does not mean “bitwise T1 arithmetic everywhere.”

---

## 3. HARD FAILURES — DO NOT WAVE THESE AWAY AS FLOATING POINT

The following are real correctness bugs if demonstrated.

### Verify/accept contract

FAIL if:

- a draft is accepted when it is not equal to the fast verifier's corresponding greedy target;
- correction token is not the verifier's greedy token at the first rejected position / continuation position;
- committed token count or accepted prefix is inconsistent with verifier outputs;
- stop/max-new truncation commits the wrong token sequence.

### Causal positions / masks / valid columns

FAIL if:

- a verify column can observe a future column;
- RoPE/cache position is wrong;
- invalid/padding columns affect valid-column results;
- results for an existing valid column depend on junk values in invalid columns beyond permitted arithmetic/layout effects.

### KV / GDN / recurrent state

FAIL if:

- committed KV does not correspond exactly to committed tokens/positions;
- rejected speculative KV survives as committed state;
- rollback/rewind/frontier is off by one;
- GDN conv/recurrent state commits the wrong snapshot/slot;
- state selected after an accepted prefix is not the state after precisely that committed prefix;
- graph/eager uses different logical state transactions.

Historical upstream guidance was particularly strict about GDN sequence state. Where the same state kernel/order is being compared, snapshot state was expected to be bit-consistent with the sequential state construction.

### Memory/indexing/synchronization

FAIL on:

- OOB access;
- stale tail consumed as valid data;
- workspace overlap/alias corruption;
- cross-column contamination;
- race/non-deterministic same-binary output;
- graph replay using stale pointers/state;
- incorrect width/tail indexing.

`38369da` remains the reliability baseline for this reason.

### Approximation changing a healthy-margin winner

For B1/B2/other approximate ranking paths, FAIL if an approximation drops or replaces the trusted exact-fast-verifier winner when that winner has a healthy exact margin and there is no validated fallback/proof.

Do **not** excuse that as ordinary BF16 reduction noise. Approximate row elimination is a different algorithmic mechanism.

---

## 4. ALLOWED DIFFERENCES — DO NOT TURN THESE INTO PERF-KILLING “FIXES”

The following are not automatically bugs:

- BF16 hidden values differ by a few representable values because T1 and T4/T5/T6 accumulate in a different order;
- Tensor Core small-T and SIMT/GEMV paths produce numerically close but non-bit-identical outputs;
- an argmax flips at a genuinely tiny top1/top2 gap that the observed allowed numerical perturbation can cross;
- after one allowed near-tie token flip, the remainder of the generated sequence is entirely different;
- MTP acceptance/trajectory differs after that sequence fork;
- different width produces different future text because it reached a legitimate numerical fork earlier.

**Do not measure “percent of suffix tokens equal” after the first justified fork.** Once the history differs, later token equality is not a meaningful correctness metric.

Do not force a faster T4/T5/T6 kernel through the T1 kernel solely to restore output SHA.

---

## 5. FROZEN OPERATOR NUMERICAL PHILOSOPHY FROM UPSTREAM

The old `docs/l1-op-test-standard.md` used maximum-precision oracles from rounded inputs and a composite tolerance rather than bit identity.

Important historical presets included:

### BF16 linear / BF16 reduction style

Typical frozen criterion:

- `atol = 2e-3`
- `rtol = 1.6e-2`
- violating tail fraction <= `2e-3`
- worst violation ratio <= `5`
- relative L2 <= `8e-3`

### BF16 elementwise

- `atol = 1e-3`
- `rtol = 8e-3`
- tail <= `1e-3`
- worst ratio <= `4`
- relative L2 <= `4e-3`

### Tensor-Core linear

Upstream intentionally used a **normwise** criterion because low-precision GEMM can have large relative error on near-zero cancellation outputs while still being mathematically correct.

Historical relative-L2 bound was approximately `4e-3`, with NaN/inf fatal.

### GDN state

Historical GDN FP32 state tolerance was much tighter than general BF16 output tolerance; state correctness is not something to hand-wave.

### Argmax

For a single fixed logit tensor, argmax itself is exact-index semantics with deterministic tie-break.

The allowance arises because two legitimate compute paths can produce slightly different logit tensors — not because argmax is fuzzy.

**Do not loosen frozen numerical criteria to make a new kernel pass. Fix the kernel or reject it.**

Current repository tests may have evolved names/locations, but preserve this philosophy: compare against the semantic/math oracle using the compute-class tolerance; do not invent per-experiment tolerances.

---

## 6. AUTOMATED FIRST-DIVERGENCE TEST — USE THIS INSTEAD OF MANUAL PROSE INSPECTION

If ordinary T1 and fast batched MTP first choose different tokens at the same logical history, capture both logit vectors at that **first divergence only**.

Let:

- `A` = T1/reference winner
- `B` = batched winner
- `Lr` = reference/T1 logits
- `Lf` = fast/batched logits

Compute:

`reference_margin = Lr[A] - Lr[B]`

and a directly observed ordering-perturbation envelope:

`observed_flip_budget = abs(Lf[A] - Lr[A]) + abs(Lf[B] - Lr[B])`

Interpretation:

### Allowed near-tie candidate

If:

`reference_margin <= observed_flip_budget + justified_numeric_epsilon`

then the measured numerical change is large enough to explain why the A/B ordering flipped.

This is not sufficient alone: operator/state invariants must also pass.

### Suspicious / bug

If:

`reference_margin >> observed_flip_budget + justified_numeric_epsilon`

then ordinary numerical drift cannot explain the flip. Investigate state/indexing/math.

Do **not** invent a magic global “near-tie threshold” if the old documentation did not specify one. Prefer this measured local explainability test plus the frozen operator tolerance envelope.

For extra robustness, also record:

- reference top1-top2 gap;
- fast top1-top2 gap;
- max/rel-L2 logit difference;
- hidden difference at the relevant target column;
- whether both candidate token logits lie inside expected operator-level numerical error;
- deterministic repeatability of the same first fork.

This can be fully automated. Human inspection of generated prose is not the correctness oracle.

---

## 7. DETERMINISM IS A SEPARATE HARD REQUIREMENT

For the **same binary, same checkpoint, same input, same settings, same deterministic/greedy mode**, repeated execution should reproduce the same fast-path result unless the runtime explicitly documents nondeterministic arithmetic.

A deterministic near-tie fork relative to T1 can be acceptable.

A run-to-run random fork under identical conditions is a different problem and should be treated as race/state/reduction nondeterminism until proven otherwise.

This distinction is why the q45g reliability work and `38369da` still matter even though we no longer demand T1 bit identity.

---

## 8. CANONICAL SHA / TRAJECTORY: KEEP THEM, BUT USE THEM CORRECTLY

Canonical SHA and MTP trajectory are excellent regression alarms.

Use them as:

- “something changed; classify it” signals;
- strong evidence that a local optimization preserved the established fast-path semantics;
- convenient reproducibility fingerprints.

Do **not** use them as an absolute definition of mathematical correctness across every legitimate kernel geometry.

If SHA changes:

1. find the first divergence;
2. check hard state/algorithm invariants;
3. check operator numerical oracle;
4. classify first argmax flip as explainable near-tie or unexplained bug;
5. only then decide whether the change is acceptable.

A different SHA after an allowed first fork is not itself a failure.

---

## 9. SPECIAL RULE FOR B1/B2 / APPROXIMATE HEADS

Do not conflate two phenomena.

### Ordinary optimized kernel

Computes the same intended mathematical operator with a different reduction/order/geometry.

Allowed: small oracle-bounded numerical difference, including explainable near-tie argmax flip.

### Approximate shortlist/head

Changes the algorithm by using coarse scores / excluding vocabulary rows before exact rerank.

Current production classification:

- B2 on current Qwen3.8 checkpoint: **empirically validated accelerator**;
- B1/W1 rescue-row/pin tricks: **research-only, rejected**;
- arbitrary fine-tune universal safety: requires a trusted fallback/proof, not “it worked on our corpus.”

If B2 disagrees with the trusted full fast verifier, compare against the full verifier, not merely ordinary T1. A healthy-margin missing winner is an approximation failure.

Do not use the batched-vs-T1 near-tie allowance as a loophole for checkpoint-specific shortlist errors.

---

## 10. MTP WIDTH 4/5: WHAT MUST BE PROVEN BEFORE CALLING IT A BUG

The fact that MTP widths generate a different long-run sequence is **not sufficient**.

Before opening an MTP-width correctness fix lane, require at least one of:

- same logical/history state but a hard verify/accept invariant fails;
- invalid/future column influences a valid column;
- KV/GDN/conv/recurrent state after committed prefix is wrong;
- width-dependent indexing/frontier/rollback is wrong;
- operator output fails its frozen numerical oracle by more than allowed compute-class tolerance;
- first argmax fork has a healthy reference margin that cannot be explained by measured fast-vs-reference logit perturbation;
- eager vs graph logical results differ under same arithmetic path;
- deterministic repeats disagree.

If none of those is demonstrated, classify the width divergence as:

`PASS_NUMERICAL_NEAR_TIE` or `NUMERICAL_NON_BITWISE_EQUIVALENCE_PENDING_MARGIN_CAPTURE`

—not `investigate_mtp_correctness` by default.

A status string like `mtp_width_gt3_changes_greedy_output_even_full_exact` is **insufficient by itself to authorize a fix lane**.

This line explicitly overrides older orchestration that branches on output inequality alone.

---

## 11. PERFORMANCE PROMOTION HEURISTIC

For a proposed SM75 kernel optimization:

### Promote / keep testing when

- math/state invariants pass;
- operator numerical tests pass frozen tolerances;
- deterministic behavior is stable;
- any sequence fork is first-divergence near-tie explainable;
- full-model speed improves reproducibly.

### Reject when

- it breaks state/accept/causal invariants;
- requires loosening oracle tolerances;
- produces unexplained healthy-margin argmax flips;
- adds checkpoint/token-specific rescue logic;
- only wins microbench but not full-model performance;
- creates run-to-run nondeterminism.

### Do not reject solely because

- output SHA differs from T1/upstream after an explainable near-tie fork;
- hidden/logits are not bitwise equal across different legal kernel geometries;
- MTP trajectory differs after a legitimate sequence fork.

---

## 12. WHAT TO DO NOW

The owner/adviser priority remains:

1. Stop ordinary-T1 bitwise parity chasing unless a **hard failure defined above** is demonstrated.
2. Preserve the MTP diagnostic artifacts; do not commit slow T1-forcing diagnostics.
3. Restore the tracked exact72 tree to `38369da` unless a genuine state/math bug has been proven and minimally fixed.
4. Resume NCU-backed single-GPU performance work.
5. Start with the bounded Q4 A8 SwiGLU load/compute overlap/software-pipeline hypothesis supported by ~549 GB/s effective traffic and ~67.5% long-scoreboard.
6. If Q4 pipeline is neutral/regressive, document and stop rather than geometry-thrashing.
7. Then pursue the other evidence-backed single-GPU work / MTP economics without imposing T1 bit identity.
8. Then move to dual-GPU layer sharding, with **full native 262144 context as the first dual-GPU milestone**.

Healthy internal benchmark UUID remains:

`GPU-020906b4-0904-3471-e4ee-e41ec2350be3`

Healthy eGPU UUID remains:

`GPU-2fffc240-5d58-fbc8-27d6-6b349448fda9`

No CUDA P2P exists between them.

---

# ONE-LINE OVERRIDE FOR AUTOMATION

**OUTPUT INEQUALITY ALONE IS NOT A BUG. Do not open/continue an MTP correctness-fix lane unless a hard state/algorithm/oracle failure or an unexplained healthy-margin first argmax flip is demonstrated. Otherwise classify the difference as allowed/pending numerical non-bitwise equivalence, restore diagnostics, and resume NCU-backed performance work.**
