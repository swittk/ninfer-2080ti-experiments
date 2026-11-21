# SUPERSEDED — OWNER CORRECTED THIS REQUIREMENT AT 14:02 +07

Do not use this file as the current acceptance gate. See `PLEASE_READ_THIS_2026-08-17_1402_CORRECTED_HARD_GATES_INT8_SINGLE80_DUAL100PLUS.md`.

# HARD OWNER ACCEPTANCE GATE — INT8 SINGLE-USER FULL-CONTEXT DECODE MUST BE >=80 TOK/S

**2026-08-17 14:00 +07 — this is a final acceptance requirement, not a suggestion.**

For the finished full-context mode, the owner will not accept less than **80 useful generated tokens/second for ONE active user** using **INT8-G64 KV**.

Unless later clarified otherwise, **"full context" for the final performance proof means a genuinely populated near-native context/KV history, not merely allocating `--max-context 262144` while decoding at a short history.** The native model context is 262,144. Benchmark points below it remain useful diagnostics, but an empty/full-capacity allocation does not satisfy this gate.

## Final gate

A final promoted full-context configuration must demonstrate:

- one active logical request/session;
- dual-GPU sharded mode as needed for capacity;
- INT8-G64 KV;
- near-native populated history, with a final proof at/near 262,144 visible tokens subject only to the output-token headroom needed to decode;
- **>=80 useful decode tok/s** sustained over a meaningful generated-token interval;
- report MTP drafted/accepted/fallback/useful-tokens-per-round so speculative accounting is transparent;
- report per-stage compute time, output-head time, boundary time, GPU DRAM bandwidth/utilization and memory headroom;
- no aggregate-throughput accounting: 40 tok/s + 40 tok/s on two users is NOT 80 tok/s single-user;
- no prefill TPS substitution;
- no benchmark that merely reserves 262K capacity while the live attention window is tiny.

**100+ tok/s single-user full-context remains the stretch target.**

Experiments and intermediate milestones may of course run below 80; this is a promotion/final-result floor.

## Current measured gap

The current checkpoint-qualified B2/MTP3 lane measured **64.96 tok/s with INT8-G64 KV** on the canonical short/moderate-context 8192-capacity workload. Thus even before paying the cost of a truly large populated attention window, the INT8 lane needs about **+23.2%** to reach 80 tok/s.

The current dual correctness prototype is ordinary synchronous T1 and ~22–24 tok/s decode. Do not compare that directly to the 80 final target: it intentionally lacks MTP/B2/turbo and is an architecture bring-up path.

## Why dual GPUs alone do not satisfy single-user latency

For one autoregressive/speculative round, stage 1 depends on stage 0 and the next round depends on the verifier/head result. Multi-user pipeline overlap can raise aggregate throughput, but it does not magically halve one user's round latency. Therefore a weak per-stage INT8 kernel cannot be hidden behind multi-user scheduling for this gate.

The path to >=80 single-user full-context must include genuine per-round efficiency gains: TU102-specific INT8 attention optimization, efficient sharded MTP/B2, high useful-token acceptance, low synchronization/boundary overhead, sensible stage balance, and any other measured architectural/kernel wins.

## Do not declare the Turing kernel frontier closed yet

The existing INT8 GQA code visibly contains non-TU102 assumptions and Ampere-oriented staging behavior:

- an INT8 launch heuristic comment targets a **170-SM wave**, while TU102 RTX 2080 Ti has 68 SMs;
- `gqa_attention_decode_i8.cuh` uses `cp_async`/commit/wait staging, but on SM75 `memory.cuh` degrades those operations to synchronous global->shared copies with no-op commit/wait;
- special INT8 split policies concentrate on particular T5/T6 and <=~8K windows, while the owner deployment path is primarily MTP3 and must remain fast through long populated windows.

This is now a required NCU/SASS investigation area. Tune against TU102's actual 68 SMs, native s8 MMA, no cp.async, and long-window INT8 KV behavior.

## Required performance curve during development

Measure INT8 single-user decode at representative populated histories such as ~1K, 8K, 32K, 64K, 128K, and near 262K, using the real intended MTP/B2 path as soon as it is available. This curve should guide optimization and expose where attention overtakes weight streaming.

Do not optimize only the 71-token canonical prompt and infer the full-context result from it.

**Final trajectory:** make INT8 TU102 fast -> integrate it into sharded MTP/B2 -> prove >=80 tok/s for one near-full-context user -> then optimize 2–3 concurrent users without sacrificing that single-user floor.
