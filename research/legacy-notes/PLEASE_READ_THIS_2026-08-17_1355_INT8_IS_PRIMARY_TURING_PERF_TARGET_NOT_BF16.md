# OWNER CORRECTION — INT8 KV IS THE PRIMARY SERVING/PERF LANE; OPTIMIZE TU102, DO NOT USE DUAL GPU TO HIDE A SLOW PER-GPU PATH

**2026-08-17 13:55 +07 — read before claiming the performance frontier.**

The owner is happy to use **INT8-G64 KV for all serving**. Treat INT8 KV as the primary deployment/performance configuration, not merely a memory-saving fallback for giant context.

The previous canonical ~72 tok/s qualification used BF16 KV. That remains useful historical evidence for MTP3+B2 and weight-kernel performance, but it is **not the deployment TPS target**.

A direct apples-to-apples run on the current exact72/B2 worktree with only `--kv-dtype int8` changed measured:

- decode: **64.96 tok/s**
- overall: **61.98 tok/s**
- 1024 generated tokens
- INT8-G64 KV payload at 8192 capacity: **280.63 MiB**
- MTP3: 380 rounds, 1139 drafted, 642 accepted, 56.37%, 2.69 useful tok/round, fallback 1

The corresponding BF16 qualified lane is ~72 tok/s with 58.49% acceptance / 2.75 useful tok/round on the canonical trajectory.

Therefore **do not continue quoting 72 as if it were already the intended INT8 serving speed.** The immediate engineering target is to recover and then exceed that class of throughput on the INT8 deployment lane where hardware permits.

## INT8 correctness is already proven; do not redo old archaeology

Early SM75 correctness work already implemented and validated native INT8-G64 KV, including the Bc32/page-offset fix, native Turing integer MMA, GDN/GQA correctness, and retrieval/capacity gates. Historical handoff explicitly records this.

The dual runtime must expose/reuse that existing INT8 path and run bounded sharding regressions, but **do not spend days re-proving INT8 KV from scratch**.

## There is clear TU102-specific INT8 optimization work left

Do not interpret the prior Q4/Q5 short-context NCU frontier as the entire Turing frontier.

### 1. Current INT8 attention split heuristics contain non-TU102 assumptions

`src/ops/launcher/gqa_attention_decode.cu` contains an INT8 special case whose comment explicitly targets **"one 170-SM wave"**. The RTX 2080 Ti TU102 has **68 SMs**. The INT8 special split policies are also concentrated around T5/T6 and windows up to ~8K; larger windows fall back to generic tiering.

Retune the INT8 GQA launch geometry for the actual TU102 device and actual production widths/windows. At minimum measure T1/T2/T3/T4/T5/T6 at representative populated windows such as ~1K, 8K, 32K, 64K, 128K, and near 262K. Do not assume a split count tuned for a 170-SM GPU is near-optimal on 68 SMs.

### 2. Current INT8 decode staging is architecturally cp.async-oriented, but Turing has no cp.async

`gqa_attention_decode_i8.cuh` describes asynchronous K/V code+scale staging and uses `cp_async`, `cp_commit`, and `cp_wait` helpers.

On SM75, `src/ops/common/memory.cuh` explicitly maps `cp_async` to a synchronous global->shared `sync_copy`; commit/wait become no-ops. Thus a latency-hiding design that is sensible on Ampere+ partially collapses into synchronous loads on Turing.

This is a major candidate for real SM75 work. Use NCU/SASS to determine whether an SM75-specific path can do better with techniques such as:

- vectorized LDG into registers followed by shared stores;
- manual software double buffering;
- earlier code/scale prefetch to increase load-to-use distance;
- warp-specialized producer/consumer scheduling designed around Turing's actual load/barrier behavior rather than cp.async semantics;
- alternate Bc / warps-per-CTA / split-KV geometry for 68 SMs;
- reducing barrier cost and register pressure while retaining enough independent memory operations;
- direct/register-resident dequant paths where they reduce shared-memory traffic;
- separate tuning for T3 MTP production because that is the main deployment width, not only T5/T6 edge cases.

Do not blindly apply the failed Q4 scale-prefetch lesson to attention; profile this kernel directly because its memory/dependency structure is different.

## Benchmark the actual deployment curve

The performance matrix must now use **INT8 KV by default** and report populated history, not only max allocation:

- 1 / 2 / 3 active users;
- representative live histories 1K, 8K, 32K, 64K, 128K, near 262K;
- MTP3+B2 first, then wider/adaptive speculation where profitable;
- per-user and aggregate useful tok/s;
- MTP acceptance/useful tok per round;
- stage0/stage1/head/boundary timing;
- NCU DRAM bytes/bandwidth, long scoreboard, occupancy, barriers, L1/L2, registers;
- per-session INT8 state bytes and remaining VRAM.

The owner wants at least **2–3 native-context-capable users around 70–80+ tok/s each**, with **100+ each as the stretch target**.

## Dual GPU is an amplifier/capacity mechanism, not an excuse

Dual-GPU sharding remains required for native full context and multi-user pipeline throughput, but do not treat a second card as a substitute for optimizing each TU102.

The desired order from here is:

**make the INT8 single-stage/per-GPU kernels genuinely Turing-fast -> wire that lane through the sharded runtime -> batch/pipeline 2–3 users so both GPUs stay occupied -> measure real per-user throughput.**

If multi-user batching allows one weight read to serve multiple sequences, exploit it aggressively. But report the underlying single-stage and one-user numbers too, so aggregate scheduling cannot hide a weak kernel.

The project is specifically about extracting unusual value from these Turing cards. Use the hardware they actually have: 68 SMs, native s8 MMA, no cp.async, no P2P, and high local GDDR6 bandwidth. Optimize for those facts rather than inheriting Ampere/Blackwell launch assumptions.
