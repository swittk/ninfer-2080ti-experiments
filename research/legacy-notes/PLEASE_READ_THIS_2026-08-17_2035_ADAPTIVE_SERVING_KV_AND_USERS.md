# PLEASE READ — production serving must adapt both compute scheduling and KV/context allocation

Owner/adviser serving-architecture note, 2026-08-17 20:35 +07. This does **not** interrupt the current dual-MTP correctness/performance qualification. Finish the current MTP3 work first. This note prevents the present fixed-session benchmark harness from accidentally becoming the production serving architecture.

## 1. The current fixed-session/full-capacity allocation is a scaffold, not the server design

`TextShardStage` currently reserves each configured session's full logical KV capacity and immediately calls `materialize_tokens(capacity_)` for both text KV and (when enabled) MTP KV. That was useful to prove 2x262K / 3x262K capacity and state isolation, but production serving should NOT require predeclaring N sessions and pre-materializing every session to its maximum context.

The production target should use the existing NInfer paged-KV concepts properly:
- one shared page pool per relevant cache;
- each live request/session owns only the page entitlement/materialized pages needed for its current frontier plus bounded speculative headroom;
- grow/materalize more pages as the conversation/context actually grows;
- return pages to the pool when a request ends, resets, trims, or otherwise no longer needs them;
- preserve the Engine's startup maximum/resource envelope, but do not confuse that maximum with every session's current materialized footprint.

Upstream runtime already has useful primitives/semantics around reserve/resize/materialize/trim of paged-KV bundles and resident-prefix append. Reuse/extend those instead of inventing a second static allocator inside `TextShardStage`.

## 2. Context must grow naturally inside a long-lived request/session

Desired product behavior:

turn 1: 500-token prompt + 100 generated
turn 2: compatible history grows to ~1K
turn 3: grows to 5K
...
turn N: can continue toward the configured/native context ceiling

No model restart should be needed merely because the live context grew. Compatible resident prefixes should reuse existing KV/GDN state and append only the new suffix. If the client resubmits the full conversation history, prefix matching/reuse should make this efficient.

The maximum context remains an Engine/resource-planning ceiling; growing beyond that ceiling may require another Engine/configuration. But growth **within** it should be page-on-demand and prefix-resident, not restart-and-reprefill behavior.

## 3. Compute scheduling must also adapt to the number of runnable users

Do not make final serving behavior a static `--sessions N` execution mode. The scheduler should dynamically react to the set of currently runnable requests:

- **1 active user:** latency/extreme-speed policy; use both GPUs as aggressively as possible for that one request. For dual single-user mode the hard product goal remains >=100 useful tok/s while preserving native/near-native 262K-class context capability.
- **2+ active users:** pipeline/pack ready MTP verification rounds across users; use continuous/in-flight scheduling so idle pipeline capacity is filled and weight reads can be amortized where profitable.
- when users join/leave or become temporarily non-runnable, rebalance automatically; a lone remaining user should regain the single-user fast path without a process restart.

Multi-user throughput should not simply be `single_user_speed / N`; aggregate throughput should usually rise as otherwise-idle pipeline slots become useful work. Report per-user TPOT/throughput and aggregate useful tok/s.

## 4. Memory admission and compute admission should be separate policies

A request can be compute-runnable yet unable to grow its context because insufficient free KV pages remain. Keep explicit admission/backpressure semantics:
- scheduler chooses which ready requests run next;
- KV allocator decides whether requested frontier growth can be admitted under the configured memory envelope;
- requests must not corrupt/steal one another's KV/GDN state;
- if a user reaches a memory limit, fail/trim according to explicit product policy rather than forcing an Engine restart for unrelated users.

Eventually consider prefix/page sharing for identical immutable prefixes only if correctness and ownership semantics stay clean; this is optional and lower priority than basic page-on-demand serving.

## 5. Preserve the three independent performance goals

1. One GPU / one user INT8: >=80 useful tok/s minimum.
2. Two GPUs / ONE user: >=100 useful tok/s minimum + native/near-native 262K-class context capability.
3. Two GPUs / multiple users: >=100 aggregate minimum, preferably materially higher, with acceptable per-user latency.

Adaptive serving should move between goal-2 and goal-3 policies automatically based on live concurrency; neither goal replaces the other.

## 6. Architecture constraint

Keep canonical model/state semantics in `TextContext` / `ProgramImplCore` / `Variant` / paged-KV primitives. `TextShardStage` should remain a thin sharding/state/transport adapter, and the serving scheduler should be a policy layer above it. Do not grow `apps/dual/main.cpp` into a permanent second serving runtime.

## Suggested sequencing

A. Finish current single-session sharded-MTP qualification and performance work.
B. Get the first correct/qualified multi-user MTP pipeline result.
C. Before calling the server architecture done, replace fixed full-capacity session materialization with dynamic paged-KV admission/materialization/reclamation and a live-request scheduler.
D. Benchmark context growth and user join/leave transitions explicitly, including 8K/32K/64K/128K/near-262K populated frontiers where feasible.

Do not optimize the benchmark scaffold into a dead-end static server.
