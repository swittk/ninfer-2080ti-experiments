# PLEASE READ — reuse the inherited RTX3090 concurrent Engine; dual harness must plug into it, not replace it

Owner/adviser correction, 2026-08-17 20:44 +07. This supplements/corrects the 20:35 adaptive-serving note after auditing the actual repository lineage. The repository was cloned from `Don-Chad/ninfer-3090` (`release/v0.6.0-rtx3090`), and the sophisticated small-scale concurrent serving architecture is still present in the current tree. **Do not rebuild this subsystem from scratch.**

This note does NOT interrupt the current single-session dual-MTP qualification/performance work. Finish the current bounded execution work first.

## 1. Important fact: the inherited 3090 fork already has the serving architecture we want

The RTX3090 release includes and documents:

- startup-bounded `max_concurrency=1..8`;
- active decode batch that dynamically grows/shrinks each round (`B=1..C`);
- late-arriving requests joining at safe round boundaries;
- exact-B batched ordinary/MTP execution rather than permanently padding to C;
- shared physical paged-KV pools;
- per-request logical context limits/entitlements rather than statically dividing KV by lanes;
- page materialization/growth/trim/reclamation;
- compatible-prefix/retained-state reuse;
- protected-head resource-aware admission/backfill;
- per-request GDN/recurrent/speculative state ownership;
- published true concurrent serving benchmarks and `tools/bench/run_serve_concurrency.py`.

Relevant existing sources to read/reuse:

- `docs/maintainer/concurrent-inference-architecture.md`
- `docs/maintainer/paged-kv-cache.md`
- `src/targets/qwen3_6/impl/runtime/program_impl.h`
- `src/targets/qwen3_6/impl/runtime/layouts_impl.h`
- `tools/bench/run_serve_concurrency.py`
- RTX3090 root README / release branch for Qwen3.8 C1/C2/C4/C8 behavior.

The existing documented scheduler already has the desired shape:

```text
[A]                 -> exact B=1
[A,B]               -> exact B=2
[A,B,C]             -> exact B=3
[B,C] after A ends  -> exact B=2
[B,C,D] after D joins -> exact B=3
```

A `max_concurrency=C` engine is a capacity bound, not a permanent C-wide decode.

## 2. We did NOT lose this architecture in the SM75 refactor

Current-tree audit shows `ProgramImplCore`, `max_concurrency`, per-lane state, MTP batch execution, graph families by exact batch size, shared paged-KV planning, admission/resource machinery, and the concurrency benchmark runner are still present. The core runtime differs from the RTX3090 base but has not been replaced/deleted.

The reason the new dual path is not using this machinery is architectural bypass, not loss: `apps/dual/main.cpp` directly constructs two `TextShardStage` objects and implements `dual()`, `dual_multi()`, and `dual_mtp3()` itself. `TextShardStage` currently reserves/materializes full per-session capacity explicitly. This was a useful experimental path for proving:

- layer-sharded correctness;
- 262K capacity;
- independent GDN/KV state;
- sharded target verification/replay/fold;
- canonical sharded MTP3.

It should NOT become the permanent serving architecture.

## 3. Desired integration direction

After the current dual-MTP execution path is correctness-qualified and its main performance bottlenecks are understood, make the dual/sharded executor participate in the existing Engine/Program scheduling contract. Prefer adapting/exposing an execution backend boundary rather than copying scheduler semantics into `apps/dual`.

Conceptually:

```text
OpenAI/Anthropic server
        |
existing admission + retained-prefix + scheduler
        |
RoundMembership / exact active B
        |
existing Program request/state lifecycle
        |
execution policy/backend
   |                 |
1-GPU SM75        2-GPU sharded SM75
canonical ops     canonical sharded MTP/target ops
```

The scheduler should remain authoritative for request join/leave, compact batch membership, output budgets, prefix reuse, cancellation and admission. The paged-KV allocator should remain authoritative for shared physical capacity and per-request entitlements/materialization. The sharded backend should own only what is genuinely multi-GPU-specific: stage-local model/state views, boundary transfer, sharded verify/proposal execution, replay/fold coordination, stage overlap and dual-specific graph/timing policy.

## 4. Do not regress the existing bounded-admission guarantee

The RTX3090 architecture deliberately reserves/entitles enough lifetime capacity for an admitted request's prompt + output contract; it does not blindly overcommit pages and hope another user finishes. Preserve that safety/fairness property.

At the same time, physical pages are a shared pool and are materialized as frontiers grow. `max_concurrency` must not statically partition context memory into C equal slices.

## 5. Context-growth product behavior

Preserve inherited compatible-prefix behavior: a long-lived conversation can grow turn by turn within the Engine's configured per-sequence ceiling without model restart. Reuse the resident prefix/KV/GDN state and append only the new suffix. The startup maximum/resource envelope is fixed, but current live length/materialized pages are not.

For dual mode, both stage-local pools/state must follow the same logical request frontier/ownership contract.

## 6. Performance goals remain three separate gates

1. One GPU / one user INT8 >=80 useful tok/s.
2. Two GPUs / ONE user >=100 useful tok/s + native/near-native 262K-class context capability.
3. Two GPUs / multiple users >=100 aggregate minimum, preferably much higher.

The existing concurrent scheduler should eventually switch naturally between exact B=1 and B>1 based on live requests. A lone user should receive the optimized dual-single-user policy; when more users are decode-ready the same Engine should use packed/pipelined multi-user work.

## 7. Immediate sequencing

A. Finish current single-session dual MTP3 qualification/optimization.
B. Get a first bounded multi-user dual-MTP proof/benchmark if useful to validate sharded batching.
C. Then integrate the sharded backend into the inherited `ProgramImplCore`/scheduler/paged-KV serving architecture rather than polishing `apps/dual` into a server.
D. Re-run the inherited concurrency benchmark style on SM75/dual: C1/C2/C4 where memory permits, active-batch join/leave, prefix reuse, resource admission, and populated-context growth.

Treat the RTX3090 concurrency implementation as a major asset we already own. Do not throw it away.
