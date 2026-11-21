# OWNER PRIORITY CORRECTION — STRICT C1 NEEDS A WHOLE-ROUND ARCHITECTURE CAMPAIGN

Effective 2026-09-15. This sharpens the existing `AGENTS.md` rule about new kernel/dataflow families and modeled complete-round savings. It does not weaken any correctness, safety, genericity, multimodal, no-training, or no-subagent rule.

## The problem

Strict C1 D3 is still roughly 80–81 useful tok/s and the owner gate is >=100. That is about a **1.24x end-to-end requirement**. A continuing sequence of independent 1–3% local kernel ideas is not an adequate primary strategy. Small isolated improvements are useful evidence, but they are not the campaign.

## Required primary strategy

After any already-bounded experiment currently in flight is classified, make the primary strict-C1 campaign an **architecture search over large fractions of a target round**. Think in terms of combined Q4 gate / Q5 down / GDN / projection / target-round weight traffic and execution ownership, not another sequence of tiny operator-local mutations.

Before implementing a large candidate, write an explicit performance budget:

- current strict C1 useful tok/s and round time;
- fraction of the complete round covered by the proposed architecture;
- current physical weight traffic / instruction / synchronization cost for that covered fraction;
- proposed reduction or reuse mechanism;
- Amdahl-predicted end-to-end ceiling;
- the minimum oracle result required to justify full implementation.

The architecture should have a credible path to a **large end-to-end movement**, ideally enough to attack most of the remaining ~24% throughput gap. A merely 1–3% complete-round ceiling is not a primary C1 candidate. Existing >=8–10% modeled-saving language is a minimum screening floor, not the desired final scale.

Examples of appropriately large search directions include, without mandating one exact implementation:

- cross-token / D3 weight-fragment reuse so one load/dequantization feeds more live verifier/proposal work before eviction;
- a different CTA/warp ownership model where physical weight traffic is organized around the whole live D3 target round rather than separate GEMV-like calls;
- larger producer/consumer pipelines spanning adjacent dense operations;
- mechanically derived SM75-specific packed layouts that change the physical dataflow materially;
- whole-round or large-slice fusion where register/shared-memory lifetime and weight reuse justify it analytically;
- another genuinely different tensor-core/dataflow family with a measured oracle that supports the required whole-round ceiling.

## Role of small experiments

Small microbenchmarks are **branch-pruning tools for a large architecture hypothesis**. They are encouraged when they cheaply test the largest uncertain premise. If the oracle fails, kill the architecture cheaply. If the oracle passes materially, **follow through into the full operation / whole-round implementation** rather than immediately moving to another cute local idea.

Do not brute-force warp/tile/cache knobs. Do not try to sum dozens of unrelated 1% wins as the main plan. Do not substitute session-copy, margin policies, workload-specific heuristics, cached answers, or serving aggregate throughput for generic strict C1 kernel/runtime progress. Keep useful optional features, but freeze them once bounded and return to the raw shared primitive.

## Genericity

Large kernel/dataflow work must remain checkpoint/fine-tune agnostic within the supported architecture/artifact contract: hardware/model geometry/quant format specialization is allowed; literal checkpoint values, learned adapters, prompt lookup tables, per-checkpoint calibration constants, or behavior-specific hacks are not. Mechanical transformations must derive from loaded weights and invalidate/rebuild with their source.

## Immediate execution rule

Finish/classify the already-bounded GDN precision/yield A/B if it is still in flight. If it does not produce a large real strict-C1 improvement, close it. Then explicitly choose and budget a **large whole-round architecture** and use targeted oracles only to decide whether to commit to that architecture. Strict C1 >=100 remains independently mandatory.
