# CORRECTED HARD OWNER PERFORMANCE GATES — INT8 SINGLE GPU >=80; DUAL GPU >=100+

**2026-08-17 14:02 +07 — this supersedes the 14:00 note.**

The owner corrected the acceptance requirement. Do NOT require 80 tok/s at a near-262K populated history on one GPU; that was an incorrect interpretation.

## Hard performance floors

### One user, ONE GPU, INT8-G64 KV

The normal intended single-GPU serving lane must achieve **at least 80 useful decode tok/s** with INT8-G64 KV.

This is the minimum performance bar for the INT8 single-GPU lane. It does not require native 262K context on one card. Benchmark representative practical contexts and preserve correctness/quality; do not promote an INT8 serving lane below 80.

Current measured reference on the exact72/B2 worktree with INT8-G64 KV is **64.96 tok/s** on the canonical 8192-capacity / 1024-generated-token workload, so the present gap to the owner's floor is **15.04 tok/s, about +23.2%**.

The historical ~72 tok/s BF16-KV lane is useful technical evidence, but it is not the owner's final deployment target. INT8 is the primary serving/performance configuration.

### DUAL GPU mode

If both GPUs are used for one logical serving mode/request, the owner only considers it worthwhile if it reaches **100+ useful decode tok/s**. Treat **100 tok/s as the minimum serious dual-GPU performance bar**, with materially above 100 preferred.

Do not claim success merely because two cards unlock 262K capacity. Capacity is necessary, but a dual-GPU performance mode that remains below 100 tok/s is an intermediate implementation, not the final result.

For multi-user dual serving, continue to report per-user and aggregate throughput separately. Do not hide weak per-request performance behind aggregate accounting.

## Implications

1. **Single-GPU TU102 optimization is not done.** Recovering 64.96 -> >=80 on INT8 is now a mandatory first-class target.
2. Continue TU102-specific INT8 attention NCU/SASS work. The existing INT8 GQA path contains a 170-SM-oriented launch heuristic and cp.async-oriented staging that degrades to synchronous copies on SM75; this is directly relevant.
3. Preserve and stack the already-qualified B2/MTP ideas where they remain numerically acceptable. Do not fall back to BF16 KV merely to make the benchmark easier.
4. Dual-GPU sharding remains important for native 262K capacity, but it is an amplifier/capacity mechanism, not an excuse for a slow per-GPU path.
5. After the INT8 single-GPU path reaches/exceeds 80, push dual single-request serving toward >=100 by combining the optimized kernels with MTP/B2, stage balance, low-overhead boundary transfer, graphing, and any safe intra-round/concurrency opportunities.
6. Multi-user work remains important after that; benchmark 1/2/3/4 active sessions with per-user and aggregate useful tok/s.

## Do not misread the targets

- **80 is the single-GPU INT8 floor**, not the BF16 floor.
- **100+ is the dual-GPU floor** for a worthwhile dual performance mode.
- Native 262K capacity is a separate functional requirement for dual sharding.
- Intermediate experiments may of course be slower; these are promotion/final-result gates.
- 100+ per user for several simultaneous users remains a stretch objective worth pursuing, but report honestly what the two TU102 cards can sustain.

Current trajectory should be:

**INT8 TU102 single-GPU >=80 -> integrate the same optimized lane into dual sharding -> dual single-request >=100+ while preserving native 262K capacity -> then maximize 2–3 user per-user/aggregate throughput.**
