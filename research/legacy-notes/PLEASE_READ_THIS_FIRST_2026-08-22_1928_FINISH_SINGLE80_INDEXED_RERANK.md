# 19:28 adviser — finish the qualified P0-M+B1 single80 package; exact indexed rerank next

Read repo-root `AGENTS.md` first. Stock GPUs only; no hardware tuning/reset/reboot. No agents or behavioral training. Hard gates remain 80 / 100 / 100 / 100 useful complete-path tok/s.

## Current stop state
At 19:20 both GPUs are stock-idle and no NInfer/build/research producer is live. Last substantive artifact is P0-BE at 17:27. Source was restored. The worker stopped after completing the K64/K128 paired benchmark but before writing its conclusion / choosing the next lane.

## Current single frontier
The important package is repaired P0-M Q4G64/Marlin factory-MTP + combined B1 target/proposal heads.

- repaired P0-M alone: **79.01 t/s** vs 76.35 baseline, exact/deterministic.
- P0-M + combined B1 initial paired E2E: **80.09 t/s**, deterministic, same output.
- six-family qualification: **66.08 -> 67.38 t/s (+2.0%)**, deterministic, quality good; general/code/Thai/Japanese exact, JSON/Arabic benign output forks under the corrected behavioral contract. Qualified for promotion.
- promoted six-run stock recert: **79.93 median**, min 79.66. Therefore single80 is NOT certified yet.

This is extremely close. 128/79.93 - 128/80 is only ~1.40 ms over the complete 128-token run, roughly **32 us per decode transaction** if spread over the 43 MTP rounds + fallback. Do not abandon the qualified package for another giant speculative architecture before exhausting one or two exact tens-of-microseconds opportunities.

## Latest micro-lanes
- P0-BD Marlin lock reuse: 79.775 -> 79.81 median; paired delta ~+0.05 t/s. Exact but below gate; effectively noise-sized.
- P0-BE B1 K64 changed BOTH proposal and target B1 together. K128 median **79.805**, K64 **79.85**; paired median delta ~-0.005 t/s / mean +0.016. Exact and below gate. Write the missing conclusion: CLOSED/NEUTRAL as a joint K64 change.

### Cheap discriminator still missing: asymmetric K
Because P0-BE changed both heads together, test exactly two asymmetric variants before closing candidate-count work:
1. proposal K64 + target K128;
2. proposal K128 + target K64.
The two paths have different economics and one may be helping while the other cancels it. Do not sweep K values beyond this. Use interleaved stock reps against promoted K128/K128. If neither gives a clear >=~0.1% complete-path win, close candidate-count tuning permanently.

## Fresh adviser Nsight profile of the qualified P0-M+B1 binary
Artifact: `.chatgpt/adviser_1924_p0mb1_profile/`. 64 generated tokens, 19 MTP rounds + 1 fallback; profiling run is attribution only, not a gate number.

The B1 exact rerank still performs explicit candidate-row gathers into temporary contiguous buffers:
- target W8 `gather`: 22 calls, **16.884 us/call**, 0.371 ms total;
- proposal Q4 `gather_exact`: 69 calls, **5.129 us/call**, 0.354 ms total;
- target `finish`: 22 calls, 2.479 us/call;
- proposal `finish_exact`: 69 calls, 2.362 us/call.

The gather traffic alone is ~0.725 ms in this short profile, on the order of **30-36 us per decode transaction**, essentially the entire remaining median single80 gap. The finish kernels add roughly another ~10 us/transaction.

The exact rerank compute itself is already tiny:
- target exact W8 candidate rerank ~13.37 us/invocation;
- proposal exact Q4 candidate rerank ~9.74 us/invocation.
So copying selected rows to temporary storage before these tiny kernels is now disproportionately expensive.

## P0-BF — highest-priority exact micro-architecture: direct indexed rerank
Do NOT alter B1 coarse ranking or model semantics. Replace gather->rerank->finish with an exact candidate-ID indexed rerank path.

### Target W8
Build one custom T4 candidate-rerank kernel that takes the K candidate token IDs and directly reads each selected row from the original W8G32 codes/scales. Preserve the current exact W8 per-row arithmetic/reduction order as closely as practical. Write candidate logits/top key directly; ideally fold the tiny `finish` reduction into the same kernel or a single follow-up if necessary. Do not materialize `s.codes` / `s.exact_scales` candidate copies.

### Proposal Q4
Likewise, make the proposal exact-Q4 rerank consume candidate IDs and original Q4 codes/scales directly rather than `gather_exact` -> contiguous Q4 GEMV -> `finish_exact`. A CTA-per-candidate / fixed-row indexed schedule is acceptable. Preserve current exact candidate scoring arithmetic; only addressing/storage changes.

Benchmark target and proposal indexed rerank separately first. If each saves real microseconds without hurting coarse-candidate semantics, combine and run complete stock E2E. The combined gather ceiling is large enough to matter to the 79.93->80 gap.

## P0-BG — second exact micro-lane if BF is insufficient: collapse B1 top-k reduction kernels
The same profile shows B1 `topk_group` ~22.97 us/invocation and `topk_final` ~17.04 us/invocation, with ~91 invocations in the 64-token trace. P0-AY only swept producer CTA geometry; it did NOT classify the group+final reduction hierarchy.

Try one bounded alternative only: after `project_b1_local_top2`, use one CTA per column to read all partial candidates and produce the final K set directly, eliminating the intermediate group buffer / second launch. Use exact `sampling_sort_key` ordering and deterministic tie behavior. Microbenchmark first; abandon immediately if direct single-CTA scan is slower. Do not start a top-k algorithm zoo.

## CUDA research closures / don't waste time
- Programmatic Dependent Launch requires compute capability >=9.0; unavailable on SM75.
- CUDA L2 persistence/residency control requires compute capability >=8.0; unavailable on SM75.
- Conditional graph/device graph control does not help: P0-AR measured host CPU bookkeeping <0.001 ms/round and device self-tail launch was slightly slower.
- Do not reopen Token Recycling, LogitSpec, Lookahead, active vocab, candidate cache, SDFP, QuantSpec, QServe/Humming, or the failed producer/consumer pipelines. Their costed ceilings are already below gates.

## After single80
Once a stock repeated recert genuinely clears >=80, freeze/record that package and immediately return to the unsolved dual100 / 65K100 / 262K100 gates. Do not spend hours polishing single from 80.0 to 81.0. The long/dual gaps require a materially larger architecture.

## Autonomy
1. Write P0-BE conclusion.
2. Test the two asymmetric K variants cheaply.
3. Implement/benchmark P0-BF direct indexed exact rerank (target then proposal, combine if wins).
4. P0-BG one-shot direct top-k reduction only if still needed.
5. Re-run stock promoted single certification.
6. If >=80, continue autonomously to dual/long gates; if still <80, research a new exact mechanism rather than reopening closed families.
