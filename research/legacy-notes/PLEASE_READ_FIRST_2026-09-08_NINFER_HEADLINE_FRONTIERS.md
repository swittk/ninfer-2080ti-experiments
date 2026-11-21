# OWNER GENERALIZATION RULE — NO CHECKPOINT-SPECIFIC SPEED TRICKS

Headline kernel/runtime optimizations, including strict-C1 work, must be **checkpoint/fine-tune agnostic within the supported model architecture and artifact format**. They may specialize to real hardware/model geometry/quantization format (e.g. stock SM75, Qwen3.8-27B tensor shapes, canonical Q4/Q5 G64), but must not depend on literal checkpoint weight values, model hashes, token/prompt tables, per-checkpoint calibration/search, learned adapters, or hand-tuned constants derived from one fine-tune. Any prepared representation must be generated mechanically from the weights actually loaded and must be invalidated/rebuilt with that source.

A candidate tested on one checkpoint is not thereby proven universal: before broad promotion, retain randomized/source-identity tests and, when another real compatible fine-tune/checkpoint is available, run a cross-check. Numerical approximations must still satisfy the owner's quality gates on each qualified checkpoint. Performance wins that only exist for one checkpoint do not count as generalizable headline progress.

# OWNER HARD GATE OVERRIDE — STRICT C1 >100 REQUIRED; RELAXED 107 DOES NOT COUNT

Effective immediately, the owner's single-GPU/single-user gate is **STRICT C1 D3 >=100 useful tok/s** on one stock RTX 2080 Ti with acceptable output quality. Strict means `greedy_margin_delta=-1`; C1 remains D3. The current strict matching-fixture floor is ~79/80.5 tok/s, so the gate is OPEN and performance work must push upward from that strict lane.

The ~107 margin7 route is preserved as historical/experimental evidence only. It has **ZERO owner C1 gate credit** because same-prompt long Thai shows severe language soup versus the clean strict control. Language soup, comparable semantic drift, pathological repetition, or quality degradation is a **HARD FAIL regardless of tok/s**. Do not call 107 a passed single-user gate, do not use it as the C1 north-star floor, and do not substitute policy-assisted throughput for strict-path kernel/runtime progress.

Immediate priority: **strict C1 D3 architecture/runtime efficiency**. Target >=100 first, then continue toward 120/150/200+ if physically real. Preserve margin7 and all prior experiment paths, but do not spend the primary campaign optimizing a relaxed lane while strict remains below100. C4=D1 and physical C8=C4+D1 per GPU x2 remain unchanged. Finish only already-bounded validation from another lane; next architectural campaign is strict C1.

Machine authority: `tests/retained_sm75_profiles.json` + `scripts/check_retained_sm75_profiles.py`. Primary prose authority: `PLEASE_READ_FIRST_2026-09-08_NINFER_HEADLINE_FRONTIERS.md`.



# OWNER QUALITY-ENVELOPE CORRECTION — MARGIN7 IS NOT UNIVERSAL LONG-THAI (2026-09-08)

Same-prompt 768-token evidence now directly separates model/runtime baseline from margin-policy damage. On the community-library Thai prompt, the current strict control stays overwhelmingly Thai (~96.0% Thai letters, ~4.0% Latin, no CJK), while retained C1 D3 margin7 falls to ~16.5% Thai letters / ~82.9% Latin plus CJK leakage and becomes largely English. This is not normal Thai code-switching. Evidence: `.chatgpt/production_serve_20260908/retained_c1/strict_unchanged/cold.json` versus `.chatgpt/production_serve_20260908/retained_c1/retained_six/cold.json`; longer review: `completion_0600/longer_quality_review.json`.

Therefore keep the performance facts distinct:
- **79.01 / ~80.5 current strict C1** is the clean strict/generalizable C1 reference.
- **107.27 / ~107-108 current margin7** remains a real short/general-code fast-path throughput result under the owner-accepted relaxed NInfer equivalence policy, but it is NOT a universal useful-token rate for arbitrary language/output length.
- Do not use the old six-family `quality_pass` alone as strong multilingual quality evidence: the historical 256-token Thai sample itself visibly contains abnormal Chinese/English mixing despite that automated pass.
- Existing 128/256 accepted receipts remain retained historical facts; long-horizon Thai requires stricter qualification before margin7 may be called a universal product default.
- For kernel/generalizable raw-power claims, do not let policy-assisted margin7 substitute for strict-path progress. Preserve and improve both lanes.
# NInfer AUTHORITATIVE HEADLINE FRONTIER LEDGER — OWNER / DO NOT RE-DERIVE

Updated Bangkok: 2026-09-08 05:31 +07:00.

This file is the primary performance/semantics authority for the active SM75 continuation. It supersedes scattered chat recollection and any older note that collapses distinct historical routes. Accepted paths are **cumulative**: a new path never erases an older accepted headline. Keep the callable/source/artifact/fixture/evidence for every retained line below.

## Owner terminology and settled topology

- **C1 = one stock RTX 2080 Ti, one real user/request, MTP D3.** Do not re-sweep D1/D2/D3 as a topology choice.
- **C4 per GPU = D1.**
- **Physical C8 production = C4+D1 on GPU0 + C4+D1 on GPU1.** One-card C8/T16/T32 are research/oracle geometries, not the production topology.
- **Strict** means the NInfer strict speculative same-same acceptance control (`greedy_margin_delta=-1`), not a claim that NInfer is bit-for-bit identical to an external upstream implementation in every internal numeric operation.
- **Margin7 / relaxed same-same** means the already owner-accepted NInfer MTP C1 policy with greedy logit margin 7. It is a distinct qualified decoding contract, not a hidden benchmark cheat and not to be silently conflated with strict acceptance.
- Raw stock-2080Ti **C1 useful decode tok/s is the kernel-generalizability north star**. C4/C8 aggregate wins are valuable serving wins but do not substitute for C1.

## Hard owner gates / service contract

| Gate | Required floor |
|---|---:|
| historical single gate | >=80 useful tok/s on its certified route |
| dual-short | >=100 useful tok/s |
| populated 65K | >=100 useful tok/s |
| genuine ~262K | >=100 useful tok/s |
| physical production C8 | >=160 aggregate tok/s AND >=20 useful tok/s per user |

The owner additionally wants raw C1 materially above the old frontiers: 120 -> 150 -> 200+ if physically real. These are aspirations, not permission to weaken output/quality policy.

## C1 headline ledger — KEEP ALL OF THESE SEPARATE

### A. Owner strict/current P0M headline: ~79
Evidence: `.chatgpt/post_reboot_20260821/p0_au_p0m_recert/final_current/result.json` and `CONCLUSION.md`.
- baseline median **76.35 tok/s**
- retained candidate median **79.01 tok/s**
- deterministic; delivered output byte-identical to baseline
- 43 rounds, 129 drafted, 83 accepted in each candidate repeat
- classification `LIVE_NEAR_SINGLE80_EXACT_CURRENT_P0M`

**This is the owner's strict headline reference.** Do not redefine strict C1 downward using later 768-token full-target server corpora.

### B. Older distinct certified SINGLE80 route: 80.185
Evidence: `.chatgpt/post_reboot_20260821/p0_bf_indexed_rerank/promoted_recert/result.json`.
- median **80.185**, mean 80.153, min 79.840, max 80.480
- deterministic, `CERTIFIED_SINGLE80`
- P0-M+B1 indexed-rerank lineage

This is a real older certified route and must remain reproducible, but it is **not a replacement label for the later 79.01 current-P0M strict reference**.

### C. Owner accepted C1 D3 margin7 headline: ~107
Historical evidence: `.chatgpt/post_reboot_20260821/p0_q4_single_delta7_p0m_recert/final_result.json`.
- fixed 128-token endpoint: **107.27 median** (107.45 / 107.33 / 107.21 / 107.12)
- D3, margin7, Q4G64 MTP + Marlin Q4, int8 KV
- 33 rounds / 99 drafted / 94 accepted, by-pos 33,32,29
- broader qualified family medians: general 107.555, code 107.27, JSON 103.90, Thai 72.68, Japanese 89.045, Arabic 88.895

Independent 2026-09-08 reproduction: `.chatgpt/production_serve_20260908/exact107_by_chatgpt/result.json`.
- **107.255 median** (107.59 / 107.35 / 107.16 / 107.05)
- exact same digest, 33/99/94 trajectory, by-pos 33,32,29, and exact same delivered stdout SHA as the historical fixed fixture

Do not call this route missing or regressed again.

### D. New strict full-target server proposer work — useful, but NO 79-headline credit yet
Evidence: `.chatgpt/production_serve_20260908/raw_c1/full_vocab_comparison.json`.
`--fast-full-vocab-draft` improved the newer **full-target/full-vocabulary strict server lane** across six sustained workloads by about 11.2-13.1% with byte-identical delivered outputs versus that lane's control:
- general 51.28 -> 57.76
- code 73.63 -> 82.70
- JSON 73.38 -> 82.30
- Thai 47.19 -> 52.47
- Japanese 45.70 -> 51.28
- Arabic 46.07 -> 52.11

**As currently implemented this does NOT yet advance the historical 79.01 strict headline.** The new option requires full proposal source + full native target verification and is explicitly incompatible with the old `--lm-head-draft` route used by the 79 fixture. The implementation reuses existing SM75 full-vocabulary ranking/rerank machinery; treat it as a valuable production integration/acceleration until a matching-fixture A/B proves otherwise.

Required experiment: reproduce the historical strict fixture on rebuilt current source, then compare the new proposer under a semantically matched strict policy. Do not compare unlike 768-token/full-target corpora to the 128-token 79 receipt and claim a new strict frontier.

## Dual / long-context headline ledger

- **Dual-short margin7:** 104.94 median, min 104.87, deterministic, passes dual100. Evidence: `.chatgpt/post_reboot_20260821/p0_fz_margin_p0m_joint/promoted_recert/result.json`.
- **Accepted Q64 short dispatch:** 106.01 median, deterministic, passes100. Evidence: `.chatgpt/post_reboot_20260821/p0_q64_retained_local_splits/final_text_matrix/result.json`.
- **65K exact Q45 path:** **107.77 tok/s**, 65,292 prompt tokens, byte-exact and trajectory-exact vs its 106.28 baseline, 26 rounds / 103 drafted / 100 accepted. Evidence: `.chatgpt/post_reboot_20260821/p0_q45_native_fp16_pv_operands/result65.json`.
- **Accepted Q64 65K dispatch:** **110.94 median**, min 110.91, deterministic, passes100. Evidence: `.chatgpt/post_reboot_20260821/p0_q64_retained_local_splits/final_text_matrix/result.json`. Keep this distinct from the Q45 exact 107.77 route.
- **Genuine ~262K accepted:** **104.53 median**, min 104.46, max 104.57, deterministic, quality pass, classification `ACCEPT_262K100`; 48 rounds / 191 drafted / 188 accepted. Evidence: `.chatgpt/post_reboot_20260821/p0_q64_retained_local_splits/recert262_f8/result.json`.
- **Exact Q45 PV genuine262 alternative:** 91.66 tok/s. It is below the 100 gate but is a preserved exact mechanism/evidence lane, not disposable. Evidence: `.chatgpt/post_reboot_20260821/p0_q45_native_fp16_pv_operands/result262.json`.
- Later production-integration text262 receipts around 103.4-103.6 tok/s exist; do not let them erase the certified 104.53 Q64 headline.

## Multimodal accepted headline ledger

Evidence roots: `.chatgpt/post_reboot_20260821/p0_q65_q64_multimodal_final/media/`.

Single accepted medians (all deterministic/quality pass; classification `SINGLE_MULTIMODAL_PASS`):
- image **108.26**
- mixed URL **104.68**
- video **105.29**

Dual accepted medians (all deterministic/quality pass; classification `DUAL_MULTIMODAL_PASS`):
- image **105.26**
- mixed URL **101.20**
- video **101.99**

Short factual media policy also preserves qualified natural D3 rather than blindly forcing D4.

## Physical C8 production-serving ledger

### First historical physical-C8 per-user gate pass
Evidence: `.chatgpt/post_reboot_20260821/p0_q109_physical_c8_proposal_b4/recert/r2/result.json`.
- **166.85 aggregate tok/s**
- **20.32 slowest/per-user floor**
- classification `C8_PHYSICAL_20_PASS`

### Current clean physical C4+C4 measurement
Evidence: `.chatgpt/production_serve_20260908/decode_architecture/node_c8/result.json`.
- warm common-window **256.969 aggregate / 26.921 slowest**
- p95 first content 0.786 s
- `MEASURED_NOT_CERTIFIED`

### Q5 batch serving measured frontier — NOT a C1 substitute
Evidence: `.chatgpt/production_serve_20260908/decode_architecture/q5_matrix_dual_1/result.json`, repeat `.chatgpt/production_serve_20260908/decode_architecture/q5_qual_repeat_1/result.json`.
- primary warm common-window **271.160 aggregate / 28.887 slowest**
- repeat **271.086 / 30.447**
- `MEASURED_NOT_CERTIFIED`; preserve as batch-serving win, not a raw-C1 breakthrough

### Latest same-source full-target dual regression
Evidence: `.chatgpt/production_serve_20260908/raw_c1/source_final_dual/result.json`.
- warm **259.417 / 30.124**, p95 0.782 s
- actual appended follow-up **240.721 / 26.497**, p95 3.266 s, 8/8 prefix hits
- post-cancel refill **248.230 / 27.421**, p95 0.789 s
- no non-cancel errors/length limits in those reported phases
- `MEASURED_NOT_CERTIFIED`

Do not compare any of these aggregate numbers to raw C1 as if they prove kernel elegance.

## Long-media / prefill / responsiveness retained headlines

Evidence: `.chatgpt/production_serve_20260908/cold_memory/serving_checkpoint.md` and indexed children.
- 15.4K long-media cold ingestion: **118.94 s p95** (still open/slow)
- warm long-media histories: **213.91 aggregate / 24.55 slowest**, p95 1.112 s
- refill after three deliberate cancellations: **212.99 / 25.35**, p95 1.083 s
- matched cold-8K HTTP cohort: **65.48 -> 55.46 s p95** after dense attention-input prefill work
- real 40-second media preparation: first **14.922 -> 1.575 s**; retained repeat **13.221 -> 0.124 s**, next **0.129 s**
- real HTTP first content for retained clip: **0.675 s warm / 0.845 s retry** versus ~14-16 s before scanner/cache fix
- late image/video join protected six established text clients at **>=23.19 useful tok/s** while newcomers waited; newcomer first-content **8.14 / 18.69 s**

These are workload-specific serving facts, not universal semantic or C8 certification claims.

## Current-source margin integration state

As of this audit, current source has gained a first-class `SpeculativeOptions::greedy_margin_delta` and `--mtp-logit-margin` CLI/server plumbing, logging, validation, and C1/D3 scoping. This is the right direction: the historical margin policy is no longer supposed to exist only as a hidden stale-binary env hook.

**No current-source headline is earned merely by wiring the option.** Required recertification sequence:
1. strict historical 79.01 fixture on rebuilt current source;
2. margin7 historical 107.27 fixture on rebuilt current source;
3. current HTTP/Engine C1 equivalents with measurement boundary labeled;
4. fast-full-vocab proposer A/B under strict and margin7 policies, preserving all old routes;
5. only then award a new C1 headline and proceed to deeper D3-specific Q4/Q5/GDN architecture work.

## Promotion / reporting rules

- Never call a different fixture, output length, target-head policy, acceptance policy, context length, or aggregate concurrency a regression/advance against a headline without a matching-fixture discriminator.
- `MEASURED_NOT_CERTIFIED` remains measured, even if faster than an older certified result.
- Preserve old binaries/evidence/source paths until current-source replacement is proven; do not delete historical paths because a new option exists.
- A microkernel/operator win earns no raw-C1 headline unless actual C1 on the correct policy/fixture moves.
- C1 strict (~79 current-P0M), legacy certified SINGLE80 (80.185), and margin7 (~107) are **three distinct retained facts**, not one number with moving wording.
- Accepted 65K/262K, multimodal, physical-C8, long-media, cancellation/prefix, and prefill/media-cache headlines are also cumulative and must remain in the retained-profile manifest.

## Current-source preservation update — 2026-09-08 06:04 Bangkok

The earlier integration checklist above is now partly completed, without replacing
any historical number or acceptance policy:

- Current rebuilt CLI margin7 + explicit `--mtp-retained-packed-prefix` exactly
  reproduces the historical128-token stdout hash, digest `ef4dc9c008396373`,
  33/99/94 counts and33,32,29 positions at108.06 and107.95 tok/s. Current HTTP
  reproduces the generated bytes/counts; CLI adds one terminal formatting newline.
- The matching current strict-P0M fixture with margin omitted and the explicit
  retained packed-prefix mode gives80.51 tok/s, exact original stdout/digest,
  43/129/83 counts and35,27,21 positions. This is preservation of the79.01 lineage,
  not a relabeling of the older certified80.185 route or a broad new performance seal.
- Full-target/full-vocabulary drafting on that same128-token input gives74.90 tok/s,
  despite identical delivered bytes/counts; it does NOT advance the79.01 lineage.
- Combining full-target drafting and margin7 on the new768-token six-family corpus
  is not a universal win. Japanese develops repeated phrases, so no new general
  promotion. The retained107 route itself has a longer Thai language-drift limitation
  on this new corpus. Preserve existing128/256 qualified facts; do not extrapolate them
  into a universal768-token quality claim.
- Both-device margin-operator tests, explicit parent/view bounds and cache lifetime,
  retained Engine warmup/repeat, cancellation/invalidation/deadline/lease, and three
  load/generate/destroy cycles pass. The prior strict full-target six-family output
  hashes remain unchanged. These checks do not rerun every genuine65K/262K profile.

Evidence: `.chatgpt/production_serve_20260908/retained_c1/` and `completion_0600/`.
All18 historical/measured profile witnesses remain protected by the manifest; final
physical C4+D1 regression is separately recorded there when terminal. No old headline
or old binary is deleted, and no aggregate C8 number substitutes for C1.

