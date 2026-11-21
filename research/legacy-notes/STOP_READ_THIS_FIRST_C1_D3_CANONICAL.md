# OWNER HARD GATE OVERRIDE — STRICT C1 >100 REQUIRED; RELAXED 107 DOES NOT COUNT

Effective immediately, the owner's single-GPU/single-user gate is **STRICT C1 D3 >=100 useful tok/s** on one stock RTX 2080 Ti with acceptable output quality. Strict means `greedy_margin_delta=-1`; C1 remains D3. The current strict matching-fixture floor is ~79/80.5 tok/s, so the gate is OPEN and performance work must push upward from that strict lane.

The ~107 margin7 route is preserved as historical/experimental evidence only. It has **ZERO owner C1 gate credit** because same-prompt long Thai shows severe language soup versus the clean strict control. Language soup, comparable semantic drift, pathological repetition, or quality degradation is a **HARD FAIL regardless of tok/s**. Do not call 107 a passed single-user gate, do not use it as the C1 north-star floor, and do not substitute policy-assisted throughput for strict-path kernel/runtime progress.

Immediate priority: **strict C1 D3 architecture/runtime efficiency**. Target >=100 first, then continue toward 120/150/200+ if physically real. Preserve margin7 and all prior experiment paths, but do not spend the primary campaign optimizing a relaxed lane while strict remains below100. C4=D1 and physical C8=C4+D1 per GPU x2 remain unchanged. Finish only already-bounded validation from another lane; next architectural campaign is strict C1.

Machine authority: `tests/retained_sm75_profiles.json` + `scripts/check_retained_sm75_profiles.py`. Primary prose authority: `PLEASE_READ_FIRST_2026-09-08_NINFER_HEADLINE_FRONTIERS.md`.

# READ THIS FIRST — HEADLINE FRONTIERS ARE CUMULATIVE

**Primary authority:** `PLEASE_READ_FIRST_2026-09-08_NINFER_HEADLINE_FRONTIERS.md`. Machine guard: `tests/retained_sm75_profiles.json` + `scripts/check_retained_sm75_profiles.py`.

Non-negotiable C1 distinction: owner strict/current P0M **79.01**, older distinct certified SINGLE80 **80.185**, accepted D3 margin7 **107.27** (independently re-hit 107.255). Do not collapse them. New full-target `fast-full-vocab-draft` work has **no strict79 headline credit yet** because it is a different proposal/target route; matching-fixture A/B is required. Preserve dual-short/65K/262K, multimodal, physical-C8, long-media and prefill/media-cache headlines from the ledger as cumulative evidence.

Current priority: current source now has explicit `--mtp-logit-margin`. Rebuild and recert both the strict79 fixture and margin7~107 fixture on current source; then test the new full-vocab proposer under both policies without deleting the historical paths. Only after matched policy/fixture proof may a new C1 headline be awarded.
