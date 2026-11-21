# READ THIS FIRST — MANDATORY PREFLIGHT + ONE CANONICAL LEDGER

The owner is explicitly calling out a second recurring failure mode: **you are doing real experiments, but you are not systematically checking whether the experiment/data/source already exists or whether the line of work was already closed.** This is causing loops, duplicate downloads, and repeated rediscovery of old conclusions.

A concrete failure just happened:
- the machine already had a complete BF16 `Qwen3.8-27B-bf16` source checkpoint with `model.safetensors.index.json` + all 18 safetensor shards (~55.56 GB);
- `.chatgpt/int8_tu102/bf16_recipe/` already referenced that complete source;
- `PLEASE_READ_THIS_2026-08-18_Q4DOWN_CANARY_CLOSED_RETURN_TO_EXACT_ENGINE.md` and `.chatgpt/int8_tu102/bf16_recipe/Q4DOWN_CANARY_CONCLUSION.md` had already explicitly closed selective Q5-down -> Q4 enumeration for the current product path;
- despite that, a new `.chatgpt/bf16_source/` campaign was started and another ~55 GB Hugging Face download was launched, then the new audit eventually re-derived essentially the same closure.

This is unacceptable duplication. Fix the workflow, not just this one download.

## Mandatory preflight BEFORE ANY new experiment, download, worktree, benchmark family, or research branch
Do **all** of the following first:

1. **Search the canonical ledger / progress state.**
   - Read `progress.md`.
   - Read the current `PLEASE_READ_THIS_FIRST*` / trajectory notes relevant to the topic.
   - Search `.chatgpt` for `CONCLUSION.md`, `classification.json`, `summary.json`, handoffs, and previous experiment names/keywords.
   - Search git history/diffs for the mechanism if code may already exist.
2. **Search for existing data/artifacts on disk before downloading or regenerating anything.**
   - Resolve paths stored in prior JSON/manifests (`model_dir`, `source_checkpoint`, model paths, generated binaries, extracted weights).
   - Check whether complete files already exist and validate them (index/shard count/size/hash where appropriate).
   - Do NOT infer “missing” merely because the *new experiment directory* is empty.
3. **Classify prior work before reopening it.**
   - If prior work says `REJECTED`, `CLOSED`, or `DO NOT CONTINUE`, do not reopen it unless you can state a materially new hypothesis that invalidates the prior closure.
   - Write that reopening rationale **before** spending compute/time. It must explicitly name the prior conclusion and what is different now.
4. **Record the new experiment in the canonical ledger BEFORE execution.**
   - State hypothesis, prior related work, why it is not duplicate work, exact success gate, and next action on win/loss.
5. **After execution, append the result immediately.**
   - `ACCEPTED`, `REJECTED`, `CLOSED`, `SUPERSEDED`, or `OPEN`.
   - Include exact artifact paths, key numbers, semantics/correctness status, and the next non-duplicate action.

## `progress.md` is stale and must be repaired
There is already an append-only mechanism: `scripts/append_progress.py` + `progress.md`.

`progress.md` currently stops at **2026-08-16 21:02** even though dozens of experiments and major findings occurred on Aug 18. This is why local per-experiment reports are becoming 50+ disconnected memories instead of a coherent research state.

Before opening another broad research branch, bring `progress.md` current with at least the major Aug-18 state:
- hardware asymmetry/archive-binary control;
- current single-GPU representative frontier and exact-vs-fast semantics separation;
- Q4/Q5/GDN profile facts and closed geometry/cache/prefetch hypotheses;
- MTP T1/T2/T4/T6 result and T6 closure;
- attention Q5 gate/value A8 status, including long canonical trajectory fork caveat;
- TP2 corrected harness, block-split calibration, layer sweep, and 64-layer chain result;
- populated-context/C4 status and still-open 262K gate;
- Q4-down BF16-source canary closure, explicitly linking the prior closure and the accidental duplicate audit/download.

Do not dump every scratch run into `progress.md`; keep detailed artifacts in `.chatgpt`, but maintain a concise authoritative frontier so future work can answer **“has this already been tried?” in seconds**.

## No duplicate downloads
Before any `hf download`, `git clone`, model conversion, weight extraction, or generated-weight materialization:
- search existing manifests/artifacts;
- validate existing source;
- reuse it if valid;
- only download missing shards/files, never restart a full checkpoint download because a new scratch directory is empty.

## Working rule
**Local reports are evidence. The canonical ledger is memory. You need both.**

A worker that produces 50 experiment folders but fails to consult/update the canonical frontier is not operating systematically.
