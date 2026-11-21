# PLEASE READ — checkpoint + progress housekeeping (2026-08-16 16:25 +07)

This is **not** asking you to stop a currently-running benchmark/mini-sweep or change technical direction if the current work is productive.

The owner cannot easily interact with this worker right now and mainly wants to make sure the rapidly-accumulating SM75 performance work is not left only in a large dirty worktree / scratch artifacts.

## At the next natural stable boundary, please do housekeeping before opening another major experimental branch

1. **Re-establish the best SAFE / exact-reference state.**
   - The most recently reconfirmed known reference lane was a full 1024-token greedy MTP3 run at about **72.09 tok/s**, output SHA beginning `ad070c...`, 58.49% MTP acceptance, 372 rounds, no fallback.
   - If you now have a newer validated exact/reference result, use that instead and record it clearly.
   - Keep trajectory-changing / approximate / turbo lanes explicitly separate from the safe lane.

2. **Create a local Git checkpoint when the source is genuinely reproducible.**
   - Current branch HEAD is still `b6c36cf` (`perf(sm75): checkpoint A8 speculative decode at 64 tok/s`) while a great deal of newer work exists only in the dirty tree.
   - Please do **not** blindly commit scratch/debug/turbo experiments just to make the tree clean.
   - Identify the source files actually required for the best validated safe state, validate them, then make a sensible **local commit** (no push required).
   - If the dirty tree is too entangled to safely commit immediately, first preserve an explicit local checkpoint/patch + handoff describing exactly what constitutes the safe state, then disentangle carefully. Do not destroy useful experimental evidence.

3. **Append a concise progress report to `progress.md`.**
   - Preserve the repository convention that `progress.md` is append-only via `scripts/append_progress.py`.
   - Please summarize, at minimum:
     - best validated safe/exact sustained result and benchmark artifact(s);
     - best approximate/turbo result and why it is not yet considered safe;
     - major optimizations that were accepted;
     - major experiment families that failed / were rejected, so future workers do not repeat them;
     - current correctness / trajectory caveats;
     - current dominant bottlenecks / profile observations;
     - the fact that Nsight Compute was attempted but initially hit `ERR_NVGPUCTRPERM` if that remains relevant;
     - the next concrete optimization hypothesis.

4. **Refresh durable resume metadata.**
   - The current `.chatgpt/work-sessions` pointer is stale relative to the work actually performed today.
   - Update the active work-session and/or write a current handoff after the checkpoint so another worker can resume without reconstructing hours of scratch history.

## Important intent

Please **continue optimizing** after this maintenance point if there are still evidence-backed ideas. The request is not to end the performance effort. It is simply to periodically convert successful research into durable source + benchmark evidence + written history before another large stack of experiments accumulates.

A good rhythm from here would be roughly:

`experiment -> validate -> if real win, preserve/checkpoint -> append progress -> continue research`

rather than letting many hours of validated work remain dependent on scratch binaries/files and memory of the current agent.

Thank you — keep the fast path moving, just leave breadcrumbs and commits behind you.
