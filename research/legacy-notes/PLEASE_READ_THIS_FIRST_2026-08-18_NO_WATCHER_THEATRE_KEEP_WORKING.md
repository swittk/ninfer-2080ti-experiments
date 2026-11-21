# READ THIS FIRST — NO WATCHER THEATRE / KEEP DOING REAL WORK

This job has repeatedly stalled because a worker launches polling loops waiting for files while **no producer is running**. The owner explicitly considers that lazy/failed execution.

A watcher is not work. If NInfer/NVCC/NCU/NSYS/real analysis is not running and only `sleep/test -f` loops remain, you are idle.

Before waiting on any artifact:
- identify the exact producer that will create it;
- verify that producer is live;
- if it is not live, fix/run it immediately;
- if the producer failed, diagnose and retry rather than waiting;
- if enough data already exists, write the classification yourself and continue.

Do not stop after a trivial command/build/setup failure. Do not leave 0-byte generated files, stale rejected patches, broken CMake references, or missing `main()` issues for someone else. Repair them and continue.

The required behavior is sustained iteration:
`measure -> hypothesis -> implement -> build -> bench -> exact qualify -> classify -> revert/promote -> next hypothesis`.

When one experiment is neutral, that closes one hypothesis only. Move immediately to the next measured hotspot. The single >=80 and dual-one-request >=100 goals are still open.

Also read:
- `PLEASE_READ_THIS_2026-08-18_PROFILE_SAYS_NOT_DONE_ITERATE_HOT_KERNELS_AND_TP2.md`
- `PLEASE_READ_THIS_2026-08-18_HARDWARE_ASYMMETRY_INTERNAL_CARD_BASE_CLOCK.md`
- `PLEASE_READ_THIS_2026-08-18_PORTABLE_MECHANISMS_CALIBRATED_POLICY.md`

## Also mandatory: preflight / canonical memory
Read `PLEASE_READ_THIS_FIRST_2026-08-18_MANDATORY_PREFLIGHT_AND_CANONICAL_LEDGER.md` before opening any new experiment family or download. Watcher discipline alone is not enough; do not repeat already-closed work or recreate/download artifacts that already exist.
