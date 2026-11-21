# PLEASE READ — sequencing guard: return to single-GPU >=80 after bounded dual-MTP milestone

Owner/adviser sequencing clarification, 2026-08-17 19:34 +07. This does not interrupt the current sharded-MTP verifier/fold work. Finish the bounded dual-MTP milestone first.

The project has TWO independent hard performance goals; success on one does not retire the other:

1. **Single GPU, single user, INT8, full-context product mode: >=80 useful tok/s minimum.**
2. **Dual GPU performance mode: >=100 aggregate useful tok/s minimum.**

The current dual work is a bounded milestone, not a replacement for the single-GPU track. After the sharded MTP3 path reaches a correctness-qualified 2-user performance result (preferably >=100 aggregate), OR produces a concrete hard blocker after the planned MTP/packed-stage work, explicitly return to the single-GPU TU102 track.

On return, do not restart from 65.3 as though the historical ~72 lane did not exist. First requalify the fastest post-38369da checkpoint-qualified ~72 construction under the upstream-oriented correctness gate, then pursue the still-unexplored single-GPU research classes already listed in the 18:18 and 19:02 notes: block32/16 sparsity audit, BF16-source calibrated PTQ, true calibrated W4A4/rotations/outlier handling, lower-bit/vector/codebook compression, architecture-aware GDN treatment, and better speculative-decoding economics.

Do not declare single-GPU >=80 exhausted unless those materially different SOTA-class avenues have been investigated with appropriate quality/performance evidence.

Relevant notes:
- `PLEASE_READ_THIS_2026-08-17_1818_SINGLE80_NOT_EXHAUSTED_REQUALIFY_72.md`
- `PLEASE_READ_THIS_2026-08-17_1829_DUAL_SCALAR_PP_CEILING_NEXT_MTP_PACKED.md`
- `PLEASE_READ_THIS_2026-08-17_1902_ADVANCED_QWEN_EXECUTION_RESEARCH_MENU.md`

Immediate work remains: fix the sharded verifier replay/fold oracle, prove continuation correctness, reconnect canonical MTP3 proposal recursion, then measure dual useful throughput. After that bounded milestone, return to single-GPU >=80 as above.
