# NInfer — producer-owned query preparation retained

Updated Bangkok: 2026-10-02T22:53:07+07:00

## Measured full-context gain

**Retained a 0.27% improvement in actual full-context warm decoding.** The final
same-binary comparison processed 260,973 genuinely populated input tokens in all
eight runs. Each adjacent candidate/control pair favored the candidate.

| Final comparison | Original preparation | Producer-owned preparation |
| --- | ---: | ---: |
| Four useful decode rates, tok/s |56.69, 56.52, 56.48, 56.39 |56.82, 56.75, 56.65, 56.46 |
| Mean, tok/s |56.52 |56.67 |

The ratio of means is 1.002653928. Pair differences were +0.13, +0.23, +0.17 and
+0.07 tok/s. Natural clocks varied; this small measured improvement is not a
universal confidence interval or a clock-controlled performance guarantee. No
hardware clock, power, driver or reset settings were changed.

The final rebuilt implementation, with its private A/B switch removed, confirmed
**56.83 useful tok/s**. That is confirmation of the retained source, not another
paired gain or a higher figure substituted for the table above.

## Preserved outputs and verification

All eight paired runs and the final-source check produced the same **241-token
naturally completed answer**, with **96 rounds, 288 drafts and 145 accepted drafts**,
and acceptance positions **71/48/26**. The output-token hash remains:
`905383785b0845c18e1917df016f8023e831d8c921692dae8bbbd15ad4d3962a`.

All three distant records were recovered: ALDER|173|EAST, BIRCH|286|WEST and
CEDAR|491|NORTH. All 15 per-layer fallback records matched the previous runtime.
The paired terminal counters also matched. Canonical prefix-snapshot sizes and
modification times remained unchanged.

The change preserves the arithmetic of the already-selected integer-coordinate
attention mode. Its earlier numerical approximation and long-output quality
limitations remain. This is not a new unquantized-model or universal quality
certificate. Target weights, key coverage, coordinate rank, selection thresholds,
proposal policy and strict verification rules did not change.

## Retained implementation

The old path separately quantized a full query, projected it through the prepared
key basis and mean, and quantized the projected query. The new kernel owns that
intermediate lifetime within one query block. Eight warps perform the original
basis projection; one producer warp computes the quantizers and the four original
ordered mean dot products. The basis warps therefore no longer execute a divergent
mean-projection instruction path inside their dot-product loops.

The original 32-lane maximum reduction, four 64-term dot products, pairwise partial
sum, FP16 projection boundary, mean shift and final signed-byte rounding are all
preserved. No learned constants or checkpoint-specific tuning are introduced.

The same grid initializes the entire existing union bitset and all 104 metadata
integers before the next graph operation can consume them, replacing two separate
memset nodes. Each word has one writer. Initialization coverage is not reduced.
There is **no added persistent device allocation, changed buffer size or changed
inference-buffer address**. The float-coordinate attention path retains its
original kernels and initialization behavior.

The improvement is now automatic within the existing integer-coordinate mode.
The private experimental switch has been removed; there is no new public option.
The prior query-owned completion, compact head and memory-ownership gains remain.

## Validation

The permanent test passed 14 source/input scenarios on each physical GPU, comparing
all original projected values, integer codes, scales and shifts, plus output guards.
Across the cards this covered **537,600 output bytes**, with **98,304 independent
CPU basis-projection values**, **1,536 ordered mean shifts** and **98,304 independently
checked projected quantizer values**. Initial full-query quantization was also
checked independently. Cases include zeros, very small and large inputs, signed
zeros, NaNs, infinities and changed query/basis contents at graph-stable addresses.

Each scenario changes later query inputs while holding the first query fixed;
its result remains unchanged. Input query, basis and mean storage are read-only.
Eighteen additional initialization-extent cases across the cards cover empty,
short, tail and full bitsets with stale data and guard regions.

Final validation also passed the existing query-completion, valid-query integer
attention, direct-metadata and ordered-union GPU tests on both cards. **Eight
focused CTest cases**, **11 preservation-policy tests** and **18 retained result-
record checks** passed. The real-artifact CTest ran with an absolute path and the
matching model-format settings, not a skipped test. No fresh C8, full-context
multimodal or historical executable-archive certificate is claimed.

## Experiments not retained

The initial basic query fusion passed its operator checks but averaged only a
mixed 0.10% model difference, with two positive and two negative pairs. It was not
promoted. Separating the independent mean role improved the isolated complete
query chain 1.46–1.51x and measured a separate 0.25% model difference. The final
producer-initialization comparison above supersedes that development measurement;
these stage gains are not added together.

A larger host-submission experiment used ordinary persistent CPU execution threads
to submit the two TP2 matrix shards concurrently. It passed 1,000 ownership/error/
lifetime checks and preserved the full-context answer and verification, but slowed
actual decoding from 56.67 to 52.09 tok/s. Its source was restored and the helper
header archived. No thread-affinity, spinning or priority search followed. These
execution threads were not delegated coding or reasoning agents.

## Source and continuation state

Five source/test/documentation files are retained in `final/scoped.patch`.
The ordinary single-GPU CLI and server hashes still equal their pre-turn hashes;
there is no new single-GPU speed measurement or improvement claimed here. The dual
executable was rebuilt and its current hash is recorded. The failed threaded TP2
source is exactly restored, both temporary query benchmark targets are archived,
and benchmark CMake equals its pre-turn copy.

Worktree: `<local-worktree>`.
Branch: `feat/sm75-dual-gpu-full-context`.
HEAD: `1697ed3b7134d3419918e2ea824401f79ee15515` plus inherited uncommitted work.
Evidence: `.chatgpt/production_serve_20261002/query_input_lifetime_fusion/` and
`.chatgpt/production_serve_20261002/device_thread_submission/`.
All our jobs are terminal and both GPUs have no CUDA compute clients. No model
artifact rewrite, training, hardware tuning, agent, commit, push or deployment
occurred. Prior status and the protected full handoff are saved for continuation.

**Both generic strict single-GPU C1 D3 >=100 and populated complete-context dual
single-request >=100 remain unmet. The retained gain this turn is 0.27%.**
