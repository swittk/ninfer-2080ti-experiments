# NInfer SM75 benchmark comparison — 2026-11-21

Live measurements on two modified RTX 2080 Ti cards, comparing the historical October 2026 SM75 experimental engine with the maintained NInfer-all November 20, 2026 release. **Each result below is one run, not a statistically qualified performance claim.**

## Environment and reproducibility boundary

- Two RTX 2080 Ti, 22,528 MiB per card, 250 W stock power limits; Linux NVIDIA driver 580.126.09. No clock changes, hardware resets or driver modifications.
- GPU 0 was otherwise idle. GPU 1 retained the unrelated REIMS QEMU/streaming stack (about 3.4 GiB resident VRAM); two-GPU runs are consequently not fully isolated.
- Old engine: executable from 2026-10-02, historical dual branch `feat/sm75-dual-gpu-full-context` at `1697ed3` plus surviving uncommitted work; published research snapshot is this repository's `main`. The old single executable and old `ninfer-dual` are distinct paths.
- NInfer-all: build `4633e49`, CUDA 12.9.86 SM75, pinned container `ghcr.io/iamwavecut/ninfer-all@sha256:15ec17e4571a8888eacd135fd98b2f7d51a0b982a2215a37d80215f05f325520`.
- Original Qwen3.8-27B NInfer v2 input SHA-256: `eec39564993d6e9c7d5e383382a760f093465c9d163ec9a1bd6b80199514bf3e`.
- That original artifact was mechanically upgraded by NInfer-all's `tools/upgrade_ninfer_v2_to_v3.py`, which reported weights preserved and changed the chat template/resources. Upgraded v3 artifact SHA-256: `f6e4aaadab685b4de93e404e94f632f36a3b5d85f42eb1a44528c9e84eeea57f`.
- Bonsai 2 is a **different model**, with approximately 7.28 GiB of resident weights with MTP. Its results are useful hardware/model comparisons, not identical-weights evidence.
- Short fixtures: same JSON message inputs, coding (`scenario_code_python.json`), English fiction (`scenario_story_en_mystery.json`), and a Thai technical-explanation prompt; each 256 generated tokens, context 4,096, INT8 KV, greedy, thinking off, MTP3.
- Long fixture: a genuine populated 64,581-token document derived from the historical `long_niah_64k.json` with a 128-token continuation request. JSON fixture SHA-256: `0cddf9b3fac01c3390ac8a297f0ecd09301ffbae9f2000eed8eb2f4c8043fe93`. The original local fixture is not published verbatim; public archived synthetic fixtures were sanitized and will not preserve its exact tokenization. The long test requires retrieval of `ORCHID=493817; COLOR=COBALT`.
- Single and upstream multi-GPU executions used prefill chunk 1,024. The older dual executable permits at most 128, used 128; this confounds prefill differences. All short and long measurements used INT8 KV and MTP3 unless noted.

## Original Qwen3.8 — short single-GPU decode

| Workload | Old SM75 engine | NInfer-all | Old / upstream |
|---|---:|---:|---:|
| Coding | 67.08 tok/s | 51.0 tok/s | 1.32x |
| English fiction | 59.46 tok/s | 37.1 tok/s | 1.60x |
| Thai technical explanation | 36.18 tok/s | 20.1 tok/s | 1.80x |
| Coding, no speculation | 24.21 tok/s | **28.6 tok/s** | 0.85x |

All runs generated 256 tokens and produced coherent short continuations. On the coding fixture, the old engine accepted 152 / 308 drafts in 103 verifier rounds over 3.802 s of decode, versus upstream 176 / 237 accepted in 79 rounds over approximately 5.0 s. The old path therefore cost ~36.9 ms per complete speculative round versus upstream ~63 ms, despite upstream's higher acceptance and faster non-speculative decode. **This does not prove which individual kernel is responsible.** The two engines' exact output bytes differ (and the artifact chat template changed), so these are matched-input/model-weight observations rather than strict matched-output kernel attribution.

## Original Qwen3.8 — short two-GPU decode

| Workload | Old dual full-computation engine | NInfer-all PP2 | NInfer-all TP2 |
|---|---:|---:|---:|
| Coding, 256 tokens | 39.15 tok/s | 53.4 tok/s | 53.6 tok/s |
| English fiction, 256 tokens | 37.56 tok/s | Not measured | Not measured |
| Thai, 256 tokens | 23.34 tok/s | Not measured | Not measured |

Old dual used `--full-context-compute` with all layers/keys and strict verification. Its stdout token IDs were decoded offline using the original artifact's embedded tokenizer, and the short responses are coherent. All new NInfer-all dual executions used pinned-host transport: peer access was unavailable. Upstream PP2 produced byte-identical code output to upstream single GPU; TP2 differed numerically.

## Ternary Bonsai 2 — short contexts

| Workload | One GPU | Two-GPU pipeline (PP2) | Two-GPU tensor parallel (TP2) |
|---|---:|---:|---:|
| Coding, 256 tokens | **115.2 tok/s** | 112.8 tok/s | 56.8 tok/s |
| English fiction, 256 tokens | 82.6 tok/s | Not measured | Not measured |
| Thai, 256 tokens | 47.9 tok/s | Not measured | Not measured |

The coding single GPU accepted 167/261 drafts (64.0%). The same single-GPU model accepted 37.1% on English fiction and 7.5% on Thai. This confirms that a published ~95.9 tok/s benchmark on an 11 GB 2080 Ti is plausible, but not universally representative across prompts. At this Turing shape, TP2 halves Bonsai decode speed whereas PP2 preserves the single-card rate. The maintained NInfer-all documentation notes that its tensor-parallel T2 shares can take a slower A16 path instead of the tuned whole-tensor A8 route; a profiler is still required before proposing a fix.

## Genuine 64,581-token populated document

| Model and hardware | Prefill tok/s | Decode tok/s | Requested facts |
|---|---:|---:|---|
| Qwen3.8 · NInfer-all single GPU | 568.4 | 31.1 | Both correct |
| Qwen3.8 · historical dual PP/full-compute | 304.39 | 34.23 | Both correct |
| Qwen3.8 · NInfer-all PP2 | **596.0** | **34.8** | Both correct |
| Qwen3.8 · NInfer-all TP2 | Not completed | Not measured | Not tested |
| Bonsai 2 · NInfer-all single GPU | 760.0 | 63.0 | Both correct |
| Bonsai 2 · NInfer-all PP2 | **801.8** | **66.9** | Both correct |

Each completed run used 64,581 actual prompt tokens, zero prefix reuse, and 128 emitted tokens. Upstream PP2 outputs were byte-identical to corresponding upstream single-GPU outputs for both Qwen and Bonsai. Historical dual Qwen decoded a coherent answer with both facts correct, but the exact tokens differed. Do not label unlike output trajectories or models as correctness-equivalent.

**Bounded TP2 limitation:** the upstream Qwen two-GPU TP2 64K run was terminated after about eight minutes of active prefill with no completed response and no CUDA error. It had fully initialized and both GPUs were busy; it was not a crash. The completed PP2 run took 108.4 seconds of prefill. A censored/incomplete TP run cannot be assigned a decode tok/s figure. The specific benchmark container was stopped normally and its GPU allocations released. The existing REIMS processes were left untouched.

**Old single-card long path:** on this configuration the October binary failed during Q4 layout initialization with `cudaErrorGraphExecUpdateFailure` at `src/ops/linear_swiglu/q4/q4_linear_swiglu_a8_sm75.cu:457`. It is a failed configuration, not a zero-throughput result.

## Conclusions and next research gates

1. The maintained NInfer-all engine already provides working two-card pipeline and tensor parallelism; no reason to port generic dual-device infrastructure wholesale.
2. On the same original dense Qwen weights, upstream is faster without MTP but our historical MTP-enabled single-GPU path is much faster across the three short prompts. Profile complete round costs and the small-width verifier/proposer/target-head operations in controlled matched-output tests before attributing a specific mechanism or filing a performance issue.
3. For these PCIe-only Turing cards, pipeline parallelism is a better long-context deployment choice. The official NInfer-all T2 tensor share path is a focused potential optimization target only if verified with operator timings and end-to-end gain.
4. Ternary Bonsai 2 gives a much faster **different-model** option, including successful 64K retrieval on one 2080 Ti; still validate multilingual output and model task quality separately.
5. A new genuinely populated 262K-token comparison was **not** run in this campaign. The historical Q64 ~104.53 tok/s 262K receipt has repetitive output despite its automated quality pass; the later October ~56.67 tok/s 260K strict configuration is a separate workload/implementation. Neither is superseded or directly validated by the new 64K runs.
6. Single-run rates may shift with GPU boost and the concurrent REIMS workload; repeat matched fixtures with clocks/telemetry logged before publishing any upstream optimization claim.

Provenance: historical code and archived investigation notes: [swittk/ninfer-2080ti-experiments](https://github.com/swittk/ninfer-2080ti-experiments). Maintained project: [iamwavecut/ninfer-all](https://github.com/iamwavecut/ninfer-all). Its documented two-GPU implementation and limitations: [pipeline stages and tensor parallelism](https://github.com/iamwavecut/ninfer-all/blob/master/docs/maintainer/pipeline-stages.md). No external issue or PR was filed in this benchmarking campaign.
