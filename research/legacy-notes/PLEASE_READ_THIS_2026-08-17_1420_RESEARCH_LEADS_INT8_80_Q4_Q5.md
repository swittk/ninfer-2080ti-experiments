# RESEARCH LEADS FOR INT8 >=80 — CHANGE BYTES/USEFUL TOKEN, NOT JUST DRAM %

Owner research update, 2026-08-17 14:20 +07. Keep the current TU102 INT8-GQA baseline work going; this is the bounded queue after/alongside it.

## Why dense NCU polishing is insufficient
Q4 SwiGLU is ~175.5 us/layer and Q5 down/add ~112.5 us/layer. Across 64 layers that is roughly 18.4 ms of a verification round before GDN/attention/head. Dense Q4 already runs near the TU102 GDDR6 ceiling. Hitting 100% bandwidth cannot by itself move 64.96 -> 80 tok/s. The new metric is **DRAM bytes per useful token / speculative round**.

## A. Highest-priority research: Prox-style sparse SwiGLU, Turing-native proxy
Paper: Prox, arXiv:2607.27591 (July 2026). It targets native SwiGLU, evaluates Qwen3/Qwen3.5-family models, uses a cheap quantized proxy only to rank intermediate channels, then computes retained channels with original weights. The same mask sparsifies up + gate + down. It reports up to 1.99x end-to-end decode speedup at 70% FFN sparsity and is compatible with quantization.

Do this in bounded stages:
1. **Oracle audit first, no new kernel.** Capture exact S=SiLU(gate)*up and test top-magnitude retention at 80/70/60/50% density on prose/code/JSON/multilingual. Measure individual-channel and 16/32/**64-channel** masks. Group64 is especially attractive because it aligns with current Q4/Q5 quant groups.
2. If oracle quality is good, implement **oracle block-sparse execution** to measure the hardware upper bound. Selected channels still use current exact Q4/Q5 math. If this does not give a large E2E win, stop.
3. Only then build the proxy. Test (a) native INT4 proxy and (b) B1/B2/BMMA proxy reusing the vocabulary-screening idea. The proxy only needs ranking, so aggressive low-bit error is isolated to mask choice.
4. Use block-friendly layouts; do not random-gather dense rows and conclude sparsity is slow. Consider consistent offline permutation of SwiGLU intermediate channels so hot blocks cluster while dense semantics remain unchanged.

A ~30-40% FFN weight-read reduction is the first mechanism we have found whose scale is actually consistent with the +23.2% E2E target.

## B. Turing native INT4 Tensor Cores are currently unused by the dominant MLP path
NVIDIA documents `mma.sync.aligned.m8n8k32.row.col.s32.s4.s4.s32` on sm75 and ~455 INT4 Tensor TOPS vs ~228 INT8 Tensor TOPS for RTX 2080 Ti. Current SM75 Q4/Q5 fast kernels expand Q4/Q5 to s8 and call `mma_s8`; repo search finds no native `.s4.s4` MMA in the dominant implementation.

Bounded spike: add an isolated `mma_s4` microbench and one packed Q4 fragment mapping. Compare SASS, registers and throughput against expand+mma_s8. Do **A4 only for the sparse proxy first**; that is low-risk because proxy values are not propagated. Only if promising consider actual W4A4. QuaRot/SpinQuant/FlatQuant show rotation-based W4A4 can preserve quality, but global rotations on this hybrid GDN model are a later project.

INT4 compute alone will not 2x a bandwidth-bound dense kernel; its value is removing expansion/issue pressure and making the sparse proxy extremely cheap.

## C. If necessary, reduce stored weight bits
Research precedents: SqueezeLLM (3-bit dense+sparse outliers), LUT-GEMM (sub-4-bit without costly dequant), QServe (compute-aware reordering/register-level dequant).

Before artifact rewrites, audit:
- Q5 -> Q4 with recalibrated scales;
- Q4 -> Q3;
- dense low-bit core + sparse residual/outlier correction;
- bytes saved versus correction density and output quality.

A Turing-specific Q5 idea worth measuring is **Q4 core + sparse residual correction**: dense core via native s4 MMA, residual separately. Reject immediately if residual density erases byte savings.

## D. Finish the current TU102 INT8 attention pass
Still worthwhile: 170-SM-oriented split assumptions on a 68-SM GPU; cp.async-oriented staging collapses to sync copies on sm75; tune T1..T6, especially T3, across long windows. This can recover part of the INT8 penalty but probably not all +23%.

## What a splendid NCU result means now
Report bytes/round and bytes/useful-token, not only % DRAM peak. A sparse kernel at 450 GB/s reading 40% fewer bytes beats a dense kernel at 600 GB/s. Also report long scoreboard, registers, occupancy, barriers, selected density, proxy/oracle mask overlap, MTP useful tok/round, and E2E tok/s.

Priority: finish INT8-GQA baseline -> oracle SwiGLU sparsity audit -> oracle block-sparse upper bound -> native s4 microbench -> proxy/layout only if upper bound wins -> deeper Q3/Q4-residual work last. Keep INT8 single-GPU >=80 as the promotion floor.
