# IMPORTANT — active cards are not performance-equivalent

Measured 2026-08-18 with identical NInfer workload and identical output/acceptance.

- GPU-2fffc240-5d58-fbc8-27d6-6b349448fda9, PCI 3c:00.0 = Razer Core X / TB3 card. Sustained ~1.68-1.86 GHz SM, ~66.5 t/s on 512-token control.
- GPU-020906b4-0904-3471-e4ee-e41ec2350be3, PCI 87:00.0 = direct internal x16 card. Sustained exactly 1350 MHz SM, ~61.6 t/s on same control.
- Both expose the same VBIOS 90.02.17.00.A3, same 250 W power limit, same 6800 MHz loaded memory clock, and identical supported clock tables up to 2100 MHz graphics.
- Internal card is cooler and reports no active SW power, thermal, HW power-brake, sync-boost, application-clock, or display-clock limiter during load.
- This is distinct from the previously removed repaired/broken card, which was stuck around 300 MHz under load with SW Power Cap active and only ~12 t/s.

Implications:
1. Do NOT generalize the current optimum split, stage placement, proposal-offload percentage, or absolute dual TPS as representative of two healthy matched 2080 Ti cards.
2. Controlled candidate-vs-reference A/Bs on the same fixed device assignment remain valid software evidence.
3. The large normal-vs-swapped-role delta is now explained at least partly by physical clock asymmetry, not merely architecture.
4. Keep the faster normally-boosting card on latency-critical roles for current experiments, but label hardware assignment.
5. Single-GPU planning should distinguish ~70.3 t/s normally-boosting card from ~65.2 t/s base-clock-limited internal card.
6. Do not tune production algorithms around the pathological 1350-MHz internal behavior; portability claims need requalification on matched healthy/floating-clock devices.

## Design rule
Use **portable mechanisms, calibrated policy**. Split layer, head/proposal ownership, TP fractions, PP/TP/context-parallel thresholds and absolute placement deltas are hardware-policy choices, not Qwen/TU102 architectural constants. Eventually choose them from a lightweight per-machine calibration profile; until then expose/manual-tune the knobs and record UUID/loaded clocks for benchmarks. See `PLEASE_READ_THIS_2026-08-18_PORTABLE_MECHANISMS_CALIBRATED_POLICY.md`.

## 2026-08-18 stronger historical control: identical archived binary proves runtime/device asymmetry

A literal archived NInfer executable from **2026-08-16 16:17 +07**, immediately after the ~72 tok/s checkpoint era, was rerun today against the same model and canonical 1024-token BF16-KV workload.

Historical clean validation from Aug 16:
- 72.08 / 71.77 decode tok/s;
- 372 MTP rounds, 1113 drafted, 651 accepted, 58.49%, accepted-by-pos 300/204/147;
- stdout SHA256 `ad070c8822b16afc4acfe2d7162269e6bd7d030d64ae442ea8fa135fa5819f3e`.

**Same archived executable today, pinned to internal GPU UUID `GPU-020906b4-0904-3471-e4ee-e41ec2350be3`:**
- 65.38 tok/s;
- exact same 372/1113/651/58.49%/300,204,147 trajectory;
- stdout is byte-identical to the Aug-16 historical output (same SHA above, `cmp` identical);
- loaded SM clock exactly 1350 MHz min/mean/max;
- memory 6800 MHz, ~98% GPU utilization.

**Same archived executable today, pinned to Core-X eGPU UUID `GPU-2fffc240-5d58-fbc8-27d6-6b349448fda9`:**
- 71.04 tok/s;
- exact same output SHA and MTP trajectory;
- loaded SM mean ~1759.9 MHz, range 1680-1860;
- memory 6800 MHz, ~100% GPU utilization.

Interpretation: software/model/prompt/trajectory are controlled. The current internal-device state alone loses ~9% throughput versus the same executable on the eGPU and versus its own historical ~72 tok/s era. The timing transition is suspiciously near the eGPU setup (`egpu_sanity/internal.stderr` at 2026-08-17 05:03 was already 62.25 tok/s), but this does **not** prove the Thunderbolt GPU caused it. NVIDIA driver/display/power/topology state can make nominally identical GPUs materially asymmetric and can change across boots/configurations.

### Engineering rule
Do NOT assume equal SKU => equal effective throughput, and do NOT infer architecture constants from one physical role assignment. Record UUID + loaded SM/memory clocks with serious benchmarks. Multi-GPU split/ownership/TP fractions must be calibrated from measured device throughput. Treat unexplained fixed-base-clock states as platform/driver/device-state conditions until isolated; do not optimize NInfer around them and do not call a board defective from throughput alone.

A cold boot with the Core-X physically disconnected remains the cleanest causal discriminator, but NInfer engineering should remain robust regardless.
