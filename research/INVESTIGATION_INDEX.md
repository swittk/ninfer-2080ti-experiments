# Historical research index

The original research included more than 100 GiB of ignored experiment artifacts. This public archive intentionally publishes code, dated notes and a small reproducible evidence index rather than all scratch logs, model weights or GPU traces.

## Semantics
Single GPU strict C1 D3 and the historical relaxed margin-7 C1 route are distinct decoding policies. C4/C8 multi-user aggregate throughput is not single-request throughput. Genuine populated 65K/262K memory must not be confused with reserved context capacity. Historical saved hashes refer to the original local corpus, whose published synthetic long-NIAH fixtures had language sanitized.

## Selected reports
- [October 2 query-lifetime/producer-owned query preparation](reports/2026-10-02-query-preparation.md): 260,973 genuinely populated tokens, 56.52 to 56.67 useful tokens/s paired means.
- [October 2 query-owned finish](reports/2026-10-02-query-finish.md): exact query-owned completion and small end-to-end gain.
- [October 2 frozen DFlash2 screen](reports/2026-10-02-dflash2-screen.md): multilingual agreement insufficient for a native integration.
- [October 2 TP2 fused input](reports/2026-10-02-tp2-fused-input.md): no reliably promoted complete-model speed gain.
- [October 2 threaded submission rejection](reports/2026-10-02-two-thread-rejection.md): measured end-to-end regression.

## Numeric retained receipts
- [single strict 79](evidence/strict_79.json)
- [certified single 80.185](evidence/single80_185.json)
- [relaxed 107](evidence/margin7_107.json), with later long-Thai output failure
- [dual short and 65K](evidence/dual_short_65k.json)
- [dual populated 262K Q64](evidence/dual_262k_q64.json), with later-identified repetition despite automated pass
- [physical C8 aggregate serving](evidence/c8_batch.json), not an individual-request rate

The original full transcripts, 223K+ ignored research files and binary model artifacts stay local. Requests for identical historical-output replay require the original full fixture bytes; the publication snapshots do not claim those exact fixtures are public.
