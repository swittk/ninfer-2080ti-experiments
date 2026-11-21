# NInfer RTX 2080 Ti Experiments

Independent historical Turing (sm_75) NInfer CUDA/C++ experimentation on two 22 GB RTX 2080 Ti cards. This is a code-and-evidence archive, NOT a currently supported NInfer release.

## Three archive branches

- main: historical dual-GPU full-context research worktree, including unfinished source and selected receipts.
- archive/single-sm75: earlier single GPU SM75 worktree.
- archive/single-int8-tu102: separate single GPU INT8 TU102 optimization worktree.

The snapshots preserve the original source states as independently reviewable branches; they are not chronological releases and have a fresh sanitized publication history.

## Performance evidence and caveats

| Historical experiment | Recorded useful tokens/s |
| --- | ---: |
| strict single-GPU C1 D3 on original fixture | 79.01 median; later source ~80.51 |
| distinct certified single-80 | 80.185 median |
| relaxed MTP logit-margin-7 single-GPU | 107.27 median, short fixture ONLY |
| dual GPU short Q64 | 106.01 median |
| dual GPU populated ~65K Q64 | 110.94 median |
| dual GPU populated ~262K Q64 | 104.53 median, automated quality checker missed repetition |
| dual GPU 260,973 genuinely populated-token October runtime | 56.67 paired mean, 56.83 final confirmation |

These differ in target layout, acceptance policy, output fixture, context length, and quality. The relaxed 107 route produces unacceptable language drift in sustained Thai output and has NO strict-path 100-tok/s qualification. The old Q64 long-context result has visibly repetitive output despite a historical automated quality-pass flag. Do not compare those routes to the later strict October runtime as same-policy A/B performance.

See [research index](research/INVESTIGATION_INDEX.md), [archived experimental notes](research/legacy-notes/), and [source provenance](PROVENANCE.md).

## Upstream and future comparison

This code descends from https://github.com/Neroued/ninfer . The best-maintained modern multi-GPU and Turing lineage now appears to be https://github.com/iamwavecut/ninfer-all . Next steps: reproduce the current NInfer-all stock-card baseline separately; benchmark pipeline and tensor parallelism; propose upstream changes with matched operator and complete-model measurements under its issue-first contribution policy.

Large raw machine logs, trained/model weight artifacts, local system paths, and credentials are deliberately not published. Historical long-context synthetic fixtures were sanitized for language and thus do not reproduce their original byte-level token hashes. Original local research worktrees and raw data were left intact.

License and contributor notices from the original source are preserved.
