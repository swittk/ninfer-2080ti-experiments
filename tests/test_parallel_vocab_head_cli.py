"""Reject unsupported parallel-head execution before opening an artifact."""
from pathlib import Path
import subprocess
import sys
binary = Path(sys.argv[1]).resolve()
base = [str(binary), "missing-test-model.ninfer", "--prompt", "x", "--parallel-vocab-head"]
cases = [
    [],
    ["--spec", "mtp"],
    ["--spec", "none", "--full-context-compute"],
    ["--spec", "mtp", "--full-context-compute", "--mtp-depth", "4"],
    ["--spec", "mtp", "--full-context-compute", "--sessions", "2"],
    ["--spec", "mtp", "--full-context-compute", "--single-device", "0"],
    ["--spec", "mtp", "--full-context-compute", "--target-verify-oracle"],
    ["--spec", "mtp", "--full-context-compute", "--mtp-batch"],
]
for args in [c+[flag] for c in cases for flag in ["--parallel-vocab-head","--parallel-greedy-head"]]:
    result = subprocess.run(base + args, capture_output=True, text=True, timeout=10)
    if result.returncode != 2 or "--parallel-vocab-head requires complete dual C1 D3" not in result.stderr:
        raise AssertionError((args, result.returncode, result.stdout, result.stderr))
print("PASS 16 unsupported parallel-head configurations rejected before model loading")
