#!/usr/bin/env python3
"""Compare one fixed greedy MTP trajectory with and without a second CUDA context."""

from __future__ import annotations
import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

PROMPT = (
    "Write a detailed technical explanation of how a modern GPU executes transformer inference, "
    "including memory hierarchy, quantized matrix-vector and matrix-matrix operations, attention, "
    "speculative decoding, and practical performance bottlenecks. Aim for at least 1500 words and "
    "continue until the output limit if necessary."
)


def args():
    p = argparse.ArgumentParser()
    p.add_argument("--exe", required=True, type=Path)
    p.add_argument("--model", required=True, type=Path)
    p.add_argument("--pressure-exe", type=Path)
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--normal-repetitions", type=int, default=2)
    p.add_argument("--pressure-repetitions", type=int, default=20)
    p.add_argument("--pressure-seconds", type=int, default=46)
    p.add_argument("--pressure-blocks", type=int, default=4)
    p.add_argument("--max-new", type=int, default=1024)
    p.add_argument("--prompt", default=PROMPT)
    p.add_argument("--route-label", default="q45g-current")
    p.add_argument("--no-cuda-graph", action="store_true")
    p.add_argument("--launch-blocking", action="store_true")
    p.add_argument("--trajectory-digest", action="store_true")
    p.add_argument("--expected-stdout-sha")
    p.add_argument("--expected-token-digest")
    p.add_argument("--expected-trajectory-digest")
    return p.parse_args()


def field(text, key):
    m = re.search(r"^summary\s+" + re.escape(key) + r"\s+(.+)$", text, re.M)
    return m.group(1).strip() if m else None


def generated_token_ids(text):
    m = re.search(r"^tokens\s+generated ids\s+(.+)$", text, re.M)
    return m.group(1).strip() if m else None


def sig(row):
    return {
        k: row[k]
        for k in (
            "stdout_sha",
            "token_ids",
            "token_digest",
            "trajectory_digest",
            "rounds",
            "drafted",
            "accepted",
            "by_position",
            "fallback",
        )
    }


def main():
    a = args()
    exe = a.exe.resolve()
    model = a.model.resolve()
    pressure = a.pressure_exe.resolve() if a.pressure_exe else None
    if not exe.is_file() or not model.is_file():
        raise SystemExit("--exe and --model must exist")
    if a.pressure_repetitions and (pressure is None or not pressure.is_file()):
        raise SystemExit("--pressure-exe is required")
    if a.normal_repetitions < 0 or a.pressure_repetitions < 0:
        raise SystemExit("repetitions must be nonnegative")
    a.output.mkdir(parents=True, exist_ok=True)
    cmd = [
        str(exe),
        str(model),
        "--prompt",
        a.prompt,
        "--max-context",
        "8192",
        "--kv-capacity",
        "auto",
        "--prefill-chunk",
        "128",
        "--max-new",
        str(a.max_new),
        "--kv-dtype",
        "bf16",
        "--spec",
        "mtp",
        "--draft-tokens",
        "3",
        "--lm-head-draft",
        "--greedy",
        "--no-thinking",
        "--print-token-ids",
    ]
    if a.no_cuda_graph:
        cmd.append("--no-cuda-graph")
    metadata = {
        "command": cmd,
        "pressure_executable": str(pressure) if pressure is not None else None,
        "normal_repetitions": a.normal_repetitions,
        "pressure_repetitions": a.pressure_repetitions,
        "pressure_seconds": a.pressure_seconds,
        "pressure_blocks": a.pressure_blocks,
        "cuda_graph": not a.no_cuda_graph,
        "cuda_launch_blocking": a.launch_blocking,
        "trajectory_digest": a.trajectory_digest,
        "route": a.route_label,
    }
    (a.output / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    expected = None
    if any(
        (a.expected_stdout_sha, a.expected_token_digest, a.expected_trajectory_digest)
    ):
        expected = {
            "stdout_sha": a.expected_stdout_sha,
            "token_ids": None,
            "token_digest": a.expected_token_digest,
            "trajectory_digest": a.expected_trajectory_digest,
            "rounds": None,
            "drafted": None,
            "accepted": None,
            "by_position": None,
            "fallback": None,
        }
    rows = []
    rep = 0
    for pressured, count in (
        (False, a.normal_repetitions),
        (True, a.pressure_repetitions),
    ):
        for _ in range(count):
            rep += 1
            burn = None
            if pressured:
                burn = subprocess.Popen(
                    [str(pressure), str(a.pressure_seconds), str(a.pressure_blocks)],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
                time.sleep(0.45)
                if burn.poll() is not None:
                    raise RuntimeError(
                        f"pressure process exited before inference (status {burn.returncode})"
                    )
            env = os.environ.copy()
            if a.launch_blocking:
                env["CUDA_LAUNCH_BLOCKING"] = "1"
            if a.trajectory_digest:
                env["NINFER_MTP_TRAJECTORY_DIGEST"] = "1"
            started = time.monotonic()
            try:
                run = subprocess.run(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    env=env,
                    timeout=240,
                )
            except subprocess.TimeoutExpired as exc:
                run = subprocess.CompletedProcess(
                    cmd,
                    124,
                    exc.stdout or b"",
                    (exc.stderr or b"") + b"\nq45g harness: inference timed out\n",
                )
            finally:
                if burn is not None:
                    if burn.poll() is None:
                        burn.terminate()
                    try:
                        burn.wait(timeout=2)
                    except subprocess.TimeoutExpired:
                        burn.kill()
                        burn.wait()
            wall = round(time.monotonic() - started, 3)
            err = run.stderr.decode("utf-8", "replace")
            row = {
                "repetition": rep,
                "pressure": pressured,
                "cuda_graph": not a.no_cuda_graph,
                "cuda_launch_blocking": a.launch_blocking,
                "route": a.route_label,
                "returncode": run.returncode,
                "wall_seconds": wall,
                "stdout_sha": hashlib.sha256(run.stdout).hexdigest(),
                "token_ids": generated_token_ids(err),
                "token_digest": field(err, "generated token digest"),
                "trajectory_digest": field(err, "mtp trajectory digest"),
                "decode": field(err, "decode speed"),
                "rounds": field(err, "mtp rounds"),
                "drafted": field(err, "mtp drafted tokens"),
                "accepted": field(err, "mtp accepted tokens"),
                "by_position": field(err, "mtp accepted by pos"),
                "fallback": field(err, "mtp fallback steps"),
            }
            current = sig(row)
            if expected is None:
                expected = current
            mismatches = {
                k: {"expected": v, "actual": current[k]}
                for k, v in expected.items()
                if v is not None and current[k] != v
            }
            row["matches_reference"] = run.returncode == 0 and not mismatches
            row["mismatches"] = mismatches
            rows.append(row)
            suffix = f"{rep:03d}_{'pressure' if pressured else 'normal'}"
            (a.output / f"{suffix}.stdout").write_bytes(run.stdout)
            (a.output / f"{suffix}.stderr").write_bytes(run.stderr)
            (a.output / "summary.json").write_text(json.dumps(rows, indent=2) + "\n")
            print(json.dumps(row), flush=True)
            if not row["matches_reference"]:
                return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
