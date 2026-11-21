#!/usr/bin/env python3
from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo


def main() -> int:
    parser = argparse.ArgumentParser(description="Append a timestamped checkpoint to progress.md")
    parser.add_argument("title", help="short checkpoint title")
    parser.add_argument("details", nargs="+", help="one or more detail paragraphs/bullets")
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    ledger = root / "progress.md"
    stamp = datetime.now(ZoneInfo("Asia/Bangkok")).strftime("%Y-%m-%d %H:%M +07")

    chunks = [f"\n### {stamp} — {args.title}\n"]
    for detail in args.details:
        detail = detail.strip()
        if not detail:
            continue
        if detail.startswith(("- ", "* ")):
            chunks.append(detail + "\n")
        else:
            chunks.append("- " + detail + "\n")
    with ledger.open("a", encoding="utf-8") as fh:
        fh.write("".join(chunks))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
