#!/usr/bin/env python3
"""Stage 0 — generate the topic bench (pillars + intersection topics).

    python run_topics.py --target 10

Writes bench/topics.json, bench/rejected.json, and docs/latest-bench.md.
"""
from __future__ import annotations

import argparse
import sys

from dotenv import load_dotenv

from forecaster.stage0 import generate_bench


def main() -> int:
    load_dotenv()
    p = argparse.ArgumentParser(description="Stage 0 — generate the topic bench.")
    p.add_argument("--target", type=int, default=10, help="Approx number of bench topics (default 10).")
    p.add_argument("--out", default="bench", help="Output directory (default bench/).")
    p.add_argument("--max-rounds", type=int, default=6, help="Max diverge rounds (default 6; saturation stops earlier).")
    p.add_argument("--model", default=None, help="Override the diverge/mining model.")
    args = p.parse_args()

    generate_bench(target=args.target, outdir=args.out, max_rounds=args.max_rounds, model=args.model)
    return 0


if __name__ == "__main__":
    sys.exit(main())
