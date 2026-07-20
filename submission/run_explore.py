#!/usr/bin/env python3
"""Run the iterating exploration loop on a topic.

    python run_explore.py "future market structure of LLMs"
    python run_explore.py "..." --max-rounds 8 --patience 2

Needs ANTHROPIC_API_KEY (in .env or the environment).
"""
from __future__ import annotations

import argparse
import sys

from dotenv import load_dotenv

from forecaster.explore import run


def main() -> int:
    load_dotenv()
    p = argparse.ArgumentParser(description="Iterate notes on a topic until no new material notes appear.")
    p.add_argument("topic", help="The topic to explore, in quotes.")
    p.add_argument("--max-rounds", type=int, default=20, help="Safety cap on note rounds (default 20; converge first).")
    p.add_argument("--patience", type=int, default=2, help="Stop after N consecutive dry rounds (default 2).")
    p.add_argument("--per-round", type=int, default=12, help="Max new items requested per round (default 12).")
    p.add_argument("--analogy-rounds", type=int, default=12, help="Max rounds for the analogy phase (default 12).")
    p.add_argument("--no-analogies", action="store_true", help="Skip the historical-analogies phase.")
    p.add_argument("--analogies-only", action="store_true",
                   help="Skip notes; load an existing run and only mine analogies.")
    p.add_argument("--model", default=None, help="Override model (default = your Claude Code model).")
    p.add_argument("--out", default=None, help="Output directory (default runs/<slug>).")
    args = p.parse_args()

    run(
        args.topic,
        max_rounds=args.max_rounds,
        patience=args.patience,
        per_round=args.per_round,
        model=args.model,
        outdir=args.out,
        notes_phase=not args.analogies_only,
        analogies_phase=not args.no_analogies,
        analogy_rounds=args.analogy_rounds,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
