#!/usr/bin/env python3
"""End-to-end orchestrator for the systems-model pipeline.

Runs the stages in order as subprocesses. Run it from inside submission/ so the
data dirs (runs/ bench/ docs/) resolve relative to the working directory:

    .venv/bin/python run_pipeline.py                 # compose -> dashboard (cheap; uses committed bench/*)
    .venv/bin/python run_pipeline.py --from gatetracker
    .venv/bin/python run_pipeline.py --only compose,systemsmap
    .venv/bin/python run_pipeline.py --topics --explore   # full rebuild incl. web-grounded stages
    .venv/bin/python run_pipeline.py --dry-run

Stage map (● = web-grounded / expensive, needs the Claude Code login):

    0 topics        stage-0 topic bench                 -> bench/topics.json      ●   (opt-in: --topics)
    1 explore       per-topic note mining               -> runs/<t>/notes.json    ●   (opt-in: --explore)
    2 decisiongraph per-topic decision graph            -> runs/<t>/decisiongraph.json ● (needs runs/)
    3 compose       merge topic graphs                  -> bench/composed-graph.json
    4 gatetracker   live gauges per binding variable    -> bench/gate-tracker.json ●
    5 systemsmap    feedback loops / branches           -> bench/systems-map.json
    6 takeoff       RSI-takeoff branch                  -> bench/takeoff.json      ●
    6 dominance     Liebig durability ladder            -> bench/dominance.json    ●
    7 part2         integrated framework write-up       -> docs/part2-final.md
    8 charts        regenerate figures                  -> docs/charts/*.png
    8 dashboard     one-file dashboard + PDF            -> docs/DASHBOARD.md / .pdf

runs/ is committed, so stages 2-4 are reproducible; or start at `compose`
(the default) — the committed bench/*.json carry everything downstream needs.
"""
from __future__ import annotations

import argparse
import glob
import os
import re
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))          # submission/
ROOT = os.getcwd()                                          # submission/ (run from here)
CHARTS = os.path.join(ROOT, "docs", "charts")

# canonical chart scripts (superseded make_charts*.py variants are intentionally omitted)
CHART_SCRIPTS = [
    "make_gate_charts.py", "make_charts_gt.py", "make_charts_gt2.py",
    "make_migration.py", "make_gt1_alt.py", "make_appendix.py",
]


def _topic_ids() -> list[str]:
    """Topic run-dir prefixes that have been explored (e.g. t1, t6stack, t9)."""
    tids = []
    for d in sorted(glob.glob(os.path.join(ROOT, "runs", "*"))):
        m = re.match(r"^(t\d+\w*)-", os.path.basename(d))
        if m:
            tids.append(m.group(1))
    return tids


def _script(name: str, *args: str) -> list[str]:
    return [sys.executable, os.path.join(HERE, name), *args]


def build_steps(topics: bool, explore: bool) -> list[tuple[str, list[list[str]]]]:
    """Ordered (stage-name, [commands]) list. Multi-command stages fan a script per item."""
    tids = _topic_ids()
    steps: list[tuple[str, list[list[str]]]] = []
    if topics:
        steps.append(("topics", [_script("run_topics.py")]))
    if explore:
        # explore takes a free-text topic argument; it is intended for interactive use.
        steps.append(("explore", [_script("run_explore.py")]))
    steps += [
        ("decisiongraph", [_script("run_decisiongraph.py", t) for t in tids]),
        ("compose", [_script("run_compose.py")]),
        ("gatetracker", [_script("run_gatetracker.py")]),
        ("systemsmap", [_script("run_systemsmap.py")]),
        ("takeoff", [_script("run_takeoff.py")]),
        ("dominance", [_script("run_dominance.py")]),
        ("part2", [_script("run_part2.py")]),
        ("charts", [[sys.executable, os.path.join(CHARTS, s)] for s in CHART_SCRIPTS
                    if os.path.exists(os.path.join(CHARTS, s))]),
        ("dashboard", [_script("build_dashboard.py"), _script("build_pdf.py")]),
    ]
    return steps


def main() -> int:
    ap = argparse.ArgumentParser(description="Run the systems-model pipeline e2e.")
    ap.add_argument("--from", dest="start", metavar="STAGE", help="start at this stage")
    ap.add_argument("--only", metavar="STAGE[,STAGE]", help="run only these stages")
    ap.add_argument("--topics", action="store_true", help="include stage-0 topic generation")
    ap.add_argument("--explore", action="store_true", help="include stage-1 explore")
    ap.add_argument("--dry-run", action="store_true", help="print commands, don't run")
    args = ap.parse_args()

    steps = build_steps(topics=args.topics, explore=args.explore)
    names = [n for n, _ in steps]

    if args.only:
        want = {s.strip() for s in args.only.split(",")}
        bad = want - set(names)
        if bad:
            print(f"unknown stage(s): {', '.join(sorted(bad))}\navailable: {', '.join(names)}")
            return 2
        steps = [(n, c) for n, c in steps if n in want]
    elif args.start:
        if args.start not in names:
            print(f"unknown stage: {args.start}\navailable: {', '.join(names)}")
            return 2
        steps = steps[names.index(args.start):]

    print(f"pipeline · cwd={ROOT}")
    print(f"stages: {' -> '.join(n for n, _ in steps)}\n")
    for name, cmds in steps:
        if not cmds:
            print(f"— {name}: nothing to do (no topic runs found) — skipping")
            continue
        for cmd in cmds:
            shown = " ".join(os.path.relpath(c, ROOT) if os.path.isabs(c) else c for c in cmd)
            print(f"▶ {name}: {shown}")
            if args.dry_run:
                continue
            r = subprocess.run(cmd, cwd=ROOT)
            if r.returncode != 0:
                print(f"✗ {name} failed (exit {r.returncode}) — stopping")
                return r.returncode
    print("\n✓ pipeline complete")
    return 0


if __name__ == "__main__":
    sys.exit(main())
