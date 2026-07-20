#!/usr/bin/env python3
"""Consolidated reference — every global variable + every loop in one doc, including the
ones NOT prioritised (non-spine variables, variables that appear in no loop).

    python build_reference.py   -> docs/variables-and-loops.md
"""
import json, os

g = json.load(open("bench/composed-graph.json", encoding="utf-8"))
sm = json.load(open("bench/systems-map.json", encoding="utf-8")) if os.path.exists("bench/systems-map.json") else {}

vars_ = g["merged_variables"]
spine = set(g.get("spine", []))
cross = set(g.get("cross_cutting", []))
tracked = {"GV1", "GV2", "GV6", "GV8", "GV10", "GV11", "GV12", "GV15", "GV21", "GV4"}  # gate-tracker set
loops = sm.get("loops", [])

# which loops each variable appears in
in_loops = {v["id"]: [] for v in vars_}
for lp in loops:
    for nid in lp.get("nodes", []):
        if nid in in_loops:
            in_loops[nid].append(lp["id"])

L = ["# Reference — all global variables & all loops (including un-prioritised)\n",
     f"*{len(vars_)} merged variables · {len(loops)} loops · assembled from composed-graph.json + systems-map.json. "
     "⭐ = spine · ◆ = cross-cutting · ⚙ = gate-tracked · ↻ = appears in a loop.*\n"]

# ---- loops ----
L.append("## Loops (R = reinforcing, B = balancing / brake)\n")
for kind in ("reinforcing", "balancing"):
    L.append(f"### {'Reinforcing (R)' if kind=='reinforcing' else 'Balancing / brakes (B)'}")
    for lp in [x for x in loops if x.get("kind") == kind]:
        dom = " · **DOMINANT**" if lp.get("dominant_now") else ""
        reg = f" · {lp['regime']}" if lp.get("regime", "n/a") != "n/a" else ""
        L.append(f"- **{lp['id']} · {lp['name']}** _[{lp.get('basis','')}{reg}]_{dom}")
        L.append(f"    - nodes: {' → '.join(lp.get('nodes', []))}")
        L.append(f"    - gauge: {lp.get('gauge','')[:80]} · flip: {lp.get('flip_threshold','')[:90]}")
    L.append("")

# ---- all variables ----
L.append("## All 42 global variables (spine + un-prioritised)\n")
L.append("| ID | Variable | Leverage | Flags | In loops |")
L.append("|---|---|---|---|---|")
for v in sorted(vars_, key=lambda x: int(x["id"][2:]) if x["id"][2:].isdigit() else 999):
    flags = "".join(["⭐" if v["id"] in spine else "", "◆" if v["id"] in cross else "",
                     "⚙" if v["id"] in tracked else "", "↻" if in_loops.get(v["id"]) else ""])
    L.append(f"| {v['id']} | {v['name'][:60]} | {v.get('leverage','')} | {flags or '—'} | {', '.join(in_loops.get(v['id'], [])) or '—'} |")

# ---- the un-prioritised ----
dropped = [v for v in vars_ if v["id"] not in spine and v["id"] not in tracked and not in_loops.get(v["id"])]
L.append(f"\n## Un-prioritised — not in spine, not gate-tracked, in no loop ({len(dropped)})\n")
for v in sorted(dropped, key=lambda x: int(x["id"][2:]) if x["id"][2:].isdigit() else 999):
    L.append(f"- **{v['id']} · {v['name']}** _(leverage {v.get('leverage','')})_ — {(v.get('current_reading','') or '')[:110]}")

open("docs/variables-and-loops.md", "w", encoding="utf-8").write("\n".join(L))
print(f"wrote docs/variables-and-loops.md — {len(vars_)} vars, {len(loops)} loops, {len(dropped)} un-prioritised")
print(f"  reinforcing: {sum(1 for x in loops if x.get('kind')=='reinforcing')} · balancing: {sum(1 for x in loops if x.get('kind')=='balancing')}")
