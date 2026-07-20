#!/usr/bin/env python3
"""Part 2 generator — the ≤3-page 'Framework & Holistic Synthesis' deliverable, generated
(not hand-written) from the pipeline outputs so it stays grounded and reproducible.

One synthesis call over: the systems-map (thesis / dominant dynamic / branches / loops),
the gate-tracker forward-calls + the takeoff analysis (= the most impactful forecasts), and
the chart inventory. Encodes the Bridgewater Part-2 brief: a coherent framework, the key
cause-and-effect dynamics, references to the most impactful Part-1 forecasts, charts welcome,
SYNTHESIS not per-forecast justification, dense 1500-2000 words, plain language (NO internal
codes/jargon). Argument-first on charts: it picks the best EXISTING chart per point and SPECS
any better chart that doesn't exist yet (printed for follow-up).

    python run_part2.py     # reads systems-map + gate-tracker + takeoff + composed-graph
"""
from __future__ import annotations
import json, os, re, sys
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from forecaster.llm import LLM

SYSMAP, TRACKER, TAKEOFF, GRAPH = "bench/systems-map.json", "bench/gate-tracker.json", "bench/takeoff.json", "bench/composed-graph.json"
DOMINANCE = "bench/dominance.json"
OUT_MD = "docs/part2-framework-synthesis.md"

CHART_INVENTORY = {
    "gt1_thesis.png": "prize evaporates (API $/M tokens 2020→26) vs moat widens (US transformer lead-time 2018→26)",
    "gt2_us_power_bind.png": "China power-adds ramp 2015→25 vs US interconnection queue 2015→24 — silicon→electrons",
    "gt3_master_derivative.png": "METR task-horizon 2019→25 (2s→2.3h) vs closed−open capability gap closing (Elo)",
    "gt4_value_capture.png": "hyperscaler capex 2019→26 vs model-API price collapse (value barbell)",
    "gt_migration.png": "the binding-constraint MIGRATION timeline: chips/EUV→power→power+aggregation as value moves training→deployment (the conclusion visual)",
    "gate_GV1.png": "leading-edge chip dashboard", "gate_GV8.png": "critical-minerals dashboard",
    "gate_GV11.png": "capability-timing dashboard", "gate_GV21.png": "Taiwan dashboard",
    "gate_GV12.png": "talent dashboard", "gate_GV4.png": "export-controls dashboard",
}


class Section(BaseModel):
    heading: str
    markdown: str = Field(description="Dense synthesis prose for this section — cause-and-effect, specific numbers, named historical analogies. No fluff.")
    chart: str = Field(default="", description="Filename of the BEST existing chart to embed here (from the inventory), or '' if none fits / a new one is needed.")
    anchoring_forecasts: list[str] = Field(description="The most impactful Part-1 forecasts this section rests on, each WITH its probability (e.g. 'No durable winner-take-all lead before 2030 (~75%)'). Reference, don't justify.")


class NewChartSpec(BaseModel):
    filename_hint: str = Field(description="e.g. 'scenario_fan.png'")
    proves: str = Field(description="The single point this chart must make.")
    spec: str = Field(description="Exactly what to plot — series/metrics (with the numbers if known), chart type, panels — so it can be built.")


class Part2(BaseModel):
    title: str
    thesis_paragraph: str = Field(description="The one-paragraph big-picture view — gripping, committed, the whole framework in ~120-180 words.")
    sections: list[Section] = Field(description="5-6 sections presenting the key cause-and-effect dynamics + the scenario synthesis. Dense.")
    holistic_synthesis: str = Field(description="Why AI and modern mercantilism are ONE story — the closing synthesis.")
    what_would_change_our_mind: str = Field(description="The specific watchable signals that would flip the view.")
    charts_to_build: list[NewChartSpec] = Field(description="NEW charts the argument needs that the inventory lacks (e.g. a scenario-probability fan, a constraint-migration chart). Empty if the existing set suffices.")
    word_count_estimate: int = Field(description="Your estimate of total words across thesis + sections + synthesis (target 1500-2000).")


SYSTEM = """You write Part 2 — 'Framework & Holistic Synthesis' — for the Bridgewater 'Forecasting the Future' competition. The brief: present a COHERENT FRAMEWORK that ties the forecasts into a big-picture view of how AI and modern mercantilism reshape the next decade; explain the key CAUSE-AND-EFFECT dynamics; REFERENCE the most impactful Part-1 forecasts to anchor the thinking; charts welcome. This is about SYNTHESIS — NOT justifying each forecast individually.

Hard constraints:
- DENSE, 1500-2000 words total. Every sentence earns its place; no hedging, no filler, committed voice.
- Plain language for a smart generalist panel. NEVER use internal codes or pipeline jargon (no variable ids, loop ids, gate ids, 'GV…', topic codes). Use named real-world referents (EUV, TSMC, Taiwan, hyperscalers) and named historical analogies.
- Reference the most impactful forecasts WITH their probabilities, woven into the argument as anchors — do not enumerate or defend them.
- CHARTS: for each point, choose the BEST existing chart from the inventory (by filename) to embed. Where the inventory lacks the ideal chart for a point, SPEC a new one in charts_to_build (series + type) rather than forcing a weak fit. Argument first, chart to match. The conclusion likely wants a chart the inventory lacks — a 'binding-constraint migration' timeline (chips/EUV early → power → power+aggregation late) or a 'difficulty vs durability vs value' map — spec it.
- THE CONCLUSION IS NOT A LATTICE. The holistic_synthesis MUST land the committed, SEQUENCED, directional call from the dominance analysis, not "a symmetric lattice, depends on relative clearing speed." Specifically: the binding constraint MIGRATES (chips/EUV in the 2024–26 training race → power/minerals in the 2026–28 buildout → power + aggregation in the 2028–30+ deployment phase); difficulty and durability are ANTI-correlated with value (the hardest gate, EUV, guards the fastest-commoditizing prize); durable value accrues to the owners of ELECTRONS + DISTRIBUTION/AGGREGATION (as in electrification and the internet), NOT the chip/tooling monopoly; geopolitically the edge tilts to the power-abundant-AND-absorptive builder (China's structural position) over the tooling holder (the US) — because energy dominance is durable only coupled with absorptive capacity; this INVERTS if RSI takeoff re-freezes the frontier (→ US/chips), and is VOIDED by a demand/ROI bust (→ capital burns, no one wins).

You are given the systems-map (thesis, dominant dynamic, branches), the gate-tracker forward-calls + takeoff analysis (forecast material), the DOMINANCE analysis (the conclusion to land), and the chart inventory. Build the tightest, boldest, most coherent synthesis the evidence supports — a shape and a set of committed calls, ending on the sequenced directional conclusion, not a survey."""


def _digest(sm, gt, tk, dm) -> str:
    L = ["=== THESIS MATERIAL (systems-map) ==="]
    L += [f"dominant dynamic: {sm.get('dominant_dynamic','')}",
          f"system-dominant loop: {sm.get('system_dominant_loop','')}",
          f"current phase: {sm.get('current_phase','')}",
          f"macro trajectory: {sm.get('macro_trajectory','')}",
          f"synthesis: {sm.get('synthesis','')}"]
    L.append("branches (the scenario fan):")
    for b in sm.get("branches", []):
        L.append(f"  {b['name']} — P={b.get('probability')}: {b.get('shape','')[:120]} | wins: {b.get('who_wins','')[:90]}")
    L.append("tail ignitions:")
    for t in sm.get("tail_ignitions", []):
        L.append(f"  {t['tail']} P={t.get('probability')} → {t.get('what_it_resets','')[:70]}")
    L.append("loops (plain-language, for cause-effect — DO NOT cite ids):")
    for l in sm.get("loops", []):
        L.append(f"  {l['name']} [{l['kind']}]: {l.get('mechanism','')[:110]}")

    L.append("\n=== FORECAST MATERIAL (gate forward-calls = Part 1 anchors) ===")
    for x in gt.get("gates", []):
        L.append(f"{x['gate_name']}: {(x.get('forward_call','') or '')[:170]}")

    if tk:
        L.append("\n=== TAKEOFF / WINNER-TAKE-ALL ===")
        L.append(f"P(winner-take-all | takeoff) = {tk.get('calibrated_probability')}; {tk.get('lead_vs_advantage','')[:300]}")

    if dm:
        L.append("\n=== DOMINANCE / THE CONCLUSION (Liebig + GPT reference class) — the framework MUST land this ===")
        L.append(f"decisive chokepoint: {dm.get('decisive_chokepoint','')}")
        L.append(f"who wins the value: {dm.get('who_wins_the_value','')}")
        L.append(f"durability ladder: {dm.get('durability_ladder','')}")
        L.append("binding sequence (the Liebig migration): " +
                 " | ".join(f"{s.get('phase','')}→{s.get('binding_constraint','')} (favors {s.get('who_it_favors','')[:40]})"
                            for s in dm.get("binding_sequence", [])))
        L.append(f"energy verdict: {dm.get('energy_verdict','')}")
        L.append("GPT precedents: " + "; ".join(f"{g.get('case','')} → value to {g.get('where_value_actually_landed','')[:50]}" for g in dm.get("gpt_precedents", [])))
        L.append(f"committed conclusion: {dm.get('committed_conclusion','')}")
        L.append(f"inverts if: {dm.get('inverts_if','')} | voided if: {dm.get('voided_if','')}")

    L.append("\n=== CHART INVENTORY (choose the best per point; spec new ones if needed) ===")
    for fn, d in CHART_INVENTORY.items():
        L.append(f"  {fn} — {d}")
    txt = "\n".join(L)
    # strip internal code tokens so they cannot leak into the reader-facing draft
    txt = re.sub(r"\s*\((?:GV|R|B)\d+[^)]*\)", "", txt)   # "(R3)", "(GV11 …)"
    txt = re.sub(r"\b(?:GV|GV\d+|R\d+|B\d+)\b", "", txt)  # stray "GV11", "R1"
    return txt


def _md(p: Part2) -> str:
    L = [f"# Part 2 — Framework & Holistic Synthesis", f"## {p.title}\n", f"**{p.thesis_paragraph}**\n"]
    for s in p.sections:
        L.append(f"\n### {s.heading}\n")
        if s.chart and os.path.exists(os.path.join("docs/charts", s.chart)):
            L.append(f"![{s.chart}](charts/{s.chart})\n")
        L.append(s.markdown)
        if s.anchoring_forecasts:
            L.append("\n> **Anchoring forecasts:** " + "; ".join(f"*{f}*" for f in s.anchoring_forecasts))
    L.append(f"\n### Synthesis — why the two forces are one story\n\n{p.holistic_synthesis}")
    L.append(f"\n### What would change our mind\n\n{p.what_would_change_our_mind}")
    return "\n".join(L)


def main() -> int:
    load_dotenv()
    if not os.path.exists(SYSMAP):
        print(f"missing {SYSMAP} — run run_systemsmap.py first."); return 1
    sm = json.load(open(SYSMAP, encoding="utf-8"))
    gt = json.load(open(TRACKER, encoding="utf-8")) if os.path.exists(TRACKER) else {}
    tk = json.load(open(TAKEOFF, encoding="utf-8")) if os.path.exists(TAKEOFF) else {}
    dm = json.load(open(DOMINANCE, encoding="utf-8")) if os.path.exists(DOMINANCE) else {}
    if not dm:
        print("  ⚠ no dominance.json — the conclusion will be weaker; run run_dominance.py first.")

    user = (_digest(sm, gt, tk, dm) +
            "\n\nWrite Part 2 per the brief: a coherent framework, the key cause-and-effect dynamics, "
            "the most impactful forecasts woven in as anchors (with probabilities), the scenario synthesis, and a "
            "close that LANDS THE SEQUENCED DOMINANCE CONCLUSION (binding constraint migrates chips/EUV → power → "
            "power+aggregation; difficulty anti-correlates with durable value; durable value → electrons + "
            "distribution/aggregation as in electrification & the internet; tilts to the power-abundant-absorptive "
            "builder; inverts in the RSI tail, voided in the bust). 1500-2000 words, dense, plain language, "
            "charts chosen per point + spec a binding-constraint-migration chart.")
    llm = LLM()
    print("generating Part 2 (synthesis, a few min)…")
    p = llm.parse(system=SYSTEM, user=user, schema=Part2, max_turns=6, call_timeout=900,
                  max_buffer_size=32 * 1024 * 1024)
    open(OUT_MD, "w", encoding="utf-8").write(_md(p))
    print(f"✓ wrote {OUT_MD} — ~{p.word_count_estimate} words, {len(p.sections)} sections")
    used = [s.chart for s in p.sections if s.chart]
    print(f"  charts used: {', '.join(used) or '(none)'}")
    if p.charts_to_build:
        print(f"  ⚠ {len(p.charts_to_build)} NEW chart(s) the argument wants (not yet built):")
        for c in p.charts_to_build:
            print(f"     - {c.filename_hint}: {c.proves}")
            print(f"       spec: {c.spec[:160]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
