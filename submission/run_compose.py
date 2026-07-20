#!/usr/bin/env python3
"""Cross-topic COMPOSITION stage — merges per-topic decision graphs into ONE graph.

Replaces the old Stage 5 (thematic driver-clustering), which over-aggregated distinct
variables into 5-7 coarse 'drivers' and destroyed the actor-specificity + interaction
structure. This composes by REFERENT (same underlying thing → one variable), unions the
edges, and reads the worldview OFF the composed graph rather than clustering it.

Prereq: run the decision-graph stage on each topic first, e.g.
    for t in t1 t2 t3 t4 t5 t6stack t7 t8 t9; do python run_decisiongraph.py $t; done
    (t6 is replaced by t6stack, which already has a decisiongraph.json)

Then:
    python run_compose.py     # reads runs/*/decisiongraph.json, writes docs/composed-graph.md
"""
from __future__ import annotations
import glob, json, os, sys
from typing import Literal
from dotenv import load_dotenv
from pydantic import BaseModel, Field

from forecaster.llm import LLM

# ------------------------------------------------------------------ schema


class ActorReading(BaseModel):
    actor: str
    reading: str


class MergedVariable(BaseModel):
    id: str = Field(description="GV1, GV2 … (global variable id).")
    name: str
    sources: list[str] = Field(description="Provenance: which per-topic variables merged, e.g. ['t6stack:V1','t5:V3'].")
    state_space: list[str] = Field(description="Unioned distinct values/regimes — NOT collapsed to yes/no.")
    actor_readings: list[ActorReading] = Field(default_factory=list, description="Reconciled per-actor readings (note disagreements).")
    current_reading: str
    trajectory: str = Field(description="Direction + clock (catch-up-ability), reconciled across topics.")
    leverage: Literal["high", "medium", "low"]
    reconciliation: str = Field(default="", description="If sources disagreed, how you reconciled (else '').")


class ComposedEdge(BaseModel):
    source: str = Field(description="GV id.")
    target: str = Field(description="GV id.")
    kind: Literal["gates", "substitutes_for", "chokepoint_controlled_by", "correlates"]
    actor: str = Field(default="")
    note: str


class ForecastSignal(BaseModel):
    id: str
    binary_question: str
    measures: str = Field(description="Which GV id it reads.")
    probability: float
    resolution_criteria: str


class ComposedGraph(BaseModel):
    merged_variables: list[MergedVariable]
    edges: list[ComposedEdge]
    spine: list[str] = Field(description="The 5-8 highest-leverage GV ids the whole worldview hangs on (the honest 'axes' — graph nodes, not themes).")
    cross_cutting: list[str] = Field(description="GV ids that recur across many topics (e.g. the diffusion clock, the demand question).")
    binding_constraint_by_actor: list[ActorReading] = Field(description="For each actor (US/China/EU/Gulf), its minimum stave across the WHOLE problem — read off the graph.")
    worldview: str = Field(description="3-6 sentences describing the STRUCTURE the composed graph reveals (actor-specific staves, mutual chokepoints, what clears on what clock, the tails). Do NOT force a single 'who wins'.")
    key_tensions: list[str] = Field(description="The genuinely open/uncertain nodes where the graph does not resolve.")
    forecast_register: list[ForecastSignal] = Field(description="The ~15-20 most decisive, non-duplicative forecast signals across all topics, each tied to a spine/high-leverage variable.")


SYSTEM = """You COMPOSE several per-topic decision graphs into ONE cross-topic decision graph. This REPLACES thematic 'driver-clustering', which over-aggregated distinct variables into 5-7 coarse drivers and destroyed actor-specificity and interaction structure (it is what turned a vector of constraints into a single 'physical constraint').

You are given N per-topic graphs, each a set of VARIABLES (with state-spaces, per-actor readings, trajectories) and EDGES (gates / substitutes_for / chokepoint_controlled_by / correlates).

Rules — the merge discipline is the entire point:
1. MERGE BY REFERENT, NOT BY THEME. Variables from different topics that denote the SAME underlying thing (e.g. 'leading-edge logic chips' appearing in several topics under different names) become ONE merged variable — union their state-spaces, reconcile their actor-readings and current readings (note any disagreement), keep the sharpest trajectory, and record provenance (source topic:var ids). NEVER lump DISTINCT things into a coarse bucket: do NOT create a 'physical constraint' variable out of chips + energy + materials. If two variables are related but distinct, KEEP THEM SEPARATE and connect them with an EDGE.
2. UNION THE EDGES across the merged variable set; dedup identical edges; keep the interaction structure intact. If merging reveals a new cross-topic edge (a variable from topic A gates one from topic B), add it.
3. SPINE: name the 5-8 highest-leverage merged variables the whole worldview hangs on — the load-bearing nodes. This is the honest version of 'axes', but they are graph NODES, not themes.
4. CROSS-CUTTING: name the variables that recur across many topics (e.g. a diffusion/commoditization clock, the demand-realization question).
5. binding_constraint_by_actor: for each actor (US, China, EU, Gulf), its minimum stave across the WHOLE problem — READ from the graph, do not assert.
6. worldview: describe the STRUCTURE the composed graph reveals — actor-specific staves, where the mutual chokepoints sit, what clears on what clock, and the tails. Do NOT force a single 'who wins'; if the honest read is contested, say so and state what it depends on.
7. key_tensions: the genuinely open/uncertain nodes.
8. forecast_register: select the ~15-20 most decisive, non-duplicative forecast signals across all topics (drop near-duplicates and near-certain restatements-of-the-present in favour of discriminating, dated, sourced ones), each tied to a spine/high-leverage variable.

Be disciplined about merge-by-referent. Preserve provenance. Encode uncertainty as a distribution over states, not a hedge."""


def _fmt_graph(tid: str, g: dict) -> str:
    L = [f"### TOPIC {tid}", f"summary: {g.get('summary','')}"]
    L.append("variables:")
    for v in g.get("variables", []):
        ars = "; ".join(f"{a['actor']}={a['reading']}" for a in v.get("actor_readings", []))
        L.append(f"  [{tid}:{v['id']}] {v['name']} (lev {v.get('leverage','?')}) "
                 f"states={v.get('state_space', [])}" + (f" | actors: {ars}" if ars else ""))
        L.append(f"      now: {v.get('current_reading','')[:220]}")
        L.append(f"      traj: {v.get('trajectory','')[:200]}")
    L.append("edges:")
    for e in g.get("edges", []):
        act = f" [{e.get('actor')}]" if e.get("actor") else ""
        L.append(f"  {tid}:{e['source']} -{e['kind']}-> {tid}:{e['target']}{act}: {e.get('note','')[:140]}")
    L.append("binding_by_actor:")
    for a in g.get("binding_constraint_by_actor", []):
        L.append(f"  {a['actor']}: {a['reading'][:200]}")
    L.append("forecast_signals:")
    for f in g.get("forecast_signals", []):
        L.append(f"  [{tid}:{f['id']}] P={f.get('probability')} {f['binary_question']}")
    return "\n".join(L)


def _render_md(g: ComposedGraph) -> str:
    L = ["# Composed cross-topic decision graph\n", f"**Worldview (read off the graph):** {g.worldview}\n"]
    L.append("## Spine (load-bearing variables)")
    byid = {v.id: v for v in g.merged_variables}
    for gid in g.spine:
        v = byid.get(gid)
        L.append(f"- **{gid} · {v.name}**" if v else f"- {gid}")
    L.append("\n## Binding constraint by actor")
    for a in g.binding_constraint_by_actor:
        L.append(f"- **{a.actor}:** {a.reading}")
    L.append("\n## Cross-cutting variables")
    for gid in g.cross_cutting:
        v = byid.get(gid); L.append(f"- **{gid} · {v.name}**" if v else f"- {gid}")
    L.append("\n## Merged variables")
    for v in g.merged_variables:
        star = " ⭐spine" if v.id in g.spine else ""
        L.append(f"\n### {v.id} · {v.name}  _(leverage {v.leverage}){star}_")
        L.append(f"- sources: {', '.join(v.sources)}")
        L.append(f"- state space: {' | '.join(v.state_space)}")
        for ar in v.actor_readings:
            L.append(f"- **{ar.actor}:** {ar.reading}")
        L.append(f"- now: {v.current_reading}")
        L.append(f"- trajectory: {v.trajectory}")
        if v.reconciliation:
            L.append(f"- _reconciled: {v.reconciliation}_")
    L.append("\n## Edges (composed interaction structure)")
    for e in g.edges:
        act = f" [{e.actor}]" if e.actor else ""
        L.append(f"- **{e.source} —{e.kind}→ {e.target}**{act}: {e.note}")
    L.append("\n## Key tensions (unresolved nodes)")
    for t in g.key_tensions:
        L.append(f"- {t}")
    L.append("\n## Forecast register (deduped, most decisive)")
    for f in g.forecast_register:
        L.append(f"- **{f.id}** (P={f.probability:g}, measures {f.measures}): {f.binary_question}")
        L.append(f"    - _resolve:_ {f.resolution_criteria}")
    return "\n".join(L)


def main() -> int:
    load_dotenv()
    graphs = []
    for path in sorted(glob.glob("runs/*/decisiongraph.json")):
        tid = os.path.basename(os.path.dirname(path)).split("-")[0]
        graphs.append((tid, json.load(open(path, encoding="utf-8"))))
    if not graphs:
        print("no decisiongraph.json files found — run run_decisiongraph.py on the topics first."); return 1

    print(f"composing {len(graphs)} topic graphs: {', '.join(t for t, _ in graphs)}")
    if len(graphs) < 3:
        print("  ⚠ only", len(graphs), "graph(s) — run the triage batch first for a real composition.")

    blocks = "\n\n".join(_fmt_graph(t, g) for t, g in graphs)
    user = ("PER-TOPIC DECISION GRAPHS:\n\n" + blocks +
            "\n\nCompose these into ONE cross-topic decision graph per the rules. "
            "Merge by referent, union edges, name the spine, read the worldview off the graph, "
            "and build a deduped forecast register. Preserve provenance (topic:var ids).")

    llm = LLM()
    print("calling deep model (a few minutes)…\n")
    g = llm.parse(system=SYSTEM, user=user, schema=ComposedGraph)

    os.makedirs("bench", exist_ok=True)
    json.dump(g.model_dump(), open("bench/composed-graph.json", "w", encoding="utf-8"), indent=2)
    open("docs/composed-graph.md", "w", encoding="utf-8").write(_render_md(g))
    print(f"✓ {len(g.merged_variables)} merged variables (from ~{sum(len(gr.get('variables',[])) for _,gr in graphs)} topic vars), "
          f"{len(g.edges)} edges, spine {g.spine}, {len(g.forecast_register)} forecasts")
    if llm.last_cost_usd:
        print(f"  cost: ${llm.last_cost_usd:.2f}")
    print("  wrote bench/composed-graph.json and docs/composed-graph.md")
    return 0


if __name__ == "__main__":
    sys.exit(main())
