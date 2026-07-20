#!/usr/bin/env python3
"""Stage 2 — decision graph for ONE topic.

The analysis unit is a decisive VARIABLE (with a state space + trajectory), not a
binary yes/no question. Runs on a topic's frozen notes (no re-explore).

    python run_decisiongraph.py t6stack

Reads runs/<id>-*/notes.json, calls the deep model once with the Variable + Edge +
ForecastSignal schema, and writes:
    runs/<id>-*/decisiongraph.json
    docs/<id>-decisiongraph.md
"""
from __future__ import annotations

import glob
import json
import os
import sys
from typing import Literal

from dotenv import load_dotenv
from pydantic import BaseModel, Field

from forecaster.llm import LLM
from forecaster.corpus import format_notes as _format_graph

# ----------------------------------------------------------------- schema (the fix)


class ActorReading(BaseModel):
    actor: str = Field(description="US | China | EU | Gulf | (other named actor)")
    reading: str = Field(description="Where this actor stands on the variable now, + confidence, cite #ids.")


class Variable(BaseModel):
    id: str = Field(description="V1, V2, …")
    name: str = Field(description="The decisive input/variable, e.g. 'leading-edge logic chips'.")
    state_space: list[str] = Field(
        description="The DISTINCT values/regimes it can take — NOT forced to yes/no. "
                    "e.g. ['US/allied-controlled','bifurcated','China-indigenized'] or a magnitude range.")
    actor_readings: list[ActorReading] = Field(
        default_factory=list,
        description="Populate ONLY if the variable is genuinely actor-specific (binds one actor more than another). "
                    "Empty for actor-neutral variables. Never assume a single global actor.")
    current_reading: str = Field(description="Where it sits NOW + confidence, grounded in notes (cite #ids).")
    trajectory: str = Field(
        description="DIRECTION + a CLOCK: how fast it moves and how catch-up-able it is, grounded in the "
                    "historical analogies (which factors were easy vs hard to catch up on). Cite #A ids.")
    leverage: Literal["high", "medium", "low"]
    base_rate: str = Field(description="Outside-view base rate from comparable past tech races, AND where the analogy breaks.")
    note_ids: list[int] = Field(default_factory=list)
    analogy_ids: list[int] = Field(default_factory=list)


class Edge(BaseModel):
    source: str = Field(description="Variable id the edge starts from.")
    target: str = Field(description="Variable id the edge points to.")
    kind: Literal["gates", "substitutes_for", "chokepoint_controlled_by", "correlates"]
    actor: str = Field(default="", description="Whose constraint this edge is about, if actor-specific (else '').")
    note: str = Field(description="One line on the mechanism, cite #ids.")


class ForecastSignal(BaseModel):
    id: str = Field(description="F1, F2, …")
    binary_question: str = Field(description="A yes/no, actor-AND-time-tagged question that MEASURES a variable's state.")
    measures: str = Field(description="Which Variable id this reads.")
    probability: float = Field(description="0.0-1.0, rounded to 0.1.")
    resolution_criteria: str = Field(description="Objective, single-source-checkable, with a deadline.")


class DecisionGraph(BaseModel):
    variables: list[Variable]
    edges: list[Edge]
    forecast_signals: list[ForecastSignal]
    binding_constraint_by_actor: list[ActorReading] = Field(
        description="For each actor, which variable is its MINIMUM STAVE (binding input) — read from the graph, don't assert.")
    summary: str = Field(
        description="3-4 sentences on the STRUCTURE the graph reveals (who is gated by what, where mutual chokepoints sit, "
                    "whose constraint clears on what clock). Describe structure; do NOT force a single 'who wins'.")


SYSTEM = """You convert a topic's research note-graph into a DECISION GRAPH. The analysis unit is a decisive VARIABLE, not a yes/no question.

This REPLACES the old step that distilled a topic into 4-6 binary yes/no cruxes. That step flattened multi-dimensional, actor-specific, interacting constraints into single binaries (e.g. it turned 'what constrains the AI buildout?' into 'is the constraint physical — yes/no', losing that chips bind China while grid binds the US). Do NOT emit yes/no questions as your primary objects.

Produce three things:

1. VARIABLES — the 6-10 decisive variables the topic's answer swings on. For each:
   - state_space: the distinct VALUES/REGIMES it can take. NEVER collapse to yes/no if the honest answer is a spectrum or a set of regimes.
   - actor_readings: where each relevant actor (US, China, EU, Gulf, …) stands — ONLY when the variable genuinely binds one actor more than another. Leave empty for actor-neutral variables. Never assume a single global actor.
   - current_reading: where it sits now, with confidence, grounded in the notes (cite #ids).
   - trajectory: DIRECTION + a CLOCK — how fast it moves and how catch-up-able it is, grounded in the historical analogies (which factors were easy vs hard to catch up on in past tech races). Cite #A ids.
   - leverage, base_rate (outside view + where the analogy breaks), note_ids, analogy_ids.

2. EDGES — the interaction structure. For every important pair of variables, state the relationship: 'gates' (A must clear before B), 'substitutes_for' (A can relieve scarcity of B), 'chokepoint_controlled_by' (one actor controls A that another needs for B), or 'correlates'. This is where minimum-stave, substitution, and mutual-chokepoint structure lives — it is REQUIRED, not optional.

3. FORECAST SIGNALS — binary, actor-and-time-tagged yes/no questions that MEASURE a variable's state (these are the competition-facing forecasts, DERIVED from the variables). Each names the variable it reads, a probability (rounded to 0.1), and objective resolution criteria with a deadline.

Then read off:
- binding_constraint_by_actor: for each actor, which variable is its minimum stave — READ from the graph, do not assert.
- summary: describe the STRUCTURE (who is gated by what, where mutual chokepoints sit, whose constraint clears on what clock). Do NOT force a single 'who wins' conclusion — if the honest read is 'contested / depends on relative clearing speed', say exactly that.

Be grounded (cite note #ids and analogy #A ids), decisive about leverage, and encode uncertainty as a distribution over states — not as a hedge."""


def _render_md(tid: str, g: DecisionGraph) -> str:
    L = [f"# {tid.upper()} — Decision-Graph (prototype)\n",
         f"**Structure summary:** {g.summary}\n",
         "## Binding constraint by actor (min-stave, read off the graph)"]
    for a in g.binding_constraint_by_actor:
        L.append(f"- **{a.actor}:** {a.reading}")
    L.append("\n## Variables")
    for v in g.variables:
        L.append(f"\n### {v.id} · {v.name}  _(leverage: {v.leverage})_")
        L.append(f"- **State space:** {' | '.join(v.state_space)}")
        if v.actor_readings:
            for ar in v.actor_readings:
                L.append(f"- **{ar.actor}:** {ar.reading}")
        L.append(f"- **Now:** {v.current_reading}")
        L.append(f"- **Trajectory / clock:** {v.trajectory}")
        L.append(f"- **Base rate:** {v.base_rate}")
        L.append(f"- _notes {v.note_ids} · analogies {v.analogy_ids}_")
    L.append("\n## Edges (interaction structure)")
    for e in g.edges:
        act = f" [{e.actor}]" if e.actor else ""
        L.append(f"- **{e.source} —{e.kind}→ {e.target}**{act}: {e.note}")
    L.append("\n## Forecast signals (derived, competition-facing)")
    for f in g.forecast_signals:
        L.append(f"- **{f.id}** (P={f.probability:g}, measures {f.measures}): {f.binary_question}")
        L.append(f"    - _resolve:_ {f.resolution_criteria}")
    return "\n".join(L)


def main() -> int:
    load_dotenv()
    tid = (sys.argv[1] if len(sys.argv) > 1 else "t6").lower()

    dirs = sorted(glob.glob(f"runs/{tid}-*"))
    if not dirs:
        print(f"no run dir matching runs/{tid}-*  (run the topic explore first)")
        return 1
    outdir = dirs[0]

    # Prefer raw notes.json: it carries the analogies (the base-rate/trajectory grounding).
    # notes.merged.json is a token-saving dedup that in this run predates the analogies (0 of them).
    notes_path = os.path.join(outdir, "notes.json")
    if not os.path.exists(notes_path):
        notes_path = os.path.join(outdir, "notes.merged.json")
    if not os.path.exists(notes_path):
        print(f"no notes.json in {outdir}")
        return 1

    data = json.load(open(notes_path, encoding="utf-8"))
    if not data.get("analogies"):
        merged = os.path.join(outdir, "notes.merged.json")
        # (defensive) if a future run puts analogies only in merged, pull them in
        if os.path.exists(merged):
            m = json.load(open(merged, encoding="utf-8"))
            if m.get("analogies"):
                data["analogies"] = m["analogies"]
    n_notes = len(data.get("notes", []))
    n_analog = len(data.get("analogies", []))
    print(f"=== {tid.upper()} decision-graph prototype ===")
    print(f"notes source: {notes_path}  ({n_notes} notes, {n_analog} analogies)\n")

    graph_text = _format_graph(data)
    user = (
        f"TOPIC RUN DIR: {outdir}\n\n"
        f"{graph_text}\n\n"
        "Build the DECISION GRAPH for this topic per the instructions: variables (with state_space + "
        "actor_readings + trajectory), edges (the interaction structure), derived forecast signals, "
        "the binding constraint per actor, and a structure summary. Cite note #ids and analogy #A ids throughout."
    )

    llm = LLM()  # default = CLI deep model (Opus)
    print("calling deep model (this can take a few minutes — run in a terminal, not Conductor)…\n")
    g = llm.parse(system=SYSTEM, user=user, schema=DecisionGraph)

    json.dump(g.model_dump(), open(os.path.join(outdir, "decisiongraph.json"), "w", encoding="utf-8"), indent=2)
    md = _render_md(tid, g)
    md_path = os.path.join("docs", f"{tid}-decisiongraph.md")
    open(md_path, "w", encoding="utf-8").write(md)

    print(f"✓ {len(g.variables)} variables, {len(g.edges)} edges, {len(g.forecast_signals)} forecast signals")
    if llm.last_cost_usd:
        print(f"  cost: ${llm.last_cost_usd:.2f}")
    print(f"  wrote {os.path.join(outdir, 'decisiongraph.json')}")
    print(f"  wrote {md_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
