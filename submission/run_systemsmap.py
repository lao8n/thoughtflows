#!/usr/bin/env python3
"""Systems-map stage (step 7) — the Dalio layer on top of the decision graphs.

Replaces the compose SPINE step (a lossy top-8 ranking) with a dynamical model.
Two calls:
  1. LOOPS + COUPLINGS — extract the reinforcing/balancing feedback loops (seeded from the
     real cycles, adding the feedback edges the dependency-first graph missed), each tagged
     basis (empirical/analogical/theoretical) + activation (unconditional/conditional) + a
     GAUGE read off gate-tracker.json + dominant_now + flip_threshold. Then the COUPLING MAP:
     where loops share nodes and contest, with a dominance rule + an adversarial verdict
     (the retired cross-implications stage, folded in as a coupling attribute; five move
     types are the checklist).
  2. BRANCHES + TEMPLATES + SYNTHESIS — theory-templates (Perez/Dalio/Bostrom/Acemoglu, nested
     by time-scale), driving forces, discrete branches (A base / B RSI-takeoff [theoretical,
     quarantined] / C bust / D fracture), the dominant dynamic, current phase, and the
     probability-weighted macro trajectory (a shape, not dated point-predictions).

Reuses bench/composed-graph.json's 42 vars + 64 edges as vocabulary (IGNORES the spine) and
reads bench/gate-tracker.json for the loop gauges. Reasoning over grounded inputs — no web.
Spec: docs/methodology-systems-model.md.

    python run_systemsmap.py
"""
from __future__ import annotations
import json, os, sys
from collections import defaultdict
from typing import Literal
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from forecaster.llm import LLM

GRAPH = "bench/composed-graph.json"
TRACKER = "bench/gate-tracker.json"
OUT_JSON = "bench/systems-map.json"
OUT_MD = "docs/systems-map.md"


# ------------------------------------------------------------------ schema: call 1
class LoopEdge(BaseModel):
    from_var: str = Field(description="GV id (or short name).")
    to_var: str
    polarity: Literal["+", "-"] = Field(description="+ = same-direction (more→more); - = inverse (more→less).")
    mechanism: str


class FeedbackLoop(BaseModel):
    id: str = Field(description="R1, R2 … (reinforcing) or B1, B2 … (balancing).")
    name: str
    kind: Literal["reinforcing", "balancing"]
    nodes: list[str] = Field(description="The GV ids the loop runs through.")
    edges: list[LoopEdge] = Field(description="The closed chain of polarised links.")
    mechanism: str = Field(description="The loop's causal story in one or two sentences.")
    regime: Literal["training", "inference", "both", "n/a"] = Field(default="n/a",
        description="Which compute regime the loop bites: leading-edge chips + concentrated power bind TRAINING; mature nodes + efficiency + distributed power serve INFERENCE; n/a for non-compute loops. Make the split explicit.")
    basis: Literal["empirical", "analogical", "theoretical"] = Field(
        description="empirical = observed across historical cases; analogical = 1-2 cases; theoretical = mechanism only (RSI is the ONLY reference-class-free loop).")
    activation: Literal["unconditional", "conditional"] = Field(
        description="conditional loops fire only inside a regime (theoretical loops are conditional).")
    activation_condition: str = Field(default="", description="If conditional, the regime/threshold that switches it on.")
    reference_class: str = Field(default="", description="The historical instances grounding it (empty if purely theoretical).")
    gauge: str = Field(description="Which tracked metric/gate reads this loop's current strength.")
    current_gain: str = Field(description="The loop's strength NOW, read from the gauge (accelerating/damped/dominant?).")
    dominant_now: bool = Field(description="Is it currently the dominant loop where it acts?")
    flip_threshold: str = Field(description="The level/condition that changes its sign or dominance — where the prediction lives. A loop with none earns no place.")


class Coupling(BaseModel):
    shared_node: str = Field(description="The GV id where loops meet/contest.")
    loops_involved: list[str] = Field(description="Loop ids.")
    contest: str = Field(description="How the loops push this node (same/opposite) and what's at stake.")
    move_type: Literal["substitution", "scope_of_constraint", "value_locus_shift", "asymmetric_adoption", "compounding", "reinforcement", "none"] = Field(
        description="The second-order move (folded-in cross-implications checklist): substitution (a route around a gate); scope_of_constraint (gated input binds only part of the value chain, e.g. training vs inference); value_locus_shift (value in use not intelligence-creation); asymmetric_adoption; compounding; reinforcement; none.")
    dominance_rule: str = Field(description="Which loop wins at this node NOW (from the gauges) and what would flip it.")
    evidence_for: str = Field(description="Grounded support for the dominant reading.")
    steelman_against: str = Field(description="The strongest case the dominance flips.")
    probability: float = Field(description="0-1: probability the dominant reading holds over the window.")


class LoopSet(BaseModel):
    loops: list[FeedbackLoop] = Field(description="The reinforcing + balancing loops (aim ~6-10).")
    couplings: list[Coupling] = Field(description="Where loops share nodes and contest — the coupling map that makes them one system.")


# ------------------------------------------------------------------ schema: call 2
class TheoryTemplate(BaseModel):
    name: str = Field(description="Perez / Dalio / Bostrom / Acemoglu-Restrepo …")
    time_scale: Literal["slow", "medium", "fast"] = Field(description="Nesting order: slow (Dalio) ⊃ medium (Perez) ⊃ fast (Bostrom).")
    governs: str = Field(description="Which loops / tempo of the system it describes.")
    phases: list[str] = Field(description="The template's phase sequence.")
    where_we_are: str = Field(description="Current phase mapping.")
    base_rate_shape: str = Field(description="The characteristic trajectory it predicts.")


class DrivingForce(BaseModel):
    name: str
    loops: list[str] = Field(description="Loop ids it aggregates.")
    reading: str = Field(description="Current state + direction.")


class Branch(BaseModel):
    id: Literal["A", "B", "C", "D"]
    name: str
    probability: float = Field(description="0-1 over the window; the four should form a coherent weighting.")
    is_theoretical: bool = Field(description="True only for B (RSI takeoff) — quarantined, its loops never touch the base case.")
    active_loops: list[str] = Field(description="Loop ids dominant in this branch.")
    trigger_threshold: str = Field(description="What tips the system into this branch.")
    shape: str = Field(description="The trajectory within this branch (phases, no date-precision).")
    who_wins: str = Field(description="How the who-captures-value picture resolves in this branch.")


class TailIgnition(BaseModel):
    tail: str
    loop: str = Field(description="Which loop crossing which threshold ignites it.")
    threshold: str
    probability: float
    what_it_resets: str


class SystemsMap(BaseModel):
    theory_templates: list[TheoryTemplate]
    driving_forces: list[DrivingForce]
    branches: list[Branch] = Field(description="A (base, modal), B (RSI, theoretical/quarantined), C (bust), D (fracture).")
    tail_ignitions: list[TailIgnition] = Field(description="Tails as loop-threshold crossings.")
    dominant_dynamic: str = Field(description="The single loop-contest that matters most now.")
    system_dominant_loop: str = Field(description="The SINGLE loop ruling the WHOLE system right now (not one-per-subdomain) + one line why.")
    current_phase: str = Field(description="Where we are, in Perez/Dalio idiom.")
    macro_trajectory: str = Field(description="The SHAPE of the decade (no dated point-predictions) — the Dalio-style read.")
    watch_thresholds: list[str] = Field(description="The gauges/levels that would switch branches — what to watch.")
    synthesis: str = Field(description="What the systems model says the contest actually is and where value lands.")


SYSTEM_LOOPS = """You extract the DYNAMICAL structure — feedback loops and their couplings — from a decision graph + live gauges. This replaces spine-ranking with a systems model; conviction lives in loops and thresholds, not in a ranked list.

You are given: merged variables (nodes), existing edges (mostly static 'gates' dependencies), candidate cycles already present in the edges, and GAUGES (per-gate measured trends = the current state and rate of each variable).

1. LOOPS — identify reinforcing (R) and balancing (B) feedback loops. Seed from the candidate cycles, but ADD the feedback edges the dependency-first graph missed (it has ~44 'gates' and almost no feedback — you must supply the closing edges, each with polarity + mechanism). For each loop set:
   - basis: empirical (observed across historical cases) / analogical / theoretical. RSI (capability→faster-R&D→capability) is the ONLY reference-class-free loop — mark it theoretical. Labor/cognitive substitution is grounded in mechanization — empirical, NOT theoretical.
   - activation: unconditional, or conditional (+ the regime that switches it on). Theoretical loops are conditional.
   - gauge (which tracked metric reads its strength), current_gain (from that gauge), dominant_now, and flip_threshold — the level/condition that changes its sign or dominance. THE FLIP-THRESHOLD IS WHERE THE PREDICTION LIVES. A loop with no threshold earns no place — drop it.
   - reference_class: the historical instances grounding it.
   - EVERY loop must have AT LEAST 3 nodes and 3 edges — a 2-node loop is just a bidirectional edge; expand it to the real mechanism (e.g. RSI is capability → R&D-productivity/optimization-power → more capability, NOT capability↔scaling).
   - regime: tag whether the loop bites TRAINING (leading-edge chips + concentrated power) or INFERENCE (mature nodes + efficiency + distributed power) — the chip gate binds training, not inference; make the split explicit.
   - Mark dominant_now on AT MOST 3 loops — only the ones actually running the whole system now, not every plausibly-active loop.
2. COUPLINGS — the loops are ONE system, not separate pieces. Find where they SHARE NODES and contest. For each: the shared node, loops involved, how they push it (same/opposite), and the DOMINANCE RULE (which wins now, from the gauges, and what flips it). Run the five second-order moves as a checklist so coverage isn't lost: substitution, scope_of_constraint (does the gated input bind the whole value chain or only part — e.g. leading-edge chips bind TRAINING not INFERENCE), value_locus_shift (value in economy-wide use, not intelligence-creation), asymmetric_adoption, compounding. Each coupling carries an adversarial verdict: evidence_for, steelman_against, and a probability the dominant reading holds.

Cite variables + gauges; never invent a loop that isn't mechanistically real."""

SYSTEM_SYNTH = """You assemble a set of feedback loops + couplings into a Dalio-shaped macro model. Do not invent new loops; reason over the ones given.

1. THEORY-TEMPLATES — make the generating patterns first-class: Perez (techno-economic/financial cycle: irruption→frenzy/installation→turning-point→synergy/deployment), Dalio (empire/debt/reserve-currency Big Cycle), Bostrom (RSI loop-gain / decisive strategic advantage), Acemoglu-Restrepo (task displacement vs reinstatement). For each: time_scale, which loops/tempo it governs, its phases, where we are, and its base-rate shape. They NEST by time-scale — Dalio (slow) ⊃ Perez (medium) ⊃ Bostrom (fast) — the outer sets the boundary conditions for the inner; do NOT average them.
2. DRIVING FORCES — cluster loops into ~5-7 aggregate forces with a current reading.
3. DISCRETE BRANCHES — A base / diffusion-deployment (modal); B RSI-takeoff (THEORETICAL, quarantined — its loops never touch the base case); C buildout-bust / Perez turning-point; D exogenous fracture (Taiwan/war). Each: probability (coherent weighting across the four), active loops, trigger threshold, shape, who-wins. Only B is theoretical.
4. dominant_dynamic — the single loop-contest that matters most now (e.g. the RSI reinforcing loop vs the diffusion balancing loop). Also name system_dominant_loop: the SINGLE loop ruling the whole system now (not one-per-subdomain).
5. current_phase (Perez/Dalio idiom), macro_trajectory (the SHAPE of the decade — no dated point-predictions), watch_thresholds (the gauges that switch branches), synthesis.

The forecast is a shape + probability-weighted branches + watch-thresholds, not a point prediction."""


# ------------------------------------------------------------------ context builders
def _seed_cycles(g: dict) -> list[list[str]]:
    adj = defaultdict(list)
    for e in g["edges"]:
        adj[e["source"]].append(e["target"])
    nodes = [v["id"] for v in g["merged_variables"]]
    cyc = set()

    def dfs(s, c, path):
        if len(path) > 5:
            return
        for n in adj.get(c, []):
            if n == s and len(path) >= 2:
                cyc.add(tuple(path if path[0] == min(path) else path))
            elif n not in path:
                dfs(s, n, path + [n])
    for n in nodes:
        dfs(n, n, [n])
    # dedup by frozenset
    seen, out = set(), []
    for c in cyc:
        k = frozenset(c)
        if k not in seen:
            seen.add(k); out.append(list(c))
    return out


def _graph_ctx(g: dict) -> str:
    byid = {v["id"]: v for v in g["merged_variables"]}
    L = ["VARIABLES (nodes):"]
    for v in g["merged_variables"]:
        L.append(f"  [{v['id']}] {v['name']} — now: {(v.get('current_reading','') or '')[:150]} | traj: {(v.get('trajectory','') or '')[:90]}")
    L.append("\nEDGES (mostly static dependencies — add the missing feedback):")
    for e in g["edges"]:
        L.append(f"  {e['source']} -{e['kind']}-> {e['target']}: {(e.get('note','') or '')[:80]}")
    cyc = _seed_cycles(g)
    L.append(f"\nCANDIDATE CYCLES already in the edges ({len(cyc)}) — validate/expand these:")
    for c in cyc:
        L.append("  " + " → ".join(c))
    return "\n".join(L)


def _gauges_ctx(t: dict) -> str:
    L = ["GAUGES (per-gate measured trends = current loop strengths):"]
    for g in t.get("gates", []):
        L.append(f"\n[{g['gate_id']}] {g['gate_name']} — {g.get('hard_or_soft','')}")
        for m in g.get("metrics", []):
            L.append(f"    {m['metric'][:64]}: {m['direction']} ({m.get('rate','')[:70]})")
        L.append(f"    forward: {(g.get('forward_call','') or '')[:150]}")
    if t.get("synthesis"):
        L.append("\nCROSS-GATE READ: " + (t["synthesis"].get("cross_gate_read", "") or "")[:500])
    return "\n".join(L)


# ------------------------------------------------------------------ render
def _md(ls: LoopSet, sm: SystemsMap) -> str:
    L = ["# Systems map — the dynamical model (loops, couplings, branches)\n",
         "*Replaces the spine with a dynamical read: feedback loops (tagged empirical/theoretical, "
         "with a live gauge + flip-threshold), the coupling map that makes them one system, and the "
         "discrete branches + theory-templates that give the decade its shape. Prediction = shape + "
         "weighted branches + watch-thresholds, not point forecasts.*\n",
         f"**Dominant dynamic:** {sm.dominant_dynamic}\n",
         f"**System-dominant loop:** {sm.system_dominant_loop}\n",
         f"**Current phase:** {sm.current_phase}\n",
         f"**Macro trajectory (the shape):** {sm.macro_trajectory}\n",
         f"**Synthesis:** {sm.synthesis}\n",
         "## Branches (discrete regimes)"]
    for b in sorted(sm.branches, key=lambda x: -x.probability):
        tag = " · *theoretical/quarantined*" if b.is_theoretical else ""
        L.append(f"\n### {b.id} · {b.name} — **P={b.probability:g}**{tag}")
        L.append(f"- trigger: {b.trigger_threshold}")
        L.append(f"- active loops: {', '.join(b.active_loops)}")
        L.append(f"- shape: {b.shape}")
        L.append(f"- who wins: {b.who_wins}")
    L.append("\n## Theory-templates (nested by time-scale)")
    for tp in sorted(sm.theory_templates, key=lambda x: {"slow": 0, "medium": 1, "fast": 2}.get(x.time_scale, 3)):
        L.append(f"\n### {tp.name} _({tp.time_scale})_")
        L.append(f"- governs: {tp.governs}")
        L.append(f"- phases: {' → '.join(tp.phases)}")
        L.append(f"- where we are: {tp.where_we_are}")
        L.append(f"- base-rate shape: {tp.base_rate_shape}")
    L.append("\n## Driving forces")
    for f in sm.driving_forces:
        L.append(f"- **{f.name}** ({', '.join(f.loops)}): {f.reading}")
    L.append("\n## Feedback loops")
    for lp in ls.loops:
        cond = f" · conditional: {lp.activation_condition}" if lp.activation == "conditional" else ""
        reg = f" · regime:{lp.regime}" if lp.regime != "n/a" else ""
        L.append(f"\n### {lp.id} · {lp.name} — {lp.kind} _[{lp.basis}{cond}{reg}]_ {'⭐dominant' if lp.dominant_now else ''}")
        L.append(f"- chain: " + " ".join(f"{e.from_var}{'→' if e.polarity=='+' else '⊣'}{e.to_var}" for e in lp.edges))
        L.append(f"- mechanism: {lp.mechanism}")
        L.append(f"- gauge: {lp.gauge} → **gain now:** {lp.current_gain}")
        L.append(f"- flip-threshold: {lp.flip_threshold}")
        if lp.reference_class:
            L.append(f"- reference class: {lp.reference_class}")
    L.append("\n## Coupling map (why the loops are one system)")
    for c in sorted(ls.couplings, key=lambda x: -x.probability):
        L.append(f"\n### @{c.shared_node} — {', '.join(c.loops_involved)} _[{c.move_type}]_ (P={c.probability:g})")
        L.append(f"- contest: {c.contest}")
        L.append(f"- dominance: {c.dominance_rule}")
        L.append(f"- for: {c.evidence_for}")
        L.append(f"- against: {c.steelman_against}")
    L.append("\n## Tail ignitions (loop-threshold crossings)")
    for ti in sorted(sm.tail_ignitions, key=lambda x: -x.probability):
        L.append(f"- **{ti.tail}** (P={ti.probability:g}) — {ti.loop} @ {ti.threshold} → {ti.what_it_resets}")
    L.append("\n## Watch-thresholds (what switches the branch)")
    for w in sm.watch_thresholds:
        L.append(f"- {w}")
    return "\n".join(L)


def main() -> int:
    load_dotenv()
    for p in (GRAPH, TRACKER):
        if not os.path.exists(p):
            print(f"missing {p} — run compose / gate tracker first."); return 1
    g = json.load(open(GRAPH, encoding="utf-8"))
    t = json.load(open(TRACKER, encoding="utf-8"))
    llm = LLM()

    # ---- call 1: loops + couplings ----
    print("call 1/2 — extracting feedback loops + coupling map (deep model, a few min)…")
    user1 = ("\n\n".join([_graph_ctx(g), _gauges_ctx(t)]) +
             "\n\nExtract the feedback loops (add the missing feedback edges; tag basis + activation; "
             "attach a gauge + current_gain + dominant_now + flip_threshold to each) and the coupling map "
             "(shared nodes, dominance rule, the five-move checklist, adversarial verdict).")
    ls = llm.parse(system=SYSTEM_LOOPS, user=user1, schema=LoopSet,
                   max_turns=12, call_timeout=1200, max_buffer_size=32 * 1024 * 1024)
    json.dump({"loops": [l.model_dump() for l in ls.loops],
               "couplings": [c.model_dump() for c in ls.couplings]},
              open(OUT_JSON, "w", encoding="utf-8"), indent=2)
    print(f"  ✓ {len(ls.loops)} loops ({sum(1 for l in ls.loops if l.kind=='reinforcing')}R/"
          f"{sum(1 for l in ls.loops if l.kind=='balancing')}B), {len(ls.couplings)} couplings")

    # ---- call 2: branches + templates + synthesis ----
    print("call 2/2 — branches, theory-templates, macro trajectory…")
    loops_blob = json.dumps({"loops": [l.model_dump() for l in ls.loops],
                             "couplings": [c.model_dump() for c in ls.couplings]}, indent=1)
    user2 = ("WORLDVIEW: " + (g.get("worldview", "") or "") + "\n\nLOOPS + COUPLINGS:\n" + loops_blob +
             "\n\nAssemble the theory-templates (nested by time-scale), driving forces, the four discrete "
             "branches (A base / B RSI-takeoff theoretical-quarantined / C bust / D fracture), the dominant "
             "dynamic, current phase, macro trajectory (a shape, no dated points), watch-thresholds, tail "
             "ignitions as loop-threshold crossings, and the synthesis.")
    sm = llm.parse(system=SYSTEM_SYNTH, user=user2, schema=SystemsMap,
                   max_turns=8, call_timeout=1200, max_buffer_size=32 * 1024 * 1024)

    out = {"loops": [l.model_dump() for l in ls.loops],
           "couplings": [c.model_dump() for c in ls.couplings],
           **sm.model_dump()}
    json.dump(out, open(OUT_JSON, "w", encoding="utf-8"), indent=2)
    open(OUT_MD, "w", encoding="utf-8").write(_md(ls, sm))
    print(f"  ✓ {len(sm.branches)} branches, {len(sm.theory_templates)} templates, {len(sm.tail_ignitions)} tails")
    print(f"  dominant dynamic: {sm.dominant_dynamic[:90]}")
    print(f"  wrote {OUT_JSON} and {OUT_MD}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
