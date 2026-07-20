#!/usr/bin/env python3
"""Dominance stage — turns "a symmetric lattice of chokepoints" into a committed, sequenced call.

Applies Liebig's law of the minimum + the general-purpose-technology (GPT) reference class:
- RANK each binding constraint by how decisive it is for VALUE capture (slowness-to-replicate
  x non-substitutability x share-of-value-chain-gated) — NOT how hard it is in isolation.
- SEQUENCE the binding constraint across the decade (it migrates: chips/lithography bind the
  early training race; power binds the deployment phase where value lands).
- Ground in MANY historical examples — every prior GPT buildout (electrification, internet,
  railroads, telegraph) shows the invention commoditizes and the physical-distribution +
  aggregation layer captures the rent.
- Answer the specific question: has the country with the MOST ENERGY won before? (coal→Britain,
  oil→US, vs the resource curse and energy-rich-but-lost cases) — energy as necessary-plus-
  absorptive-capacity, and what that implies for China's power abundance.

Web-grounded. Reads gate-tracker + systems-map + composed graph + corpus. Feeds Part 2's
conclusion (run_part2.py can ingest bench/dominance.json).

    python run_dominance.py
"""
from __future__ import annotations
import glob, json, os, re, sys
from typing import Literal
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from forecaster.llm import LLM

SYSMAP, TRACKER, GRAPH = "bench/systems-map.json", "bench/gate-tracker.json", "bench/composed-graph.json"
OUT_JSON, OUT_MD = "bench/dominance.json", "docs/dominance-analysis.md"


class ConstraintRank(BaseModel):
    constraint: str = Field(description="Plain-language binding input: power/electrons, EUV/lithography, leading-edge chips, critical minerals, talent, capital, demand/ROI, etc.")
    slowness_to_replicate: str = Field(description="How slow to replicate/clear + why (the decisive Liebig property).")
    non_substitutability: str = Field(description="How hard to route around / substitute.")
    value_share_gated: str = Field(description="Which layer of the value chain it gates, and how much of the DURABLE value that layer captures (training frontier commoditizes; deployment/distribution captures value).")
    binds_in_phase: Literal["early_training_frenzy", "late_deployment", "throughout", "tail_only"]
    decisiveness_for_value: float = Field(description="0-1: how decisive this constraint is for who captures the DECADE'S VALUE (not how hard it is in isolation).")
    historical_precedent: str = Field(description="The GPT-buildout analog for THIS constraint (e.g. power ↔ electrification's grid/utilities; models ↔ commoditized TCP/IP).")
    dominance_precedents: str = Field(description="Historical cases of whether CONTROLLING THIS SPECIFIC INPUT conferred DURABLE dominance — and its durability class: durable (energy+absorptive capacity, standards lock-in), caught-up (manufacturing/DRAM), or routed-around-by-tech (materials chokepoints broken by substitution/synthesis).")


class GPTPrecedent(BaseModel):
    case: str = Field(description="e.g. 'Electrification, 1880-1930'.")
    the_invention_that_commoditized: str = Field(description="The clever core that leaked/commoditized (dynamo, TCP/IP, locomotive).")
    where_value_actually_landed: str = Field(description="The physical-distribution + aggregation layer that captured durable rent (grid/utilities, fiber+hyperscalers, right-of-way).")
    lesson_for_ai: str


class DominanceCase(BaseModel):
    constraint: str = Field(description="Which constraint this case bears on: energy, manufacturing capacity, precision tooling / gate-behind-the-gate, materials/minerals, talent/knowledge, capital/finance, or standards/networks.")
    case: str = Field(description="The historical episode, e.g. 'Coal & British industrial primacy 1780-1850'; 'WWII arsenal-of-democracy manufacturing'; 'Chile nitrate monopoly broken by Haber-Bosch'; 'Operation Paperclip'; 'Amsterdam→London→NY financial primacy'; 'Wintel / dollar standards lock-in'.")
    did_control_confer_dominance: Literal["yes", "no", "partial"]
    durability: Literal["durable", "caught_up", "routed_around_by_tech"] = Field(description="Was the advantage lasting, competed away by imitation, or dissolved by a substituting technology?")
    why: str = Field(description="The decisive mechanism, incl. any coupling requirement (e.g. energy needs absorptive capacity; materials chokepoints invite synthesis).")


class DominanceAnalysis(BaseModel):
    method: str = Field(description="The Liebig + GPT-reference-class ranking method, stated.")
    ranked_constraints: list[ConstraintRank] = Field(description="All binding constraints, ranked MOST→LEAST decisive for value capture.")
    binding_sequence: list[dict] = Field(description="The Liebig migration across the decade: a list of {phase, binding_constraint, why, who_it_favors} — how the binding minimum moves over time (e.g. lithography/chips early → power late).")
    gpt_precedents: list[GPTPrecedent] = Field(description="3-5 general-purpose-technology buildouts showing the invention commoditizes and distribution+aggregation captures value.")
    dominance_cases: list[DominanceCase] = Field(description="For EACH constraint, historical cases of whether CONTROLLING it conferred DURABLE dominance — as many as possible, spanning energy, manufacturing, tooling, materials, talent, capital, standards; both confirming and dis-confirming, each tagged with its durability class.")
    durability_ladder: str = Field(description="Rank the constraint TYPES by historical durability of the advantage they confer: which controls have been LASTING (energy+absorptive capacity, standards/network lock-in), which get CAUGHT UP (manufacturing capacity), and which get ROUTED AROUND by technology (materials/minerals via substitution/synthesis, software moats via diffusion).")
    energy_verdict: str = Field(description="The specific answer on energy: does the energy-rich power win? (energy = necessary + absorptive-capacity), and what it implies for China's power abundance vs the US chip lead — as one case within the broader ladder.")
    decisive_chokepoint: str = Field(description="THE committed answer: which constraint dominates, and WHEN — the sequenced call, not a tie.")
    who_wins_the_value: str = Field(description="The directional conclusion: who captures the decade's durable value, given the ranking + sequence.")
    inverts_if: str = Field(description="The condition that flips it (RSI takeoff re-elevates chips/training).")
    voided_if: str = Field(description="The condition under which no chokepoint wins (demand/ROI bust — capital just burns).")
    committed_conclusion: str = Field(description="The bold synthesis that REPLACES 'a symmetric lattice' — a sequenced, directional bet.")


SYSTEM = """You convert a set of AI-race chokepoints into a COMMITTED, SEQUENCED conclusion using Liebig's law of the minimum + the general-purpose-technology (GPT) reference class. The framework so far says 'a symmetric lattice, depends on relative clearing speed' — that is a shrug. Your job is to rank and sequence, and land a directional call. You have WebSearch — use it heavily for historical cases and to verify figures.

Principles:
1. LIEBIG: at any moment one input is the binding minimum; as it clears, dominance migrates to the next. So do not ask 'which chokepoint wins' — establish the SEQUENCE of binding constraints across the decade, and identify which one binds during the phase that CAPTURES THE VALUE.
2. GPT REFERENCE CLASS: in every prior general-purpose-technology buildout the clever core commoditized and durable rent accrued to the physical-distribution + aggregation layer — electrification (dynamo commoditized; grid/utilities + GE/Westinghouse won), internet (TCP/IP free; fiber + hyperscalers won), railroads (locomotive vs right-of-way/network), telegraph. Rank each AI constraint on slowness-to-replicate x non-substitutability x share-of-durable-value-gated — NOT on how hard it is in isolation. (EUV is the hardest gate but gates the training frontier, which commoditizes — hardest gate, least-valuable prize. Power is universal, slowest, and gates deployment where value lands.)
3. THE DOMINANCE TEST — for EVERY constraint, not just energy: has controlling THIS input let a power win before, and was the advantage DURABLE? Pull as many cases as you can across all of them, and classify each as durable / caught-up / routed-around-by-tech:
   - ENERGY: coal→British primacy, oil→US hegemony, Japan's 1941 oil embargo, hydro→US aluminium; disconfirming: petrostate resource curse, Soviet energy abundance that still lost. Lesson: energy wins ONLY coupled with absorptive capacity — pure abundance is the resource curse. (China is power-abundant AND absorptive, unlike petrostates; the US is absorptive but power-constrained.)
   - MANUFACTURING CAPACITY (chips analog): WWII 'arsenal of democracy'; but DRAM leadership went Japan→Korea→China — manufacturing gets CAUGHT UP.
   - PRECISION TOOLING / gate-behind-the-gate (EUV analog): machine-tool leadership shifting Britain→Germany→US→Japan; Zeiss/ASML optics — tooling monopolies are rare but DURABLE when they exist.
   - MATERIALS / MINERALS: Chile's nitrate monopoly broken by Haber-Bosch; natural rubber → synthetic; these chokepoints get ROUTED AROUND by synthesis/substitution. Materials = a tax, not a wall.
   - TALENT / KNOWLEDGE: Operation Paperclip, Manhattan Project concentration; but Fuchs/espionage and émigré flows show knowledge DIFFUSES.
   - CAPITAL / FINANCE: Amsterdam→London→New York financial primacy, reserve-currency status — enabling and fairly durable, but not sufficient alone.
   - STANDARDS / NETWORKS: railway gauge, Wintel, the dollar/SWIFT — network lock-in is among the MOST durable advantages.
Then build the DURABILITY LADDER: which control types last, which get caught up, which get routed around — and map our AI chokepoints onto it.

Land: the ranked constraints, the binding sequence over time, the GPT precedents, the per-constraint dominance cases + the durability ladder + the energy verdict, and a committed 'which dominates and when / who wins the value' conclusion — with the honest inverts-if (RSI takeoff) and voided-if (demand bust). Be bold; the point is to conclude, not to survey."""


def _corpus(k: int = 20) -> str:
    pat = re.compile(r"(energy|power|electrif|coal|\boil\b|grid|hydro|resource curse|industrial revolution|"
                     r"dynamo|standard oil|opec|hegemon|dominance|railroad|railway|telegraph|distribution|"
                     r"absorptive|Liebig|binding constraint|deployment|electricity)", re.I)
    seen, hits = set(), []
    for f in glob.glob("runs/*/notes*.json"):
        tid = os.path.basename(os.path.dirname(f)).split("-")[0]
        d = json.load(open(f, encoding="utf-8"))
        for n in d.get("notes", []):
            t = f"{n.get('claim','')} {n.get('detail','')}"
            if pat.search(t):
                key = n.get("claim", "")[:56]
                if key not in seen:
                    seen.add(key); hits.append(f"  [note] {n.get('claim','')[:110]}")
        for a in d.get("analogies", []):
            if pat.search(a.get("case", "") + a.get("what_happened", "")):
                key = a["case"][:44]
                if key not in seen:
                    seen.add(key); hits.append(f"  [analogy] {a['case']}: {(a.get('what_happened','') or '')[:120]}")
    return "RELEVANT CORPUS (energy / GPT-buildout / dominance):\n" + "\n".join(hits[:k])


def _gates(gt) -> str:
    L = ["THE CHOKEPOINTS TO RANK (from the gate tracker):"]
    for x in gt.get("gates", []):
        L.append(f"  {x['gate_name']} [{x.get('hard_or_soft')}]: {(x.get('forward_call','') or '')[:150]}")
    return "\n".join(L)


def _md(r: DominanceAnalysis) -> str:
    L = ["# Dominance analysis — which chokepoint wins, and when (Liebig + the GPT reference class)\n",
         f"**Decisive chokepoint:** {r.decisive_chokepoint}\n",
         f"**Who wins the value:** {r.who_wins_the_value}\n",
         f"**Committed conclusion:** {r.committed_conclusion}\n",
         f"**Inverts if:** {r.inverts_if}",
         f"**Voided if:** {r.voided_if}\n",
         f"*Method: {r.method}*\n",
         "## Binding-constraint sequence (the Liebig migration)"]
    for s in r.binding_sequence:
        L.append(f"- **{s.get('phase','')}** → binds: **{s.get('binding_constraint','')}** — {s.get('why','')} _(favors {s.get('who_it_favors','')})_")
    L.append("\n## Constraints ranked by decisiveness for VALUE capture")
    for c in sorted(r.ranked_constraints, key=lambda x: -x.decisiveness_for_value):
        L.append(f"\n### {c.constraint} — {c.decisiveness_for_value:g} · binds {c.binds_in_phase}")
        L.append(f"- slow to replicate: {c.slowness_to_replicate}")
        L.append(f"- non-substitutable: {c.non_substitutability}")
        L.append(f"- value gated: {c.value_share_gated}")
        L.append(f"- GPT precedent: {c.historical_precedent}")
        L.append(f"- dominance history: {c.dominance_precedents}")
    L.append("\n## Durability ladder — which controls last, which erode")
    L.append(f"{r.durability_ladder}\n")
    L.append("## Has controlling each constraint won before? (by constraint)")
    bycon: dict[str, list] = {}
    for e in r.dominance_cases:
        bycon.setdefault(e.constraint, []).append(e)
    for con, cases in bycon.items():
        L.append(f"\n**{con}**")
        for e in cases:
            L.append(f"- {e.case} — control→dominance? **{e.did_control_confer_dominance}** · *{e.durability}*. {e.why}")
    L.append(f"\n*Energy verdict:* {r.energy_verdict}")
    L.append("\n## GPT-buildout precedents (invention commoditizes; distribution+aggregation wins)")
    for g in r.gpt_precedents:
        L.append(f"- **{g.case}**: {g.the_invention_that_commoditized} commoditized → value landed at {g.where_value_actually_landed}. *AI lesson:* {g.lesson_for_ai}")
    return "\n".join(L)


def main() -> int:
    load_dotenv()
    if not os.path.exists(TRACKER):
        print(f"missing {TRACKER} — run the gate tracker first."); return 1
    gt = json.load(open(TRACKER, encoding="utf-8"))
    sm = json.load(open(SYSMAP, encoding="utf-8")) if os.path.exists(SYSMAP) else {}
    user = ("\n\n".join([_gates(gt), _corpus(),
                         f"SYSTEMS-MAP dominant dynamic: {sm.get('dominant_dynamic','')[:300]}",
                         f"macro trajectory: {sm.get('macro_trajectory','')[:300]}"]) +
            "\n\nRank the constraints by decisiveness for VALUE capture, sequence the binding minimum across the "
            "decade, and for EVERY constraint pull historical cases of whether controlling it conferred DURABLE "
            "dominance (energy: coal/Britain, oil/US, resource curse; manufacturing: arsenal-of-democracy, DRAM "
            "caught-up; tooling: machine-tools, Zeiss/ASML; materials: nitrates→Haber-Bosch routed-around; talent: "
            "Paperclip/Manhattan/espionage; capital: Amsterdam→London→NY; standards: Wintel/dollar). Build the "
            "durability ladder and land a committed 'which dominates and when / who wins the value' conclusion "
            "with the inverts-if and voided-if.")
    llm = LLM()
    print("dominance ranking (web-grounded, a few min)…")
    r = llm.parse(system=SYSTEM, user=user, schema=DominanceAnalysis,
                  tools=["WebSearch"], max_turns=22, call_timeout=1200, max_buffer_size=32 * 1024 * 1024)
    json.dump(r.model_dump(), open(OUT_JSON, "w", encoding="utf-8"), indent=2)
    open(OUT_MD, "w", encoding="utf-8").write(_md(r))
    print(f"✓ decisive chokepoint: {r.decisive_chokepoint[:100]}")
    print(f"  energy verdict: {r.energy_verdict[:100]}")
    print(f"  {len(r.ranked_constraints)} constraints ranked, {len(r.dominance_cases)} dominance cases, {len(r.gpt_precedents)} GPT precedents")
    if llm.last_cost_usd:
        print(f"  cost: ${llm.last_cost_usd:.2f}")
    print(f"  wrote {OUT_MD} and {OUT_JSON}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
