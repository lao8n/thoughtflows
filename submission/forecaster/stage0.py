"""Stage 0 — topic generation (the bench). See docs/architecture.md §14, Stage 0.

Diverge → compress, same grounded-termination shape as explore. Writes frozen
bench/topics.json (pillars + intersection topics, each with rationale, axis tags,
and the forecasts it should spawn) + bench/rejected.json (cut candidates + why).

Two altitudes: PILLARS (run pure, single-axis) and INTERSECTION topics (run
crossed — where the interaction cruxes that anchor the framework come from).
"""
from __future__ import annotations

import json
import os
from typing import Literal

from pydantic import BaseModel, Field

from .explore import _saturated  # reuse the saturation judge
from .llm import DEEP_MODEL, FAST_MODEL, LLM

# --------------------------------------------------------------- schemas


class CandidateTopic(BaseModel):
    title: str = Field(description="The topic as a crisp question/area, e.g. 'Do AI scaling returns hold through 2030?'")
    kind: Literal["pillar", "intersection"]
    one_line: str = Field(description="One line on what it covers.")
    relevant: bool = Field(description="True only if it bears on modern mercantilism, AI, or their intersection.")
    is_novel: bool = Field(description="False if it just rephrases an existing candidate.")


class CandidateBatch(BaseModel):
    topics: list[CandidateTopic]


class BenchTopic(BaseModel):
    id: str = Field(description="Short id, T1, T2, ...")
    title: str
    kind: Literal["pillar", "intersection"]
    run_mode: Literal["pure", "crossed"] = Field(description="pillar→pure (single-axis), intersection→crossed.")
    rationale: str = Field(description="Why it earns a slot, and why it's BROAD enough to sustain a deep explore.")
    axis_tags: list[str] = Field(description="Which mercantilist lever / AI layer / actor it covers.")
    sub_questions: list[str] = Field(description="Narrower questions/cruxes that live INSIDE this broad topic (incl. any narrow predictions folded up into it).")
    candidate_forecasts: list[str] = Field(description="2-4 ILLUSTRATIVE example forecasts this area could spawn — not commitments; final forecasts are chosen later at Stage 7.")


class RejectedTopic(BaseModel):
    title: str
    reason: str = Field(description="Out-of-scope / immaterial / duplicative — one line.")


class Selection(BaseModel):
    bench: list[BenchTopic]
    rejected: list[RejectedTopic]


class MissingTopic(BaseModel):
    title: str
    kind: Literal["pillar", "intersection"]
    why_material: str = Field(description="How its answer could change the overall forecast.")
    lens: str = Field(default="", description="Filled in by the caller.")


class LensFinding(BaseModel):
    missing: list[MissingTopic]


# --------------------------------------------------------------- prompts

BRIEF = "modern mercantilism, AI, or their intersection (the Bridgewater 'Forecasting the Future 2026' brief)"

AXIS = (
    "AXIS SCAFFOLD for coverage — mercantilist lever {tariffs/trade barriers, export controls, "
    "industrial policy/subsidies, resource & supply-chain nationalism, financial/currency statecraft, "
    "standards/regulation-as-protectionism} × AI value-chain layer {compute/chips, energy/datacenters, "
    "frontier models, data, applications/agents, distribution/devices} × actor {US, China, EU, Gulf, "
    "India, hyperscalers, frontier labs}."
)

DIVERGE_SYSTEM = f"""You generate candidate BROAD TOPICS for a forecasting bench on {BRIEF}.

WHAT A TOPIC IS (critical): a topic is a BROAD question-AREA — rich enough to sustain a deep multi-lens exploration and to spawn SEVERAL distinct binary forecasts. It is NOT a single prediction.
- GOOD (broad area): "How does the AI + industrial-policy spending wave resolve macro-financially?" (spawns forecasts on rates, debt, currency, productivity, bust).
- BAD (too narrow — that's a forecast, not a topic): "Will China blockade Taiwan before 2028?" → that belongs as a CRUX inside a broader "US–China tech decoupling" topic.
Pitch every candidate at the broad-area level. If a candidate is really one yes/no prediction, broaden it to the area it lives in (or it's a sub-question of an existing topic, not its own topic).

Two altitudes:
- PILLARS (run pure, single-axis): deep pure-AI topics (e.g. do scaling returns hold, model-layer commoditization, agentic diffusion) and pure-mercantilism topics (new industrial-policy era, weaponized interdependence, resource nationalism). LLM market structure is a pillar. The macro-financial loop (AI capex + industrial-policy spending → rates, debt, currency debasement) is a required pillar.
- INTERSECTION topics (run crossed): explicit AI×mercantilism mechanisms (e.g. does export-control compute scarcity decide who wins the model layer).

{AXIS}

Each round, surface candidate topics NOT already listed — both pillars and intersections, hunting coverage gaps on the scaffold.
RELEVANCE GATE: keep only topics bearing on the brief (set `relevant` honestly). A materially-relevant adjacency (e.g. 'quantum as a compute-chokepoint obsolescence risk') is in-scope; standalone biotech/climate is not.
Mark `is_novel` false for rephrasings of existing candidates. Return an EMPTY list when nothing genuinely new and relevant remains."""

SELECT_SYSTEM = """You curate the FINAL bench of BROAD topics for a modern-mercantilism × AI submission.

Select about {target} BROAD topics, balanced roughly 40% PILLARS (run pure) / 60% INTERSECTION topics (run crossed).

GRANULARITY (critical): every topic must be a BROAD question-AREA — rich enough to (a) sustain a deep multi-lens explore and (b) spawn ≥3 distinct forecasts. A candidate that is really ONE binary prediction is NOT a topic: MERGE it UP into the broad area it belongs to and list it as a sub_question (e.g. "will China blockade Taiwan before 2028?" → a sub_question inside a "US–China tech decoupling" topic, not its own slot). Fewer, broader, richer beats many narrow ones.
Coverage discipline: across the bench, the major mercantilist levers, AI value-chain layers, and actors should each be represented or consciously skipped. Depth within Mercantilism×AI, NOT breadth across all futures.

For each: id (T1..), title (BROAD), kind, run_mode (pillar→pure, intersection→crossed), rationale, axis_tags, sub_questions (the narrower cruxes/predictions folded in), and 2-4 ILLUSTRATIVE candidate_forecasts (not commitments).

REQUIRED: include the macro-financial loop (AI capex + industrial-policy spending → rates, debt, currency debasement, external balance) as ONE BROAD PILLAR — do not split it into separate rate/debt/currency topics; the competing scenarios (debt-stress vs productivity-offset vs deflationary-bust) are sub_questions within it.

Everything excluded as out-of-scope / immaterial / duplicative goes in `rejected` with a one-line reason (the breadth-of-consideration record for the roundtable)."""


# Multi-lens completeness critic (architecture §14 Stage 0). The lever×layer grid is just one lens
# and 36 cells is arbitrary; real completeness = surviving several ORTHOGONAL "what's missing?" lenses.
COMPLETENESS_LENSES = [
    ("macro / economic", "fiscal, monetary, rates, debt-sustainability, currency-debasement, or balance-of-payments feedback from the AI + industrial-policy spending wave"),
    ("actor", "an actor missing or under-weighted — EU/Brussels-effect, India, Russia, the open-source commons, standards bodies"),
    ("mechanism", "a battleground mechanism missing — talent/immigration, data/IP/copyright, standards & protocol capture, payment rails / financial statecraft, energy"),
    ("tail scenario", "a discontinuity that would dominate the gradualist forecasts — AI-safety incident → regulatory clampdown, a capability jump, a kinetic shock"),
    ("discipline", "what a political scientist, economic historian, or AI-safety researcher would say the bench ignores"),
]

CRITIC_SYSTEM = """You are an adversarial completeness critic for a forecasting bench on modern mercantilism × AI.

You are given the current bench and ONE lens. Through that lens ONLY, find BROAD topic-areas MISSING from the bench whose answer could materially CHANGE the overall forecast or framework. Be specific and material — not "consider X more." Pitch each as a broad area, not a single prediction. Mark each pillar (pure single-axis) or intersection (AI×mercantilism mechanism). If the bench already covers this lens well, return an EMPTY list — the honest, valued answer."""

MERGE_SYSTEM = """You finalize a forecasting bench. You have the current bench plus candidate gaps proposed by completeness lenses.

Incorporate the genuinely MATERIAL gaps (add as full BROAD topics with rationale, axis_tags, sub_questions, candidate_forecasts), MERGE near-duplicates and fold any narrow single-prediction gaps UP as sub_questions of a broad topic, and keep the bench at about {target} BROAD topics, balanced ~40% pillars / 60% intersections. Every topic must be broad enough to sustain a deep explore and spawn ≥3 forecasts. The macro-financial loop MUST be present as ONE broad pillar. Everything you decline to add goes in `rejected` with a one-line reason."""


def _diverge_user(cands: list[CandidateTopic], per_round: int, gap: str | None) -> str:
    lines = [f"BRIEF: {BRIEF}", ""]
    if cands:
        lines.append(f"EXISTING CANDIDATES ({len(cands)}) — do not repeat:")
        for c in cands:
            lines.append(f"  - [{c.kind}] {c.title}")
        lines.append("")
    else:
        lines.append("No candidates yet. Lay down the pillars first, then intersection topics across the scaffold.")
    if gap:
        lines.append(f"\nPRIORITISE this under-covered area this round: {gap}\n")
    lines.append(f"Return up to {per_round} new, novel, relevant candidate topics. Empty list when the space is covered.")
    return "\n".join(lines)


# --------------------------------------------------------------- stage


def generate_bench(target: int = 10, outdir: str = "bench", max_rounds: int = 6,
                   per_round: int = 12, model: str | None = None) -> Selection:
    llm_fast = LLM(model or FAST_MODEL)
    llm_deep = LLM(DEEP_MODEL)
    os.makedirs(outdir, exist_ok=True)

    print(f"Stage 0 — topic generation (target ~{target})")
    print(f"Diverge: {llm_fast.model or '(CLI default)'}  |  select/judge: {llm_deep.model or '(CLI default)'}\n")

    # 0a Diverge — accumulate relevant candidates until saturated.
    print("0a — diverge")
    cands: list[CandidateTopic] = []
    seen: set[str] = set()
    gap: str | None = None
    for rnd in range(1, max_rounds + 1):
        batch = llm_fast.parse(system=DIVERGE_SYSTEM, user=_diverge_user(cands, per_round, gap), schema=CandidateBatch)
        added = 0
        for t in batch.topics:
            key = t.title.strip().lower()
            if t.relevant and t.is_novel and key not in seen:
                seen.add(key)
                cands.append(t)
                added += 1
        print(f"  round {rnd}: +{added} (proposed {len(batch.topics)})  total {len(cands)}")
        if added == 0:
            print("  stop — dry round")
            break
        if rnd >= 2:
            sat = _saturated(llm_deep, BRIEF, "Candidate topics", [c.title for c in cands])
            if sat.saturated:
                print(f"  stop — saturated: {sat.reason}")
                break
            gap = sat.major_gap or None
            if gap:
                print(f"    -> next round targets gap: {gap}")

    # 0b Compress — select the bench + completeness critic + rejection log.
    print("\n0b — compress + select")
    listing = "\n".join(f"  - [{c.kind}] {c.title} — {c.one_line}" for c in cands)
    user = (
        f"BRIEF: {BRIEF}\n\nCANDIDATE TOPICS ({len(cands)}):\n{listing}\n\n"
        f"Curate the final bench of about {target} topics."
    )
    sel = llm_deep.parse(system=SELECT_SYSTEM.format(target=target), user=user, schema=Selection)

    # 0c Multi-lens completeness critic — orthogonal "what's missing?" attacks (not grid-coverage).
    print("\n0c — multi-lens completeness critic")
    bench_listing = "\n".join(f"  - [{t.kind}] {t.title}" for t in sel.bench)
    missing: list[MissingTopic] = []
    for name, desc in COMPLETENESS_LENSES:
        res = llm_deep.parse(
            system=CRITIC_SYSTEM,
            user=f"CURRENT BENCH:\n{bench_listing}\n\nLENS: {name} — {desc}\n\nWhat MATERIAL topic is missing under THIS lens? Empty list if none.",
            schema=LensFinding,
        )
        for m in res.missing:
            m.lens = name
        missing.extend(res.missing)
        print(f"  {name}: +{len(res.missing)}" + (f"  ({'; '.join(m.title for m in res.missing)})" if res.missing else ""))

    # 0d Finalize — fold material gaps in, merge dupes, keep ~target, log the rest.
    print("\n0d — finalize")
    miss_listing = "\n".join(f"  - ({m.lens}) [{m.kind}] {m.title} — {m.why_material}" for m in missing) or "  (none)"
    final = llm_deep.parse(
        system=MERGE_SYSTEM.format(target=target),
        user=(f"CURRENT BENCH:\n" + "\n".join(f"  - [{t.kind}] {t.title} — {t.rationale}" for t in sel.bench)
              + f"\n\nCANDIDATE GAPS FROM LENSES:\n{miss_listing}\n\nProduce the final bench (~{target}) and rejection log."),
        schema=Selection,
    )
    # Keep the original select rejections too (append-only breadth-of-consideration record).
    final.rejected = sel.rejected + final.rejected
    sel = final

    json.dump({"brief": BRIEF, "topics": [t.model_dump() for t in sel.bench]},
              open(os.path.join(outdir, "topics.json"), "w", encoding="utf-8"), indent=2)
    json.dump({"rejected": [r.model_dump() for r in sel.rejected]},
              open(os.path.join(outdir, "rejected.json"), "w", encoding="utf-8"), indent=2)

    os.makedirs("docs", exist_ok=True)
    open("docs/latest-bench.md", "w", encoding="utf-8").write(_render(sel))

    pillars = [t for t in sel.bench if t.kind == "pillar"]
    inter = [t for t in sel.bench if t.kind == "intersection"]
    print(f"\nDone. Bench: {len(sel.bench)} topics ({len(pillars)} pillars, {len(inter)} intersections); {len(sel.rejected)} rejected.")
    print("Written: bench/topics.json, bench/rejected.json  and  docs/latest-bench.md")
    return sel


def _render(sel: Selection) -> str:
    L = ["# Bench — topic generation (Stage 0)\n"]
    for kind, label in (("pillar", "Pillars (run pure)"), ("intersection", "Intersection topics (run crossed)")):
        items = [t for t in sel.bench if t.kind == kind]
        L.append(f"## {label} ({len(items)})\n")
        for t in items:
            L.append(f"### {t.id}. {t.title}")
            L.append(f"_{t.rationale}_")
            L.append(f"- axis: {', '.join(t.axis_tags)}")
            if t.sub_questions:
                L.append("- sub-questions / cruxes inside this topic:")
                for sq in t.sub_questions:
                    L.append(f"  - {sq}")
            L.append("- illustrative forecasts it could spawn:")
            for f in t.candidate_forecasts:
                L.append(f"  - {f}")
            L.append("")
    L.append(f"## Rejected ({len(sel.rejected)}) — breadth of consideration\n")
    for r in sel.rejected:
        L.append(f"- **{r.title}** — {r.reason}")
    L.append("")
    return "\n".join(L) + "\n"
