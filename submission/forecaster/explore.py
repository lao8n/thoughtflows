"""The iterating exploration loop (docs/architecture.md §5).

Keep surfacing new, material notes on a topic until the model stops finding
any — the grounded termination rule. The loop is ordinary control flow; the
model is one step inside it. Convergence is defined by *notes that stop moving*
(K consecutive dry rounds), NOT by the model declaring itself done — an LLM
asked "what's missing?" will always invent something, so we filter hard on
novelty + materiality and stop when nothing new survives.
"""
from __future__ import annotations

import json
import os
from typing import Literal

from pydantic import BaseModel, Field

from .llm import DEEP_MODEL, FAST_MODEL, LLM
from .notes import NOTE_TYPES, NoteStore, TYPE_LABEL, slugify

NoteType = Literal["theory", "hypothesis", "evidence_for", "evidence_against", "crux", "question"]


class CandidateNote(BaseModel):
    type: NoteType
    claim: str = Field(description="A terse key, <= ~15 words — the headline of the note.")
    detail: str = Field(description="One or two sentences expanding the claim.")
    is_novel: bool = Field(description="True only if NOT already covered by the existing notes shown.")
    is_material: bool = Field(description="True only if it would meaningfully move the forecast or analysis.")
    rationale: str = Field(description="One line: why novel and material (or why you included it).")


class NoteBatch(BaseModel):
    notes: list[CandidateNote]


SYSTEM = """You are a rigorous forecasting analyst building a structured note graph for a single topic.

Method (from a council-of-lenses approach):
- Hold many competing theories/lenses in parallel — name them where they have names (e.g. natural-monopoly economics, aggregation theory, Christensen disruption), unnamed basic mechanisms are fine too.
- Decompose into supporting evidence AND, with equal weight, disconfirming evidence. Actively try to refute, not just confirm.
- Surface CRUXES: the specific fact or assumption that, if known, would flip which lens governs.
- Note key open questions / uncertainties.

Your job each round: find what is MISSING from the notes shown to you — a missing lens, an un-stated counter-argument, an unsurfaced crux, a piece of evidence on either side.

Discipline (critical):
- Only return notes that are genuinely NEW (not already covered) AND MATERIAL (would actually move the analysis). Set is_novel / is_material honestly.
- A rephrasing, a narrower SPECIAL CASE, or a sub-variant of an existing note is NOT novel — it is a duplicate. Only a materially DISTINCT idea (different mechanism, different lens, different direction of evidence) counts. When in doubt, mark is_novel false.
- Do NOT pad. If you cannot find anything genuinely new and material, return an EMPTY list. An empty list is the correct, valued answer when the topic is well-covered — never invent filler to look productive.
- Prefer disconfirming evidence and missing lenses; those are the most undervalued and the easiest to overlook."""


class Saturation(BaseModel):
    saturated: bool = Field(description="True if further mining would mostly yield restatements / marginal sub-cases.")
    reason: str = Field(description="One line of justification.")
    major_gap: str = Field(description="If NOT saturated: the single most important under-covered area to mine next. Empty string if saturated.")


SATURATION_SYSTEM = """You judge whether a topic's idea-space is SATURATED for forecasting purposes.

Saturated means the space of *materially-distinct* considerations is covered — so further rounds would mostly produce rephrasings. Concretely, do NOT declare saturation while ANY of these holds:
- one SIDE of the central question is materially under-represented relative to the other (e.g. lots of "it commoditizes" evidence but little "a durable moat forms") — both sides must be roughly balanced;
- a major lens, actor, mechanism, or **value-capture / who-wins-and-where** angle is missing;
- a significant **tail / discontinuity** scenario is absent;
- the analysis has only argued the salient / consensus framing.

When not saturated, name the single biggest gap so the next round can target it.

Balance breadth against bloat: a mature graph is typically ~60-90 materially-distinct notes. Don't chase encyclopedic coverage of minor angles — but don't stop short of the both-sides / value-capture / tail coverage above just to be lean. (An earlier over-tight version stopped a broad topic at 32 notes with an 11-vs-3 disconfirming skew; that is the failure mode to avoid.)"""


class CandidateAnalogy(BaseModel):
    case: str = Field(description="The historical case + era, e.g. 'US railroads, 1860s–1900s'.")
    what_happened: str = Field(description="How that market actually structured / the outcome.")
    similarities: list[str] = Field(description="Concrete ways it resembles the current situation.")
    differences: list[str] = Field(description="Where the analogy BREAKS — be honest, this matters most.")
    implication: str = Field(description="What it suggests for the focal question, given the (dis)analogy.")
    is_novel: bool = Field(description="True only if NOT already among the cases shown.")
    is_material: bool = Field(description="True only if the comparison genuinely informs the forecast.")


class AnalogyBatch(BaseModel):
    analogies: list[CandidateAnalogy]


ANALOGY_SYSTEM = """You are a forecasting analyst mining HISTORICAL ANALOGIES for a topic — Ray Dalio's archetype method: find past cases the current situation resembles, see how they actually played out, and reason about where the analogy holds vs breaks.

Pick genuinely comparable cases across domains, not just the obvious one:
- technology platforms (internet/dot-com build-out, PC/Wintel, browsers, search, cloud, app stores, social)
- capital-intensive industrials (steel, railroads, automobiles, aircraft)
- infrastructure / utilities (electricity, telecom/telephony, canals)
- resource/commodity markets (oil majors, semiconductors/fabs)

For each analogy:
- state what ACTUALLY happened to that market's structure (consolidation? commoditization? regulation? vertical integration?),
- list concrete similarities AND concrete differences — naming where the analogy BREAKS is the most valuable part; a forced analogy is worse than none,
- give the implication for the focal question given both.

Discipline (critical):
- Only return analogies that are genuinely NEW (not already listed) and MATERIAL. Set is_novel / is_material honestly.
- Do NOT pad. If no further genuinely useful analogy remains, return an EMPTY list — that is the correct answer when the space is well-covered."""


def _gap_line(gap: str | None) -> str:
    return (f"\nPRIORITISE this under-covered area this round (target the gap, don't repeat what's covered): {gap}\n"
            if gap else "")


def _build_analogy_user(topic: str, store: NoteStore, per_round: int, gap: str | None = None) -> str:
    lines = [f"FOCAL TOPIC: {topic}", ""]
    cases = store.analogy_cases()
    if cases:
        lines.append(f"EXISTING ANALOGIES ({len(cases)}) — do not repeat; find genuinely different cases:")
        for c in cases:
            lines.append(f"  - {c}")
        lines.append("")
    else:
        lines.append("No analogies yet. Lay down the strongest, most diverse historical comparisons.")
    lines.append(_gap_line(gap))
    lines.append(f"Return up to {per_round} new, novel, material analogies. Empty list if nothing genuine remains.")
    return "\n".join(lines)


def _build_user(topic: str, store: NoteStore, per_round: int, gap: str | None = None) -> str:
    lines = [f"TOPIC: {topic}", ""]
    by_type = store.claims_by_type()
    total = len(store.notes)
    if total == 0:
        lines.append("There are no notes yet. Lay down an initial spread: the main competing lenses, "
                     "the key hypotheses, the strongest evidence on each side, and the central crux.")
    else:
        lines.append(f"EXISTING NOTES ({total}) — do not repeat these; find what they are MISSING:")
        lines.append("")
        for t in NOTE_TYPES:
            claims = by_type.get(t, [])
            if not claims:
                continue
            lines.append(f"{TYPE_LABEL[t]}:")
            for c in claims:
                lines.append(f"  - {c}")
            lines.append("")
    lines.append(_gap_line(gap))
    lines.append(f"Return up to {per_round} new, novel, material notes. Empty list if nothing genuine remains.")
    return "\n".join(lines)


def _saturated(llm_deep: LLM, topic: str, items_label: str, items: list[str]) -> Saturation:
    listing = "\n".join(f"  - {x}" for x in items)
    user = (
        f"FOCAL TOPIC: {topic}\n\n"
        f"{items_label} so far ({len(items)}):\n{listing}\n\n"
        "Is this space saturated for forecasting purposes? If not, name the single biggest gap."
    )
    return llm_deep.parse(system=SATURATION_SYSTEM, user=user, schema=Saturation)


def _notes_loop(llm_fast: LLM, llm_deep: LLM, topic: str, store: NoteStore,
                max_rounds: int, per_round: int, max_notes: int = 70, min_rounds: int = 3) -> None:
    print("Phase 1 — notes")
    gap: str | None = None
    for rnd in range(1, max_rounds + 1):
        batch = llm_fast.parse(system=SYSTEM, user=_build_user(topic, store, per_round, gap), schema=NoteBatch)
        added = 0
        for n in batch.notes:
            if n.is_novel and n.is_material and store.add(type=n.type, claim=n.claim, detail=n.detail, round=rnd):
                added += 1
        store.save()
        print(f"  round {rnd:>2}: +{added:>2} new  (proposed {len(batch.notes)})  total {len(store.notes):>3}")
        if len(store.notes) >= max_notes:
            print(f"  stop — note cap ({max_notes})")
            return
        if added == 0:
            print("  stop — dry round")
            return
        if rnd >= min_rounds:
            sat = _saturated(llm_deep, topic, "Note claims", [n.claim for n in store.notes])
            if sat.saturated:
                print(f"  stop — saturated: {sat.reason}")
                return
            gap = sat.major_gap or None
            if gap:
                print(f"    -> next round targets gap: {gap}")


def _analogy_loop(llm_fast: LLM, llm_deep: LLM, topic: str, store: NoteStore,
                  max_rounds: int, per_round: int, max_analogies: int = 40, min_rounds: int = 3) -> None:
    print("Phase 2 — historical analogies")
    gap: str | None = None
    for rnd in range(1, max_rounds + 1):
        batch = llm_fast.parse(system=ANALOGY_SYSTEM, user=_build_analogy_user(topic, store, per_round, gap), schema=AnalogyBatch)
        added = 0
        for a in batch.analogies:
            if a.is_novel and a.is_material and store.add_analogy(
                case=a.case, what_happened=a.what_happened, similarities=a.similarities,
                differences=a.differences, implication=a.implication, round=rnd,
            ):
                added += 1
        store.save()
        print(f"  round {rnd:>2}: +{added:>2} new  (proposed {len(batch.analogies)})  total {len(store.analogies):>3}")
        if len(store.analogies) >= max_analogies:
            print(f"  stop — analogy cap ({max_analogies})")
            return
        if added == 0:
            print("  stop — dry round")
            return
        if rnd >= min_rounds:
            sat = _saturated(llm_deep, topic, "Analogy cases", store.analogy_cases())
            if sat.saturated:
                print(f"  stop — saturated: {sat.reason}")
                return
            gap = sat.major_gap or None
            if gap:
                print(f"    -> next round targets gap: {gap}")


def _seed_from_disk(store: NoteStore, outdir: str) -> None:
    """Reload an existing run's notes/analogies so a later phase appends to them."""
    path = os.path.join(outdir, "notes.json")
    if not os.path.exists(path):
        raise FileNotFoundError(f"no existing run to seed from at {path}")
    data = json.load(open(path, encoding="utf-8"))
    for n in data.get("notes", []):
        store.add(type=n["type"], claim=n["claim"], detail=n["detail"], round=n["round"])
    for a in data.get("analogies", []):
        store.add_analogy(
            case=a["case"], what_happened=a["what_happened"], similarities=a["similarities"],
            differences=a["differences"], implication=a["implication"], round=a["round"],
        )


def run(
    topic: str,
    *,
    max_rounds: int = 20,
    patience: int = 2,
    per_round: int = 12,
    model: str | None = None,
    outdir: str | None = None,
    notes_phase: bool = True,
    analogies_phase: bool = True,
    analogy_rounds: int = 12,
    max_notes: int = 130,   # soft runaway backstop only — saturation is the real stop, and a
    max_analogies: int = 60,  # downstream merge pass (pipeline) trims granularity, not breadth
) -> NoteStore:
    # Model tiering: a FAST model mines (high volume), a DEEP model judges
    # saturation (the stop decision needs good judgment). --model overrides the
    # fast/mining model only.
    llm_fast = LLM(model or FAST_MODEL)
    llm_deep = LLM(DEEP_MODEL)
    outdir = outdir or os.path.join("runs", slugify(topic))
    # Tracked mirror so the rendered notes are viewable in Conductor / editors.
    store = NoteStore(topic, outdir, mirror_md=os.path.join("docs", "latest-notes.md"))

    print(f"Topic: {topic}")
    print(f"Mining: {llm_fast.model or '(CLI default)'}  |  saturation judge: {llm_deep.model or '(CLI default)'}  |  out: {outdir}\n")

    # Auto-resume: if notes.json already exists, seed it into the store before
    # running any explore phases — so a kill mid-explore resumes rather than restarts.
    notes_path = os.path.join(outdir, "notes.json")
    if os.path.exists(notes_path) and notes_phase:
        _seed_from_disk(store, outdir)
        n_notes = len(store.notes)
        n_analogies = len(store.analogies)
        if n_notes > 0:
            print(f"Resuming explore from disk: {n_notes} notes, {n_analogies} analogies already saved.\n")

    if not notes_phase:
        if len(store.notes) == 0:
            _seed_from_disk(store, outdir)
        print(f"Skipping notes phase ({len(store.notes)} notes on disk).\n")
    else:
        _notes_loop(llm_fast, llm_deep, topic, store, max_rounds, per_round, max_notes=max_notes)

    if analogies_phase:
        # Analogies are verbose (paragraph + two lists each), so request fewer
        # per call to keep each structured response from truncating.
        _analogy_loop(llm_fast, llm_deep, topic, store, analogy_rounds, per_round=min(per_round, 6), max_analogies=max_analogies)

    print(f"\nDone. {len(store.notes)} notes, {len(store.analogies)} analogies.")
    print("  " + "  ".join(f"{TYPE_LABEL[t]}={c}" for t, c in store.counts().items()))
    print(f"\nWritten: {os.path.join(outdir, 'notes.md')}  and  docs/latest-notes.md")
    return store
