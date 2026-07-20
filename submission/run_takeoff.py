#!/usr/bin/env python3
"""Takeoff / unassailable-lead stage — deep-dives Branch B (the quarantined ~12% RSI tail).

The crux question the whole submission rests on: GIVEN a lab has a lead (say a 1-year
superintelligence head-start), what is the MECHANISM by which trailing labs cannot catch up?
Recursive self-improvement happens — but a trailing lab is also recursing; the leader does
NOT instantly control all compute; it will not bomb rival labs. So why would the lead become
*unassailable* rather than competed away by diffusion (open weights, talent, papers) and the
physical compute/power/fab substrate?

This stage: (1) enumerates Bostrom's lock-in mechanisms, adversarially tested against the
catch-up case (diffusion + distributed compute + no-kinetic-option + physical substrate),
each paired with a HISTORICAL precedent (theory ↔ history); (2) separates an unassailable
LEAD (rivals can't catch up) from a usable ADVANTAGE (turned into dominance), analysing the
AFFORDANCES and BOUNDS of a lead under two control regimes — human-controlled and ai-breakout
(what a US lab could actually do to China: cure cancer yes, mass-coercion no); (3) calibrates
P(decisive winner-take-all | takeoff) as the CONJUNCTION of the two — the number Branch B
rides on — grounded in the decisive-tech-lead reference class (nuclear monopoly eroded ~4yr).

    python run_takeoff.py     # reads bench/systems-map.json + composed-graph.json + corpus
"""
from __future__ import annotations
import glob, json, os, re, sys
from typing import Literal
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from forecaster.llm import LLM

SYSMAP = "bench/systems-map.json"
OUT_JSON = "bench/takeoff.json"
OUT_MD = "docs/takeoff-analysis.md"


class LockInMechanism(BaseModel):
    name: str = Field(description="The lock-in mechanism, e.g. 'Recursion amplifies leads (optimization-power / recalcitrance)'.")
    bostrom_basis: str = Field(description="How Bostrom frames it (fast takeoff / crossover, optimization power vs recalcitrance, decisive strategic advantage, singleton, wealth-into-capability).")
    how_it_locks_in: str = Field(description="The concrete mechanism by which it PREVENTS a trailing lab from catching up.")
    requires: str = Field(description="The condition it needs to hold (e.g. takeoff fast enough that 1yr = many capability doublings).")
    catch_up_counter: str = Field(description="The strongest reason a trailing lab catches up anyway — diffusion (open weights/talent/papers), distributed compute, no kinetic option, or the physical compute/power/fab substrate the leader ALSO needs.")
    historical_test: str = Field(description="The closest HISTORICAL precedent for THIS specific mechanism and what it showed — did a first-mover lead lock in or erode via this exact channel? (e.g. nuclear monopoly for cognition→resource/coercion lead; Manhattan espionage for diffusion-beats-secrecy; DRAM/semiconductor for compute-compounding; naval/dreadnought for a tech lead neutralised by fast-following). Pair the theory with the instance.")
    net_verdict: str = Field(description="Does this mechanism actually produce unassailability, and only under what condition — reconciling the theory (does it hold) with the historical test (has it ever held).")
    probability_conditional: float = Field(description="0-1: P(this mechanism holds | an RSI takeoff has begun).")


class ControlRegimeAffordance(BaseModel):
    regime: Literal["human_controlled", "ai_breakout"] = Field(
        description="human_controlled = a lab/state directs the superintelligence as a tool; ai_breakout = misaligned/autonomous, not taking human instruction.")
    description: str = Field(description="What this regime means concretely.")
    affordances: list[str] = Field(description="What the lead can ACTUALLY DO in this regime — concrete levers: scientific/biomedical (cure cancer, materials), economic (out-earn, corner compute/energy), cyber (compromise rival training runs, infrastructure), military/ISR, bioweapon, mass persuasion/info-ops, compute-cornering, regulatory capture.")
    bounds: list[str] = Field(description="What LIMITS the translation of the lead into real-world power: political will + legitimacy (a US lab will not genocide China even if capable), physical/kinetic reality (the world stays physical; you can't think your way to controlling fabs/grids/armies overnight), retaliation/MAD (cyber + nuclear), verification/attribution, allies, and time.")
    what_it_can_do_to_a_rival: str = Field(description="Concretely: what could a US lab with this lead actually DO to China (and what it CANNOT) — e.g. out-innovate and out-earn, degrade rival AI via cyber, tighten chokepoints; but NOT mass-casualty coercion (politics), NOT instant physical control.")
    dominance_verdict: str = Field(description="Does the lead translate into actual DOMINANCE in this regime, and how bounded.")
    probability_of_decisive_use: float = Field(description="0-1: P(the lead is translatable into a decisive REAL-WORLD advantage | the lead exists in this regime), given the bounds.")


class TakeoffAnalysis(BaseModel):
    framing: str = Field(description="The question, sharply stated.")
    mechanisms: list[LockInMechanism] = Field(description="The 4-6 candidate lock-in mechanisms, each adversarially tested.")
    lead_vs_advantage: str = Field(description="The key distinction: an UNASSAILABLE LEAD (rivals can't catch up) is NOT the same as a USABLE ADVANTAGE (the lead can be turned into real-world dominance). A cognitive lead is bounded by what is politically and physically ACTIONABLE — so branch B (winner-take-all) requires BOTH conjuncts, and the second is also < 1.")
    control_regimes: list[ControlRegimeAffordance] = Field(description="The two variations — human_controlled and ai_breakout — with their affordances, bounds, and what a lead-holder could actually do to a rival.")
    physical_brake: str = Field(description="The compute/power/fab substrate as the key balancing check — the leader needs ever-more physical compute to feed recursion, and that is fab/energy-gated and cannot be recursed away fast.")
    diffusion_brake: str = Field(description="Algorithm/weight/talent diffusion as the other check — a software lead is copyable in months (the 3-6mo floor).")
    headstart_analysis: str = Field(description="Does a ~1-year lead actually suffice? Work it: under FAST takeoff (steep capability curve) 1yr = many doublings → the time-lead becomes a capability chasm; under SLOW takeoff, 1yr is copied away. State the takeoff-speed threshold that flips it.")
    crux_conditions: list[str] = Field(description="What must ALL hold for the lead to be unassailable (e.g. crossover beats diffusion AND the leader can secure disproportionate compute AND translate cognition into resource/control lead non-kinetically).")
    unassailable_if: str = Field(description="The committed conditional: the lead becomes unassailable IF …")
    catch_up_if: str = Field(description="The committed conditional for the opposite: the lead is competed away IF …")
    calibrated_probability: float = Field(description="0-1: P(a lead that is BOTH unassailable AND usable for decisive real-world dominance | RSI takeoff begins) — the conjunction of catch-up-impossible AND politically/physically actionable. This is the number Branch B rides on — justify it, and show it as the product of the two conjuncts.")
    historical_analogies: list[str] = Field(description="Decisive-tech-lead reference class: what each shows about whether first-mover leads become unassailable (nuclear monopoly eroded ~4yr via espionage+parallel programs; no instant dominance; vs cases where a lead did lock in).")
    synthesis: str = Field(description="The bottom line: is Bostrom's unassailable-lead a real risk or an artefact of assuming fast takeoff — and what that does to Branch B's weight.")


SYSTEM = """You are a rigorous analyst stress-testing the single load-bearing bet of an AI-forecasting submission: whether a lead in AI capability can become UNASSAILABLE (Bostrom's 'decisive strategic advantage'). You have WebSearch — use it for current evidence on takeoff speed (RE-bench, METR task-horizons), lead persistence at the frontier, and compute distribution.

The user's sharp challenge, which you must answer mechanistically, not hand-wave: recursive self-improvement happens — but a TRAILING lab is also recursing; the leader does NOT instantly control all compute; it will NOT bomb rival labs. So by what MECHANISM does a ~1-year head-start become unassailable rather than get competed away by (a) diffusion (open weights, talent, published methods) and (b) the physical compute/power/fab substrate the leader ALSO depends on?

Be BOTH theory-driven and history-grounded, mechanism by mechanism — pair each theoretical channel with its closest historical precedent and reconcile them.

Do this:
1. Enumerate Bostrom's candidate LOCK-IN mechanisms (fast-takeoff crossover; optimization-power/recalcitrance so the one ahead recurses FASTER and the gap WIDENS; translating a cognitive lead into a resource/compute/control lead non-kinetically — capital, cyber, persuasion, cornering compute/energy, regulatory capture; wealth→capability compounding; singleton formation). For EACH: how it locks in, what it requires, the strongest catch-up counter, the closest HISTORICAL precedent for that exact channel and what it showed (nuclear monopoly, Manhattan espionage, DRAM/semiconductor compounding, naval/dreadnought fast-following, etc.), a net verdict reconciling theory-vs-history, and P(mechanism holds | takeoff begun).
2. Name the two brakes explicitly — the PHYSICAL substrate (recursion needs ever-more fab/energy-gated compute that can't be recursed away fast) and DIFFUSION (a software lead copyable in ~3-6 months). These are why leads normally erode.
3. Work the HEAD-START math: a 1-year lead is decisive only if takeoff is fast enough that a year equals many capability doublings; under slow takeoff it is copied away. State the takeoff-speed threshold that flips it.
4. Land the CRUX CONDITIONS (what must ALL hold), the committed 'unassailable_if' / 'catch_up_if'.
5. THE SECOND QUESTION — what can be DONE with a lead? An unassailable LEAD (rivals can't catch up) is NOT a usable ADVANTAGE (dominance). Analyse the AFFORDANCES and BOUNDS under TWO control regimes: (a) human_controlled (a US lab/state directs it as a tool) and (b) ai_breakout (misaligned, ignoring humans). For each: what it can concretely do (biomedical/economic/cyber/military/bio/persuasion/compute-cornering), what BOUNDS it (political will + legitimacy — a US lab will not mass-kill in China even if able; physical/kinetic reality — you can't think your way to controlling fabs/grids/armies overnight; retaliation/MAD; verification), and specifically WHAT A US LAB COULD ACTUALLY DO TO CHINA and what it could not. The regimes differ sharply.
6. CALIBRATE P(decisive winner-take-all | takeoff) as the CONJUNCTION: P(unassailable lead) × P(usable for real-world dominance | lead) — both < 1. Justify, because Branch B rides on it.
7. Ground in the decisive-tech-lead reference class — above all the nuclear monopoly (US first, but the monopoly eroded in ~4 years via espionage + parallel programs; no instant dominance) — and say what the base rate implies.

Be bold but adversarial: default to skepticism that leads lock in (the historical base rate is erosion), and make the fast-takeoff assumption carry the weight it actually requires."""


def _corpus(k: int = 14) -> str:
    pat = re.compile(r"(recursi|self-improv|decisive|takeoff|singleton|nuclear|manhattan|monopoly|first-mover|winner-take|espionage|lead (erod|persist)|catch.?up)", re.I)
    seen, hits = set(), []
    for f in glob.glob("runs/*/notes*.json"):
        tid = os.path.basename(os.path.dirname(f)).split("-")[0]
        for n in json.load(open(f, encoding="utf-8")).get("notes", []):
            if n.get("type") not in ("theory", "hypothesis", "crux", "evidence_for", "evidence_against"):
                continue
            t = f"{n.get('claim','')} {n.get('detail','')}"
            if pat.search(t):
                key = n.get("claim", "")[:60]
                if key in seen:
                    continue
                seen.add(key)
                hits.append(f"  [{tid} {n['type']}] {n.get('claim','')}: {(n.get('detail','') or '')[:150]}")
    return "RELEVANT CORPUS NOTES:\n" + "\n".join(hits[:k])


def _branchB(sm: dict) -> str:
    b = next((x for x in sm.get("branches", []) if x.get("id") == "B"), None)
    r1 = next((l for l in sm.get("loops", []) if "RSI" in l.get("name", "") or l.get("basis") == "theoretical"), None)
    L = ["BRANCH B (from the systems-map — the tail this stage deepens):"]
    if b:
        L.append(f"  {b['name']} — P={b.get('probability')}; trigger: {b.get('trigger_threshold','')}; who wins: {b.get('who_wins','')[:200]}")
    if r1:
        L.append(f"  RSI loop: {r1.get('mechanism','')} | flip: {r1.get('flip_threshold','')}")
    L.append(f"  dominant dynamic: {sm.get('dominant_dynamic','')[:200]}")
    return "\n".join(L)


def _md(r: TakeoffAnalysis) -> str:
    L = ["# Takeoff / unassailable-lead analysis (Branch B deep-dive)\n",
         f"*{r.framing}*\n",
         f"**Calibrated P(unassailable lead | RSI takeoff begins): {r.calibrated_probability:g}**\n",
         f"**Unassailable if:** {r.unassailable_if}\n",
         f"**Competed away if:** {r.catch_up_if}\n",
         f"**Head-start math:** {r.headstart_analysis}\n",
         "## Lock-in mechanisms (each adversarially tested)"]
    for m in sorted(r.mechanisms, key=lambda x: -x.probability_conditional):
        L.append(f"\n### {m.name} — P(holds | takeoff)={m.probability_conditional:g}")
        L.append(f"- Bostrom basis: {m.bostrom_basis}")
        L.append(f"- how it locks in: {m.how_it_locks_in}")
        L.append(f"- requires: {m.requires}")
        L.append(f"- catch-up counter: {m.catch_up_counter}")
        L.append(f"- historical test: {m.historical_test}")
        L.append(f"- **verdict (theory ↔ history):** {m.net_verdict}")
    L.append(f"\n## The two brakes\n- **Physical substrate:** {r.physical_brake}\n- **Diffusion:** {r.diffusion_brake}")
    L.append(f"\n## Lead ≠ advantage: what can actually be done with it\n{r.lead_vs_advantage}")
    for cr in r.control_regimes:
        L.append(f"\n### Regime: {cr.regime} — P(decisive real-world use | lead) = {cr.probability_of_decisive_use:g}")
        L.append(f"_{cr.description}_")
        L.append("- **can do:** " + "; ".join(cr.affordances))
        L.append("- **bounded by:** " + "; ".join(cr.bounds))
        L.append(f"- **US lab → China:** {cr.what_it_can_do_to_a_rival}")
        L.append(f"- **dominance verdict:** {cr.dominance_verdict}")
    L.append("\n## Crux conditions (all must hold for unassailability)")
    for c in r.crux_conditions:
        L.append(f"- {c}")
    L.append("\n## Historical reference class (decisive-tech leads)")
    for a in r.historical_analogies:
        L.append(f"- {a}")
    L.append(f"\n## Synthesis\n{r.synthesis}")
    return "\n".join(L)


def main() -> int:
    load_dotenv()
    sm = json.load(open(SYSMAP, encoding="utf-8")) if os.path.exists(SYSMAP) else {}
    user = ("\n\n".join([_branchB(sm), _corpus()]) +
            "\n\nAnswer the crux: by what MECHANISM (if any) does a ~1-year lead become unassailable rather than "
            "competed away? Enumerate + adversarially test Bostrom's lock-in mechanisms, name the physical + diffusion "
            "brakes, work the head-start math, land the crux conditions and a calibrated P(unassailable | takeoff), and "
            "ground it in the decisive-tech-lead reference class (esp. the nuclear monopoly).")
    llm = LLM()
    print("takeoff / unassailable-lead analysis (web-grounded, a few min)…")
    r = llm.parse(system=SYSTEM, user=user, schema=TakeoffAnalysis,
                  tools=["WebSearch"], max_turns=20, call_timeout=1200, max_buffer_size=32 * 1024 * 1024)
    json.dump(r.model_dump(), open(OUT_JSON, "w", encoding="utf-8"), indent=2)
    open(OUT_MD, "w", encoding="utf-8").write(_md(r))
    print(f"✓ {len(r.mechanisms)} mechanisms; P(unassailable | takeoff) = {r.calibrated_probability:g}")
    print(f"  unassailable_if: {r.unassailable_if[:100]}")
    if llm.last_cost_usd:
        print(f"  cost: ${llm.last_cost_usd:.2f}")
    print(f"  wrote {OUT_MD} and {OUT_JSON}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
