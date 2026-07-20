#!/usr/bin/env python3
"""Gate-tracker stage — turn each load-bearing gate into a DATA-DRIVEN trend with a
mechanistic explanation, so conviction rests on the derivative, not the level.

For every spine / cross-cutting gate in the composed graph this stage:
  1. picks 2-3 trackable time-series metrics,
  2. pulls the ACTUAL data points (WEB-GROUNDED — this stage runs with WebSearch on,
     so numbers are sourced, not confabulated; each point is tagged with its source
     and a grounding level),
  3. reads the trend (closing / widening / stuck + a quantified rate) AND EXPLAINS THE
     MECHANISM — e.g. *how* China closed 14nm->7nm so fast (DUV multipatterning + SMEE
     + state capital, NOT EUV), and why that same path stalls at ~5nm,
  4. names the DRIVING FACTORS that explain the past AND predict the future (the levers
     whose state decides whether the next leg is easier or harder),
  5. grounds the forward call in HISTORICAL ANALOGIES (preferring the mined corpus),
  6. classifies the gate hard-vs-soft (physical wall vs cost/time tax vs policy-reversible)
     and gives a committed forward call, and
  7. spells out the IMPLICATIONS FOR ACTUAL AI OUTCOMES if the trend holds (concrete,
     second-order, per-actor — e.g. China EUV-capped => competes via efficiency/open
     weights/inference not out-training; US firm-power-short => less usable inference,
     pushes build-out offshore/behind-the-meter) and the counterfactual if it breaks.

It is timeout-safe: writes bench/gate-tracker.json after EACH gate and auto-resumes
(skips gates already done). Pass `fresh` to start over.

    python run_gatetracker.py           # all load-bearing gates, web-grounded
    python run_gatetracker.py fresh      # ignore any partial run and redo all
    python run_gatetracker.py GV1 GV6    # only these gates
"""
from __future__ import annotations
import glob, json, os, sys
from typing import Literal
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from forecaster.llm import LLM

GRAPH = "bench/composed-graph.json"
OUT_JSON = "bench/gate-tracker.json"
OUT_MD = "docs/gate-tracker.md"


# ------------------------------------------------------------------ schema
class DataPoint(BaseModel):
    date: str = Field(description="Point in time, e.g. '2019', '2024-Q4', 'May 2023'.")
    value: float
    unit: str = Field(description="e.g. 'nm', '%', 'GW', 'months', 'Elo', 'pp'.")
    source: str = Field(description="Concrete source: org + report/year (e.g. 'LBNL 2024 DC Energy Report'). Empty only if pure estimate.")
    grounding: Literal["web_verified", "notes_corpus", "model_estimate"] = Field(
        description="web_verified = pulled/confirmed via web search this run; model_estimate = unverified, treat as approximate.")


class MetricSeries(BaseModel):
    metric: str = Field(description="The specific tracked quantity, e.g. 'SMIC best node in volume production (nm)'.")
    why_it_matters: str = Field(description="Why this series reads the gate.")
    unit: str = Field(description="THE single unit for the whole series (e.g. 'nm', '%', 'GW', 'months', '$/M tokens'). Every point must be in THIS unit — do not mix units in one series; if you need two units, make two metrics.")
    chart_type: Literal["line", "bar"] = Field(description="line = a true time-series trend; bar = a few discrete/categorical points or a snapshot comparison.")
    points: list[DataPoint] = Field(description="The time-series, oldest->newest, AT LEAST 5 points, ALL in `unit`. Span real history: reach back to a PRE-2020 baseline (e.g. 2015-2019) and forward to 2025/2026, so the trend and its inflections are visible — not just two recent dots.")
    direction: Literal["closing", "widening", "stuck", "accelerating", "reversing"]
    rate: str = Field(description="QUANTIFIED rate, e.g. '~1 node every 2yr', 'gap -20pp over 30 months', 'flat since 2022'.")


class DrivingFactor(BaseModel):
    factor: str = Field(description="A key lever that governs this gate's trajectory (e.g. 'EUV access', 'firm-power interconnection', 'algorithmic efficiency').")
    shaped_past: str = Field(description="HOW this factor drove the trend so far (e.g. 'DUV multipatterning let SMIC close 14->7nm without EUV').")
    governs_future: str = Field(description="HOW it binds or accelerates the trend going forward (e.g. 'sub-3nm needs EUV, so the same path now stalls — the next leg is much harder').")


class AnalogyRef(BaseModel):
    case: str = Field(description="The historical race/precedent (e.g. 'Japan DRAM 1980s', 'Soviet lithography', 'US-Japan semiconductor accord').")
    what_happened: str
    what_it_implies_here: str = Field(description="What this precedent predicts for THIS gate — the transferable lesson (and where the analogy breaks).")
    grounding: Literal["corpus", "web_verified", "model_knowledge"] = Field(
        description="corpus = from the provided analogy set; web_verified = confirmed via search; model_knowledge = from memory.")


class GateTrend(BaseModel):
    gate_id: str
    gate_name: str
    binds_actor: str = Field(description="Which actor(s) this gate binds most.")
    metrics: list[MetricSeries] = Field(description="5-7 tracked time-series (each single-unit, chart-ready). Cover the gate from several angles — the core binding metric, the catch-up/erosion metric, the accelerant metric, the counter-lever, and a cost/scale metric — so the gate can be read as a small dashboard.")
    hard_or_soft: Literal["hard_physical", "soft_slow", "soft_fast", "policy_reversible"] = Field(
        description="Is the gate a physical impossibility, a slow cost/time tax, a fast-eroding moat, or a policy switch?")
    hardness_mechanism: str = Field(description="WHY it is hard or soft — the physical / tacit-knowledge / supply-chain / policy mechanism. Be concrete (e.g. 'Zeiss sub-atomic mirrors + tin-plasma source + ~3 firms of tacit process knowledge').")
    trend_explanation: str = Field(description="WHY the trend moved as it did — the causal driver. MUST answer the 'how did they do it so fast / why is it stuck' question mechanistically, not just restate the direction.")
    driving_factors: list[DrivingFactor] = Field(description="2-4 key levers that EXPLAIN THE PAST AND PREDICT THE FUTURE of this trend — the factors whose state determines whether the trend continues, accelerates, or stalls. This is where past-explanation becomes future-prediction.")
    historical_analogies: list[AnalogyRef] = Field(description="2-3 historical precedents that ground the forward call — prefer the provided corpus analogies, supplement via web/knowledge. Each must say what it predicts here and where it breaks.")
    inflection_risks: str = Field(description="What would accelerate the trend, and what would stall/reverse it.")
    forward_call: str = Field(description="A COMMITTED call: where this gate is by 2028 and 2030, with a probability where possible. Not 'it depends'.")
    implications_if_holds: str = Field(description="THE SO-WHAT FOR ACTUAL AI OUTCOMES if this trend holds. Be concrete and second-order: e.g. 'if China cannot bridge the EUV gap, it stays capped at ~5nm → higher $/token and power/chip for frontier TRAINING, so it competes via efficiency + open weights + inference on mature nodes, not by out-training the US'; or 'if the US cannot add firm power fast enough, does it fundamentally have less inference capacity? does it push build-out offshore (Gulf/allies), behind-the-meter, or into efficiency?' Name the downstream capability, cost, and strategic-response consequences per actor.")
    implications_if_breaks: str = Field(description="The so-what if the trend instead inflects the other way (the gate clears, or the leader's edge collapses) — the counterfactual that a forecaster must price.")
    confidence: Literal["low", "medium", "high"]


class Synthesis(BaseModel):
    cross_gate_read: str = Field(description="3-6 sentences: which gates are closing vs stuck vs widening, the contrast that matters, and what the pattern of derivatives implies about who is ahead when.")
    boldest_departures: list[str] = Field(description="The 4-8 highest-conviction, non-consensus calls that fall OUT of the trend data (not restatements of the present).")


SYSTEM_GATE = """You are a data-driven forecasting analyst tracking ONE gate in a multi-input technology race (AI x mercantilism). Conviction comes from the DERIVATIVE, not the level — so your job is to measure the trend, EXPLAIN its mechanism, PREDICT its future via the factors that govern it, and spell out the SO-WHAT for actual AI outcomes.

You have WebSearch. USE IT to pull real, current (2024-2026) numbers — do not rely on memory for figures. For every data point, name a concrete source and set grounding='web_verified' only if you actually confirmed it via search this run; otherwise 'model_estimate' and treat it as approximate. NEVER present an unverified number as sourced. Better 3 solid sourced points than 6 invented ones. You are given a set of CORPUS ANALOGIES mined for this gate — prefer and cite these for historical grounding, supplement with web/knowledge, and mark grounding accordingly.

Do ALL of the following for the gate:
1. METRICS — 5-7 genuinely trackable time-series that read the gate from several angles (the core binding metric, the catch-up/erosion metric, an accelerant, a counter-lever, and a cost/scale metric — a small dashboard). Each is a SINGLE-UNIT, chart-ready series with AT LEAST 5 real, dated, sourced points that SPAN a pre-2020 baseline (2015-2019) through 2025/2026 — a real trend with history, not two recent dots. Give a direction + a QUANTIFIED rate. NEVER mix units within one series (if you need two units, make two metrics); set chart_type (line for a trend, bar for discrete/snapshot). Prefer 5+ solid grounded series; if a gate genuinely supports fewer clean time-series, return what you can honestly ground and say so via grounding/confidence rather than padding with invented numbers.
2. TREND MECHANISM — explain causally WHY it moved. If a gap closed fast, say exactly HOW (which technology / policy / capital / workaround), and what that path's ceiling is. Discipline: 'China went 14nm->7nm fast via DUV multi-patterning + SMEE tooling + state capital + repatriated talent — NOT EUV; that path reaches ~5nm density then stalls on cost/yield because sub-3nm needs EUV, the near-hard gate behind it.'
3. DRIVING FACTORS — 2-4 levers that both explain the past AND predict the future. For each: how it shaped the trend so far, and how it governs the trend ahead (this is where past-explanation becomes forward-prediction — e.g. 'the gap closed fast while DUV sufficed; it gets HARDER now because the EUV constraint binds the next leg').
4. HISTORICAL ANALOGIES — 2-3 precedents (prefer the provided corpus) that ground the forward call; each states what it predicts here and where it breaks.
5. HARD-vs-SOFT — classify (physical wall / slow cost-tax / fast-eroding / policy switch) with the concrete mechanism.
6. FORWARD CALL — committed 2028/2030 position + a probability where you can. Not 'it depends'.
7. IMPLICATIONS FOR AI OUTCOMES — the crux the competition rewards. If the trend HOLDS, what does it mean for real AI capability, cost, and strategy, second-order and per-actor? (e.g. 'if China can't bridge EUV → capped for frontier TRAINING → it competes via efficiency + open weights + inference on mature nodes, not by out-training'; 'if the US can't add firm power fast enough → does it fundamentally have less usable inference capacity? does it push build-out offshore to the Gulf/allies, behind-the-meter, or lean on efficiency?'). Then give the counterfactual if the trend BREAKS.

Be bold but every call rests on data + mechanism + analogy. If the data is thin, say so via grounding and confidence — do not fabricate."""

SYSTEM_SYN = """You synthesize a set of per-gate trend analyses into the cross-gate read. Do not add new data. Identify which gates are closing vs stuck vs widening, name the contrast that carries the thesis, and extract the highest-conviction NON-CONSENSUS calls that fall out of the derivatives (not restatements of the present). Boldness = high-conviction + non-consensus + resolvable."""


# ------------------------------------------------------------------ analogy provenance
# Same chain as build_part3.py: composed var.sources (topic:Vid) -> per-topic
# decisiongraph variable (analogy_ids) -> that topic's notes.json analogies.
def load_provenance() -> tuple[dict, dict]:
    tvar: dict[str, dict] = {}          # "t6stack:V1" -> per-topic variable
    anlg_by_topic: dict[str, dict] = {}  # tid -> {analogy_id: analogy}
    for dg in glob.glob("runs/*/decisiongraph.json"):
        tid = os.path.basename(os.path.dirname(dg)).split("-")[0]
        graph = json.load(open(dg, encoding="utf-8"))
        for v in graph.get("variables", []):
            tvar[f"{tid}:{v['id']}"] = {**v, "topic": tid}
        nj = os.path.join(os.path.dirname(dg), "notes.json")
        if os.path.exists(nj):
            nd = json.load(open(nj, encoding="utf-8"))
            anlg_by_topic[tid] = {a["id"]: a for a in nd.get("analogies", [])}
    return tvar, anlg_by_topic


def gate_analogies(mv: dict, tvar: dict, anlg_by_topic: dict, k: int = 6) -> list[dict]:
    found: dict[str, dict] = {}          # dedup by case-prefix, keep richest
    for src in mv.get("sources", []):
        tv = tvar.get(src)
        if not tv:
            continue
        for aid in tv.get("analogy_ids", []):
            a = anlg_by_topic.get(tv["topic"], {}).get(aid)
            if not a:
                continue
            key = a["case"][:48].lower()
            if key not in found or len(a.get("what_happened", "")) > len(found[key].get("what_happened", "")):
                found[key] = a
    return sorted(found.values(), key=lambda a: len(a.get("what_happened", "")), reverse=True)[:k]


# ------------------------------------------------------------------ context
def gate_context(g: dict, gid: str, analogies: list[dict] | None = None) -> str:
    byid = {v["id"]: v for v in g["merged_variables"]}
    v = byid.get(gid)
    if not v:
        return ""
    L = [f"GATE {gid}: {v['name']}",
         f"state space: {' | '.join(v.get('state_space', []))}",
         f"current reading: {v.get('current_reading','')}",
         f"trajectory (our prior on catch-up): {v.get('trajectory','')}",
         f"leverage: {v.get('leverage','')}"]
    if v.get("actor_readings"):
        L.append("per-actor readings:")
        for ar in v["actor_readings"]:
            L.append(f"  - {ar['actor']}: {ar['reading']}")
    edges = [e for e in g["edges"] if e["source"] == gid or e["target"] == gid]
    if edges:
        L.append("connected edges (interaction structure):")
        for e in edges:
            act = f" [{e.get('actor')}]" if e.get("actor") else ""
            L.append(f"  {e['source']} -{e['kind']}-> {e['target']}{act}: {e.get('note','')}")
    if analogies:
        L.append("\nCORPUS ANALOGIES mined for this gate (prefer & cite these; grounding='corpus'):")
        for a in analogies:
            diff = "; ".join((a.get("differences") or [])[:2])
            L.append(f"  - {a['case']}: {a.get('what_happened','')[:300]}"
                     + (f" | breaks: {diff}" if diff else "")
                     + (f" | implies: {a.get('implication','')[:200]}" if a.get("implication") else ""))
    return "\n".join(L)


# ------------------------------------------------------------------ render
def render_md(gates: list[dict], syn: dict | None) -> str:
    L = ["# Gate tracker — data-driven trends per gate\n",
         "*Each load-bearing gate as a tracked time-series with a mechanistic trend explanation "
         "and a hard-vs-soft verdict. Web-grounded where marked `web_verified`; `model_estimate` "
         "points are approximate. Conviction rests on the derivative, not the level.*\n"]
    if syn:
        L.append("## Cross-gate read\n")
        L.append(syn.get("cross_gate_read", "") + "\n")
        if syn.get("boldest_departures"):
            L.append("**Boldest departures from consensus (fall out of the trends):**")
            for b in syn["boldest_departures"]:
                L.append(f"- {b}")
            L.append("")
    # summary table
    L.append("## Summary\n")
    L.append("| Gate | Binds | Direction | Hard/Soft | Forward call |")
    L.append("|---|---|---|---|---|")
    dirn = {m: "" for m in ()}
    for gt in gates:
        d = " · ".join(sorted({m["direction"] for m in gt["metrics"]}))
        L.append(f"| {gt['gate_id']} {gt['gate_name'][:34]} | {gt['binds_actor'][:18]} | "
                 f"**{d}** | {gt['hard_or_soft']} | {gt['forward_call'][:90]} |")
    L.append("")
    # per-gate detail
    for gt in gates:
        L.append(f"\n## {gt['gate_id']} · {gt['gate_name']}")
        L.append(f"_binds {gt['binds_actor']} · **{gt['hard_or_soft']}** · confidence {gt['confidence']}_\n")
        for m in gt["metrics"]:
            L.append(f"### {m['metric']} — **{m['direction']}** ({m['rate']})")
            L.append(f"*{m['why_it_matters']}*\n")
            L.append("| date | value | source | grounding |")
            L.append("|---|---|---|---|")
            for p in m["points"]:
                L.append(f"| {p['date']} | {p['value']:g} {p['unit']} | {p['source']} | {p['grounding']} |")
            L.append("")
        L.append(f"**Why it's {gt['hard_or_soft']}:** {gt['hardness_mechanism']}\n")
        L.append(f"**Why the trend moved this way:** {gt['trend_explanation']}\n")
        if gt.get("driving_factors"):
            L.append("**Driving factors (past → future):**")
            for df in gt["driving_factors"]:
                L.append(f"- **{df['factor']}** — *past:* {df['shaped_past']} *→ future:* {df['governs_future']}")
            L.append("")
        if gt.get("historical_analogies"):
            L.append("**Historical analogies:**")
            for a in gt["historical_analogies"]:
                L.append(f"- **{a['case']}** ({a['grounding']}): {a['what_happened']} — *implies here:* {a['what_it_implies_here']}")
            L.append("")
        L.append(f"**Inflection risks:** {gt.get('inflection_risks','')}\n")
        L.append(f"**Forward call:** {gt.get('forward_call','')}\n")
        L.append(f"**Implications for AI outcomes — if the trend HOLDS:** {gt.get('implications_if_holds','')}\n")
        L.append(f"**…if it BREAKS:** {gt.get('implications_if_breaks','')}\n")
    return "\n".join(L)


# ------------------------------------------------------------------ main
def main() -> int:
    load_dotenv()
    if not os.path.exists(GRAPH):
        print(f"no {GRAPH} — run run_compose.py first."); return 1
    g = json.load(open(GRAPH, encoding="utf-8"))

    argv = [a for a in sys.argv[1:]]
    fresh = "fresh" in argv
    explicit = [a for a in argv if a.startswith("GV")]

    # gate set: explicit, else spine ∪ cross_cutting (deduped, order-preserving)
    if explicit:
        gate_ids = explicit
    else:
        seen, gate_ids = set(), []
        for gid in g["spine"] + g.get("cross_cutting", []):
            if gid not in seen:
                seen.add(gid); gate_ids.append(gid)

    done: dict[str, dict] = {}
    if not fresh and os.path.exists(OUT_JSON):
        prev = json.load(open(OUT_JSON, encoding="utf-8"))
        # only treat a cached gate as done if it has the CURRENT schema fields;
        # stale entries (older schema, missing implications/driving_factors) re-run.
        required = set(GateTrend.model_fields)
        done, stale = {}, []
        for x in prev.get("gates", []):
            mets = x.get("metrics", [])
            # current schema: all GateTrend fields present, AND metrics carry `unit`
            # (added with the 5-7-metric / >=5-point upgrade) and there are >=5 of them
            fresh_metrics = len(mets) >= 5 and all("unit" in m for m in mets)
            if required.issubset(x) and fresh_metrics:
                done[x["gate_id"]] = x
            else:
                stale.append(x["gate_id"])
        if done:
            print(f"resuming — {len(done)} gate(s) cached & current: {', '.join(done)}")
        if stale:
            print(f"re-running {len(stale)} stale gate(s) (old schema): {', '.join(stale)}")

    todo = [gid for gid in gate_ids if gid not in done]
    print(f"gate set ({len(gate_ids)}): {', '.join(gate_ids)}")
    print(f"to run ({len(todo)}): {', '.join(todo) or '(none — all cached)'}\n")

    tvar, anlg_by_topic = load_provenance()
    byid = {v["id"]: v for v in g["merged_variables"]}
    print(f"provenance: {len(tvar)} topic-vars, {sum(len(a) for a in anlg_by_topic.values())} analogies available\n")

    # canonical file order: spine ∪ cross_cutting, then any extras
    seen_c, canon = set(), []
    for gid in g["spine"] + g.get("cross_cutting", []):
        if gid not in seen_c:
            seen_c.add(gid); canon.append(gid)

    # PRESERVE all current-schema gates already on disk — an explicit-subset run
    # must MERGE into the file, never clobber gates outside the run set.
    result: dict[str, dict] = dict(done)

    def _ordered() -> list[dict]:
        ids = list(result.keys())
        return [result[gid] for gid in canon if gid in result] + \
               [result[gid] for gid in ids if gid not in canon]

    def _dump(extra: dict | None = None) -> None:
        payload = {"gates": _ordered()}
        if extra:
            payload.update(extra)
        json.dump(payload, open(OUT_JSON, "w", encoding="utf-8"), indent=2)

    llm = LLM()
    total_cost = 0.0
    for i, gid in enumerate(todo, 1):
        anas = gate_analogies(byid.get(gid, {}), tvar, anlg_by_topic, k=6)
        ctx = gate_context(g, gid, anas)
        if not ctx:
            print(f"  ! {gid} not in graph — skipping"); continue
        print(f"[{i}/{len(todo)}] {gid} — web-grounded trend analysis ({len(anas)} corpus analogies, a few min)…")
        user = (ctx + "\n\nTrack this gate per the instructions: (1) 5-7 single-unit, chart-ready trackable metrics, "
                "each with AT LEAST 5 real dated sourced points spanning pre-2020 (2015-2019) to 2025/2026 (use WebSearch); "
                "(2) direction + quantified rate; (3) the MECHANISM of the "
                "trend; (4) 2-4 DRIVING FACTORS that explain the past AND predict the future (what makes the next "
                "leg easier/harder); (5) 2-3 HISTORICAL ANALOGIES (prefer the corpus ones above); (6) hard-vs-soft "
                "with mechanism; (7) a committed 2028/2030 forward call; (8) the IMPLICATIONS FOR AI OUTCOMES if "
                "the trend holds (concrete, second-order, per-actor) and the counterfactual if it breaks.")
        gt = llm.parse(system=SYSTEM_GATE, user=user, schema=GateTrend,
                       tools=["WebSearch"], max_turns=20, call_timeout=900, retries=1,
                       max_buffer_size=32 * 1024 * 1024)   # 32 MB: a single web-search result can exceed the 1 MB default
        result[gid] = gt.model_dump()
        if llm.last_cost_usd:
            total_cost += llm.last_cost_usd
        _dump()   # incremental, timeout-safe, MERGE-not-clobber
        wv = sum(1 for m in gt.metrics for p in m.points if p.grounding == "web_verified")
        print(f"      ✓ {len(gt.metrics)} metrics, {wv} web-verified points, {gt.hard_or_soft}, conf {gt.confidence}")

    # synthesis over ALL gates on file
    ordered = _ordered()
    print("\nsynthesizing cross-gate read…")
    syn = None
    try:
        blocks = "\n\n".join(
            f"{x['gate_id']} {x['gate_name']}: " +
            "; ".join(f"{m['metric']} [{m['direction']}, {m['rate']}]" for m in x["metrics"]) +
            f" | {x['hard_or_soft']} | forward: {x['forward_call']}" +
            f" | if-holds: {x.get('implications_if_holds','')}"
            for x in ordered)
        syn = llm.parse(system=SYSTEM_SYN, user="PER-GATE TRENDS:\n\n" + blocks +
                        "\n\nGive the cross-gate read and the boldest non-consensus departures.",
                        schema=Synthesis, max_buffer_size=32 * 1024 * 1024).model_dump()
    except Exception as e:
        print(f"  (synthesis skipped: {e})")

    _dump(extra={"synthesis": syn})
    open(OUT_MD, "w", encoding="utf-8").write(render_md(ordered, syn))
    print(f"\n✓ {len(ordered)} gates tracked; wrote {OUT_JSON} and {OUT_MD}")
    if total_cost:
        print(f"  cost this run: ${total_cost:.2f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
