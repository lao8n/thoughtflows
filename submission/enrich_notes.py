#!/usr/bin/env python3
"""Manually enrich a topic's frozen notes with fresh 2026 data + a fatter analogy list.

Non-destructive + idempotent: reads runs/<id>-*/notes.json, appends the curated
manual notes/analogies below (tagged round=99), writes notes.enriched.json.
Re-running regenerates the same file (ids computed off the frozen notes.json).

    python enrich_notes.py t6
"""
from __future__ import annotations
import glob, json, os, sys

# ---- curated NOTES (fresh 2026 data + the disaggregation the old notes lacked) ----
MANUAL_NOTES = [
    ("evidence_for",
     "Capital is abundant for the US and is NOT its binding constraint",
     "US big-4 hyperscaler capex is ~$725B for 2026 (Amazon ~$200B, Google ~$185B, Meta ~$125B, Microsoft ~$120B), up from ~$410B in 2025; OpenAI has ~$1.15T of multi-year infra commitments and xAI/CoreWeave are additive, so total US private AI-infra capex is ~$500-650B. The US can raise and commit capital far faster than it can deploy it."),
    ("evidence_for",
     "The US AI buildout is EXECUTION-gated (grid/transformers/parts), not capital-gated",
     "30-50% of planned US 2026 data centers are delayed or cancelled on power, transformers, and parts (many China-sourced). Morgan Stanley projects a ~44-49 GW US data-center power shortfall through 2028; PJM wholesale power rose +76% YoY (Q1'25 ~$78 → Q1'26 ~$137/MWh); high-voltage transformer lead times are 2.5-4 years with one domestic grain-oriented-steel supplier. Behind-the-meter routes around it only locally (~8.9 GW operational vs a 44 GW gap) — money cannot clear an interconnection queue."),
    ("evidence_for",
     "China is not energy-, capital-, speed-, or materials-gated on the buildout",
     "China added ~429 GW of generating capacity in 2024 vs ~49 GW in the US; its data-center electricity costs less than half the US's; projects go from plan to operation in months not years; and it has ~125 GW of DC capacity across ~550 sites in the pipeline vs the US bringing ~5-7 GW online next year. US installed DC capacity leads today (~54 vs ~32 GW) but the trajectory favors China on every physical input except chips."),
    ("crux",
     "The binding constraint is ACTOR-SPECIFIC, not a single global variable",
     "The US minimum stave is grid interconnection + transformers + parts (execution); China's minimum stave is leading-edge logic chips (frontier training). 'Physical constraint' is not one thing — it is a vector that binds each actor at a different input, so any single-variable framing (physical-vs-capital) is mis-specified."),
    ("evidence_for",
     "China has de facto decoupled from NVIDIA internally; the chip gate binds TRAINING not inference",
     "Huawei Ascend 910C delivers ~60% of H100 inference performance and Huawei holds ~60% of China's DOMESTIC AI-chip market by end-2026 (CloudMatrix-384 is system-level competitive via scale). The leading-edge chip chokepoint binds frontier TRAINING (5nm-class dense clusters); inference and deployment run fine on mature nodes plus efficiency."),
    ("evidence_against",
     "'ASML EUV = 100%, China 10 years behind' is overstated as of 2026",
     "ASML is ~98% of sub-5nm, but China is ~4 years behind, not 10: an indigenous LDP-source EUV tool entered trial production in late 2025 at ~50% of ASML's throughput; SMIC produces 7nm via DUV multipatterning (Kirin 9000S/9030) and targets 5nm in 2027-28 at <50% yield. The chip chokepoint is eroding with a finite leverage window (~2026-28)."),
    ("theory",
     "The chip chokepoint's leverage window closes on two fronts at once",
     "It erodes both because China indigenizes (EUV/Ascend/CXMT) AND because demand shifts away from it — Ascend being 'good enough' plus algorithmic efficiency make frontier-5nm nice-to-have rather than essential for most Chinese workloads. A chokepoint whose product is becoming less necessary loses value even before it is technically broken."),
    ("evidence_for",
     "Chokepoint durability varies sharply by layer — EDA controls already reversed",
     "US EDA export controls to China were reversed in July 2025 (Synopsys/Cadence resumed sales) under industry pressure — a weak chokepoint. EUV is far more durable (single-source, tacit-knowledge, one-per-fab, hard to lobby away). Ranking hardest→softest: EUV > TSMC/Taiwan fabs > HBM > logic design (bifurcated) > packaging > EDA."),
    ("evidence_for",
     "China holds a counter-chokepoint on the West's buildout via materials refining",
     "China controls refining of the materials the buildout needs: rare earths ~87-90%, permanent magnets ~90%, gallium ~98%, germanium ~68%, plus expanding copper smelting. It has weaponized this: gallium/germanium licensing (2023) → US ban (Dec 2024) → rare-earth/magnet controls (Apr 2025) → tactical suspension (Nov 2025). Prices spiked (EU gallium ~6x, germanium ~3x, magnets ~6x)."),
    ("theory",
     "The two chokepoints are asymmetric in KIND — mutual assured disruption, not destruction",
     "The US/allied lever (EUV/EDA/CUDA) PREVENTS China from doing the thing (frontier chips) — hard but eroding (~4yr). China's lever (materials refining + electrical-equipment supply) RAISES the cost/timeline of the Western buildout (+15-25%, +6-12mo) — soft but durable (Western reshoring reaches only ~20-25% of China output by 2028). Each holds a chokepoint on the other; neither is a knockout."),
    ("evidence_for",
     "Advanced packaging (CoWoS) and HBM are the near-term binding chip sub-constraints",
     "TSMC CoWoS (~85-90%) and HBM (SK Hynix/Samsung/Micron) have been the physical bottleneck for H100/H200-class supply — more binding than raw wafer starts. China's CXMT is ~2-3 years behind (HBM3 by 2026-27) and Ascend uses older HBM2E at a bandwidth penalty."),
    ("theory",
     "Algorithmic efficiency substitutes for chips and helps the compute-poor more — but mainly for inference",
     "Training-compute for a fixed capability falls ~10-15x/yr (DeepSeek V3 final run ~$5.6M), which relieves China's chip constraint more than the US's — but chiefly for INFERENCE, not frontier training, and Jevons rebound keeps aggregate compute demand (and thus the chokepoints) binding at the system level."),
    ("hypothesis",
     "The buildout race is won by whoever clears their own minimum stave first, on a ~2-4yr clock",
     "The US clears via permitting reform + behind-the-meter + domestic transformers; China clears via indigenous EUV + Ascend sufficiency + efficiency. Each can also slow the other (mutual chokepoints). Neither is likely to secure a DURABLE decisive lead, because the frontier keeps diffusing and each side's edge is a depreciating asset — the honest read is a contested race whose direction depends on relative clearing speed."),
    ("evidence_against",
     "TSMC/Taiwan is the single hardest and most persistent chokepoint — a catastrophic tail",
     "TSMC holds >90% of sub-5nm and is a single point of failure for all frontier training ('silicon shield'). TSMC Arizona reaches only ~20% of capacity by 2030. This concentration risk dwarfs the (eroding) EUV question and is the dominant discontinuity in the whole stack."),
]

# ---- curated ANALOGIES (the fatter reference class, each with a disanalogy + catch-up signal) ----
MANUAL_ANALOGIES = [
    ("Japan overtakes US in DRAM/consumer electronics (1970s-80s) then loses the software/PC era",
     "Japan won manufacturing/hardware (≈80% of DRAM by 1988, dominant consumer electronics) via process excellence and MITI coordination, but the value migrated to software, standards and platforms (Wintel, then the internet) where the US led — so Japan won the layer that stopped mattering.",
     ["A rising manufacturing power overtaking the incumbent on scale/process/cost",
      "State-coordinated industrial policy (MITI then, Big Fund now)"],
     ["This time the binding/decisive layer may be PHYSICAL (energy, fabs, deployment) rather than software — which would invert who wins",
      "AI weights diffuse; PC software value locked to platforms"],
     "The whole thesis is a bet on WHICH layer is decisive: if value stays in software/frontier (US), it's Japan-in-software redux; if it sits in the physical buildout, the follower can win."),
    ("China overtakes the West in solar PV (2005-2020)",
     "The West invented and led solar; China scaled to >80% of global manufacturing (>95% of wafers) via state subsidy, cheap energy, and manufacturing scale, driving costs down ~90% and capturing the industry.",
     ["Follower wins on manufacturing scale + energy + cost + state capital — exactly China's AI-buildout advantages",
      "Invented-in-the-West, scaled-by-China pattern"],
     ["Solar had no EUV-equivalent single-source chokepoint; AI does (leading-edge chips China cannot yet make)",
      "Solar cells are commodity; frontier AI training is not"],
     "China's playbook wins the physical/manufacturing layer decisively — IF that layer is where AI's decisive advantage sits and no upstream chokepoint (chips) gates it."),
    ("COCOM Cold-War export controls (1949-94); Toshiba-Kongsberg 1987",
     "A 17-ally regime denied advanced tech to the Soviet bloc for 45 years; it slowed Soviet computing but leaked via neutral transshipment and allied commercial pressure (the Toshiba submarine-propeller sale), and never stopped catch-up.",
     ["Multilateral tech-denial aimed at a strategic rival — the direct template for chip controls",
      "Leaks via third countries and commercial pressure"],
     ["The modern Japan-NL-US trilateral has held TIGHTER at the equipment layer than COCOM ever did",
      "Chips are physical/trackable; some Cold-War tech was more diffuse"],
     "Controls buy years, not denial — the base rate is 'slow, not stop,' so the durable lever is the equipment layer, not finished chips."),
    ("China's 2010 rare-earth embargo on Japan (Senkaku dispute)",
     "China halted rare-earth exports to Japan; prices spiked ~10x. Japan cut consumption ~50% and funded diversification (Lynas), but China still held ~80% of refining a decade later; the embargo itself lasted ~months.",
     ["Materials-refining chokepoint weaponized against a tech rival — the exact template for gallium/germanium/magnets today",
      "Processing (not mining) is the concentrated, hard-to-replicate stage"],
     ["AI-relevant materials (gallium/germanium) are even more processing-concentrated in China and less substitutable than 2010 rare earths",
      "Diversification took a decade and never broke China's refining share"],
     "China's materials lever is durable and weaponizable but reversible in the short run — it raises Western cost/timeline (a tax), it does not halt the buildout (not a knockout)."),
    ("US atomic monopoly → first Soviet test, 1945-49 (~4 years)",
     "Even a secret, materials-gated, genuinely decisive lead under total secrecy diffused in about four years via espionage and parallel development.",
     ["Sets the base rate that decisive technological leads are transient"],
     ["AI diffuses FASTER (weeks; open weights) — so this understates diffusion for the model layer",
      "But the PHYSICAL/materials layer (China's analog) diffuses far slower than software"],
     "The model/frontier layer diffuses fast (no durable lead there); the physical layer does not — which is why the contest has moved to the physical stack."),
    ("US nuclear submarines & stealth aircraft (1950s-present; 30-50yr leads)",
     "The US held qualitative leads in SSBNs and stealth for decades; rivals never fully closed the gap.",
     ["Proof that SOME technological leads endure for decades"],
     ["Those are physical PLATFORMS with irreproducible tacit manufacturing; AI capability is software and publishes",
      "So durable AI leads transfer to the TOOLING layer (EUV/fabs), not the model layer"],
     "Durability in AI will live at the physical tooling/fab layer, not the model — a lead in weights cannot be defended the way a stealth airframe can."),
    ("OPEC oil embargo, 1973",
     "Arab OPEC states embargoed oil; prices quadrupled, US GDP fell ~2.5%, stagflation followed — but diversification happened within months because oil is fungible and refining is globally distributed.",
     ["Weaponization of a concentrated strategic input against a rival bloc"],
     ["Oil is fungible with globally-distributed refining; China's rare-earth/gallium REFINING is concentrated and non-fungible",
      "So the China-materials analog is a LONGER-duration constraint than the oil shock"],
     "A concentrated-refining chokepoint (China materials) is stickier than the oil embargo was — diversification takes years, not months."),
    ("Wintel x86 → ARM disruption (Apple M1 2020; AWS Graviton ~25% of EC2)",
     "A 30-year x86 interface moat held via backward-compatibility lock-in until ARM's efficiency advantage justified the switching cost.",
     ["An interface/ecosystem moat (CUDA today) can persist for decades",
      "It breaks when an efficiency shock justifies switching"],
     ["Rewriting an API call (CUDA) is far cheaper than recompiling billions of x86 binaries — so CUDA's moat may be WEAKER than x86's",
      "China is being forced to build the ARM-equivalent (Ascend/CANN) under sanction pressure"],
     "Software-interface moats (CUDA) erode faster than physical/manufacturing moats — so the durable chokepoints are the fab/equipment layers, not CUDA."),
]

def main() -> int:
    tid = (sys.argv[1] if len(sys.argv) > 1 else "t6").lower()
    dirs = sorted(glob.glob(f"runs/{tid}-*"))
    if not dirs:
        print(f"no run dir matching runs/{tid}-*"); return 1
    outdir = dirs[0]
    data = json.load(open(os.path.join(outdir, "notes.json"), encoding="utf-8"))

    notes = [n for n in data["notes"] if n.get("round") != 99]      # drop any prior manual additions
    analogies = [a for a in data.get("analogies", []) if a.get("round") != 99]
    nid = max((n["id"] for n in notes), default=0)
    aid = max((a["id"] for a in analogies), default=0)

    for t, claim, detail in MANUAL_NOTES:
        nid += 1
        notes.append({"id": nid, "type": t, "claim": claim, "detail": detail, "round": 99})
    for case, what, sim, diff, impl in MANUAL_ANALOGIES:
        aid += 1
        analogies.append({"id": aid, "case": case, "what_happened": what,
                          "similarities": sim, "differences": diff, "implication": impl, "round": 99})

    out = {**data, "notes": notes, "analogies": analogies}
    path = os.path.join(outdir, "notes.enriched.json")
    json.dump(out, open(path, "w", encoding="utf-8"), indent=2)
    print(f"wrote {path}")
    print(f"  notes: {len(notes)} (+{len(MANUAL_NOTES)} manual)  |  analogies: {len(analogies)} (+{len(MANUAL_ANALOGIES)} manual)")
    return 0

if __name__ == "__main__":
    sys.exit(main())
