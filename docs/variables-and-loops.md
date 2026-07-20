# Reference — all global variables & all loops (including un-prioritised)

*42 merged variables · 10 loops · assembled from composed-graph.json + systems-map.json. ⭐ = spine · ◆ = cross-cutting · ⚙ = gate-tracked · ↻ = appears in a loop.*

## Loops (R = reinforcing, B = balancing / brake)

### Reinforcing (R)
- **R1 · RSI takeoff (recursive self-improvement)** _[theoretical · both]_
    - nodes: GV11 → GV10 → GV9
    - gauge: METR RE-bench: AI-agent vs human-expert on frontier ML R&D · flip: AI ≥ human expert on 32h+ ML-R&D horizons, sustained — at which point the loop self-closes
- **R2 · Compute–capex flywheel** _[analogical · training]_ · **DOMINANT**
    - nodes: GV11 → GV15 → GV16 → GV1
    - gauge: Combined hyperscaler capex; NVIDIA DC revenue; TSMC CoWoS capacity · flip: Demand realization (GV14) fails to validate ROI — revenue-per-token stays below infra amor
- **R3 · Commoditization / diffusion flywheel** _[empirical · inference]_ · **DOMINANT**
    - nodes: GV10 → GV13 → GV14 → GV11
    - gauge: GV10: $/capability, algorithmic-efficiency halving time, open-weight Elo gap, di · flip: An RSI/decisive-capability threshold (GV11) crosses BEFORE commoditization completes → win
- **R4 · Chip-chokepoint / export-control moat** _[empirical · training]_
    - nodes: GV1 → GV11 → GV27 → GV4
    - gauge: GV4: Entity-list growth, NVIDIA China share, Singapore transshipment, smuggled-G · flip: China indigenizes EUV at volume (GV2 prototype→production, ~2026-28 window) OR efficiency/
- **R5 · Talent concentration flywheel** _[empirical]_
    - nodes: GV12 → GV11 → GV15
    - gauge: GV12: US elite-researcher work-location share, stay-rate, comp packages, green-c · flip: Stay-rate + H-1B/green-card friction crosses the point where net elite-talent flow reverse
- **R6 · Deployment / implementation flywheel (Kai-Fu Lee)** _[empirical · inference · asymmetric — favours the scaled, fast, low-friction deployer]_
    - nodes: GV13 → GV14 → GV15 → GV13
    - mechanism: once models commoditize, deployment velocity → adoption/demand → proprietary usage-data & value capture → reinvestment → more deployment. Lee's "age of implementation": favours China (data, execution, techno-utilitarian state, physical-AI manufacturing ~50-60% of global robot installs). Note the US still keeps the standards/distribution rent (CUDA, cloud) — a two-tier split.
    - gauge: China open-weight token/derivative share; China robot/physical-AI installs · flip: US closes the deployment-friction gap (regulation/liability) or China's data flywheel stalls
    - _(added manually to match the submission; systems-map.json still lists 10 loops — re-run run_systemsmap.py to regenerate with R6)_

### Balancing / brakes (B)
- **B1 · Demand/ROI capital-fragility ceiling** _[empirical · both]_
    - nodes: GV16 → GV1 → GV14
    - gauge: GV14 (no killer app, revenue-per-token) + GV16 (neocloud/SPV fragility, circular · flip: Revenue-per-token < infra amortization sustained ~2-3 quarters at scale → neocloud impairm
- **B2 · Power/grid execution ceiling** _[empirical · both]_ · **DOMINANT**
    - nodes: GV11 → GV6 → GV13
    - gauge: GV6: interconnection queue, request-to-COD time, LPT lead time, DC demand, HV tr · flip: Interconnection reform + behind-the-meter gas/nuclear + SMRs clear the queue faster than d
- **B3 · Minerals counter-lever MAD** _[empirical · both]_
    - nodes: GV4 → GV8 → GV6
    - gauge: GV8: rare-earth processing share, gallium price, magnet-export switch, US transf · flip: Ex-China processing reaches ~30-40% (removes the lever, ~post-2028) OR US chip control bec
- **B4 · Fiscal-dominance / bond-market loop** _[empirical]_
    - nodes: GV39 → GV37 → GV36 → GV42
    - gauge: GV39 r* drift, GV37 dollar/UST bid, GV38 plumbing stress, GV36 CB-independence · flip: AI productivity offset (GV40) arrives contemporaneously → deficits self-liquidate and brea
- **B5 · Labor reinstatement / underconsumption brake** _[empirical · both]_
    - nodes: GV11 → GV17 → GV18 → GV14
    - gauge: GV17 reinstatement ratio, GV18 wage-share/appropriability, GV13 diffusion lag · flip: Reinstatement ratio <1 sustained + credit buffer exhausts + MPC differential bites → aggre

## All 42 global variables (spine + un-prioritised)

| ID | Variable | Leverage | Flags | In loops |
|---|---|---|---|---|
| GV1 | Leading-edge logic fabrication (sub-5nm capacity & indigeniz | high | ⭐◆⚙↻ | R2, R4, B1 |
| GV2 | EUV / EDA semiconductor tooling chokepoint (gate behind the  | high | ⭐⚙ | — |
| GV3 | Advanced packaging (CoWoS) + HBM | medium | — | — |
| GV4 | Chip-access export-control regime (BIS controls + coalition  | high | ◆⚙↻ | R4, B3 |
| GV5 | Software/platform moat (CUDA / cluster interconnect / EDA-so | medium | — | — |
| GV6 | Power & grid execution (electrons, transformers, interconnec | high | ⭐◆⚙↻ | B2, B3 |
| GV7 | Energy-cost trajectory (abundance vs J-curve) | high | — | — |
| GV8 | Critical-minerals & grid-equipment refining (China counter-l | high | ⭐⚙↻ | B3 |
| GV9 | Scaling axis & data wall (where capability originates) | high | ↻ | R1 |
| GV10 | Efficiency compression & open-weight commoditization (the di | high | ⭐◆⚙↻ | R1, R3 |
| GV11 | Decisive-capability threshold & its timing (master clock: TA | high | ⭐◆⚙↻ | R1, R2, R3, R4, R5, B2, B5 |
| GV12 | Frontier AI talent concentration & mobility | high | ◆⚙↻ | R5 |
| GV13 | Diffusion / deployment velocity (capability-to-deployment ga | high | ↻ | R3, B2 |
| GV14 | Demand realization / killer-app ROI | high | ↻ | R3, B1, B5 |
| GV15 | Value-capture locus (where durable rent accrues) | high | ⭐◆⚙↻ | R2, R5 |
| GV16 | AI-buildout capital structure & financial fragility | medium | ↻ | R2, B1 |
| GV17 | Labor reinstatement ratio (task creation vs displacement) | high | ↻ | B5 |
| GV18 | Wage-appropriability & aggregate-demand feedback (distributi | high | ↻ | B5 |
| GV19 | US–China power-transition trajectory | high | — | — |
| GV20 | Security-fragmentation / bloc structure (weaponization of in | high | — | — |
| GV21 | Taiwan/TSMC discontinuity & Beijing's action calculus | high | ⭐◆⚙ | — |
| GV22 | Great-power war scope, nuclear firebreak & exogenous chokepo | high | — | — |
| GV23 | Supply-chain integration floor (stickiness, agglomeration, r | high | — | — |
| GV24 | Trade-order anchor / rule-writer | medium | — | — |
| GV25 | Institutional adaptation capacity (WTO / dispute settlement  | high | — | — |
| GV26 | Swing-state / connector alignment & Global-South standards c | medium | — | — |
| GV27 | Great-power auction / second-pole credibility (China as outs | high | ↻ | R4 |
| GV28 | Gulf neutrality-premium / dual-stack hub viability | medium | — | — |
| GV29 | Managed-trade fiscal & state-capacity sustainability | medium | — | — |
| GV30 | AI regulation, liability & standards regime | medium | — | — |
| GV31 | Training-data legality / IP regime | high | — | — |
| GV32 | Political backlash channel, coalition speed & national-secur | high | — | — |
| GV33 | Antitrust / structural intervention capacity | medium | — | — |
| GV34 | State-capture / control structure of decisive AI advantage | medium | — | — |
| GV35 | Safety-catastrophe clampdown regime | medium | — | — |
| GV36 | Central-bank independence vs fiscal-dominance regime | high | ↻ | B4 |
| GV37 | Dollar reserve status / marginal UST demand ('who buys the b | high | ↻ | B4 |
| GV38 | Treasury-market plumbing / non-linear stress capacity | high | — | — |
| GV39 | Neutral real rate (r*) / global loanable-funds balance | medium | ↻ | B4 |
| GV40 | AI productivity offset (macro TFP magnitude × timing) | high | — | — |
| GV41 | Inflation transmission / distributional channel | medium | — | — |
| GV42 | Political resolution path (who bears the adjustment) | high | ↻ | B4 |

## Un-prioritised — not in spine, not gate-tracked, in no loop (21)

- **GV3 · Advanced packaging (CoWoS) + HBM** _(leverage medium)_ — The binding NEAR-TERM chip sub-constraint, more than wafer starts: TSMC CoWoS ~85-90%, HBM an SK Hynix-dominat
- **GV5 · Software/platform moat (CUDA / cluster interconnect / EDA-software / interface protocols)** _(leverage medium)_ — Layer-by-layer variance: CUDA + NVLink/InfiniBand interconnect remain sticky (17-yr head start, 30-60% efficie
- **GV7 · Energy-cost trajectory (abundance vs J-curve)** _(leverage high)_ — Split: solar+storage LCOS already sub-$50/MWh in prime sites, but SMRs 10-15yr out and transmission (GV6) seve
- **GV19 · US–China power-transition trajectory** _(leverage high)_ — Between acute parity conflict and early peak-China deceleration. The trade war is epiphenomenal to the transit
- **GV20 · Security-fragmentation / bloc structure (weaponization of interdependence)** _(leverage high)_ — At strategic-sector carve-outs escalating toward pervasive tech-stack bifurcation, trending multipolar not cle
- **GV22 · Great-power war scope, nuclear firebreak & exogenous chokepoint shocks** _(leverage high)_ — Gray-zone baseline. War is a superposition that transforms the probability space of every other discontinuity 
- **GV23 · Supply-chain integration floor (stickiness, agglomeration, rerouting)** _(leverage high)_ — Between China+1 partial and rerouting-only: trillions in sunk MNC costs and agglomeration lock-in plus friend-
- **GV24 · Trade-order anchor / rule-writer** _(leverage medium)_ — Between US-led managed-mercantilist and no-liberal-anchor: the architect has affirmatively abandoned comparati
- **GV25 · Institutional adaptation capacity (WTO / dispute settlement / plurilaterals)** _(leverage high)_ — Rump-but-functional trending zombie: Appellate Body dead since 2019, but MPIA ~53 members, first new multilate
- **GV26 · Swing-state / connector alignment & Global-South standards contest** _(leverage medium)_ — Connector equilibrium with durable non-alignment and INCREASING hedging (India, Brazil, Vietnam, Gulf). Global
- **GV28 · Gulf neutrality-premium / dual-stack hub viability** _(leverage medium)_ — Operative and partially self-reinforcing — Gulf hosts US hyperscalers (energy/land hunger inverts dependence),
- **GV29 · Managed-trade fiscal & state-capacity sustainability** _(leverage medium)_ — Between self-limiting-transitional and rent-capture: strategic-trade theory gives managed trade a welfare rati
- **GV30 · AI regulation, liability & standards regime** _(leverage medium)_ — Fragmented and contingent, with the EU actively HOLLOWING its own AI Act (Draghi report, digital-omnibus, CoP 
- **GV31 · Training-data legality / IP regime** _(leverage high)_ — Contested and litigated: US trending fair-use + emerging bilateral licensing; EU already has an ex-ante opt-ou
- **GV32 · Political backlash channel, coalition speed & national-security override** _(leverage high)_ — Organized labor largely absent; safety/accountability frame outpolls jobs frame and is cross-partisan. Nationa
- **GV33 · Antitrust / structural intervention capacity** _(leverage medium)_ — Mostly ineffective against AI velocity: investigations run 12-24mo, remedies 3-7yr, vs 12-18mo model generatio
- **GV34 · State-capture / control structure of decisive AI advantage** _(leverage medium)_ — Currently private-lab with de facto hyperscaler leverage (compute + distribution is the binding cost input). U
- **GV35 · Safety-catastrophe clampdown regime** _(leverage medium)_ — No durable brake currently plausible; most likely triggers on a single high-salience acute event, not gradual 
- **GV38 · Treasury-market plumbing / non-linear stress capacity** _(leverage high)_ — Fragile: SLR/GSIB constraints cap dealer warehousing exactly as issuance grows; ON-RRP buffer drained; basis t
- **GV40 · AI productivity offset (macro TFP magnitude × timing)** _(leverage high)_ — Sits in the lag/uncertain band: unit costs falling in AI-exposed sectors but economy-wide TFP not yet visible;
- **GV41 · Inflation transmission / distributional channel** _(leverage medium)_ — Bifurcated regime most likely: AI rents concentrate in equity without CPI diffusion, goods deflate via China o