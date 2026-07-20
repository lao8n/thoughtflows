# Bench — topic generation (Stage 0)

## Pillars (run pure) (4)

### T1. Do AI scaling returns (compute → frontier capability) hold through 2028?
_Foundational capability question that sits upstream of every mercantilist bet. If the scaling regime breaks, the strategic value of compute chokepoints, capex, and chip controls all reset — so it must be run pure as the load-bearing assumption the rest of the bench leans on._
- axis: AI layer: frontier models/compute, lever: none (pure capability), actor: frontier labs
- candidate forecasts:
  - Will at least one publicly released model score ≥90% on GPQA-Diamond by Dec 31 2027?
  - Will the largest publicly disclosed training run exceed 1e27 FLOP by Dec 31 2028?
  - Will a frontier lab publicly state that pretraining-scaling returns have plateaued by Dec 31 2027?

### T2. Does the 2024–2028 AI capex wave (~$1T) earn its ROIC, or trigger a hyperscaler investment reversal?
_The financial sustainability spine. A capex bust (or proof of circular vendor-financing) would unwind sovereign industrial-policy bets and reorder the whole field; absorbs the circular-financing and NVIDIA-rent candidates as sub-questions._
- axis: AI layer: compute/infrastructure economics, lever: capital markets, actor: hyperscalers + frontier labs
- candidate forecasts:
  - Will any of Microsoft, Google, Amazon, or Meta report a year-over-year decline in AI-related capex in any fiscal year through 2027?
  - Will a major hyperscaler take a write-down ≥$5B explicitly attributed to AI infrastructure by Dec 31 2028?
  - Will NVIDIA datacenter-GPU gross margin fall below 60% in any reported quarter through 2028?

### T3. Is the post-Washington Consensus industrial-policy era structurally durable or a cyclical overshoot?
_The core mercantilism pillar independent of AI specifics: whether the subsidy-and-tariff paradigm survives fiscal stress and political turnover. Sets the base rate for every crossed industrial-policy intersection on the bench._
- axis: lever: industrial subsidy + tariffs, actor: US/EU/China states, AI layer: n/a (paradigm)
- candidate forecasts:
  - Will the US CHIPS Act remain unrepealed and funded through Dec 31 2027?
  - Will the US, EU, and China each have active semiconductor/AI subsidy programs as of Dec 31 2027?
  - Will any G7 government announce a material rollback of an AI/chip subsidy program citing fiscal pressure by Dec 31 2028?

### T4. Does AI white-collar automation hit disruptive workforce scale by 2028, pivoting AI politics to protectionism?
_The labor/political-economy swing variable. Whether AI substitutes or complements labor determines if a worker-protection protectionist coalition forms in advanced economies — the domestic-politics hinge for the entire mercantilist narrative._
- axis: AI layer: applications/agents, lever: labor protectionism, actor: advanced-economy electorates
- candidate forecasts:
  - Will US BLS data show a net decline in computer-programmer (SOC 15-1251) employment between 2025 and 2027?
  - Will any G7 government enact AI-specific worker-displacement legislation (automation tax, layoff restriction, retraining mandate) by Dec 31 2028?

## Intersection topics (run crossed) (7)

### T5. Do US export controls on advanced AI chips decide the frontier race vs. Chinese challengers?
_The single most central Mercantilism×AI intersection: the chip-compute-capability chokepoint theory tested directly. Crosses export-control lever × compute layer × US/China actors; absorbs the China-domestic-fab candidate as the other side of the same question._
- axis: lever: export controls, AI layer: chips/compute, actor: US vs China
- candidate forecasts:
  - Will US BIS advanced-AI-chip controls on China be net-tightened or held (no net loosening) through Dec 31 2027?
  - Will a Chinese-developed model rank in the global top 5 on Chatbot Arena at any point in 2027?
  - Will Huawei Ascend or SMIC ship a ≤5nm-class AI accelerator at scale by Dec 31 2028?

### T6. Does open-weight model release (Llama, Qwen, DeepSeek, Mistral) nullify US export controls?
_The strongest refutation lens against T5. If frontier capability diffuses via weights independent of hardware access, the entire chip-chokepoint theory weakens. Crosses model layer × export-control lever; pairs the inference-scaling-decoupling candidate as a second arbitrage path._
- axis: AI layer: open-weight models, lever: export-control arbitrage, actor: Meta/Alibaba/China labs
- candidate forecasts:
  - Will an open-weight model rank within the top 10 on Chatbot Arena as of Dec 31 2027?
  - Will Meta, Alibaba, or DeepSeek release open weights within 6 months of frontier-benchmark parity at any point in 2027?
  - Will the US impose controls specifically restricting open-weight model release by Dec 31 2028?

### T7. Is Taiwan/TSMC the decisive AI-compute chokepoint, and does a kinetic shock dwarf fab diversification?
_The dominant tail-risk: a step-function compute shock that renders incremental diversification moot. Crosses resource-concentration lever × fab layer × US/China/Taiwan; folds the TSMC-concentration and wartime-commandeering candidates into one chokepoint topic._
- axis: lever: supply-chain concentration/kinetic risk, AI layer: advanced fabrication, actor: Taiwan/US/China
- candidate forecasts:
  - Will a blockade or kinetic conflict disrupt TSMC production at any point before Dec 31 2028?
  - Will TSMC Arizona produce ≥15% of TSMC's leading-edge (≤3nm) output by Dec 31 2027?

### T8. Does cheap scalable power displace chips as the binding AI-compute constraint by 2027?
_Covers the energy layer, which the chip-centric debate underweights. If power, not silicon, is the chokepoint, mercantilist competition shifts to energy assets. Crosses energy lever × compute-infrastructure layer × US/Gulf/China._
- axis: lever: energy/resource control, AI layer: datacenter power, actor: US/Gulf states + utilities
- candidate forecasts:
  - Will at least one announced AI datacenter ≥1GW be publicly delayed or cancelled citing grid/power constraints by Dec 31 2027?
  - Will a hyperscaler bring ≥1GW of dedicated nuclear (restart or SMR) capacity online for AI compute by Dec 31 2028?

### T9. Does China's critical-mineral processing chokehold give durable AI supply-chain leverage even if mining diversifies?
_Captures resource nationalism at the refining stage — the durable chokepoint distinct from extraction. Crosses resource-nationalism lever × hardware-materials layer × China; the answer-changing variable for whether mining diversification actually de-risks anything._
- axis: lever: resource nationalism (processing), AI layer: hardware materials, actor: China
- candidate forecasts:
  - Will China maintain export restrictions on gallium, germanium, or rare-earth processing technology through Dec 31 2027?
  - Will non-China gallium or germanium refining capacity exceed 30% of global output by Dec 31 2028?

### T10. Can Gulf sovereign capital build a third AI-compute pole, and does inbound sovereign capital create insider leverage over the US stack?
_Brings in a non-US/China actor and the financial-statecraft-via-capital vector that CFIUS/export controls under-address. Crosses sovereign-capital lever × compute layer × Gulf actor; folds the sovereign-capital-insider-leverage candidate._
- axis: lever: sovereign capital/financial statecraft, AI layer: compute infrastructure, actor: Gulf states (G42/PIF/Humain)
- candidate forecasts:
  - Will a Gulf entity operate a frontier-scale (≥100k H100-equivalent) compute cluster by Dec 31 2027?
  - Will a US frontier lab (OpenAI, Anthropic, xAI) accept Gulf sovereign-fund equity by Dec 31 2027?

### T11. Does defense/IC demand and civil-military fusion become the decisive selection pressure on which AI labs and architectures survive?
_Covers the military/national-security layer — a large candidate cluster — in one decisive topic: whether Pentagon/PLA demand, not commercial SaaS, anchors frontier-lab economics, and whether China's MCF is a structural asymmetry controls can't reach. Crosses defense-procurement lever × applications layer × US/China._
- axis: lever: defense procurement/civil-military fusion, AI layer: frontier models/applications, actor: US DoD-IC vs PLA
- candidate forecasts:
  - Will US DoD/IC award ≥$500M cumulative in contracts to frontier labs (OpenAI, Anthropic, Google DeepMind) by Dec 31 2027?
  - Will the US invoke the Defense Production Act or designate AI compute clusters as critical defense infrastructure by Dec 31 2028?

## Rejected (45) — breadth of consideration

- **Is the frontier model layer commoditizing, shifting value to distribution/applications?** — Value-capture nuance subsumed by T2's economics and T5/T6's market-structure questions; not separately answer-changing.
- **How fast does agentic AI diffuse into enterprise (SaaS vs infrastructure S-curve)?** — Adoption-speed detail; the politically decisive form is captured by T4 (labor) and T2 (monetization).
- **Is weaponized financial statecraft (dollar sanctions, SWIFT) losing deterrent potency?** — Important lever but largely AI-orthogonal; AI-relevant slice folded into T10 (sovereign capital).
- **Does the EU AI Act + data-localisation function as de facto protectionism?** — EU is a regulator/taker, not a frontier-determining actor; consciously skipped to preserve compute-race depth.
- **Does data-localisation shatter the AI application layer into national gardens?** — Duplicative of the EU/India regulatory-garden theme; not decisive for who wins the frontier.
- **Will AI productivity divergence reorder the mercantilist balance of power by 2030?** — Broad outcome variable rather than a testable lever; effectively the sum of the bench, not a slot.
- **Will China lock AI standards across the Global South via ISO/ITU/BRI?** — Standards-warfare lever is slow-moving and normative; consciously skipped as second-order to compute control.
- **Can India build a credible third AI pole (IndiaAI, DPDP, UPI stack)?** — India is currently an import-substitution actor without frontier leverage; consciously skipped vs Gulf (T10) as third-pole proxy.
- **Will competition extend to physical AI (humanoids, AVs, drone swarms) and trigger new controls?** — Embodied-AI export-control frontier is largely post-2028; partially captured by T11's autonomous-weapons demand.
- **Will sovereign national-champion model programs (Falcon, Mistral, Saudi) succeed?** — Effectiveness sub-question of T3 (industrial policy) and T10 (Gulf); duplicative.
- **Does AI surveillance/social-scoring export harden an authoritarian AI bloc?** — Normative-bifurcation theme; consciously skipped as adjacent to standards warfare, not compute-race decisive.
- **Is AI infrastructure revenue structurally circular (vendor financing)?** — Folded into T2 as a sub-forecast on capex fragility.
- **Is NVIDIA's GPU monopoly rent durable through 2028?** — Folded into T2 (margin-compression forecast); not a standalone slot.
- **Does inference-scaling (test-time compute) obsolete chip export controls?** — Same refutation logic as T6 (open weights); folded there to avoid duplication.
- **Do capex write-downs strand sovereign industrial-policy bets?** — Second-order coupling of T2 (capex bust) × T3 (policy durability); emerges from running both, no separate slot.
- **Does competitive subsidy racing reach balance-sheet-threatening scale, forcing defection?** — Fiscal-endgame variant of T3; consciously skipped as a sub-scenario.
- **Does AI capex sustain high real rates, crowding out fiscal capacity for industrial policy?** — Macro feedback loop derivable from T2×T3; too indirect to be objectively resolvable as its own forecast.
- **Does AI supply-chain concentration create 'too-interconnected-to-fail' systemic risk?** — Contagion framing overlaps T2 and T7; consciously skipped as a downstream consequence.
- **Does USD/capital-market access weaponization give the US a second AI chokepoint?** — US-offense financial lever; partially in T10, otherwise consciously skipped to keep the bench at depth.
- **Will AI-agent-mediated trade/procurement create a new mercantilist intermediary chokepoint?** — Speculative and largely post-horizon; immaterial within the 2026–2028 window.
- **Will defense procurement militarise the AI stack as the decisive selection pressure?** — Core claim retained as T11; this phrasing duplicative.
- **Do AI-enabled autonomous lethal weapons become the dominant investment rationale?** — Folded into T11's defense-demand forecasts.
- **Does Taiwan war-risk probability alone lock in over-investment in AI infrastructure?** — Option-value framing of T7; the kinetic and diversification forecasts already capture the decision-relevant signal.
- **Does AI in nuclear C2 create a stability crisis forcing US–China arms control?** — High-consequence but low-probability within horizon and hard to resolve objectively; consciously skipped.
- **Does AI offensive cyber make AI infrastructure a primary military target?** — Adjacent security dimension; consciously skipped as not decisive for the mercantilist supply-chain answer.
- **Is Wassenaar/legacy arms-export architecture inadequate for autonomous-weapons proliferation?** — Governance-adequacy question downstream of T11; immaterial to the compute-race core.
- **Would governments commandeer NVIDIA/hyperscaler capacity in a Taiwan conflict?** — Folded into T7 as wartime-override risk.
- **Does AI intelligence superiority make frontier capability fungible with military power?** — Pillar framing of the arms-race claim; captured operationally by T11.
- **Does 'compute-as-munitions' framing act as an implicit sovereign backstop to capex?** — Sub-thesis of T2×T11; consciously skipped as derivable, not separately resolvable.
- **Does militarised industrial policy bias AI away from civilian-productivity objectives?** — Long-run divergence claim; too diffuse to resolve by 2028, consciously skipped.
- **Does AI automation foreclose the Global South export-industrialization ladder?** — Major distributional consequence but Global South is a taker, not a frontier-determining actor; consciously skipped with the trade-theory cluster.
- **Do digital services taxes become de facto AI tariffs, triggering a WTO-fracturing trade war?** — Plausible but a goods-vs-services tariff mechanism orthogonal to the compute-race answer; consciously skipped.
- **Does US services-productivity superiority widen the invisibles surplus (Triffin variant)?** — BOP-arithmetic specialist topic; immaterial to who controls the AI stack.
- **Does AI displacement of BPO/IT exports shock India/Philippines/EE current accounts?** — Distributional EM consequence within the trade-theory cluster; consciously skipped.
- **Does AI productivity divergence produce a Prebisch-Singer North–South terms-of-trade deterioration?** — Trade-theory analog, not a datable resolvable lever; consciously skipped.
- **Does AI-driven reshoring reverse GVC integration (trade-to-GDP decline by 2028)?** — Reshoring-pace question overlaps T4; the trade-flow metric is too slow-moving to resolve decisively by 2028.
- **Does 'factory-less' AI manufacturing divorce industrial policy from domestic jobs?** — Political-coalition erosion captured better by T4; duplicative.
- **Will the Global South adopt AI-specific defensive protectionism (AI tariffs, local-content)?** — Emerging-market defensive toolkit; consciously skipped as downstream of the foreclosure thesis already cut.
- **Does the AI capital-goods import surge cause twin-deficit BOP pressure in AI leaders?** — Near-term BOP mechanics overlapping T2's financing question; consciously skipped as too indirect.
- **Does AI make non-tradeable services tradeable, triggering a GATS-unequipped trade shock?** — Trade-theory cluster; significant long-run but not 2026–2028 answer-changing, consciously skipped.
- **Does 'immiserizing AI growth' make the Global South a net welfare loser?** — Bhagwati-paradox framing within the skipped trade-theory cluster; not a datable forecast.
- **Does AI-services import dependence cause Dutch Disease / premature deindustrialization?** — Final trade-theory-cluster variant; consciously skipped to preserve compute-race depth over breadth.
- **Do US hyperscalers act as primary instruments of American AI mercantilism, provoking countermeasures?** — Cloud-dominance-as-leverage theme distributed across T5, T6, and T10; not separately decisive.
- **Does hyperscaler vertical integration permanently foreclose independent frontier labs?** — Market-structure question largely answered by T2 (economics) and T10 (capital dependence); duplicative.
- **Is the global AI-researcher talent race (H-1B/O-1, comp inflation) a decisive mercantilist lever?** — Talent lever is real but consciously skipped under the ~10-slot ceiling as lower-leverage than compute/energy/minerals chokepoints.

