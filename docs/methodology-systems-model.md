# Systems-Model Methodology — turning notes into macro predictions

*How we get from a corpus of notes/analogies + per-topic decision graphs to a Dalio-shaped
macro read, without (a) collapsing into status-quo derivatives or (b) combinatorial explosion.
This is both the reasoning we settled on and the spec for the next stage (`run_systemsmap.py`).*

---

## The problem we're solving

Two failure modes bracket the design:
- **Status-quo derivatives.** Track each factor's local trend and you get linear extrapolations — "this continues." That's the status-quo bias the clearing stage fell into.
- **Combinatorial explosion.** Model every interaction between every factor and it's unusable (~N² edges), and mostly noise.

And a framing tension: **Dalio's macro-cycles are powerful but rely on a rich history** (empires rising/falling). AI's core dynamics may be genuinely new — so a pure "Dalio for AI" over-claims historical basis.

## Principle 1 — Reference-class each loop, not "AI"

The error is choosing the reference class at the level of *"AI transitions"* (no history). Decompose the system and almost every loop has a deep reference class:

| Loop / mechanism | Reference class (historical basis) |
|---|---|
| chip / fab catch-up | semiconductor + DRAM generational races |
| buildout finance cycle | every technological revolution's boom/bust |
| export controls | COCOM, strategic embargoes, the oil weapon |
| **labor displacement** | **mechanization, electrification, computerization** |
| power-transition friction | naval races, Thucydides, nuclear standoff |
| recursive capability takeoff (RSI) | **none — genuinely reference-class-free** |

So "AI is new" is true of **one** loop, not the system. **Ground the loops that have history; quarantine the one that doesn't.**

## Principle 2 — Two grounding modes, explicitly tagged

Every loop/edge carries:
- **basis**: `empirical` (observed across N cases → real base rate) · `analogical` (1–2 cases + stated disanalogy) · `theoretical` (mechanism only, no cases).
- **activation**: `unconditional` (always on) · `conditional-on-regime` (fires only inside a branch).

This lets a theoretical takeoff loop coexist with an empirical finance-cycle loop, each weighted honestly.

> **DECISION 1 (branch-only theory):** theoretical/conditional loops are **quarantined to their own regime branch**. They never move the modal/base-case call — they only operate inside the branch they define, carried with a probability.

## Principle 3 — The prediction lives in thresholds, not derivatives

A derivative is a linear local extrapolation; it structurally says "continues." Nonlinearity — hence non-status-quo content — comes from only two places: **feedback loops** (reinforcing loop → acceleration/phase-change) and **thresholds** (a balancing loop breaks, or a stock crosses a level and a loop flips sign/dominance).

- Derivatives (the gate-tracker metrics) are **gauges** — they tell you *where each loop currently sits*.
- The **forecast** is the map of **thresholds** and *which one trips next* + the resulting phase/shape.
- **Test for a loop earning its place:** if it has no threshold where it changes sign or dominance, it adds nothing. Drop it.

## Principle 4 — Tame combinatorics with structure (never compute N²)

1. **Feedback is sparse.** Chains don't produce sustained dynamics — only interactions that *close into loops* do (we found 9 cycles among 42 nodes, not ~1,700 edges). Enumerate loops, not pairs.
2. **Dominance (Liebig's minimum).** At any moment only 2–3 loops are *in charge*; track the dominant loop and what hands dominance to another. O(loops), not O(pairs). (This is Dalio's "which force is running the show.")
3. **Theory as a prior over edges.** Mechanism (economics, IR, Bostrom) gates which edges are even plausible — pruning the graph before scoring.

## The model — one substrate + discrete branches

> **DECISION 2 (discrete regimes):** model a small set of discrete regime-**branches**, each with its own active-edge graph and dominant loops — not one graph with continuous gains.

**Substrate (always-on, historically grounded).** The political-economy machinery of a technology shock, all `empirical`/`analogical`:
- **R1 · Buildout flywheel** — capex → compute → demand/ROI → capex
- **B1 · Diffusion brake** — any lead → efficiency + open weights copy it → lead erodes
- **B2 · Physical brake** — demand → power/chip/EUV limits → delay caps buildout
- **R3 · Mercantilist spiral** — controls → retaliation → decoupling → more controls
- **B3→R4 · Labor–demand loop** — deployment → displacement → wage/demand loss → (if transfers lag) demand destruction undercuts ROI. *Grounded in mechanization/automation history — NOT quarantined (see Decision 3).*
- **B4 · Monetary brake** — buildout + fiscal dominance → UST demand/rates → financing ceiling

**Branches (discrete regimes):**
- **A · Base / diffusion-deployment** *(modal)* — substrate runs its course; two-tier world; value migrates to deployment; RSI loop dormant.
- **B · RSI takeoff** *(theoretical, branch-only, probability-weighted)* — the capability loop's gain crosses the diffusion brake → decisive-advantage / winner-take-all dynamics; the widening physical gaps become decisive. **This is the only quarantined branch.**
- **C · Buildout bust / Perez turning-point** *(historical)* — demand/ROI fails to arrive before financing tightens → frenzy → crash → institutional recomposition (railways, dotcom).
- **D · Exogenous fracture** *(tail)* — Taiwan / great-power war shock zeroes inputs.

> **DECISION 3 (quarantine scope):** the **only** reference-class-free novelty is **RSI takeoff (branch B)**. Labor/cognitive substitution is grounded in the mechanization reference class and lives in the substrate (B3→R4), *not* quarantined.

## The three scaffolds

- **Carlota Perez** (*Technological Revolutions and Financial Capital*) — the finance↔new-tech investment cycle: irruption → **frenzy/installation** (financial-capital-led bubble) → **turning point** (crash + recomposition) → **synergy/deployment**. Places today's buildout in *frenzy* → predicts a turning-point/shakeout → deployment. **Historical.**
- **Ray Dalio** (*Changing World Order*) — the empire/debt/reserve-currency Big Cycle the tech cycle runs inside. **Historical.**
- **Nick Bostrom** (*Superintelligence*) — takeoff = loop gain; decisive strategic advantage = reinforcing loop outrunning the balancing (diffusion) loop. **Theoretical — branch B only.**

Macro model = **Perez × Dalio substrate (historical) + Bostrom driver (theoretical, branch B) + the coupling.**

## Three registers per loop

Every loop binds three registers — keep them distinct:
- **Theory-template** — the generating pattern (Perez / Dalio / Bostrom / Acemoglu-Restrepo). First-class object: name + its loops + phases + base-rate shape. *(Currently missing — must be added.)*
- **Historical instances** — the analogy corpus (458 analogies): the empirical cases that *validate* the template. *(Have it.)*
- **Live gauges** — the gate-tracker metrics that *locate us now* in the pattern. *(Have it.)*

An analogy is an *instance*; a theory is the *template* that instance validates. A loop points to all three.

## Composition — the loops/theories are ONE coupled system, not separate pieces

Composition is a **deliverable**, not an afterthought. Four mechanisms:
1. **Shared nodes → one network.** Loops overlap on variables (buildout-flywheel & labor-loop both act on *demand/ROI*; diffusion-brake & RSI-loop both act on *capability*; flywheel & physical-brake both act on *compute*). A change circulating one loop enters others through the shared node. Composition = the union graph, coupled at shared variables.
2. **Dominance arbitration at contested nodes.** Where two loops push a shared node opposite ways, the NET direction = whichever has the larger *current gain* (read from gauges). Dalio's "which force is in charge," node-by-node.
3. **Theories nest by TIME-SCALE — nested, not averaged.** Dalio (slow: empire/debt/reserve-currency) sets the boundary conditions inside which Perez (medium: tech-finance cycle) runs, inside which Bostrom (fast: capability loop) may break out. Outer layer = the constraint/context for the inner.
4. **Branches compose in two steps.** *Within* a branch: active loops compose via shared-node dominance into that branch's shape. *Across* branches: hold a probability-weighted fan — never average (Decision 1).

**Requirement:** the systems map must emit a **coupling map** — shared nodes + cross-loop edges + the dominance rule at each contested node — not just a list of loops.

## Forecast procedure

1. **Read the gauges** (gate-tracker metrics) → locate each substrate loop's current strength.
2. **Identify the dominant loops** → the current **phase** (Perez/Dalio idiom).
3. **Name the thresholds** that would switch branches (e.g. RSI gain > diffusion gain → branch B; demand-ROI fails before financing tightens → branch C; Taiwan shock → branch D).
4. **Forecast = the shape** of the modal branch (no dates) + the **probability-weighted branches** + the **watch-thresholds**. Tails are just branch triggers.

## Design implications for `run_systemsmap.py` (the next stage)

- **No re-run of compose (step 3).** Reuse `bench/composed-graph.json`'s 42 merged variables + 64 edges as node vocabulary and simply **ignore the `spine` field** — spine-selection was the only discarded step; the merge is fine plumbing. Nothing is thrown away.
- Extract **loops** (seed from the 9 real cycles + let the model add the feedback edges the dependency-first stages missed), each tagged **basis** + **activation**, with a **gauge** (from `gate-tracker.json`), **dominant_now**, and **flip_threshold**.
- Define the **discrete branches** (A base / B RSI-takeoff / C bust / D fracture), each with its active-loop set; **branch B loops are conditional and never touch the base case**.
- Cluster loops → Dalio-style **driving forces**; make **theory-templates** first-class (Perez/Dalio/Bostrom/Acemoglu) and bind each loop to its template + instances + gauge.
- Emit the **coupling map**: shared nodes, cross-loop edges, and the dominance rule at each contested node (this is what makes them one system, not separate pieces).
- **Absorbs the retired step 6 (cross-implications).** Two carry-overs: (i) a **coupling-finding checklist** covering the five second-order moves — *substitution, scope_of_constraint* (forces the training/inference node split), *value_locus_shift, asymmetric_adoption, compounding*; (ii) an **adversarial-verdict attribute** on each load-bearing coupling/threshold — `evidence_for` + `steelman_against` + `probability` — instead of a separate generative stage. `run_crossimplications.py` stays on disk, repurposable later as a thin *post-hoc* verifier of the load-bearing joints if step 7's couplings feel under-tested (do NOT pre-build).
- Output: **substrate loops + coupling map + branches + current phase + watch-thresholds + probability-weighted trajectory** — a systems machine, not a ranked list.

## Pipeline (current)

```
0 Topics → 1 Explore (notes + 458 analogies = INSTANCES)
2 Decision-graph per topic  [run_prototype.py]
3 Compose → 42 vars + 64 edges  [run_compose.py]   ── reuse vars/edges; IGNORE spine; no re-run
4 Gate tracker (5–7 metrics × ≥5 pts, web-grounded = GAUGES)  [run_gatetracker.py]
5 Auto-charter (dashboard per gate)  [make_gate_charts.py]
7 Systems-map (loops + coupling map + branches + theory TEMPLATES + phase + trajectory)  [run_systemsmap.py]  ← builds on 2/3/4
  → outputs: Part 2 framework, Part 1 forecasts, Part 3 appendix
```
Retired: **step 3 spine-selection** (merge kept), **step 6 cross-implications** (folded into 7; code retained). The three registers line up: instances = 1, gauges = 4, templates = 7.
