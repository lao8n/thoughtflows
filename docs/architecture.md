# Architecture — Forecasting Engine

> Status: design spec, pre-code. Built fresh. Last year's `archive/` (nearest-neighbour
> analogue mining) is **inspiration only** — not a codebase to extend.

## 1. What we're building

A **stateful belief engine** that produces calibrated probabilistic forecasts by holding many
competing theories in parallel, grounding each in historical track record and adversarial
evidence, and adjudicating their disagreements against an external anchor (markets / consensus).

Two horizons:

- **Near-term deliverable** — the Bridgewater × Global Citizen *Forecasting the Future 2026*
  submission (deadline **2026-08-01**). The submission is a **snapshot** of the engine's belief
  store at submission time.
- **Long-term vision** — a continuously-running system: beliefs update on incoming news, are
  constantly compared to prediction markets, and a sufficiently large, *justified* divergence
  becomes a trade signal. The near-term design is a strict subset of this; we build the parts that
  serve both and defer execution.

### Competition constraints (these are hard design inputs)

| Part | Requirement | Scored on |
|------|-------------|-----------|
| 1. Forecasts | ≥10 binary yes/no, probabilities, 1–5yr, **objective** resolution criteria | calibration + resolution accuracy |
| 2. Framework | ≤3 pages, cause-and-effect dynamics | clarity, originality, rigor |
| 3. Appendix | ~5 pages, supporting reasoning + evidence | analytical rigor |

Topics: **modern mercantilism, AI, or their intersection.** AI tools allowed *if output shows "a
sharp human point of view."* Top forecasters defend live (Stage 2 roundtable) → **every number
must be defensible from its reasoning trace.**

**How we approach the topic space.** The brief is a *bounded collision* — modern mercantilism × AI —
not "forecast the next decade," so the engine runs over a **curated bench** rather than one topic.
The bench has two altitudes: a few **pure pillars** (deep single-axis runs — pure AI *and* pure
mercantilism — for divergence and base rates; *LLM market structure is one pillar, not the whole
submission*) and a larger set of **intersection hypotheses** (explicit AI×mercantilism mechanisms,
where the framework's interaction cruxes and the binary forecasts come from). Integration is
discovered bottom-up at the cross-topic stage, not pre-baked into topic wording. Bench construction
is its own stage — see §14, **Stage 0**.

## 2. Epistemic principles (the *why* — do not lose these)

1. **A council of lenses, not one synthesized world-model.** "The model believes X" is
   unaccountable and undefendable. Each forecast traces to *named, internally-coherent theories*
   with known assumptions. We reason holistically *within* each lens, and stay structured *across*
   lenses.
2. **Structure is scaffolding the full-context model reasons over — never an assembly line that
   mechanically emits the answer.** Decomposition that exists only to help a weak model is
   obsolete (frontier models hold the whole picture). Decomposition for *epistemic discipline*
   (adversarial separation, auditability) is more valuable than ever — a stronger model is a more
   persuasive rationalizer of its own narrative, so forcing the counter-case into a separate pass
   matters more, not less.
3. **Disagreement is the signal.** Where independent lenses *converge despite different
   assumptions* → robust, high-confidence (and probably already priced in). Where they *diverge* →
   that's where the uncertainty, the originality, and the edge live. We **adjudicate cruxes, we do
   not average.** Mechanical averaging of lens probabilities is forbidden — it launders structure
   into a number accountable to nothing.
4. **Disconfirmation is first-class.** Bridgewater's *own published* reasoning (archetype + if/then,
   "better to roughly model than not model at all") is strong but skips counter-evidence and
   falsification. Our edge: use their method *plus* rigorous disconfirmation and explicit
   change-mind criteria on every node. This is exactly what "calibration + analytical rigor"
   rewards.
5. **Divergence ≠ edge.** The market/consensus is usually right; a large gap more often means we're
   missing what it knows than that we've found alpha. Before any contrarian forecast (or trade),
   a hard gate: *state explicitly what we know or weight that the consensus doesn't. If we can't
   articulate it, the divergence is our error.*
6. **Probabilities are derived, with a trace — never fabricated up front.** The number falls out
   of the adjudication (how many lenses converge, which governs *this* case, how the crux
   resolves). Last year's fatal flaw was a `weight: float` the model just emitted.

## 3. Core objects

### Theory (reusable, slowly-changing, expensive — built once, amortized across all hypotheses)

A reusable causal mechanism + its historical track record. May be **named** (Mearsheimer's
offensive realism, liberal institutionalism, power-transition, economic nationalism / List,
Schumpeterian disruption, Turchin structural-demographic) or **basic/unnamed** ("tariffs invite
proportional retaliation," "scarce compute gets rationed by price"). Unnamed is fine.

```
Theory {
  id
  statement                 # the mechanism, in one or two sentences
  assumptions[]             # core axioms it rests on (cite canonical source where named)
  domain                    # where it claims to apply
  backtest {
    pro_cases[]             # strongest historical cases where it held (mined from history)
    counter_cases[]         # strongest cases where it FAILED (equally important)
    track_record            # distilled: where it holds, where it breaks
  }
  change_mind_conditions[]  # what observation would falsify / retire this theory
  status                    # draft | converged
}
```

### Hypothesis (specific, current, fast-updating — one per forecast)

```
Hypothesis {
  id
  statement                 # binary yes/no forecast
  resolution_criteria       # objective, measurable (competition requires this)
  timeframe                 # 1–5 years
  belief {
    log_odds                # current belief (see §6 on representation)
    probability             # sigmoid(log_odds), for display
    history[]               # append-only: (timestamp, log_odds, provenance)
  }
  anchor {
    source                  # market (Polymarket/Kalshi) | Metaculus | base-rate | none
    value                   # consensus probability, if any
    divergence              # |our prob − anchor|
    edge_justification      # REQUIRED if divergence large (principle 5)
  }
  evidence_ledger[]         # Evidence nodes
  applications[]            # Application nodes (theory → this hypothesis)
  cruxes[]                  # Crux nodes
  synthesis                 # adjudicated probability + reasoning trace
}
```

### Evidence (a line item in a ledger)

```
Evidence {
  id
  claim
  direction                 # supporting | disconfirming
  llr                       # log-likelihood-ratio: how many "decibels" it shifts belief
  source                    # pointer to real source/data (or model-knowledge, flagged)
  justification             # one line
}
```

### Application (a theory's verdict on a hypothesis — this is where last year's engine lives)

```
Application {
  theory_id
  hypothesis_id
  prediction                # what this theory predicts for this case
  applicability_score       # how well current case matches this theory's pro_cases vs counter_cases
                            # ← nearest-neighbour analogue match (last year's idea, precise job now)
  reasoning
}
```

### Crux → Decision graph (the unit of synthesis)

> **Superseded (see §15).** The binary `Crux` below over-flattened synthesis (yes/no question, no actor, no interaction). The topic unit is now a **decision graph** — `Variable` (state-space + per-actor readings + trajectory), `Edge` (gates / substitutes_for / chokepoint_controlled_by / correlates), `ForecastSignal` (the binary crux, demoted to a derived probe of a variable's state). Cross-topic synthesis composes these graphs by *referent*, not by theme. Schemas in `run_prototype.py` / `run_compose.py`, pending fold-in to `pipeline.py`.

```
Crux {                        # LEGACY
  id
  competing_applications[]  # which theories disagree here
  question                  # the single fact/assumption that, if known, flips which theory wins
  resolution                # how we adjudicate it, and on what evidence
}
```

## 4. The pipeline

```
                 ┌─────────────────────────────────────────────────┐
                 │  THEORY LIBRARY  (built once, reused everywhere)  │
                 │  each theory backtested: pro_cases, counter_cases,│
                 │  change_mind_conditions  — via §5 training loop   │
                 └───────────────────────┬─────────────────────────-┘
                                         │ select relevant theories
        topics ──► hypotheses ───────────┤
        (mercantilism /                  ▼
         AI / intersection)     ┌──────────────────────┐
                                │  APPLICATION          │  each theory predicts;
                                │  per (theory,hyp)     │  weighted by analogue-match
                                └──────────┬───────────-┘  (applicability_score)
                                           ▼
                                ┌──────────────────────┐
                                │  CRUX ADJUDICATION    │  find where lenses disagree;
                                │  (NOT averaging)      │  resolve the crux; cross-examine
                                └──────────┬───────────-┘  across fields for contradictions
                                           ▼
                                ┌──────────────────────┐
                                │  ANCHOR + EDGE GATE   │  compare to market/consensus;
                                │                       │  large gap ⇒ justify edge or fold
                                └──────────┬───────────-┘
                                           ▼
                                  probability + trace
                                           ▼
                                ┌──────────────────────┐
                  news events ─►│  BELIEF STORE         │◄─ calibration log (Brier on resolution)
                  (long-term)   │  log-odds, versioned  │
                                └──────────────────────-┘
```

## 5. The per-node training loop (theories *and* hypotheses)

Every node is "battle-tested" through the same loop. This is where the **grounded termination
signal** lives.

1. State the claim.
2. Adversarially gather **supporting** evidence.
3. **Separate pass:** adversarially gather **disconfirming** evidence (no stake in the original
   claim).
4. Specify **change-mind conditions** (falsification criteria).
5. *(theories only)* Backtest: mine strongest historical **pro-cases** and **counterexamples**;
   distil track record.
6. **Converged when** another adversarial round surfaces no new *strong* evidence on either side
   **and** change-mind conditions are stable. ← stop here, not when "the model says done."

This is the only legitimate stopping rule. An LLM asked "what's a missing argument?" will always
invent one; convergence must be defined by *evidence that stops moving*, anchored in real sources.

**As implemented (explore + Stage 0).** A naive "K dry rounds" counter never fires on a rich topic —
the model always finds a marginal angle (the 176-note run proved this). So the grounded stop is
operationalized as a **saturation judge**: a deep-model call asking *"is the materially-distinct
space covered, such that further rounds would mostly restate?"* — plus **gap-targeting**: when not
saturated it names the single biggest gap, which the next round is told to attack. This **cuts depth
while protecting breadth** (it won't stop with a major angle missing, and it actively hunts the gap).
**Model tiering:** the high-volume *mining* runs on a fast model (Sonnet); the *saturation judge* and
all downstream judgment stages run on a deep model (Opus).

### Separate vs. unified loops: one graph, typed routing, dirty-propagation

Theories and hypotheses are **not** trained in isolated loops (intellectually dishonest — ignores
theory-falsifying evidence found downstream; theories never improve from use), nor in one
undifferentiated soup (every fact re-runs everything; cost explodes; legibility lost). Instead:
**one dynamically-built graph, per-node local loops, reactive dataflow between them.**

- **Typed evidence routing (the key primitive).** Evidence surfaced anywhere is classified:
  - (a) challenges a **theory's general mechanism** → update Theory, **propagate** to dependents;
  - (b) challenges **this case's match** to a theory → update that `Application.applicability_score`,
    stays local;
  - (c) bears on **this hypothesis's outcome** directly → update Hypothesis belief, stays local.
  This is the §6 "update vs. regime-change" branch generalized. A theory's
  `change_mind_conditions` are the explicit trip-wires for case (a).
- **Dirty-propagation, not global rerun.** A *material* theory change marks dependent hypotheses
  dirty; only those re-adjudicate. Incremental recomputation keeps us under the token budget.
- **Cadence.** Seed the theory library first (stable substrate), but keep edges **live** so
  downstream discoveries flow back up. Theories update on the slow path (material falsification
  only); hypotheses on the fast path (§6).
- **Guards.** Propagate only above a materiality threshold; cap propagation depth per round
  (prevents oscillation).
- **Global convergence:** no node dirty **and** no local loop finds new strong evidence.

## 6. Belief representation & updates

- Belief stored as **log-odds**; evidence as additive **log-likelihood-ratios**. Updates compose
  cleanly; the ledger is legible and auditable.
- **The log-odds sum is a diagnostic, not the authority.** The final probability is the model's
  *adjudicated* judgment with the full ledger in view (principles 3 & 6). If the mechanical sum and
  the adjudicated number diverge sharply, that's a flag to investigate — not an error to "fix" by
  picking one.
- **Two update paths (long-term system):**
  - *Fast path* — a news event proposes a log-odds **delta** + justification → nudge the belief,
    append to history. Cheap, frequent.
  - *Slow path* — periodic full re-adjudication (§4) from scratch. Catches drift the fast path
    accumulates.
  - **Regime-change branch:** before applying a fast update, ask *"is this an update to an existing
    belief, or a signal that the hypothesis/theory itself must be retired/rebuilt?"* Pure Bayesian
    updating fails silently on regime shifts (the thing geopolitics is full of).

## 7. Storage & token management

We will generate **>> 1M tokens** of analysis. The rule: **the filesystem is long-term memory, the
context window is the working set, the graph index is the addressing scheme.** No reasoning step
ever loads the whole corpus.

- **Index** (small, in-context): distilled node summaries + evidence ledgers (LLR line items) +
  pointers. This is the graph.
- **Corpus** — mostly **not stored**. Most relevant knowledge already lives in the model (§12);
  on disk we keep brief **keys** (e.g. `"Smoot-Hawley 1930 → retaliation → -66% world trade"`) that
  let the model reconstitute analysis on demand, plus the **novel deltas** it can't regenerate
  (adjudications, cruxes, LLRs). Full expansions persisted only when load-bearing *and*
  non-reproducible. Time-sensitive facts (current data/news) are the exception: stored verbatim,
  externally sourced (§12).
- Any step loads only the **handful of relevant node summaries** it needs.
- Theories stored **separately and once** — the expensive artifacts, amortized across all 10
  hypotheses instead of regenerated. This is most of the token budget saved.

Proposed layout:

```
/theories/<theory-id>/
    theory.json            # distilled node (index entry)
    backtest/<case-id>.md  # raw historical case analysis (corpus)
/hypotheses/<hyp-id>/
    hypothesis.json        # distilled node: belief, anchor, resolution criteria
    evidence/<ev-id>.md
    application/<theory-id>.md
    crux/<crux-id>.md
    synthesis.md           # adjudicated probability + trace
/events/<date>-<id>.md     # news events + touched nodes + deltas (long-term)
/index.json                # the graph: all node summaries + pointers
/submission/               # generated Part 1 / 2 / 3
```

## 8. How the three deliverables fall out

- **Part 1 (10 forecasts)** = snapshot of the belief store: hypotheses with adjudicated
  probabilities + resolution criteria.
- **Part 2 (framework)** = the council + crux-adjudication described as a cause-and-effect engine.
  This is the *original* contribution and stays **holistic** (couplings intact). **Where
  originality comes from:** not a novel theory minted from scratch (one fragile hedgehog), but
  (i) the **crux resolutions** — novel claims built from legible parts — and (ii) an emergent
  **meta-theory of applicability**: the map of *which lens governs which regime, and why*. That
  emergent map is a synthesized theory, but an accountable one. Part 2 must still land a sharp
  thesis; the council *produces* it rather than asserting it.
- **Part 3 (appendix)** = per-hypothesis lens predictions, cruxes, cross-examination, evidence
  ledgers — i.e. the structured backing.
- **Calibration log** doubles as the trading backtest *and* the calibration the judges score.

## 9. Decisions & open items

- **Stack — DECIDED: Claude Agent SDK.** The Karpathy-style refinement loop is ordinary control
  flow (`while not converged: agent.run(...)`) — the loop is ours, the model is one step inside it.
  SDK supplies programmatic agent calls, tool use (grounding later), structured outputs, subagents.
- **Grounding — DECIDED: model-internal knowledge first; data fetches added later.** Evidence
  sourced from model knowledge is flagged `model-knowledge` so it can be re-grounded when fetches
  land. (See also §12 — most relevant data already lives in the model.)
- **Anchor coverage.** Polymarket/Kalshi are liquid only for near-term, sharply-defined events;
  the multi-year structural questions the competition rewards have thin/no markets. Metaculus +
  base rates fill the gap. Anchor source is per-hypothesis.

## 10. Lens-selection discipline (was parked — resolved via Munger, §13)

- **What disciplines the choice of theories on the council?** Answer: a curated cross-disciplinary
  **latticework checklist** (Munger, §13) applied to every hypothesis — a principled bench, not an
  arbitrary one. Avoids "man with a hammer."
- The QA check that the council is *real*: measure the **spread** across lenses. If lenses almost
  always agree, we have one model in costumes, not a council — the spread is both signal and
  diagnostic.

## 11. Non-goals / explicitly rejected

- ❌ Mechanical averaging of lens probabilities.
- ❌ A single synthesized "world-model" as the source of truth.
- ❌ Model-emitted `weight`/probability floats with no derivation (last year's flaw).
- ❌ "Recurse until the model stops talking" (never terminates; see §5).
- ❌ Reusing `archive/` code. Inspiration only.

## 12. Leaning on the model's own knowledge (efficiency)

The token volume lives in the **thinking** (transient), not the **storage**. Let thinking be
ephemeral; persist only distilled residue + keys. Splits on one fault line:

- **Timeless layer** (theories, historical cases, base rates) — well-represented and stable in the
  model. *Trust parametric memory; store brief keys; use compacted prompts that reference knowledge
  by name* (`"Apply Mearsheimer offensive realism to X — you know the theory and its canonical
  cases"`). This is where the big savings come from.
- **Time-sensitive layer** (current data, news, post-training facts) — model is sparse and
  confabulates. *Must be externally sourced and stored verbatim.* This is where future data-fetches
  (§9) aim.
- **Guardrails:** pin load-bearing facts (number, direction, date) in every note so re-expansion is
  constrained, not free-form. Store full expansions only when load-bearing *and* non-reproducible.

Net: notes are very brief, prompts compact, and the >>1M-token concern largely dissolves — provided
time-sensitive facts stay externally pinned.

## 13. Intellectual lineage: foxes (Tetlock) + latticework (Munger)

The design is the operationalization of two methods; division of labour:

- **Tetlock → the process.** Structure is **hedgehog lenses, fox synthesis** (each lens a committed
  single theory; the system a self-critical aggregator). Concrete techniques bolted into the loops:
  - **Base-rate first (outside view), then adjust (inside view)** — the reference-class base rate is
    the log-odds *prior*; theories supply adjustments. Biggest calibration lever.
  - Frequent small updates (→ fast path); fine probability granularity (calibration rewards it);
    Fermi-ize genuinely hard quantities; self-criticism (→ disconfirmation pass).
- **Munger → the library curation.**
  - **Latticework checklist** — core models from each major discipline (economics, psychology, game
    theory, feedback/physics, evolutionary biology, statistics, history), applied to every
    hypothesis. This is the §10 lens-selection discipline.
  - **Inversion** — the change-mind / disconfirmation pass, named.
  - **"Man with a hammer"** — the fake-diversity failure mode; latticework is the antidote, lens
    **spread** the diagnostic.
  - **Depth over sprawl** — few foundational models understood deeply (backtested) > a long
    arbitrary list.

## 14. Pipeline: distinct stages, append-only artifacts, reference-by-id

**Core rule: compression produces a NEW layer that points back — it never overwrites the layer
below.** Each stage reads prior stages and writes its own frozen artifact; no upstream artifact is
mutated. This keeps every level (raw notes → cruxes → conclusion) inspectable and referenceable, and
makes each stage independently re-runnable.

### Stage 0 — topic generation (the bench)

Everything downstream is bounded by topic selection (garbage in → garbage out), so topic
generation is **its own stage**, with the same diverge-then-compress shape as explore, writing
frozen artifacts the rest of the pipeline consumes:

```
bench/
  topics.json       # Stage 0 — the curated bench: pillars + intersection hypotheses, each with
                    #            rationale, axis-coverage tags, and the forecasts it should spawn.
                    #            FROZEN once written; Stage 1 runs once per entry.
  rejected.json     # out-of-scope / immaterial candidates + WHY they were cut (roundtable defense)
```

**Topic granularity (a topic is not a forecast).** A topic is a BROAD question-area, rich enough to
sustain a deep multi-lens explore and to spawn *several* forecasts. The binary, dated forecasts are
the **output** (chosen at Stage 7, across topics), not the bench entries. A candidate that is really
one yes/no prediction ("will China blockade Taiwan before 2028?") is folded UP as a **sub-question /
crux** inside a broad topic ("US–China tech decoupling"), never its own slot. `candidate_forecasts`
on a bench entry are *illustrative* of what the area could spawn, not commitments. Fewer, broader,
richer topics beats many narrow ones.

**Two altitudes (mirrors Theory vs. Hypothesis, §3):**

- **Pillars (~6, run *pure*).** Deep single-axis topics — pure AI (e.g. do scaling returns hold,
  model-layer commoditization, agentic diffusion) and pure mercantilism (the new industrial-policy
  era, weaponized interdependence, resource nationalism). Explored with **no cross-framing** — this
  is where divergence is widest, and where the reusable theory substrate + exogenous base rates come
  from. *LLM market structure is a pillar*, not contrived into an intersection. **The macro-financial
  loop is a required pillar** — AI capex + industrial-policy spending → rates, debt, currency
  debasement. It is the most Bridgewater-native topic (their worldview *is* debt cycles + currency
  debasement); omitting it is the loudest silence to these judges.
- **Intersection hypotheses (~12–14, run *crossed*).** Explicit AI×mercantilism mechanisms ("does
  export-control compute scarcity decide who wins the model layer"). Forecast-bearing and on-prompt.
  The **interaction cruxes — the heart of the framework — are generated HERE**, at Stage 1–3.

Why two altitudes: **Stage 5 *clusters* cruxes; it does not *manufacture* an interaction crux from
two separate domain cruxes.** Pillars alone would never surface "do export controls relocate the
chokepoint from training to inference." So interactions are front-loaded into intersection topics,
while pillars supply the divergence and the clean drivers that the intersections condition on.

**Stage 0 internals (diverge → compress, grounded termination — same saturation rule as §5):**

- **0a Diverge.** Seed with the pillars + an axis scaffold (*mercantilist lever × AI value-chain
  layer × actor*) used **generatively** (to range across the space), iterate "what adjacent topic
  are we missing?", **relevance-gated** (kept only if it bears on Mercantilism, AI, or their
  intersection), deduped, and **saturation-stopped with gap-targeting** (§5). The relevance gate
  pulls in materially-relevant adjacencies (e.g. quantum as a compute-chokepoint *obsolescence*
  risk) while rejecting off-prompt sprawl.
- **0b Compress.** Dedup near-identical candidates; select the bench; then run the **multi-lens
  completeness critic** and **log every cut to `rejected.json`** with its reason (the breadth-of-
  consideration record — the answer to "you forgot about X" at the roundtable).

**Completeness is NOT grid-coverage (the 36-cell trap).** A lever×layer grid is `|levers|×|layers|`
cells — but those counts are *arbitrary* (I picked 6×6), the grid forces a 2-D decomposition the
space doesn't have, and it is only as complete as the hand-picked axes (the organic run surfaced
*defense procurement / civil-military fusion* — a lever the grid omitted entirely). "We filled
30/36 cells" is false precision. So we **rejected the rigid grid sweep as the generator.** Instead,
completeness = surviving **several orthogonal "what could change the answer and isn't here?" lenses**,
each catching blind spots the others can't:

| Lens | Asks |
|------|------|
| lever × layer | the IO grid — *one* lens, not the lens |
| actor | who's missing? (EU/Brussels-effect, India, the open-source commons, standards bodies) |
| mechanism | is each battleground covered? (talent/immigration, data/IP/copyright, standards capture, payment rails, energy) |
| discipline | what would a macroeconomist / political scientist / historian / safety researcher say we ignored? |
| tail scenario | what discontinuity dominates the gradualist forecasts? (safety incident → clampdown, capability jump, Taiwan kinetic) |
| reflexivity | what *couplings between topics* change the answer? (deferred to cross-topic Stage 5) |

**Scope discipline (why not "forecast everything").** The prompt is a bounded collision
(Mercantilism, AI, or their intersection), and Stage 5 drivers only emerge from *convergence* — a
bench sprawled across unrelated domains (quantum, biotech, climate as topics in their own right)
shares no cruxes, so no drivers form and the worldview never integrates. Comprehensiveness here
means **depth within the Mercantilism×AI space, not breadth across all futures.**

Per-topic, the run directory accretes (it does not replace):

```
runs/<slug>/
  notes.json        # Stage 1 (explore)        — notes + analogies. FROZEN once written.
  notes.md
  notes.merged.json # Stage 1.5 (merge)        — within-type near-duplicates compressed;
                    #                             member_ids keep the trace back to raw notes
  frame_audit.json  # Stage 2a (frame audit)   — assumptions the notes treat as SETTLED but are
                    #                             contestable + options ASSUMED AWAY (attacks the FRAME)
  cruxes.json       # Stage 2 (crux synthesis) — decisive cruxes DERIVED from notes+analogies +
                    #                             frame audit: leverage + analogy base-rate, citing #/#A ids
  debate.json       # Stage 3 (debate)         — per crux: yes-advocate / no-advocate / judge
                    #                             (separate calls), citing the #ids used
  conclusion.json   # Stage 4 (topic conclude) — TOPIC thesis + resolved cruxes + CANDIDATE
  conclusion.md     #                             forecasts (NOT the final priced 10) + attacks
```

- **Stable ids + citations** are what make "refer back" work. Every note is `#7`, every analogy
  `#A10`; downstream artifacts cite those ids, so any conclusion traces down to the raw note that
  supports it (and, in the live debate, you pull that exact note when attacked). Because stages are
  distinct, `notes.json` is frozen when Stage 1 ends — ids never shift under downstream readers.
- **Two levels both survive:** `conclusion.json` is the 3-page-framework seed (compressed);
  `notes.json` is the 5-page appendix + defense ammunition (full graph). Present compressed, defend
  from the full graph — which is exactly how the competition is scored.
- **Stage 2 = crux synthesis** (derived, not generated inline): distil the notes+analogies into the
  ~4–5 *decisive* cruxes, each with **leverage** (how much it swings the answer) and an **analogy
  base-rate** (how comparable cases resolved, discounted by the disanalogy). Analogies feed cruxes on
  both dimensions here — *does it matter* and *which way does it lean*.
- **Stage 3 = the debate** (matches the Stage-2 roundtable, stress-tests harder than quiet synthesis):
  per crux a "yes" advocate and "no" advocate (SEPARATE calls — real adversarial separation) marshal
  cited notes/analogies; a judge rules lean + confidence + what-would-flip-it.
- **Stage 4 = topic conclude:** the *topic's* thesis from how its cruxes resolved + candidate
  forecasts + top-N attacks/rebuttals. **It does NOT price the final 10** — that is global (Stage 7),
  so a topic over-mining hypotheses can't inflate the forecast count.

**Cross-topic (after the bench is run) — same append-only pattern, three more artifacts:**

```
drivers.json     # Stage 5 — cluster cruxes across topics' cruxes/debate into a few MASTER DRIVERS
worldview.json   # Stage 6 — causal model over drivers; coherence pass flags contradictions BETWEEN
                 #            topics (bugs to resolve). This is the Part-2 framework / central thesis.
forecasts.json   # Stage 7 — hypothesis shaping: SELECT & price the final 10 forecasts ACROSS topics
                 #            to express the worldview — binary, dated, objectively resolvable (Part 1)
```

Topics need not all be mercantilism+AI — the bench aims for **comprehensiveness** (Stage 0 enforces
this: pure pillars + intersection hypotheses). Shared drivers are discovered bottom-up from
recurring cruxes, not imposed; the coherence pass catches *contradictions*, it does not force one
story (some topics are genuinely independent).

- **Open choice — how active is Stage 5.** As a pure *clusterer* it only groups recurring cruxes
  into drivers (cheap; relies on the intersection topics to have already generated the interaction
  cruxes at Stages 2–3). As a *clusterer + bounded cross* it additionally crosses the top few
  pillar-drivers pairwise to generate interaction cruxes the bench didn't pre-pick — a safety net
  for blind spots *between* pillars — at the cost of a real adversarial sub-stage. **Default: cluster
  + a bounded cross over the few highest-leverage pillar-drivers**, so the framework's interactions
  come primarily from intersection topics but Stage 5 still catches what the bench missed.

## 15. Design log — what we changed and the problem that drove it

The pipeline was shaped by running it and finding failures. Each mechanism exists to fix a specific
observed problem — recorded here so the *why* survives. Through-line: **the same disconfirmation
principle, applied at successively higher levels** (claim → crux → frame; completeness at
topic-selection → within-topic).

| Mechanism | Problem observed | What we changed |
|---|---|---|
| **Saturation judge + gap-targeting** (Stage 1) | The dry-round stop never fired on rich topics — an LLM asked "what's missing?" always finds a marginal note. T1 explore ran to 176 then 113 notes without converging. | A deep-model judge decides when the idea-space is *covered* and names the single biggest gap to target next — cuts depth, preserves breadth. |
| **Model tiering** | 9 topics × full pipeline = many hours / large spend. | Fast model (Sonnet) does high-volume mining; deep model judges saturation and does all synthesis. |
| **Merge pass (Stage 1.5)** replacing a hard note cap | We *assumed* 113 notes were duplication and added a guillotine cap at 70 — then inspected the notes and found they were mostly **distinct mechanisms**. A count cap amputated whole late-round themes (war-finance, energy, housing), not dupes. | Reverted to soft backstops; added a per-type merge that compresses near-duplicate **granularity** while keeping every distinct mechanism (`member_ids` preserve the trace). Trim granularity, never breadth. |
| **Debate resilience** (Stage 3) | One exhausted-retry call killed the whole pipeline mid-debate, losing cruxes + partial debate. | Each crux-debate is wrapped (skip-on-failure) and `debate.json` is checkpointed after every crux — a transient failure costs one crux, not the run. |
| **Broad-topic re-pitch** (Stage 0) | Topics came out forecast-shaped — several were single binary predictions (Gulf, Pentagon, Taiwan had one forecast each), i.e. forecast-granularity not topic-granularity. | A topic is now a broad *area* that spawns several forecasts; narrow predictions fold up as `sub_questions`; `candidate_forecasts` are illustrative; the final 10 are chosen at Stage 7. |
| **Multi-lens completeness critic** (Stage 0) | Free-association clustered on salient topics and missed whole axes/actors (the macro-financial loop, EU, India, safety-catastrophe). Grid-coverage over an arbitrary 6×6 lever×layer matrix was false precision. | Five *orthogonal* completeness lenses (macro / actor / mechanism / tail-scenario / discipline) run as separate adversarial calls; the macro-financial pillar is required. |
| **Frame audit (Stage 2a)** | Reviewing T1 we found it disconfirms *claims* but never the *frame*: it baked "austerity is politically impossible" in as a settled premise (#60) and never surfaced DOGE-style efficiency / AI-in-government as a resolution path. The saturation judge *shares* the frame, so it couldn't see the blind spot. | An adversarially-separate call attacks the framing — surfacing (a) load-bearing assumptions treated as settled and (b) options assumed away — and crux synthesis **promotes the high-leverage ones to cruxes**, so premises get interrogated, not inherited. |
| **Decision-graph synthesis (Stage 2, replaces binary cruxes)** | The `Crux` schema forced every decisive question into a **binary yes/no** with no actor field and no relation between cruxes. On the buildout topic this flattened a multi-input, actor-specific, *interacting* constraint vector into one "physical constraint: yes/no", and the conclusion was unstable because every new datapoint moved the single lumped variable. Diagnosed as: the flattening was baked into the schema, not the notes. | The topic unit is now a **decision graph**: *variables* (with a **state-space**, **per-actor readings**, and a **trajectory/clock** — not yes/no), *edges* (`gates` / `substitutes_for` / `chokepoint_controlled_by` / `correlates` — the interaction structure, now first-class), and *derived binary **forecast-signals*** (the old crux, demoted to a probe of a variable's state). Analysis unit ≠ forecast unit. Validated on T6: produced actor-split binding constraints (US→grid, China→chips) and 14 interaction edges the binary stage could not represent. (`run_prototype.py`) |
| **Graph composition (Stage 5, replaces thematic driver-clustering)** | The cross-topic stage clustered binary cruxes into 5–7 themed "drivers" — the over-aggregation step that **re-flattened** the structure at the top even when the topics were rich. | Stage 5 now **composes** the per-topic decision graphs into one graph: *merge variables by **REFERENT*** (same underlying thing across topics → one variable; union edges) — **never by theme**. Validated across 9 topics: 89 topic variables → 42 merged (23 spanning ≥2 topics; leading-edge chips merged from 8) with **no** "physical constraint" mega-bucket. **Note (superseded output):** its `spine` (top-8 highest-leverage) is **no longer used** — spine-selection was a lossy ranking that discarded the labor/macro/tail layer; downstream stages reuse the merged **variables + edges** as vocabulary and ignore the spine. (`run_compose.py`) |
| **Gate tracker (gauges) + Systems-map (replaces spine)** | The composed graph was a *static dependency graph* (44 `gates` vs 8 `substitutes` / 8 `correlates`, ~no feedback) read via a top-8 *spine* — a ranked list that gave status-quo, isolated per-node reads and dropped the consequence layer (labor/macro/tails sit in the 42 but never made the spine). | Two stages on top of the decision graphs: **gate tracker** (`run_gatetracker.py`, web-grounded, 5–7 single-unit metrics × ≥5 points pre-2020→2026 per gate = the *live gauges*; auto-charted by `make_gate_charts.py`); then **systems-map** (`run_systemsmap.py`, *planned*) which extracts **feedback loops** (basis + activation tags, gauge, flip-threshold), a **coupling map** (shared nodes + dominance arbitration), **discrete branches** (A base / B RSI-takeoff [theory, quarantined] / C bust / D fracture), and Dalio-style **theory-templates** (Perez / Dalio / Bostrom). Prediction = phase + watch-thresholds + probability-weighted trajectory, not a ranked list. Design in `docs/methodology-systems-model.md`. The old *cross-implications* stage (`run_crossimplications.py`) is **retired** — its five move-types become a coupling checklist inside the systems-map, its adversarial verdict an attribute on couplings; code retained as a possible post-hoc verifier. |

**Current flow.** topics → explore (notes + analogies = *instances*) → **decision-graph per topic** (variables + edges + forecast-signals) → **composition** (merge-by-referent → 42 vars + 64 edges; **reuse vars/edges, ignore the spine, no re-run**) → **gate tracker** (per-gate trends = *gauges*) → **systems-map** (loops + coupling map + discrete branches + theory *templates* → phase + weighted trajectory) → Part 1 forecasts / Part 2 framework / Part 3 appendix. Retired: the binary crux (survives only as a derived forecast-signal), thematic driver-clustering, the composition **spine** step, and the standalone **cross-implications** stage (folded into the systems-map). The three grounding registers line up: *instances* = explore, *gauges* = gate tracker, *templates* = systems-map. Unresolved fold-in: the validated stages still live as standalone scripts rather than inside `pipeline.py`.
