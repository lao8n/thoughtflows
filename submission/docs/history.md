# How the pipeline evolved

A changelog of the *method*, not the code — why each stage exists and what it
replaced. The live design is [`architecture.md`](architecture.md) and
[`methodology-systems-model.md`](methodology-systems-model.md); this doc records
the dead ends so we don't re-walk them.

## 1. Binary cruxes → decision graphs

The first pipeline distilled each topic into **4–6 binary yes/no cruxes**, then
ran a merge → debate → conclude chain over them (the old `forecaster/pipeline.py`
+ `stage5–8`, `stage_dense`, `stage_review`, and their `run_*` drivers).

The crux step **flattened** multi-dimensional, actor-specific, interacting
constraints into single binaries — e.g. "what constrains the AI buildout?" became
"is the constraint physical: yes/no", which lost that *chips* bind China while
*grid/transformers* bind the US. It also erased distinctions that mattered
(inference vs. training economics) and second-order effects (labour, society).

**Replacement:** the **decision graph** (`run_decisiongraph.py`). The analysis unit is
a decisive *variable* with a `state_space` (regimes, not yes/no), per-actor
readings, a trajectory-with-a-clock, and an `edges` structure (`gates`,
`substitutes_for`, `chokepoint_controlled_by`, `correlates`). Binary forecast
signals are *derived* from the variables, not the primary object.

## 2. Driver clustering / compose → systems map

An intermediate "drivers"/`compose` step merged the per-topic graphs into one
`composed-graph.json`. Compose survived (it still does the cross-topic merge), but
the *analytical* successor to hand-clustered drivers is the **systems map**
(`run_systemsmap.py`): explicit feedback loops (R reinforcing / B balancing), a
coupling map, and discrete branches (base / bust / fracture / RSI-takeoff) with a
quarantined takeoff branch. `docs/drivers.md`, `docs/bench-organic.md` and the
various `docs/latest-*` snapshots are from this era and are gone.

## 3. Soft lenses → data-driven gate tracker

Early analyses argued trends qualitatively ("soft lenses"). The **gate tracker**
(`run_gatetracker.py`, web-grounded) makes conviction come from the *derivative*:
for each binding variable it pulls ≥5 metrics × ≥5 points spanning pre-2020→2026
and reads the trend. This is where the "how did China close 14nm→7nm so fast?"
kind of question gets a grounded answer instead of a vibe.

## 4. Clearing analysis → Liebig dominance

`docs/clearing-analysis.md` argued the status quo clears. It was replaced by the
**dominance** stage (`run_dominance.py`): Liebig's law of the minimum — the
binding stave, and the historical record of when the actor with the most of a
single input (energy, minerals, fabrication) actually won. `takeoff` sits beside
it as the quarantined RSI branch.

## 5. Integrated write-up + one-file view

`run_part2.py` regenerates the raw integrated framework
(`part2-framework-synthesis.md`); `part2-final.md` is the hand-finalised
submission. `build_dashboard.py` / `build_pdf.py` assemble the one-file dashboard.
Charts moved from ad-hoc (`make_charts*.py` v1–v3) to the current
`make_gate_charts` + `make_charts_gt*` + `make_migration` + `make_gt1_alt` +
`make_appendix` set, sharing `docs/charts/_style.py`.

## Engineering lessons carried forward

- **Web-grounded calls can exceed the SDK's default 1 MB message buffer.** The LLM
  layer now threads a `max_buffer_size` (32 MB) for gate-tracker / takeoff /
  dominance.
- **Merge, don't clobber.** An early gate-tracker writer rewrote the whole file on
  a subset run and dropped other gates; the writer now merges and preserves
  canonical ordering.
- **Three registers, combined.** Dalio's big-cycle method needs a rich history to
  work; where history is thin we lean on theory (Perez tech-finance cycle, Bostrom
  takeoff, Kai-Fu-Lee deployment loop) as templates, historical instances as
  reference classes, and the gate tracker as live gauges — and combine them rather
  than picking one.
