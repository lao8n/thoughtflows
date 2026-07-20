# Forecasting belief engine

Design: [`docs/architecture.md`](docs/architecture.md). Last year's code is in `archive/` (reference only).

## Setup

```sh
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

No API key needed — the engine runs through the **Claude Agent SDK**, which uses
your existing Claude Code login (the `claude` CLI). Optionally set
`FORECASTER_MODEL` in a `.env` to pin a model; otherwise it uses your current
Claude Code model.

## The exploration loop

The first runnable piece (architecture §5): iterate notes on a topic until the
model stops finding anything new and material — the grounded termination rule.

```sh
.venv/bin/python run_explore.py "future market structure of LLMs"
```

Each round the model is shown the existing notes and asked what they're
*missing* — a missing lens, an unstated counter-argument, an unsurfaced crux,
evidence on either side. It returns only notes it judges genuinely **new** and
**material** (empty list when the topic is covered). The loop stops after
`--patience` consecutive empty rounds (default 2) or `--max-rounds` (default 12).

Output: `runs/<slug>/notes.md` (readable) and `notes.json` (structured), saved
after every round.

Notes are typed — theory / hypothesis / evidence_for / evidence_against / crux /
question — matching the core objects in the architecture.
