"""The note store: brief distilled keys + a markdown view.

Notes are deliberately terse (docs/architecture.md §12) — a claim + one or two
sentences, not a transcript. The model already holds the supporting knowledge;
the note is a trigger, not the corpus. Persisted after every round so a run is
inspectable and resumable.
"""
from __future__ import annotations

import json
import os
import re
from dataclasses import asdict, dataclass

# Aligned with the core objects in docs/architecture.md §3.
NOTE_TYPES = ["theory", "hypothesis", "evidence_for", "evidence_against", "crux", "question"]
TYPE_LABEL = {
    "theory": "Theories / lenses",
    "hypothesis": "Hypotheses",
    "evidence_for": "Supporting evidence",
    "evidence_against": "Disconfirming evidence",
    "crux": "Cruxes",
    "question": "Open questions",
}


@dataclass
class Note:
    id: int
    type: str
    claim: str
    detail: str
    round: int


@dataclass
class Analogy:
    """A historical archetype to compare the current situation against
    (Dalio's method: reason about the deviations). See docs/architecture.md §3."""
    id: int
    case: str
    what_happened: str
    similarities: list[str]
    differences: list[str]
    implication: str
    round: int


def slugify(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", text.strip().lower()).strip("_")[:60]


class NoteStore:
    def __init__(self, topic: str, outdir: str, mirror_md: str | None = None) -> None:
        self.topic = topic
        self.outdir = outdir
        # Optional tracked copy (e.g. docs/latest-notes.md) so the rendered
        # markdown is visible in editors/Conductor without un-ignoring runs/.
        self.mirror_md = mirror_md
        self.notes: list[Note] = []
        self.analogies: list[Analogy] = []
        self._seen: set[str] = set()
        self._seen_analogy: set[str] = set()
        self._next_id = 1
        self._next_analogy_id = 1
        os.makedirs(outdir, exist_ok=True)

    @staticmethod
    def _norm(claim: str) -> str:
        return re.sub(r"\s+", " ", claim.strip().lower())

    def add(self, *, type: str, claim: str, detail: str, round: int) -> bool:
        """Add a note. Returns False if it duplicates an existing claim."""
        key = self._norm(claim)
        if key in self._seen:
            return False
        self._seen.add(key)
        self.notes.append(Note(self._next_id, type, claim, detail, round))
        self._next_id += 1
        return True

    def add_analogy(
        self, *, case: str, what_happened: str, similarities: list[str],
        differences: list[str], implication: str, round: int,
    ) -> bool:
        key = self._norm(case)
        if key in self._seen_analogy:
            return False
        self._seen_analogy.add(key)
        self.analogies.append(Analogy(
            self._next_analogy_id, case, what_happened, similarities, differences, implication, round,
        ))
        self._next_analogy_id += 1
        return True

    def analogy_cases(self) -> list[str]:
        return [a.case for a in self.analogies]

    def claims_by_type(self) -> dict[str, list[str]]:
        out: dict[str, list[str]] = {t: [] for t in NOTE_TYPES}
        for n in self.notes:
            out.setdefault(n.type, []).append(n.claim)
        return out

    def counts(self) -> dict[str, int]:
        c: dict[str, int] = {t: 0 for t in NOTE_TYPES}
        for n in self.notes:
            c[n.type] = c.get(n.type, 0) + 1
        return c

    def _markdown(self) -> str:
        lines = [f"# Notes — {self.topic}\n"]
        grouped: dict[str, list[Note]] = {t: [] for t in NOTE_TYPES}
        for n in self.notes:
            grouped.setdefault(n.type, []).append(n)
        for t in NOTE_TYPES:
            items = grouped.get(t, [])
            if not items:
                continue
            lines.append(f"## {TYPE_LABEL.get(t, t)} ({len(items)})\n")
            for n in items:
                lines.append(f"- **{n.claim}** — {n.detail}  _(#{n.id}, r{n.round})_")
            lines.append("")
        if self.analogies:
            lines.append(f"## Historical analogies ({len(self.analogies)})\n")
            for a in self.analogies:
                lines.append(f"- **{a.case}** — {a.what_happened}")
                if a.similarities:
                    lines.append(f"  - _Similar:_ {'; '.join(a.similarities)}")
                if a.differences:
                    lines.append(f"  - _Different:_ {'; '.join(a.differences)}")
                lines.append(f"  - _Implies:_ {a.implication}  _(#A{a.id}, r{a.round})_")
            lines.append("")
        return "\n".join(lines) + "\n"

    def save(self) -> None:
        with open(os.path.join(self.outdir, "notes.json"), "w", encoding="utf-8") as f:
            json.dump(
                {
                    "topic": self.topic,
                    "notes": [asdict(n) for n in self.notes],
                    "analogies": [asdict(a) for a in self.analogies],
                },
                f,
                indent=2,
            )
        md = self._markdown()
        with open(os.path.join(self.outdir, "notes.md"), "w", encoding="utf-8") as f:
            f.write(md)
        if self.mirror_md:
            os.makedirs(os.path.dirname(self.mirror_md) or ".", exist_ok=True)
            with open(self.mirror_md, "w", encoding="utf-8") as f:
                f.write(md)
