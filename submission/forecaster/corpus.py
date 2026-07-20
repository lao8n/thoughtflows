"""Shared corpus helpers — notes/analogies loading, cross-topic provenance, formatting.

Used by the decision-graph (run_decisiongraph) and gate-tracker (run_gatetracker).
Extracted to kill the copies that had drifted across those stages.
"""
from __future__ import annotations

import glob
import json
import os
import re


def format_notes(data: dict, analogy_detail: bool = True) -> str:
    """Format a topic's notes + analogies into a prompt string."""
    lines = ["NOTES:"]
    for n in data["notes"]:
        lines.append(f"[#{n['id']} {n['type']}] {n['claim']} — {n['detail']}")
    lines.append("\nHISTORICAL ANALOGIES:")
    for a in data.get("analogies", []):
        lines.append(f"[#A{a['id']}] {a['case']} — {a['what_happened']}")
        if analogy_detail:
            if a.get("similarities"):
                lines.append(f"    similar: {'; '.join(a['similarities'])}")
            if a.get("differences"):
                lines.append(f"    different: {'; '.join(a['differences'])}")
            lines.append(f"    implies: {a.get('implication', '')}")
    return "\n".join(lines)


def load_provenance() -> tuple[dict, dict]:
    """Cross-topic index of per-topic decision-graph variables + analogies.

    Returns (tvar, anlg_by_topic):
      tvar['<tid>:<Vid>']  -> the per-topic variable (with 'topic' added)
      anlg_by_topic[tid]   -> {analogy_id: analogy}
    """
    tvar: dict[str, dict] = {}
    anlg_by_topic: dict[str, dict] = {}
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
    """The k richest distinct analogies mined for a composed variable, via its provenance."""
    found: dict[str, dict] = {}
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


def grep_notes(pattern: str, k: int = 20,
               types: tuple = ("theory", "hypothesis", "crux", "evidence_for", "evidence_against")) -> list[dict]:
    """Distinct notes across runs/ whose claim+detail match `pattern` (regex, case-insensitive)."""
    pat = re.compile(pattern, re.I)
    seen, hits = set(), []
    for f in glob.glob("runs/*/notes*.json"):
        for n in json.load(open(f, encoding="utf-8")).get("notes", []):
            if types and n.get("type") not in types:
                continue
            if pat.search(f"{n.get('claim', '')} {n.get('detail', '')}"):
                key = n.get("claim", "")[:60]
                if key not in seen:
                    seen.add(key)
                    hits.append(n)
    return hits[:k]
