#!/usr/bin/env python3
"""Assemble one big scrollable document — docs/DASHBOARD.md — from the current-pipeline
outputs, with every chart inlined so it renders in Conductor's single-file preview.

Spine = the comprehensive writeup; then the systems map, the takeoff deep-dive, and full
chart galleries. `[chart: file.png]` text refs are converted to real inline embeds.
Regenerable: re-run after any stage updates.

    python build_dashboard.py
"""
from __future__ import annotations
import base64, io, os, re
from PIL import Image

MAX_W = 820   # downscale charts to this width before embedding (preview-friendly)

DOCS = "docs"
OUT = os.path.join(DOCS, "DASHBOARD.md")

HEROES = [
    ("gt1_thesis.png", "The thesis — the prize evaporates (inference cost) while the moat widens (transformer lead time)"),
    ("gt2_us_power_bind.png", "The US power bind — China builds ~8× the power; the US can't connect its grid"),
    ("gt3_master_derivative.png", "The master switch — capability accelerating vs the diffusion lag floored at 3–6 months"),
    ("gt4_value_capture.png", "The value-capture barbell — hyperscaler capex vs collapsing model-API price"),
]
GATES = [
    ("gate_GV1.png", "Leading-edge chips"), ("gate_GV2.png", "EUV / lithography"),
    ("gate_GV6.png", "Power & grid"), ("gate_GV8.png", "Critical minerals"),
    ("gate_GV10.png", "Efficiency / diffusion"), ("gate_GV11.png", "Capability timing"),
    ("gate_GV12.png", "Talent"), ("gate_GV15.png", "Value capture"),
    ("gate_GV21.png", "Taiwan"), ("gate_GV4.png", "Export controls"),
]
APPENDIX = [
    ("gt5_chips.png", "Chips — node timeline & no-EUV tax"), ("gt6_euv.png", "EUV — indigenous progress vs the wall"),
    ("gt7_minerals.png", "Minerals — China's counter-lever"), ("gt8_talent.png", "Talent — produce vs concentrate"),
    ("gt9_taiwan.png", "Taiwan — pressure & silicon-shield thinning"), ("gt10_export_controls.png", "Export controls — denial → delay-and-tax"),
]

CHART_RE = re.compile(r"\[chart:\s*([A-Za-z0-9_./-]+\.png)\s*\]")


def read(name: str) -> str:
    p = os.path.join(DOCS, name)
    return open(p, encoding="utf-8").read() if os.path.exists(p) else ""


def inline_chart_refs(md: str) -> str:
    # turn "[chart: foo.png]" into a rendered embed (strip any leading charts/ dir)
    def repl(m):
        fn = os.path.basename(m.group(1))
        return f"\n\n![{fn}](charts/{fn})\n"
    return CHART_RE.sub(repl, md)


def gallery(title: str, items) -> str:
    L = [f"\n\n## {title}\n"]
    for fn, cap in items:
        if os.path.exists(os.path.join(DOCS, "charts", fn)):
            L.append(f"**{cap}**\n\n![{fn}](charts/{fn})\n")
    return "\n".join(L)


def demote(md: str, by: int = 1) -> str:
    # push an included doc's headings down so the TOC/structure nests cleanly
    return re.sub(r"^(#{1,5}) ", lambda m: "#" * min(6, len(m.group(1)) + by) + " ", md, flags=re.M)


def embed_data_uris(md: str) -> tuple[str, int, int]:
    """Convert every ![alt](charts/FN) into reference-style ![alt][ref] and append one
    base64 data-URI definition per DISTINCT chart, so images are self-contained (no
    file:// access) and each chart's bytes appear only once regardless of reference count."""
    used = list(dict.fromkeys(re.findall(r"]\(charts/([^)]+)\)", md)))  # distinct, order-preserving

    def refid(fn: str) -> str:
        return "img_" + re.sub(r"[^A-Za-z0-9]", "_", fn)

    # inline ![alt](charts/FN) -> reference ![alt][refid]
    md = re.sub(r"]\(charts/([^)]+)\)", lambda m: f"][{refid(m.group(1))}]", md)

    defs, total_bytes = ["\n\n"], 0
    for fn in used:
        p = os.path.join(DOCS, "charts", fn)
        if not os.path.exists(p):
            continue
        img = Image.open(p)
        if img.width > MAX_W:
            img = img.resize((MAX_W, round(img.height * MAX_W / img.width)), Image.LANCZOS)
        img = img.convert("RGB")                       # flatten RGBA (charts are white-bg)
        img = img.quantize(colors=128, method=Image.Quantize.FASTOCTREE)  # charts use few colors → palette PNG is tiny
        buf = io.BytesIO()
        img.save(buf, format="PNG", optimize=True)
        raw = buf.getvalue()
        total_bytes += len(raw)
        b64 = base64.b64encode(raw).decode("ascii")
        defs.append(f"[{refid(fn)}]: data:image/png;base64,{b64}")
    return md + "\n".join(defs), len(used), total_bytes


def main() -> int:
    parts = [
        "# AI × Modern Mercantilism — Master Dashboard",
        "*One assembled document: hero charts → comprehensive analysis → systems map → "
        "takeoff/breakout deep-dive → gate dashboards. Charts inlined. Regenerate with "
        "`python build_dashboard.py`. Raw underlying: `gate-tracker.md`, `composed-graph.md`.*",
        "\n## Contents",
        "1. Hero charts\n2. Comprehensive analysis (the narrative)\n3. Systems map (loops, couplings, branches)\n"
        "4. Takeoff / breakout deep-dive\n5. Gate dashboards\n6. Appendix charts",
    ]

    parts.append("\n\n---\n\n# 1 · Hero charts")
    parts.append(gallery("The four charts that carry the thesis", HEROES))

    comp = read("comprehensive-analysis.md")
    if comp:
        parts.append("\n\n---\n\n# 2 · Comprehensive analysis")
        parts.append(inline_chart_refs(demote(comp, 1)))

    sm = read("systems-map.md")
    if sm:
        parts.append("\n\n---\n\n# 3 · Systems map")
        parts.append(inline_chart_refs(demote(sm, 1)))

    tk = read("takeoff-analysis.md")
    if tk:
        parts.append("\n\n---\n\n# 4 · Takeoff / breakout deep-dive")
        parts.append(inline_chart_refs(demote(tk, 1)))

    parts.append("\n\n---\n\n# 5 · Gate dashboards")
    parts.append(gallery("Ten load-bearing gates — measured trends (solid = web-verified)", GATES))

    parts.append("\n\n---\n\n# 6 · Appendix charts")
    parts.append(gallery("Per-gate detail", APPENDIX))

    out = "\n".join(parts)
    out, n_distinct, img_bytes = embed_data_uris(out)   # self-contained: no file:// access needed
    open(OUT, "w", encoding="utf-8").write(out)
    print(f"wrote {OUT} — {len(out):,} chars ({len(out)/1e6:.1f} MB), "
          f"{n_distinct} distinct charts embedded as base64 ({img_bytes/1e6:.1f} MB of PNG)")
    print(f"  sections: heroes + {'comprehensive ' if comp else ''}{'systems-map ' if sm else ''}"
          f"{'takeoff ' if tk else ''}+ gate gallery + appendix")
    print("  images are inlined (data-URIs) — renders without local file access")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
