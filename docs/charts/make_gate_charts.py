"""
Generic gate-dashboard charter — renders EVERY tracked metric in
bench/gate-tracker.json automatically, one dashboard figure per gate
(a grid of that gate's 5-7 metric panels). No hand-coding: the charts
scale with the data. Web-verified points are drawn solid; model_estimate
points are drawn hollow so the eye can see the grounding.

    ../../.venv/bin/python make_gate_charts.py            # all gates
    ../../.venv/bin/python make_gate_charts.py GV6 GV1    # only these
Writes gate_<GVid>.png per gate.
"""
import json, math, os, re, sys, textwrap
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

JSON = "../../bench/gate-tracker.json"
US = "#1F5FA8"; CN = "#C0392B"; DIFF = "#157A6E"; GOLD = "#C08A1E"; GREY = "#8A8A8A"
DIRCOLOR = {"closing": DIFF, "accelerating": DIFF, "widening": CN,
            "reversing": GOLD, "stuck": US}
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9,
                     "axes.titlesize": 9.5, "axes.titleweight": "bold",
                     "axes.edgecolor": "#333333", "figure.dpi": 140})

_MON = {m: i for i, m in enumerate(
    ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"], 1)}


def parse_x(s: str):
    """Best-effort date -> float year. Returns None if unparseable."""
    s = re.sub(r"\(.*?\)", "", str(s)).strip()
    m = re.search(r"\bFY\s?'?(\d{2,4})", s, re.I)
    if m:
        y = int(m.group(1)); return float(y if y > 100 else 2000 + y)
    m = re.search(r"(\d{4})[-\s]?Q([1-4])", s, re.I)
    if m:
        return int(m.group(1)) + (int(m.group(2)) - 0.5) / 4
    m = re.search(r"([A-Za-z]{3,})[-\s]+(\d{4})", s)          # "May 2023"
    if m and m.group(1)[:3].lower() in _MON:
        return int(m.group(2)) + (_MON[m.group(1)[:3].lower()] - 0.5) / 12
    m = re.search(r"\b([A-Za-z]{3})-?(\d{2})\b", s)           # "Jul-24"
    if m and m.group(1).lower() in _MON:
        return 2000 + int(m.group(2)) + (_MON[m.group(1).lower()] - 0.5) / 12
    m = re.search(r"(\d{4})[-/](\d{1,2})\b", s)               # "2023-08"
    if m:
        return int(m.group(1)) + (int(m.group(2)) - 0.5) / 12
    m = re.search(r"\b(19|20)(\d{2})\b", s)                   # bare year
    if m:
        return int(m.group(0))
    return None


def dominant_unit(points):
    counts = {}
    for p in points:
        counts[p.get("unit", "")] = counts.get(p.get("unit", ""), 0) + 1
    return max(counts, key=counts.get) if counts else ""


def draw_metric(ax, m):
    unit = m.get("unit") or dominant_unit(m["points"])
    pts = [p for p in m["points"] if (p.get("unit", unit) == unit)] or m["points"]
    xs = [parse_x(p["date"]) for p in pts]
    numeric = all(x is not None for x in xs) and len(set(xs)) == len(xs)
    color = DIRCOLOR.get(m.get("direction", ""), US)
    ys = [p["value"] for p in pts]
    labels = [re.sub(r"\(.*?\)", "", p["date"]).strip() for p in pts]
    solid = [p.get("grounding") == "web_verified" for p in pts]

    ctype = m.get("chart_type") or ("line" if numeric and len(pts) > 3 else "bar")
    if numeric and ctype == "line":
        order = sorted(range(len(xs)), key=lambda i: xs[i])
        xo = [xs[i] for i in order]; yo = [ys[i] for i in order]
        ax.plot(xo, yo, "-", color=color, lw=2.2, zorder=3)
        for i in order:
            ax.plot(xs[i], ys[i], "o", ms=6, zorder=4,
                    color=color if solid[i] else "white",
                    markeredgecolor=color, markeredgewidth=1.6)
    else:
        pos = range(len(pts))
        bars = ax.bar(pos, ys, color=color, width=0.62, zorder=3)
        for b, sol in zip(bars, solid):
            if not sol:
                b.set_facecolor("white"); b.set_edgecolor(color); b.set_linewidth(1.6)
        ax.set_xticks(list(pos))
        ax.set_xticklabels(labels, fontsize=7, rotation=20, ha="right")
        for b, v in zip(bars, ys):
            ax.annotate(f"{v:g}", (b.get_x()+b.get_width()/2, b.get_height()),
                        textcoords="offset points", xytext=(0, 2), ha="center", fontsize=7, fontweight="bold")

    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    ax.grid(axis="y", color="#E4E4E4", linewidth=0.6, zorder=0); ax.set_axisbelow(True)
    ax.set_title(textwrap.fill(m["metric"], 46), fontsize=8.6)
    ax.set_ylabel(unit, fontsize=7.5)
    wv = sum(solid)
    # grounding tag top-right (out of the way of the title)
    ax.text(1.0, 1.02, f"{wv}/{len(pts)} ✓", transform=ax.transAxes,
            fontsize=6.3, color=GREY, ha="right", va="bottom")
    # direction + quantified rate as a wrapped caption BELOW the panel
    rate = (m.get("rate", "") or "").strip()
    if len(rate) > 150:
        rate = rate[:147].rsplit(" ", 1)[0] + "…"
    cap = textwrap.fill(f"{m.get('direction','').upper()} · {rate}", 60)
    ax.text(0.0, -0.34, cap, transform=ax.transAxes, fontsize=6.7,
            color=color, va="top", ha="left")
    # original source(s) — the distinct real sources behind the series, not "gate-tracker"
    srcs = list(dict.fromkeys(re.sub(r"\s+", " ", (p.get("source") or "")).strip()
                              for p in pts if p.get("source")))
    if srcs:
        short = "; ".join(s[:34] for s in srcs[:3])
        ax.text(0.0, -0.60, textwrap.fill("Source: " + short, 66), transform=ax.transAxes,
                fontsize=6.0, color=GREY, va="top", ha="left")


def render_gate(g):
    metrics = g.get("metrics", [])
    n = len(metrics)
    if not n:
        return None
    ncols = 3 if n >= 5 else (2 if n >= 2 else 1)
    nrows = math.ceil(n / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.9 * ncols, 4.6 * nrows),
                             squeeze=False)
    flat = [a for row in axes for a in row]
    for ax, m in zip(flat, metrics):
        draw_metric(ax, m)
    for ax in flat[n:]:
        ax.axis("off")
    fig.suptitle(f"{g['gate_id']} · {g['gate_name']}   —   {g.get('hard_or_soft','')}"
                 f"  ·  conf {g.get('confidence','')}",
                 fontsize=12, fontweight="bold", y=1.002)
    fig.text(0.5, 0.005, "solid = web-verified point · hollow = model estimate  ·  sources cited per panel",
             ha="center", fontsize=7, color=GREY)
    fig.tight_layout(rect=[0, 0.02, 1, 0.98], h_pad=4.4, w_pad=2.0)
    out = f"gate_{g['gate_id']}.png"
    fig.savefig(out, bbox_inches="tight", facecolor="white"); plt.close(fig)
    return out


def main():
    if not os.path.exists(JSON):
        print(f"no {JSON} — run run_gatetracker.py first."); return 1
    data = json.load(open(JSON, encoding="utf-8"))
    want = [a for a in sys.argv[1:] if a.startswith("GV")]
    total_panels = 0
    for g in data.get("gates", []):
        if want and g["gate_id"] not in want:
            continue
        out = render_gate(g)
        if out:
            total_panels += len(g["metrics"])
            print(f"wrote {out}  ({len(g['metrics'])} metric panels)")
    print(f"\n{total_panels} metric panels across gates.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
