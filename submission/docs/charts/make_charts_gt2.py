"""
Part 3 (appendix) charts — the remaining six gates, one figure each, so every
tracked gate has a chart. Series transcribed from bench/gate-tracker.json
(web-verified points), gate id noted per figure.
  gt5  leading-edge chips (GV1)      gt8  talent (GV12)
  gt6  EUV / lithography (GV2)       gt9  Taiwan (GV21)
  gt7  critical minerals (GV8)       gt10 export controls (GV4)
Run: ../../.venv/bin/python make_charts_gt2.py
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

US = "#1F5FA8"; CN = "#C0392B"; DIFF = "#157A6E"; GOLD = "#C08A1E"; GREY = "#8A8A8A"
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 10,
    "axes.titlesize": 11, "axes.titleweight": "bold",
    "axes.edgecolor": "#333333", "figure.dpi": 150,
})


def style(ax, ylabel="", source=""):
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    ax.grid(axis="y", color="#DDDDDD", linewidth=0.7, zorder=0); ax.set_axisbelow(True)
    if ylabel: ax.set_ylabel(ylabel, fontsize=9)
    if source: ax.text(0, -0.26, source, transform=ax.transAxes, fontsize=7.3, color=GREY, ha="left", va="top")


def barlabels(ax, bars, vals, fmt="{:g}", dy=3, fs=9):
    for b, v in zip(bars, vals):
        ax.annotate(fmt.format(v), (b.get_x()+b.get_width()/2, b.get_height()),
                    textcoords="offset points", xytext=(0, dy), ha="center", fontsize=fs, fontweight="bold")


# ============================ GT5 — leading-edge chips (GV1)
fig, (a1, a2) = plt.subplots(1, 2, figsize=(11.5, 4.6))
fig.suptitle("Leading-edge chips: China climbs, but the frontier gap widens", fontsize=12.5, fontweight="bold", y=1.03)
a1.plot([2020, 2022, 2025], [5, 3, 2], "-o", color=US, lw=2.5, ms=7, label="TSMC (frontier)", zorder=4)
a1.plot([2019, 2023, 2025], [14, 7, 5], "-s", color=CN, lw=2.5, ms=7, label="SMIC (China, no EUV)", zorder=4)
for x, n in [(2020, 5), (2022, 3), (2025, 2)]:
    a1.annotate(f"{n}nm", (x, n), textcoords="offset points", xytext=(0, -15), ha="center", fontsize=8, color=US)
for x, n in [(2019, 14), (2023, 7), (2025, 5)]:
    a1.annotate(f"{n}nm", (x, n), textcoords="offset points", xytext=(0, 9), ha="center", fontsize=8, color=CN)
a1.invert_yaxis(); a1.set_ylim(16, 1); a1.set_xticks([2019, 2021, 2023, 2025])
a1.legend(loc="lower left", fontsize=8.5, frameon=False)
a1.set_title("Best node in volume production (nm)"); style(a1, "Process node (nm) — lower better", "Source: TechInsights")
b = a2.bar(["2020", "2023", "2025"], [2, 2, 3], color=CN, width=0.55, zorder=3)
barlabels(a2, b, [2, 2, 3], fmt="{:g} nodes")
a2.set_ylim(0, 4)
a2.text(0.03, 0.94, "The frontier NODE GAP widens from 2 to 3 as\nHigh-NA EUV ramps on tools China can't buy.\nTSMC Arizona: 0 → 95,000 wafers/mo in 2025\n(off-Taiwan capacity finally real, but late).",
        transform=a2.transAxes, fontsize=7.8, color=GREY, va="top", ha="left")
a2.set_title("Frontier node gap (TSMC lead, # nodes)"); style(a2, "Nodes behind frontier", "Source: TechInsights; TSMC")
fig.tight_layout(); fig.savefig("gt5_chips.png", bbox_inches="tight", facecolor="white"); print("wrote gt5_chips.png")


# ============================ GT6 — EUV / lithography (GV2)
fig, (a1, a2) = plt.subplots(1, 2, figsize=(11.5, 4.6))
fig.suptitle("EUV lithography: the one hard gate — indigenous progress, but the HVM gap holds", fontsize=12, fontweight="bold", y=1.03)
a1.plot([2025, 2026], [125, 250], "-o", color=CN, lw=2.5, ms=8, zorder=4)
a1.axhline(250, ls="--", color=GREY, lw=1.3)
a1.annotate("~250 W = ASML HVM-class source power", (2025.02, 258), fontsize=7.8, color=GREY, va="bottom")
for x, w in [(2025, 125), (2026, 250)]:
    a1.annotate(f"{w} W", (x, w), textcoords="offset points", xytext=(0, -16), ha="center", fontsize=8.5, color=CN, fontweight="bold")
a1.set_ylim(0, 300); a1.set_xticks([2025, 2026]); a1.set_xlim(2024.8, 2026.2)
a1.text(0.03, 0.60, "China's indigenous EUV light-source power is\nramping — but a lab source ≠ an integrated\nHVM scanner. Risk production ~2028+; the full\ntool-and-ecosystem gap is ~a decade.",
        transform=a1.transAxes, fontsize=7.8, color=CN, va="top", ha="left")
a1.set_title("China indigenous EUV light-source power (W)"); style(a1, "Source power (watts)", "Source: SSMB/LDP reports")
b = a2.bar(["2023", "2025"], [100, 100], color=US, width=0.5, zorder=3)
barlabels(a2, b, [100, 100], fmt="{:g}%")
a2.set_ylim(0, 120)
a2.text(0.03, 0.60, "ASML holds ~100% of economically-viable\nsub-5nm EUV tooling — and the ABSOLUTE gap\nWIDENS first: High-NA EUV ships to TSMC/Intel\non tools China cannot buy at all.",
        transform=a2.transAxes, fontsize=7.8, color=US, va="top", ha="left")
a2.set_title("ASML share of sub-5nm EUV tooling (%)"); style(a2, "% of viable EUV tooling", "Source: ASML filings; SemiAnalysis")
fig.tight_layout(); fig.savefig("gt6_euv.png", bbox_inches="tight", facecolor="white"); print("wrote gt6_euv.png")


# ============================ GT7 — critical minerals (GV8)
fig, (a1, a2) = plt.subplots(1, 2, figsize=(11.5, 4.6))
fig.suptitle("Critical minerals: China's counter-lever — a durable tax, not a knockout", fontsize=12.5, fontweight="bold", y=1.03)
seg = ["Mining", "Refining", "Magnets", "Gallium/\nGermanium"]
val = [60, 90, 92, 98]
b = a1.bar(seg, val, color=CN, width=0.62, zorder=3); barlabels(a1, b, val, fmt="{:g}%")
a1.set_ylim(0, 110)
a1.text(0.03, 0.96, "China dominates the REFINING, not the ore —\na cost/time/permitting tax the US can pay down\n(MP Materials, Lynas ramping), not a wall.",
        transform=a1.transAxes, fontsize=7.8, color=GREY, va="top", ha="left")
a1.set_title("China share of the rare-earth value chain, 2024 (%)"); style(a1, "% of global", "Source: USGS, IEA")
b = a2.bar(["2023", "2024", "2025"], [1, 2, 3], color=CN, width=0.5, zorder=3)
barlabels(a2, b, [1, 2, 3], fmt="{:g}")
a2.set_ylim(0, 4)
a2.text(0.03, 0.94, "Weaponization is escalating: export-control\nactions 1→2→3/yr; gallium spiked to ~$687/kg\n(2025). It taxes the US buildout — the mirror\nof US chip controls. Mutual disruption.",
        transform=a2.transAxes, fontsize=7.8, color=CN, va="top", ha="left")
a2.set_title("China mineral export-control actions (per year)"); style(a2, "Coercive actions / year", "Source: CSIS; press reports")
fig.tight_layout(); fig.savefig("gt7_minerals.png", bbox_inches="tight", facecolor="white"); print("wrote gt7_minerals.png")


# ============================ GT8 — talent (GV12)
fig, (a1, a2) = plt.subplots(1, 2, figsize=(11.5, 4.6))
fig.suptitle("Talent: China produces more, the US still concentrates the elite — but the door is closing", fontsize=11.5, fontweight="bold", y=1.03)
a1.plot([2017, 2020, 2024], [51, 59, 72], "-o", color=US, lw=2.5, ms=7, label="US institutions' share of ELITE researchers (workplace)", zorder=4)
a1.plot([2017, 2019, 2024], [27, 29, 38], "-s", color=CN, lw=2.5, ms=7, label="China share of top AI authors (by training)", zorder=4)
a1.set_ylim(0, 85); a1.set_xticks([2017, 2020, 2024])
a1.legend(loc="upper left", fontsize=7.4, frameon=False)
a1.set_title("Two different talent stories (%)"); style(a1, "% share", "Source: MacroPolo, Stanford AI Index")
b = a2.bar(["FY21", "FY23", "FY24"], [274, 484, 781], color=US, width=0.55, zorder=3)
barlabels(a2, b, [274, 484, 781], fmt="{:g}k")
a2.axhline(85, ls="--", color=CN, lw=1.5); a2.annotate("85k statutory cap", (0.0, 92), fontsize=8, color=CN, va="bottom")
a2.set_ylim(0, 880)
a2.text(0.30, 0.94, "Demand for H-1B visas is ~9× the fixed cap\nand rising — a self-inflicted US bottleneck\nthat throttles the very talent inflow the\nconcentration advantage depends on.",
        transform=a2.transAxes, fontsize=7.7, color=GREY, va="top", ha="left")
a2.set_title("H-1B registrations vs the fixed cap (000s)"); style(a2, "Registrations (thousands)", "Source: USCIS")
fig.tight_layout(); fig.savefig("gt8_talent.png", bbox_inches="tight", facecolor="white"); print("wrote gt8_talent.png")


# ============================ GT9 — Taiwan (GV21)
fig, (a1, a2) = plt.subplots(1, 2, figsize=(11.5, 4.6))
fig.suptitle("Taiwan: pressure rising, the silicon shield slowly thinning", fontsize=12.5, fontweight="bold", y=1.03)
sx = ["2020", "2021", "2023", "2024", "2025"]; sy = [380, 972, 1700, 3615, 4000]
a1.plot(sx, sy, "-o", color=CN, lw=2.6, ms=7, zorder=4)
for x, y in zip(sx, sy):
    a1.annotate(f"{y:,}", (x, y), textcoords="offset points", xytext=(0, 9), ha="center", fontsize=8, color=CN, fontweight="bold")
a1.set_ylim(0, 4600)
a1.set_title("PLA aircraft incursions into Taiwan's ADIZ (sorties/yr)"); style(a1, "Annual sorties", "Source: Taiwan MND")
a2.plot(["2023", "2025", "2028e"], [92, 88, 80], "-o", color=US, lw=2.6, ms=8, zorder=4)
for x, y in [("2023", 92), ("2025", 88), ("2028e", 80)]:
    a2.annotate(f"{y}%", (x, y), textcoords="offset points", xytext=(0, 10), ha="center", fontsize=9, color=US, fontweight="bold")
a2.set_ylim(60, 100)
a2.text(0.03, 0.40, "Leading-edge foundry capacity in Taiwan\nthins only slowly (Arizona/Japan). The live\ntail is a BLOCKADE/quarantine (~25–30% by\nearly 2030s), not a full invasion (~15%).",
        transform=a2.transAxes, fontsize=7.8, color=US, va="top", ha="left")
a2.set_title("Share of <=7nm foundry capacity physically in Taiwan (%)"); style(a2, "% of global leading-edge", "Source: CSIS")
fig.tight_layout(); fig.savefig("gt9_taiwan.png", bbox_inches="tight", facecolor="white"); print("wrote gt9_taiwan.png")


# ============================ GT10 — export controls (GV4)
fig, (a1, a2) = plt.subplots(1, 2, figsize=(11.5, 4.6))
fig.suptitle("Export controls: leverage inverts from denial to delay-and-tax", fontsize=12.5, fontweight="bold", y=1.03)
b = a1.bar(["2024", "2025 Q1", "2025 Q2"], [13.5, 4.6, 0], color=US, width=0.55, zorder=3)
barlabels(a1, b, [13.5, 4.6, 0], fmt="${:g}B")
a1.set_ylim(0, 16)
a1.text(0.03, 0.94, "NVIDIA's H20 China revenue was cut to zero…\nthen restored under a 15% revenue-share deal\n(Aug-25), then case-by-case (Jan-26).\nDenial → transactional in 10 months.",
        transform=a1.transAxes, fontsize=7.8, color=US, va="top", ha="left")
a1.set_title("NVIDIA H20 China data-centre revenue ($B/qtr)"); style(a1, "$ billions", "Source: NVIDIA filings")
b = a2.bar(["Restricted-class chips\nthat reached China, 2025"], [660], color=CN, width=0.42, zorder=3)
barlabels(a2, b, [660], fmt="{:g}k H100e")
a2.set_ylim(0, 820)
a2.text(0.30, 0.94, "~660,000 H100-equivalents (median est.)\nreached China in 2025 despite the rules.\nThe negotiating chip depreciates precisely\nas it is used — leverage leaks.",
        transform=a2.transAxes, fontsize=7.8, color=CN, va="top", ha="left")
a2.set_title("The controls are porous"); style(a2, "Thousands of H100-equivalents", "Source: smuggling case estimates")
fig.tight_layout(); fig.savefig("gt10_export_controls.png", bbox_inches="tight", facecolor="white"); print("wrote gt10_export_controls.png")
