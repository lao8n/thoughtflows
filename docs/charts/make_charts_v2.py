"""
Key charts for the data-led worldview (v2). Paired figures, house style.
Thesis: frontier diffuses; the build-out is CHIP-gated not energy-gated;
US leads today on chips+capital; China's energy/materials edge is stranded
behind the chip chokepoint. All data sourced in comments; approx points flagged.
Run: ../../.venv/bin/python make_charts_v2.py
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

US = "#1F5FA8"      # US / allied
CN = "#C0392B"      # China
GREY = "#8A8A8A"
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 10,
    "axes.titlesize": 11, "axes.titleweight": "bold",
    "axes.edgecolor": "#333333", "figure.dpi": 150,
})

def style(ax, ylabel="", source=""):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", color="#DDDDDD", linewidth=0.7, zorder=0)
    ax.set_axisbelow(True)
    if ylabel: ax.set_ylabel(ylabel, fontsize=9)
    if source: ax.text(0, -0.24, source, transform=ax.transAxes, fontsize=7.5,
                        color=GREY, ha="left", va="top")

def barlabels(ax, bars, vals, fmt="{:g}", dy=3, fs=9):
    for b, v in zip(bars, vals):
        ax.annotate(fmt.format(v), (b.get_x()+b.get_width()/2, b.get_height()),
                    textcoords="offset points", xytext=(0, dy), ha="center",
                    fontsize=fs, fontweight="bold")

# =====================================================================
# FIGURE 1 — Axis 1: Diffusion. Gap collapses | frontier no longer scarce
fig, (a1, a2) = plt.subplots(1, 2, figsize=(11.5, 4.4))

# 1a: best-closed minus best-open, Chatbot Arena (Stanford AI Index 2025)
x = [2024.05, 2024.5, 2025.1]; y = [8.0, 4.2, 1.7]
a1.plot(x, y, "-o", color=US, lw=2.4, ms=7)
for xi, yi in zip(x, y):
    a1.annotate(f"{yi:.1f}%", (xi, yi), textcoords="offset points", xytext=(0, 9),
                ha="center", fontsize=9, color=US)
a1.set_ylim(0, 9); a1.set_xticks(x); a1.set_xticklabels(["Jan-24", "Mid-24", "Feb-25"])
a1.set_title("The gap to the leading model has collapsed")
style(a1, "Best closed − best open (Chatbot Arena, %)", "Source: Stanford AI Index 2025")

# 1b: number of developers with a frontier-scale (>1e25 FLOP) model (Epoch AI)
#   endpoint sourced (12 developers, >30 models, mid-2025); trajectory approx.
yr = [2022, 2023, 2024, 2025]; n = [1, 4, 9, 12]
bars = a2.bar(yr, n, color=US, width=0.6, zorder=3)
barlabels(a2, bars, n)
a2.set_ylim(0, 14); a2.set_xticks(yr)
a2.annotate(">30 models have crossed GPT-4 scale;\n17 have held the #1 spot since GPT-4",
            (2022, 13), fontsize=8, color=GREY, va="top")
a2.set_title("Frontier capability is no longer scarce")
style(a2, "Developers with a GPT-4-scale (>10²⁵ FLOP) model", "Source: Epoch AI (2025; trajectory approx.)")
fig.tight_layout(); fig.savefig("v2_1_diffusion.png", bbox_inches="tight", facecolor="white")

# =====================================================================
# FIGURE 2 — HERO. China has the power; America has the compute → chips bind
fig, (a1, a2) = plt.subplots(1, 2, figsize=(11.5, 4.6))
fig.suptitle("China builds ~9× the power — but less AI compute. The binding input is silicon, not electrons.",
             fontsize=12, fontweight="bold", y=1.02)

# 2a: new power capacity added 2024 (Ember / EIA)
b = a1.bar(["China", "United States"], [429, 48.6], color=[CN, US], width=0.55, zorder=3)
barlabels(a1, b, [429, 48.6], fmt="{:g} GW")
a1.set_ylim(0, 480)
a1.set_title("China builds the power\n(new generating capacity, 2024)")
style(a1, "GW added in 2024", "Sources: Ember (China); EIA (US)")

# 2b: installed AI data-center capacity (2025, digitalinformationworld)
b = a2.bar(["United States", "China"], [53.7, 31.9], color=[US, CN], width=0.55, zorder=3)
barlabels(a2, b, [53.7, 31.9], fmt="{:g} GW")
a2.set_ylim(0, 62)
a2.set_title("…but America builds the compute\n(installed AI data-center capacity, 2025)")
style(a2, "GW of AI data-center capacity", "Source: data-center capacity trackers, 2025")
fig.tight_layout(); fig.savefig("v2_2_hero_powervchips.png", bbox_inches="tight", facecolor="white")

# =====================================================================
# FIGURE 3 — Axis 5: Controls leak | but the two real chokepoints hold
fig, (a1, a2) = plt.subplots(1, 2, figsize=(11.5, 4.6))

# 3a: restricted chips reaching China despite tightening (The Information / CNAS)
yr = [2022, 2023, 2024]; chips = [10, 60, 140]  # thousands; 2024 ~140k sourced, earlier approx
b = a1.bar(yr, chips, color=CN, width=0.6, zorder=3)
barlabels(a1, b, chips, fmt="~{:g}k")
a1.set_ylim(0, 170); a1.set_xticks(yr)
a1.axvline(2022, color=GREY, ls=":", lw=1); a1.axvline(2023, color=GREY, ls=":", lw=1)
a1.annotate("controls tightened\n2022 · 2023 · 2025", (2022.1, 160), fontsize=7.5, color=GREY, va="top")
a1.set_title("Chip-flow controls leak\n(restricted accelerators reaching China / yr)")
style(a1, "Thousands of advanced chips", "Source: The Information; CNAS (2022–23 approx.)")

# 3b: the two chokepoints that hold — mutual. US/allied tool vs China materials
items = ["EUV lithography\n(ASML)", "Rare-earth\nrefining", "Gallium", "Germanium"]
share = [100, 90, 98, 68]
colors = [US, CN, CN, CN]
b = a2.barh(items[::-1], share[::-1], color=colors[::-1], zorder=3)
for bar, v in zip(b, share[::-1]):
    a2.annotate(f"{v}%", (v, bar.get_y()+bar.get_height()/2), textcoords="offset points",
                xytext=(4, 0), va="center", fontsize=9, fontweight="bold")
a2.set_xlim(0, 115)
a2.set_title("…but the real chokepoints hold — on both sides\n(blue = US/allied, red = China)")
style(a2, "", "Sources: ASML; IEA; USGS; CSIS")
fig.tight_layout(); fig.savefig("v2_3_chokepoint.png", bbox_inches="tight", facecolor="white")

# =====================================================================
# FIGURE 4 — Axis 4: America's edge is capital | outspends China ~4x
fig, (a1, a2) = plt.subplots(1, 2, figsize=(11.5, 4.6))

# 4a: AI/hyperscaler capex as % of US GDP vs historical build-outs
labels = ["Manhattan\n(~0.3%)", "Apollo\n(~0.4%)", "Dot-com\nfiber (~1%)", "AI capex\n2026 (~2.2%)", "Railroads\npeak (~6%)"]
vals = [0.3, 0.4, 1.0, 2.2, 6.0]
cols = [GREY, GREY, GREY, US, GREY]
b = a1.bar(labels, vals, color=cols, width=0.62, zorder=3)
barlabels(a1, b, vals, fmt="{:g}%", fs=8)
a1.set_ylim(0, 6.8)
a1.set_title("America's edge is capital:\na private-capital Apollo every ~10 months")
style(a1, "Peak/annual capex, % of US GDP", "Sources: hyperscaler filings; USITC/BEA (historical approx.)")

# 4b: AI infrastructure investment 2025, US vs China
b = a2.bar(["US hyperscalers", "China (total AI)"], [370, 98], color=[US, CN], width=0.55, zorder=3)
barlabels(a2, b, [370, 98], fmt="${:g}B")
a2.set_ylim(0, 420)
a2.annotate("of which ~$56B is Chinese state funds", (1, 98), textcoords="offset points",
            xytext=(0, 20), ha="center", fontsize=7.5, color=GREY)
a2.set_title("…and outspends China ~4×\n(AI infrastructure investment, 2025)")
style(a2, "$ billion, 2025", "Sources: company disclosures; Goldman Sachs")
fig.tight_layout(); fig.savefig("v2_4_capital.png", bbox_inches="tight", facecolor="white")

print("wrote v2_1_diffusion.png, v2_2_hero_powervchips.png, v2_3_chokepoint.png, v2_4_capital.png")
