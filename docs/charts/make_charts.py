"""
Appendix chart page for the Bridgewater submission — "the five axes, in data".
House style: takeaway titles, blue actuals, red dashed forecasts/estimates,
light y-grid, source line bottom-left. All data points are sourced in comments;
interpolated/approx points are flagged.

Run: ../../.venv/bin/python make_charts.py   (writes PNGs into this dir)
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

BLUE = "#1F5FA8"      # actuals
RED  = "#C0392B"      # forecasts / estimates
GREY = "#8A8A8A"
GREEN = "#2E7D5B"

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.titleweight": "bold",
    "axes.edgecolor": "#333333",
    "figure.dpi": 150,
})

def style(ax, ylabel="", source=""):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", color="#DDDDDD", linewidth=0.7, zorder=0)
    ax.set_axisbelow(True)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=9)
    if source:
        ax.text(0, -0.22, source, transform=ax.transAxes, fontsize=7.5,
                color=GREY, ha="left", va="top")

fig, axes = plt.subplots(3, 2, figsize=(11.5, 14.5))
fig.suptitle("Appendix — The five axes, in data", fontsize=15, fontweight="bold", y=0.995)

# ---------------------------------------------------------------------------
# Panel A · Axis 1 — Diffusion: open models catch the closed frontier
# Source: Stanford AI Index 2025 — best-closed minus best-open on Chatbot Arena
# 8.0% (Jan-24) -> 4.2% (mid-24) -> 1.7% (Feb-25)
ax = axes[0, 0]
x = [2024.05, 2024.5, 2025.1]
y = [8.0, 4.2, 1.7]
ax.plot(x, y, "-o", color=BLUE, lw=2.2, ms=6, zorder=3)
for xi, yi in zip(x, y):
    ax.annotate(f"{yi:.1f}%", (xi, yi), textcoords="offset points",
                xytext=(0, 8), ha="center", fontsize=8, color=BLUE)
ax.set_ylim(0, 9)
ax.set_xticks([2024.05, 2024.5, 2025.1])
ax.set_xticklabels(["Jan-24", "Mid-24", "Feb-25"])
ax.set_title("Diffusion beats concentration: the best open model\nnow trails the closed frontier by <2%")
style(ax, "Best closed − best open (Chatbot Arena, %)", "Source: Stanford AI Index 2025")

# ---------------------------------------------------------------------------
# Panel B · Axis 2 — Physical: the interconnection queue hockey-stick
# Source: LBNL "Queued Up" editions. end-2020 ~1,400 GW; end-2022 ~2,020 GW;
# end-2023 ~2,600 GW (up 27% YoY; ~2x total US installed capacity; median wait 5 yrs)
ax = axes[0, 1]
x = [2020, 2022, 2023]
y = [1400, 2020, 2600]
ax.fill_between(x, y, color=BLUE, alpha=0.12, zorder=1)
ax.plot(x, y, "-o", color=BLUE, lw=2.2, ms=6, zorder=3)
ax.annotate("2,600 GW\n≈2× total US installed capacity\nmedian wait: 5 years",
            (2023, 2600), textcoords="offset points", xytext=(-8, -6),
            ha="right", va="top", fontsize=8, color=BLUE)
ax.set_ylim(0, 3000)
ax.set_xticks([2020, 2022, 2023])
ax.set_xticklabels(["end-2020", "end-2022", "end-2023"])
ax.set_title("The binding constraint is the grid, not the chip:\nthe US interconnection queue hit ~2,600 GW")
style(ax, "Capacity seeking interconnection (GW)", "Source: LBNL, Queued Up 2024")

# ---------------------------------------------------------------------------
# Panel C · Axis 2/F — China vs US power capacity additions, 2024
# Source: Ember (China 429 GW net new, 2024) ; EIA (US 48.6 GW installed, 2024)
ax = axes[1, 0]
bars = ax.bar(["China", "United States"], [429, 48.6],
              color=[RED, BLUE], width=0.55, zorder=3)
for b, v in zip(bars, [429, 48.6]):
    ax.annotate(f"{v:g} GW", (b.get_x() + b.get_width()/2, v),
                textcoords="offset points", xytext=(0, 4), ha="center",
                fontsize=9, fontweight="bold")
ax.set_ylim(0, 480)
ax.set_title("The constraint binds asymmetrically: China added\n~9× more power capacity than the US in 2024")
style(ax, "Net new generating capacity, 2024 (GW)", "Sources: Ember (China); EIA (US)")

# ---------------------------------------------------------------------------
# Panel D · Axis 3 — Rent flees the model (dual axis): price collapses, margin holds
# Left  (log): cheapest GPT-4-class output price $/1M tokens (published pricing)
#   GPT-4 $60 (Mar-23) -> GPT-4o $15 (May-24) -> GPT-4o-mini $0.60 (Jul-24) -> Gemini Flash $0.40 (2026)
# Right: NVIDIA gross margin % (SEC 8-Ks): 65% (early-23) -> 76% (Q4 FY24) -> ~73% (FY26)
ax = axes[1, 1]
px = [2023.2, 2024.4, 2024.55, 2026.2]
py = [60, 15, 0.60, 0.40]
ax.plot(px, py, "-o", color=BLUE, lw=2.2, ms=5, zorder=3, label="Cheapest GPT-4-class price ($/1M tok)")
ax.set_yscale("log")
ax.set_ylim(0.2, 100)
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"${v:g}"))
ax.annotate("−99%", (2026.2, 0.40), textcoords="offset points",
            xytext=(-4, 10), ha="right", fontsize=9, color=BLUE, fontweight="bold")
style(ax, "Price $/1M output tokens (log)",
      "Sources: published API pricing (OpenAI/Google); NVIDIA SEC filings")
ax2 = ax.twinx()
mx = [2023.2, 2024.0, 2026.0]
my = [65, 76, 73]
ax2.plot(mx, my, "-s", color=RED, lw=2.2, ms=5, zorder=3, label="NVIDIA gross margin (%)")
ax2.set_ylim(0, 100)
ax2.set_ylabel("NVIDIA gross margin (%)", fontsize=9, color=RED)
ax2.tick_params(axis="y", colors=RED)
ax2.spines["top"].set_visible(False)
ax2.annotate("~75% held\n(ASML EUV ≈100% share)", (2026.0, 73),
             textcoords="offset points", xytext=(-6, -4), ha="right", va="top",
             fontsize=8, color=RED)
ax.set_xlim(2022.9, 2026.6)
ax.set_title("Rent flees the model: the model layer commoditizes (−99%)\nwhile the tooling layer keeps its margin (~75%)")

# ---------------------------------------------------------------------------
# Panel E · Axis 4 — Macro: interest costs vs revenue, past the 1991 record
# Source: CBO / CRFB / PGPF. 2024 = 18.5% (new record, beats 1991); CBO 25.8% by 2036.
# ~2021 low ~8% (approx, low-rate era). 1991 prior high ~18% (reference line).
ax = axes[2, 0]
ax_actual_x = [2021, 2024]
ax_actual_y = [8.5, 18.5]
fc_x = [2024, 2030, 2036]
fc_y = [18.5, 22.0, 25.8]
ax.plot(ax_actual_x, ax_actual_y, "-o", color=BLUE, lw=2.2, ms=6, zorder=3, label="Actual")
ax.plot(fc_x, fc_y, "--o", color=RED, lw=2.2, ms=5, zorder=3, label="CBO projection")
ax.axhline(18, color=GREY, ls=":", lw=1)
ax.annotate("1991 record (~18%)", (2021.1, 18), textcoords="offset points",
            xytext=(0, 4), fontsize=7.5, color=GREY)
ax.annotate("25.8% by 2036", (2036, 25.8), textcoords="offset points",
            xytext=(-4, 6), ha="right", fontsize=8, color=RED, fontweight="bold")
ax.set_ylim(0, 30)
ax.set_xticks([2021, 2024, 2030, 2036])
ax.legend(frameon=False, fontsize=8, loc="upper left")
ax.set_title("The bill before the payoff: US interest costs blew past\ntheir 1991 record; CBO sees 26% of revenue by 2036")
style(ax, "Federal net interest, % of revenue", "Sources: CBO, CRFB (2021 approx.)")

# ---------------------------------------------------------------------------
# Panel F · Axis 5 — Controls leak / fragmentation atop flows: US-China goods trade
# Source: US Census. 2018 ~$659B; 2020 ~$560B (COVID+tariffs); 2022 ~$690B; 2024 ~$661B.
ax = axes[2, 1]
x = [2018, 2020, 2022, 2024]
y = [659, 560, 690, 661]
ax.plot(x, y, "-o", color=BLUE, lw=2.2, ms=6, zorder=3)
for xi, yi in zip(x, y):
    ax.annotate(f"${yi:g}B", (xi, yi), textcoords="offset points",
                xytext=(0, 8), ha="center", fontsize=8, color=BLUE)
ax.axhline(690 * 0.8, color=GREY, ls=":", lw=1)
ax.annotate("80% of 2022 peak", (2018, 690*0.8), textcoords="offset points",
            xytext=(0, 4), fontsize=7.5, color=GREY)
ax.set_ylim(0, 780)
ax.set_xticks([2018, 2020, 2022, 2024])
ax.set_title("The tools of control leak: two stacks, but goods keep\nflowing — 2024 trade = 96% of the 2022 peak")
style(ax, "US-China two-way goods trade ($B)",
      "Source: US Census Bureau  ·  (~140k restricted chips still reached China in 2024)")

fig.tight_layout(rect=[0, 0.01, 1, 0.985])
fig.subplots_adjust(hspace=0.55, wspace=0.28)
fig.savefig("appendix_charts.png", bbox_inches="tight", facecolor="white")
print("wrote appendix_charts.png")
