"""
v3 chart — the US firm-power ceiling on its OWN build-out.
Thesis (actor-specific): the US is energy/execution-gated on the slow clock —
data-centre load surges onto a grid that was flat for a decade, ~80% of new US
supply is intermittent, and reshored fabs compete for the same firm power and the
same 3-5yr interconnection queue. China faces the opposite problem: power oversupply,
chips bind. This is the US half of the "two clocks" story (pairs with the hero chart).
All data sourced in comments; approximate points flagged.
Run: ../../.venv/bin/python make_charts_v3.py
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

US = "#1F5FA8"       # US / allied
CN = "#C0392B"       # China
FIRM = "#1F5FA8"     # dispatchable
INT = "#8FB8DE"      # intermittent (light blue)
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
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=9)
    if source:
        ax.text(0, -0.26, source, transform=ax.transAxes, fontsize=7.5,
                color=GREY, ha="left", va="top")


def barlabels(ax, bars, vals, fmt="{:g}", dy=3, fs=9):
    for b, v in zip(bars, vals):
        ax.annotate(fmt.format(v), (b.get_x() + b.get_width() / 2, b.get_height()),
                    textcoords="offset points", xytext=(0, dy), ha="center",
                    fontsize=fs, fontweight="bold")


# =====================================================================
# FIGURE — the US is energy-gated on its own build-out
fig, (a1, a2) = plt.subplots(1, 2, figsize=(11.5, 4.7))
fig.suptitle("The US is energy-gated on its own build-out — China isn't",
             fontsize=12.5, fontweight="bold", y=1.03)

# --- 2a: data-centre share of total US electricity, onto a flat grid ---
# LBNL 2024 US Data Center Energy Usage Report: 1.9% (2018), 4.4% (2023),
# 6.7-12% (2028), 9.5-15.3% (2030). Historical share flat because efficiency
# offset growth; total US demand ~flat 2007-2020 (~3,900-4,000 TWh).
hx = [2018, 2020, 2023]
hy = [1.9, 2.5, 4.4]
a1.plot(hx, hy, "-o", color=US, lw=2.6, ms=6, zorder=4)
# projection band to 2030 (LBNL scenario range), from 2023 anchor
px = [2023, 2028, 2030]
lo = [4.4, 6.7, 9.5]
hi = [4.4, 12.0, 15.3]
mid = [4.4, 9.35, 12.4]
a1.fill_between(px, lo, hi, color=US, alpha=0.16, zorder=2, label="LBNL 2028/2030 scenario range")
a1.plot(px, mid, "--", color=US, lw=2.0, zorder=3)
for xi, yi in [(2018, 1.9), (2023, 4.4)]:
    a1.annotate(f"{yi:g}%", (xi, yi), textcoords="offset points", xytext=(0, 9),
                ha="center", fontsize=9, color=US, fontweight="bold")
a1.annotate("9.5–15.3%\nby 2030", (2030, 12.4), textcoords="offset points",
            xytext=(-4, -2), ha="right", va="center", fontsize=9, color=US, fontweight="bold")
a1.annotate("US total electricity demand was ~flat 2007–2020;\ndata-centre load alone now adds a Germany.",
            (2018, 15.0), fontsize=8, color=GREY, va="top")
a1.set_ylim(0, 16)
a1.set_xlim(2017.5, 2030.5)
a1.set_xticks([2018, 2020, 2023, 2028, 2030])
a1.set_title("Data centres surge onto a grid that stopped growing\n(share of total US electricity)")
style(a1, "Data centres, % of US electricity", "Source: LBNL 2024 US Data Center Energy Usage Report (2028/30 = scenario range)")

# --- 2b: new power capacity added 2024 — US (mostly intermittent) vs China ---
# EIA: US added ~56 GW in 2024, ~81% solar+storage (intermittent), ~19% firm.
# Ember: China added ~429 GW in 2024. Data-centre firm-load growth needs 24/7 power.
us_firm = 56 * 0.19       # ~10.6 GW dispatchable
us_int = 56 * 0.81        # ~45.4 GW intermittent
b1 = a2.bar(["United\nStates"], [us_firm], color=FIRM, width=0.5, zorder=3, label="Firm / dispatchable")
b1i = a2.bar(["United\nStates"], [us_int], bottom=[us_firm], color=INT, width=0.5, zorder=3,
             label="Intermittent (solar+storage)")
b2 = a2.bar(["China"], [429], color=CN, width=0.5, zorder=3)
a2.annotate("56 GW\n(~81% intermittent)", (0, 56), textcoords="offset points",
            xytext=(0, 4), ha="center", fontsize=8.5, fontweight="bold")
a2.annotate("429 GW", (1, 429), textcoords="offset points", xytext=(0, 4),
            ha="center", fontsize=9.5, fontweight="bold")
a2.annotate("~8×", (0.5, 205), ha="center", fontsize=13, color=GREY, fontweight="bold")
a2.annotate("US: ½ of builds\ndelayed (power+\nparts); fabs share\nthe 3–5yr queue", (0.5, 375),
            ha="center", va="top", fontsize=7.8, color=US, fontweight="bold")
a2.annotate("China: power glut,\nchips bind —\nnot electrons", (0.5, 120),
            ha="center", va="top", fontsize=7.8, color=CN, fontweight="bold")
a2.set_ylim(0, 470)
a2.legend(loc="upper center", fontsize=7.3, frameon=False, ncol=1, bbox_to_anchor=(0.28, 1.0))
a2.set_title("China builds ~8× the power — and has too much\n(new generating capacity added, 2024)")
style(a2, "GW added in 2024", "Sources: EIA (US, incl. fuel mix); Ember (China); Tom's Hardware (delays)")

fig.tight_layout()
fig.savefig("v3_1_us_energy_gate.png", bbox_inches="tight", facecolor="white")
print("wrote v3_1_us_energy_gate.png")


# =====================================================================
# FIGURE 2 — the DIFFUSION gate: US-China model gap is closing fast
fig, (a1, a2) = plt.subplots(1, 2, figsize=(11.5, 4.6))
fig.suptitle("The model gate is closing fast — the diffusion clock",
             fontsize=12.5, fontweight="bold", y=1.03)

# 2a: US-China top-model performance gap, over time (Stanford HAI AI Index)
# May-2023: 17.5-31.6 pp; by late-2024 ~5 pp; 2026 AI Index: 2.7 pp.
gx = [2023.4, 2024.0, 2024.9, 2026.0]
gy = [24.0, 12.0, 5.0, 2.7]
glo = [17.5, 9.0, 3.0, 2.7]
ghi = [31.6, 16.0, 7.0, 2.7]
a1.fill_between(gx, glo, ghi, color=US, alpha=0.15, zorder=2)
a1.plot(gx, gy, "-o", color=US, lw=2.6, ms=7, zorder=4)
a1.annotate("17.5–31.6 pp\n(May 2023)", (2023.4, 24), textcoords="offset points",
            xytext=(8, 6), fontsize=8.5, color=US, va="bottom")
a1.annotate("2.7 pp\n(2026)", (2026.0, 2.7), textcoords="offset points",
            xytext=(-6, 14), ha="right", fontsize=9, color=US, fontweight="bold")
a1.set_ylim(0, 34)
a1.set_xlim(2023.2, 2026.4)
a1.set_xticks([2023.4, 2024.0, 2024.9, 2026.0])
a1.set_xticklabels(["May-23", "Jan-24", "late-24", "2026"])
a1.set_title("US–China best-model gap has all but closed\n(performance-benchmark gap, pp)")
style(a1, "Best-US minus best-China (percentage points)", "Source: Stanford HAI AI Index (2025, 2026)")

# 2b: lag in months for China's best to match a given US frontier release
labels = ["GPT-4 class\n(2023→2024)", "o1→R1\n(2024→2025)", "frontier\n(2026)"]
lag = [14, 4, 8]
bars = a2.bar(labels, lag, color=[US, US, US], width=0.58, zorder=3)
barlabels(a2, bars, lag, fmt="{:g} mo")
a2.set_ylim(0, 17)
a2.annotate("Time for China's best to match\na given US frontier release —\ncollapsed from 14mo to single\ndigits.",
            (0.52, 16.5), ha="left", fontsize=7.9, color=GREY, va="top")
a2.annotate("3 Chinese models now in Arena\ntop-10 (0 a year earlier); Qwen >\nLlama at >50% of open-weight\ndownloads.",
            (0.52, 11.3), ha="left", fontsize=7.6, color=CN, va="top")
a2.set_title("China's lag behind the US frontier\n(months to match a given release)")
style(a2, "Lag, months", "Sources: Stanford AI Index; CAISI (2026); Chatbot Arena")
fig.tight_layout()
fig.savefig("v3_2_diffusion_gate.png", bbox_inches="tight", facecolor="white")
print("wrote v3_2_diffusion_gate.png")


# =====================================================================
# FIGURE 3 — the CHIP gate: node gap stuck; the no-EUV tax caps it
fig, (a1, a2) = plt.subplots(1, 2, figsize=(11.5, 4.6))
fig.suptitle("The chip gate is stuck — a soft gate with a heavy tax",
             fontsize=12.5, fontweight="bold", y=1.03)

# 3a: best node in VOLUME production, TSMC vs SMIC, by year (lower nm = better)
tx = [2018, 2020, 2022, 2025]; tn = [7, 5, 3, 2]
sx = [2019, 2022, 2025];       sn = [14, 7, 5]
a1.plot(tx, tn, "-o", color=US, lw=2.5, ms=7, zorder=4, label="TSMC (leading)")
a1.plot(sx, sn, "-s", color=CN, lw=2.5, ms=7, zorder=4, label="SMIC (China, no EUV)")
for x, n in zip(tx, tn):
    a1.annotate(f"{n}nm", (x, n), textcoords="offset points", xytext=(0, -14),
                ha="center", fontsize=8, color=US)
for x, n in zip(sx, sn):
    a1.annotate(f"{'~' if n==5 else ''}{n}nm", (x, n), textcoords="offset points", xytext=(0, 9),
                ha="center", fontsize=8, color=CN)
a1.invert_yaxis()
a1.set_ylim(16, 1)
a1.set_xlim(2017.5, 2026)
a1.set_xticks([2018, 2020, 2022, 2025])
a1.annotate("Gap stuck at ~4–5 years —\nand SMIC's 5nm is DUV density,\nnot true leading-edge.", (2018.2, 3),
            fontsize=8, color=GREY, va="top")
a1.legend(loc="lower left", fontsize=8.5, frameon=False, bbox_to_anchor=(0.0, 0.02))
a1.set_title("Node gap isn't closing on leading-edge logic\n(best node in volume production)")
style(a1, "Process node (nm) — lower is better", "Sources: Semiecosystem; TrendForce (2025)")

# 3b: the no-EUV tax — SMIC ~5nm-equiv (N+3) vs TSMC 5nm
metrics = ["Yield", "Cost / wafer\n(TSMC=100)"]
tsmc = [90, 100]; smic = [33, 150]
xp = range(len(metrics)); w = 0.36
b1 = a2.bar([p - w/2 for p in xp], tsmc, w, color=US, zorder=3, label="TSMC 5nm (EUV)")
b2 = a2.bar([p + w/2 for p in xp], smic, w, color=CN, zorder=3, label="SMIC ~5nm (DUV)")
barlabels(a2, b1, tsmc, fmt="{:g}")
barlabels(a2, b2, smic, fmt="{:g}")
a2.set_xticks(list(xp)); a2.set_xticklabels(metrics)
a2.set_ylim(0, 170)
a2.annotate("Soft gate, heavy tax:\nChina reaches ~5nm density\nat ⅓ yield, +50% cost →\nsecond-tier economics.",
            (0.5, 158), ha="center", fontsize=7.7, color=GREY, va="top")
a2.legend(loc="upper left", fontsize=8, frameon=False)
a2.set_title("…because the no-EUV workaround is a heavy tax\n(SMIC N+3 vs TSMC 5nm)")
style(a2, "Index (yield %, cost=100)", "Source: Semiecosystem (2025); yields approximate")
fig.tight_layout()
fig.savefig("v3_3_chip_gate.png", bbox_inches="tight", facecolor="white")
print("wrote v3_3_chip_gate.png")
