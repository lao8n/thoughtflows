"""
gt1_alt.png — an alternative to gt1: the pure DIFFUSION / commoditization story.
Left: the prize evaporates (blended GPT-4-class API price collapse, 2020→2026).
Right: open weights take over (open-weight share of token usage, 2023→late-2025).
Together: the model layer is commoditizing on both price AND share — diffusion winning.
Run: ../../.venv/bin/python make_gt1_alt.py
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DIFF = "#157A6E"; GREY = "#8A8A8A"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                     "axes.titlesize": 11, "axes.titleweight": "bold",
                     "axes.edgecolor": "#333333", "figure.dpi": 150})


def style(ax, ylabel="", source=""):
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    ax.grid(axis="y", color="#DDDDDD", linewidth=0.7, zorder=0); ax.set_axisbelow(True)
    if ylabel: ax.set_ylabel(ylabel, fontsize=9)
    if source: ax.text(0, -0.24, source, transform=ax.transAxes, fontsize=7.3, color=GREY, ha="left", va="top")


fig, (a1, a2) = plt.subplots(1, 2, figsize=(11.5, 4.7))
fig.suptitle("The model layer is commoditizing — price craters as open weights take over",
             fontsize=12.5, fontweight="bold", y=1.03)

# Panel A — blended GPT-4-class API price collapse (log)
px = [2020, 2023.2, 2024.4, 2025, 2026]; py = [60, 36, 10, 1.9, 1.1]
a1.plot(px, py, "-o", color=DIFF, lw=2.6, ms=7, zorder=4); a1.set_yscale("log")
for x, y in zip(px, py):
    a1.annotate(f"${y:g}", (x, y), textcoords="offset points", xytext=(0, 9), ha="center",
                fontsize=8.5, color=DIFF, fontweight="bold")
a1.text(0.03, 0.30, "~55× cheaper, 2020→2026.\nThe rent in a proprietary model\nis arbitraged away as fast as\nit is created.", transform=a1.transAxes,
        fontsize=8, color=GREY, va="top")
a1.set_ylim(0.7, 90); a1.set_xticks([2020, 2022, 2024, 2026])
a1.set_title("The prize evaporates\n(blended GPT-4-class API price, $/M tokens)")
style(a1, "$ per million tokens (log)", "Source: blended GPT-4-class API list prices (OpenAI, TokenCost/BenchLM)")

# Panel B — open-weight share of token usage (%)
lx = ["2023", "2024-06", "2024-late", "2025-mid", "2025-late"]; ly = [4, 9, 13, 22, 33]
b = a2.bar(lx, ly, color=DIFF, width=0.62, zorder=3)
for bar, v in zip(b, ly):
    a2.annotate(f"{v}%", (bar.get_x()+bar.get_width()/2, bar.get_height()),
                textcoords="offset points", xytext=(0, 3), ha="center", fontsize=8.5, fontweight="bold")
a2.text(0.03, 0.95, "Open-weight share of tokens roughly\ntripled 2023→late-2025 — Chinese\nopen models alone went from ~1.2%\nto ~13% of weekly usage. The moat\nleaks to free weights, not just cheap APIs.",
        transform=a2.transAxes, fontsize=7.9, color=DIFF, va="top")
a2.set_ylim(0, 40)
a2.set_title("…and open weights take over\n(open-weight share of token usage, %)")
style(a2, "% of tokens served", "Source: OpenRouter / a16z 100T-token study")

fig.tight_layout()
fig.savefig("gt1_alt.png", bbox_inches="tight", facecolor="white")
print("wrote gt1_alt.png")
