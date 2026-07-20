"""Shared chart style — palette, rcParams, and the two helpers every make_*.py used.

Extracted to kill the copies that had drifted across the chart scripts. Import:

    from _style import US, CN, DIFF, GOLD, GREY, style, barlabels

Used by make_charts_gt.py, make_appendix.py, make_gt1_alt.py. Three scripts keep
local styles on purpose and are NOT rewired: make_charts_gt2.py (source caption at
-0.26, barlabels fs=9), make_gate_charts.py (denser rcParams for grid dashboards),
make_migration.py (distinct CONTEST/greys palette).
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# canonical two-actor palette
US = "#1F5FA8"; CN = "#C0392B"; DIFF = "#157A6E"; GOLD = "#C08A1E"; GREY = "#8A8A8A"

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                     "axes.titlesize": 11, "axes.titleweight": "bold",
                     "axes.edgecolor": "#333333", "figure.dpi": 150})


def style(ax, ylabel="", source=""):
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    ax.grid(axis="y", color="#DDDDDD", linewidth=0.7, zorder=0); ax.set_axisbelow(True)
    if ylabel: ax.set_ylabel(ylabel, fontsize=9)
    if source: ax.text(0, -0.24, source, transform=ax.transAxes, fontsize=7.3, color=GREY, ha="left", va="top")


def barlabels(ax, bars, vals, fmt="{:g}", dy=3, fs=8.5):
    for b, v in zip(bars, vals):
        ax.annotate(fmt.format(v), (b.get_x()+b.get_width()/2, b.get_height()),
                    textcoords="offset points", xytext=(0, dy), ha="center", fontsize=fs, fontweight="bold")
