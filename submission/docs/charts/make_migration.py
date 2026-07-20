"""
The binding-constraint migration chart (gt_migration.png) — the on-thesis visual for
Part 2's conclusion. Transcribed from bench/dominance.json's binding_sequence: the binding
minimum hands off chips/EUV → power/minerals → power+aggregation across 2024–2030+, as
durable value migrates from the (commoditizing) frontier-training layer to the deployment/
distribution layer. Silicon → electrons; difficulty → durability.
Run: ../../.venv/bin/python make_migration.py
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Rectangle

US = "#1F5FA8"; CN = "#C0392B"; CONTEST = "#7D5BA6"; DIFF = "#157A6E"; GREY = "#9AA0A6"
plt.rcParams.update({"font.family": "DejaVu Sans", "axes.titlesize": 12.5, "axes.titleweight": "bold", "figure.dpi": 150})

fig, ax = plt.subplots(figsize=(11.5, 4.9))
ax.set_xlim(2024, 2030.4); ax.set_ylim(-0.2, 2.75)


def block(x0, x1, y, h, color, title, sub, favors, tcol="white"):
    ax.add_patch(Rectangle((x0, y), x1 - x0, h, facecolor=color, edgecolor="white", lw=2, zorder=3))
    xm = (x0 + x1) / 2
    ax.text(xm, y + h - 0.16, title, ha="center", va="top", fontsize=9.6, fontweight="bold", color=tcol, zorder=4)
    ax.text(xm, y + 0.30, sub, ha="center", va="center", fontsize=7.6, color=tcol, zorder=4, wrap=True)
    if favors:
        ax.text(xm, y - 0.12, favors, ha="center", va="top", fontsize=8, fontstyle="italic", color="#333", zorder=4)


# Row 1 — the binding constraint (the Liebig minimum) migrating
ax.text(2024, 2.62, "The binding constraint (Liebig minimum)", fontsize=8.5, color="#333", fontweight="bold")
block(2024, 2026, 1.75, 0.75, US, "Chips + EUV", "leading-edge training capacity", "▲ favors US / TSMC / ASML")
block(2026, 2028, 1.75, 0.75, CONTEST, "Power + minerals", "grid execution & transformers", "▲ favors China (power-abundant)")
block(2028, 2030.4, 1.75, 0.75, CN, "Power + aggregation", "inference at scale + distribution lock-in", "▲ favors power-rich, absorptive builder")

# Row 2 — where durable value sits
ax.text(2024, 1.48, "Where durable value sits", fontsize=8.5, color="#333", fontweight="bold")
block(2024, 2027, 0.55, 0.75, GREY, "Frontier model", "commoditizes in 3–6 months", "the least-durable prize", tcol="white")
block(2027, 2030.4, 0.55, 0.75, DIFF, "Deployment / inference + distribution", "recurring revenue · durable rent", "electrification & internet rhyme", tcol="white")

# migration arrow: silicon (top-left) -> electrons+distribution (bottom-right)
ax.add_patch(FancyArrowPatch((2025.0, 1.72), (2029.2, 1.33), connectionstyle="arc3,rad=-0.25",
             arrowstyle="-|>", mutation_scale=22, lw=2.4, color="#222", zorder=5))
ax.text(2027.1, 1.30, "silicon → electrons ·  value migrates toward the slowest gate",
        ha="center", fontsize=8.3, fontstyle="italic", color="#222")

ax.set_xticks([2024, 2025, 2026, 2027, 2028, 2029, 2030])
ax.set_xticklabels(["2024", "2025", "2026", "2027", "2028", "2029", "2030+"])
for s in ["top", "right", "left"]:
    ax.spines[s].set_visible(False)
ax.set_yticks([])
ax.set_title("The binding constraint migrates toward the value — the hardest gate guards the fastest-decaying prize", pad=14)
ax.text(2024, -0.15, "Sources: METR; LMArena; LBNL 'Queued Up'; NERC/IEA; China NEA; SEC filings — authors' Liebig sequencing. The frontier winner (US, training) is not the value winner (deployment).",
        fontsize=7.2, color=GREY)
fig.tight_layout()
fig.savefig("gt_migration.png", bbox_inches="tight", facecolor="white")
print("wrote gt_migration.png")
