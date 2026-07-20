"""
Two paired appendix figures for Part 3.
  appendix1_china.png   — the deployment engine (Kai-Fu-Lee loop evidence for §3):
                          China vs US industrial-robot installs | open-weight token share
  appendix2_structure.png — the framework structure:
                          durability ladder | scenario probability fan
Run: ../../.venv/bin/python make_appendix.py
"""
from _style import US, CN, DIFF, GOLD, GREY, style, barlabels
import matplotlib.pyplot as plt

GOOD = "#2E7D5B"; MID = "#C08A1E"; BAD = "#C0392B"


# ================= FIGURE 1 — China's deployment engine
fig, (a1, a2) = plt.subplots(1, 2, figsize=(11.5, 4.6))
fig.suptitle("China's deployment engine — the affirmative case for winning the inference tier",
             fontsize=12.3, fontweight="bold", y=1.03)
# 1a: industrial-robot installations, China vs US (IFR, approx)
yrs = ["2019", "2021", "2023"]; china = [140, 268, 276]; usa = [33, 35, 37]
x = range(len(yrs)); w = 0.38
bc = a1.bar([i - w/2 for i in x], china, w, color=CN, zorder=3, label="China")
bu = a1.bar([i + w/2 for i in x], usa, w, color=US, zorder=3, label="United States")
barlabels(a1, bc, china, fmt="{:g}k"); barlabels(a1, bu, usa, fmt="{:g}k")
a1.set_xticks(list(x)); a1.set_xticklabels(yrs); a1.set_ylim(0, 320)
a1.legend(loc="upper left", fontsize=8.5, frameon=False)
a1.text(0.30, 0.80, "China installs ~52% of the world's\nindustrial robots — ~7× the US.\nThe manufacturing/physical-AI base\nthat converts cheap compute into\nrobots, drones, and deployed value.",
        transform=a1.transAxes, fontsize=7.8, color=GREY, va="top")
a1.set_title("China owns the physical-AI base\n(annual industrial-robot installations)")
style(a1, "Installations (thousands/yr)", "Source: IFR World Robotics (approx.)")
# 1b: open-weight token share, Chinese portion highlighted
lx = ["2023", "2024-06", "2024-late", "2025-mid", "2025-late"]; ly = [4, 9, 13, 22, 33]
b = a2.bar(lx, ly, color=DIFF, width=0.62, zorder=3); barlabels(a2, b, ly, fmt="{:g}%")
a2.set_ylim(0, 40)
a2.text(0.03, 0.95, "Open weights went 4%→33% of tokens;\nChinese open models (Qwen, DeepSeek)\nare ~13% of weekly usage and rising —\nthe diffusion tier China leads.",
        transform=a2.transAxes, fontsize=7.8, color=DIFF, va="top")
a2.set_title("…and leads the open-weight tier\n(open-weight share of token usage, %)")
style(a2, "% of tokens served", "Source: OpenRouter / a16z 100T-token study")
fig.tight_layout(); fig.savefig("appendix1_china.png", bbox_inches="tight", facecolor="white"); print("wrote appendix1_china.png")


# ================= FIGURE 2 — the framework structure
fig, (a1, a2) = plt.subplots(1, 2, figsize=(11.5, 4.6))
fig.suptitle("The framework in two pictures — what lasts, and how the decade resolves",
             fontsize=12.3, fontweight="bold", y=1.03)
# 2a: durability ladder (horizontal, most→least durable)
tiers = ["Standards / networks", "Energy + absorptive cap.", "Precision tooling",
         "Capital / finance", "Manufacturing capacity", "Talent / knowledge",
         "Materials / minerals", "Inventions (models)"]
score = [7, 6, 5, 4, 3, 2.2, 1.6, 1]
cols = [GOOD, GOOD, GOOD, MID, MID, BAD, BAD, BAD]
maps = ["CUDA, cloud, dollar", "China power + industry", "ASML/Zeiss", "hyperscaler/Gulf",
        "chips → caught up", "diffuses (Fuchs)", "routed around", "open weights"]
ypos = range(len(tiers))
a1.barh(list(ypos), score, color=cols, zorder=3, height=0.7)
a1.set_yticks(list(ypos)); a1.set_yticklabels(tiers, fontsize=8.2)
a1.invert_yaxis()
for i, (s, m) in enumerate(zip(score, maps)):
    a1.text(s + 0.1, i, m, va="center", fontsize=7, color="#444")
a1.set_xlim(0, 8.6); a1.set_xticks([])
a1.spines["top"].set_visible(False); a1.spines["right"].set_visible(False); a1.spines["bottom"].set_visible(False)
a1.set_title("Durability ladder\n(how long control has historically lasted)")
a1.text(0, -0.06, "green = durable · gold = fades · red = eroded/routed-around · AI mapping at right",
        transform=a1.transAxes, fontsize=7, color=GREY)
# 2b: scenario probability fan
sc = ["Base:\ntwo-tier\ndeployment", "Perez\nbust", "Taiwan\nfracture", "RSI\ntakeover"]
pr = [55, 20, 13, 12]; scol = [DIFF, GOLD, CN, "#5B2A86"]
b = a2.bar(sc, pr, color=scol, width=0.62, zorder=3); barlabels(a2, b, pr, fmt="{:g}%")
a2.set_ylim(0, 65)
a2.set_title("Scenario probabilities\n(the probability-weighted fan)")
style(a2, "Probability (%)", "Source: author's systems-map synthesis")
fig.tight_layout(); fig.savefig("appendix2_structure.png", bbox_inches="tight", facecolor="white"); print("wrote appendix2_structure.png")
