"""
Part 2 hero charts — LONG-SPAN versions, transcribed from the fresh gate-tracker
(bench/gate-tracker.json; 5-7 metrics x >=5 points, pre-2020 -> 2026). Four figures
carry the integrated argument, each now with real history (2018-2020 baselines):
  gt1  the thesis      — the prize evaporates (API $ since 2020) while the moat widens (transformer lead since 2018)
  gt2  the US bind     — China's power ramp since 2015 vs the US interconnection queue since 2015
  gt3  master switch   — METR task-horizon since 2019 vs the closed-open capability gap closing since 2023
  gt4  value-capture   — hyperscaler capex since 2019 vs the model-API price collapse since 2020
Run: ../../.venv/bin/python make_charts_gt.py
"""
from _style import US, CN, DIFF, GOLD, GREY, style, barlabels
import matplotlib.pyplot as plt


# ============ GT1 — THESIS: prize evaporates vs moat widens (long span)
fig, (a1, a2) = plt.subplots(1, 2, figsize=(11.5, 4.7))
fig.suptitle("The hardware moats are hardening exactly as the capability they guard goes free",
             fontsize=12.5, fontweight="bold", y=1.03)
# blended frontier LLM API price ($/M tokens, GPT-4-class) — GV15
px = [2020, 2023.2, 2024.4, 2025, 2026]; py = [60, 36, 10, 1.9, 1.1]
a1.plot(px, py, "-o", color=DIFF, lw=2.6, ms=7, zorder=4); a1.set_yscale("log")
for x, y in zip(px, py):
    a1.annotate(f"${y:g}", (x, y), textcoords="offset points", xytext=(0, 9), ha="center", fontsize=8.5, color=DIFF, fontweight="bold")
a1.text(0.03, 0.30, "~55× cheaper, 2020→2026.\nThe capability the moats guard\nis commoditizing toward zero rent.",
        transform=a1.transAxes, fontsize=8, color=GREY, va="top")
a1.set_ylim(0.7, 90); a1.set_xticks([2020, 2022, 2024, 2026])
a1.set_title("The prize evaporates\n(blended GPT-4-class API price, $/M tokens)")
style(a1, "$ per million tokens (log)", "Source: blended GPT-4-class API list prices (OpenAI, TokenCost/BenchLM)")
# US large power transformer lead time (months) — GV6
tx = [2018, 2020, 2022, 2024, 2025, 2026]; ty = [12, 16, 24, 28, 33, 40]
b = a2.bar([str(x) for x in tx], ty, color=US, width=0.6, zorder=3); barlabels(a2, b, ty, fmt="{:g}")
a2.text(0.03, 0.95, "1 year → 3.3 years. The physical\nbottleneck to DEPLOYING compute\ntripled as the compute got ~55×\ncheaper. ~2,300 GW waits in the\nUS grid queue.",
        transform=a2.transAxes, fontsize=7.9, color=US, va="top")
a2.set_ylim(0, 48)
a2.set_title("…while the moat widens\n(US power-transformer lead time, months)")
style(a2, "Months (order → delivery)", "Source: US large-power-transformer lead time (NERC, IEA)")
fig.tight_layout(); fig.savefig("gt1_thesis.png", bbox_inches="tight", facecolor="white"); print("wrote gt1_thesis.png")


# ============ GT2 — US POWER BIND (long span)
fig, (a1, a2) = plt.subplots(1, 2, figsize=(11.5, 4.6))
fig.suptitle("Power, not chips, is the binding US constraint — silicon → copper-and-steel",
             fontsize=12.5, fontweight="bold", y=1.03)
# China annual net power additions since 2015 — GV6
cx = [2015, 2018, 2020, 2023, 2024, 2025]; cy = [130, 125, 190, 350, 429, 540]
a1.plot(cx, cy, "-o", color=CN, lw=2.6, ms=7, zorder=4, label="China net additions")
a1.axhline(60, ls="--", color=US, lw=1.6); a1.annotate("US ≈ 50–65 GW/yr (≈ flat)", (2015.1, 72), fontsize=8.5, color=US, va="bottom")
for x, y in [(2015, 130), (2025, 540)]:
    a1.annotate(f"{y} GW", (x, y), textcoords="offset points", xytext=(0, 9), ha="center", fontsize=9, color=CN, fontweight="bold")
a1.set_ylim(0, 610); a1.set_xticks([2015, 2018, 2021, 2024])
a1.set_title("China builds ~8× the power — and pulling away\n(annual net generating capacity added, GW/yr)")
style(a1, "GW added per year", "Source: annual net power additions (China NEA, IEA, Ember)")
# US interconnection queue since 2015 — GV6
qx = [2015, 2018, 2021, 2022, 2023, 2024]; qy = [500, 1000, 1400, 2020, 2600, 2290]
a2.plot(qx, qy, "-o", color=US, lw=2.6, ms=7, zorder=4)
for x, y in [(2015, 500), (2023, 2600), (2024, 2290)]:
    a2.annotate(f"{y:,}", (x, y), textcoords="offset points", xytext=(0, 9), ha="center", fontsize=8.5, color=US, fontweight="bold")
a2.text(0.04, 0.95, "The queue grew ~5× since 2015 and\nthe median project now waits ~5 years.\nNo export-control lever exists\nagainst electrons.",
        transform=a2.transAxes, fontsize=7.9, color=US, va="top")
a2.set_ylim(0, 3000); a2.set_xticks([2015, 2018, 2021, 2024])
a2.set_title("…onto a grid that can't connect fast enough\n(US interconnection queue, GW waiting)")
style(a2, "GW seeking grid connection", "Source: US interconnection queue (LBNL 'Queued Up')")
fig.tight_layout(); fig.savefig("gt2_us_power_bind.png", bbox_inches="tight", facecolor="white"); print("wrote gt2_us_power_bind.png")


# ============ GT3 — MASTER SWITCH (long span)
fig, (a1, a2) = plt.subplots(1, 2, figsize=(11.5, 4.6))
fig.suptitle("The whole export-control thesis rides on one contest: capability vs diffusion",
             fontsize=12.5, fontweight="bold", y=1.03)
# METR task-horizon since 2019 (log, minutes) — GV11
mx = [2019.4, 2023.2, 2024.8, 2024.9, 2025.1, 2025.6]; my = [0.03, 5, 18, 39, 59, 137]
a1.plot(mx, my, "-o", color=US, lw=2.6, ms=6, zorder=4); a1.set_yscale("log")
for x, y in [(2019.4, 0.03), (2023.2, 5), (2025.6, 137)]:
    lab = f"{y:g}min" if y >= 1 else "2 sec"
    lab = "2.3 h" if y == 137 else lab
    a1.annotate(lab, (x, y), textcoords="offset points", xytext=(6, 8), ha="center", fontsize=8.3, color=US, fontweight="bold")
a1.text(0.04, 0.96, "Capability IS accelerating — the task\na model does doubled from 2 seconds\n(2019) to 2.3 hours (2025). The AGI-\nrace premise is real.",
        transform=a1.transAxes, fontsize=7.9, color=GREY, va="top")
a1.set_ylim(0.01, 500); a1.set_xticks([2019, 2021, 2023, 2025])
a1.set_title("Capability is accelerating\n(METR autonomous task-horizon, 50% reliability)")
style(a1, "Minutes (log)", "Source: METR, Measuring AI Ability to Complete Long Tasks (2025)")
# closed − open capability gap (Elo) closing — GV11
gx = ["Jul-23", "Mar-24", "Jan-25", "2026"]; gy = [250, 150, 125, 48]
b = a2.bar(gx, gy, color=DIFF, width=0.6, zorder=3); barlabels(a2, b, gy, fmt="{:g}")
a2.text(0.30, 0.95, "…but a fast-follower closes the gap\nto the frontier within ~months, and\nthe lag has floored at 3–6 months.\nA lead that expires in a quarter\nisn't a strategic asset — so\nwinner-take-all is a ~12% tail.",
        transform=a2.transAxes, fontsize=7.7, color=GREY, va="top")
a2.set_ylim(0, 290)
a2.set_title("…but it diffuses in a quarter\n(best-closed − best-open capability gap, Elo)")
style(a2, "Elo points behind frontier", "Source: best-closed − best-open (LMSYS / LMArena Chatbot Arena)")
fig.tight_layout(); fig.savefig("gt3_master_derivative.png", bbox_inches="tight", facecolor="white"); print("wrote gt3_master_derivative.png")


# ============ GT4 — VALUE-CAPTURE BARBELL (long span)
fig, (a1, a2) = plt.subplots(1, 2, figsize=(11.5, 4.6))
fig.suptitle("Rent flees the model layer to the barbell ends: silicon + power + aggregation",
             fontsize=12.5, fontweight="bold", y=1.03)
# combined hyperscaler capex since 2019 — GV15
kx = ["2019", "2022", "2024", "2025", "2026E"]; ky = [90, 150, 256, 443, 700]
b = a1.bar(kx, ky, color=US, width=0.62, zorder=3); barlabels(a1, b, ky, fmt="${:g}B")
a1.text(0.04, 0.95, "Big-4 hyperscaler capex — the silicon/\npower/DC end of the barbell — compounding\ntoward $1T/yr. NVIDIA data-centre revenue:\n$2.9B (FY19) → ~$185B (FY26E).",
        transform=a1.transAxes, fontsize=7.8, color=US, va="top")
a1.set_ylim(0, 830)
a1.set_title("The physical end captures the rent\n(combined MSFT+GOOG+AMZN+META capex, $B/yr)")
style(a1, "$ billions / year", "Source: company 10-Ks & CreditSights; NVIDIA DC rev (SEC 10-Ks)")
# blended API price collapse since 2020 — GV15
ax_ = [2020, 2023.2, 2024.4, 2025, 2026]; ay = [60, 36, 10, 1.9, 1.1]
a2.plot(ax_, ay, "-o", color=CN, lw=2.6, ms=7, zorder=4); a2.set_yscale("log")
for x, y in zip(ax_, ay):
    a2.annotate(f"${y:g}", (x, y), textcoords="offset points", xytext=(0, 9), ha="center", fontsize=8.3, color=CN, fontweight="bold")
a2.text(0.30, 0.95, "…while the model layer — the hollow\nmiddle — collapses to marginal cost.\nPure model-API margins evaporate;\nlabs must integrate up (workflow /\nagentic lock-in) or down (compute).",
        transform=a2.transAxes, fontsize=7.7, color=CN, va="top")
a2.set_ylim(0.7, 90); a2.set_xticks([2020, 2022, 2024, 2026])
a2.set_title("…the model layer collapses to marginal cost\n(blended GPT-4-class API price, $/M tokens)")
style(a2, "$ per million tokens (log)", "Source: blended GPT-4-class API list prices (OpenAI, TokenCost/BenchLM)")
fig.tight_layout(); fig.savefig("gt4_value_capture.png", bbox_inches="tight", facecolor="white"); print("wrote gt4_value_capture.png")
