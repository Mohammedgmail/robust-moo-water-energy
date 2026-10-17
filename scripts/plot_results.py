"""Figures for the README from results/*.csv.

python scripts/plot_results.py     # -> docs/pareto_fronts.png, docs/stress_test.png
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parents[1]
SURFACE, INK, INK_2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
# fixed categorical order (validated palette, light mode); colour follows the approach
APPROACHES = {
    "deterministic": ("Deterministic (nominal future only)", "#2a78d6"),
    "expected": ("Stochastic, expected cost", "#eb6834"),
    "cvar": ("Robust, CVaR₉₀ of cost", "#1baf7a"),
    "worst": ("Robust, worst-case cost", "#eda100"),
}


def style(ax) -> None:
    ax.set_facecolor(SURFACE)
    ax.grid(True, color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(colors=INK_2, length=0)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)


def pareto(fronts: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(8.6, 5.2), facecolor=SURFACE)
    style(ax)
    for key, (label, color) in APPROACHES.items():
        f = fronts[fronts.approach == key].sort_values("eval_expected_co2_t")
        x, y = f["eval_expected_co2_t"], f["eval_cvar90_cost_usd"] / 1000
        ls = "--" if key == "deterministic" else "-"  # not efficient once stress-tested
        ax.plot(x, y, ls, color=color, lw=2, label=label, zorder=2)
        ax.plot(x, y, "o", ms=8, color=color, mec=SURFACE, mew=2, zorder=3)
    c = fronts[fronts.compromise].iloc[0]
    cx, cy = c["eval_expected_co2_t"], c["eval_cvar90_cost_usd"] / 1000
    ax.plot(cx, cy, "*", ms=18, color=APPROACHES["cvar"][1], mec=INK, mew=1, zorder=4)
    ax.annotate(
        f"TOPSIS compromise\nPV {c['pv']:.0f} kWp · battery {c['bat']:.0f} kWh\n"
        f"diesel {c['dg']:.0f} kW · +{c['tank']:.0f} m³ tank",
        (cx, cy), xytext=(18, 26), textcoords="offset points", fontsize=8.5, color=INK,
        arrowprops=dict(arrowstyle="-", color=INK_2, lw=0.8),
    )  # fmt: skip
    d = fronts[(fronts.approach == "deterministic")].iloc[0]
    ax.annotate(
        "deterministic plans: water trucking\nduring grid outages roughly\ntriples their cost",
        (d["eval_expected_co2_t"], d["eval_cvar90_cost_usd"] / 1000), xytext=(-170, -58),
        textcoords="offset points", fontsize=8.5, color=INK,
        arrowprops=dict(arrowstyle="-", color=INK_2, lw=0.8),
    )  # fmt: skip
    ax.annotate(
        "expected-cost, CVaR and worst-case\nfronts converge as the CO₂ cap tightens",
        (4.6, 27.6), xytext=(40, 38), textcoords="offset points", fontsize=8.5, color=INK,
        arrowprops=dict(arrowstyle="-", color=INK_2, lw=0.8),
    )  # fmt: skip
    ax.set_xlabel("expected CO₂ emissions (t / year)", color=INK_2)
    ax.set_ylabel("CVaR₉₀ of annual cost (k$ / year)", color=INK_2)
    ax.set_ylim(0, None)
    ax.set_xlim(0, None)
    ax.set_title(
        "Cost-risk vs emissions, every design stress-tested in all 18 futures",
        loc="left", fontsize=12, fontweight="bold", color=INK,
    )  # fmt: skip
    ax.legend(loc="upper left", bbox_to_anchor=(0, -0.13), ncol=2, frameon=False, labelcolor=INK)
    fig.tight_layout()
    fig.savefig(ROOT / "docs" / "pareto_fronts.png", dpi=130, facecolor=SURFACE)


def stress(per: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(8.6, 4.2), facecolor=SURFACE)
    style(ax)
    ax.grid(True, axis="x", color=GRID, lw=0.8)
    ax.grid(False, axis="y")
    rng = np.random.default_rng(0)
    keys = list(APPROACHES)[::-1]  # deterministic on top
    for row, key in enumerate(keys):
        label, color = APPROACHES[key]
        df = per[per.design.str.startswith(key)]
        jitter = rng.uniform(-0.18, 0.18, len(df))
        for grid, marker in ((6, "o"), (0, "^")):
            sel = (df.grid_hours == grid).to_numpy()
            ax.plot(
                df.cost_usd[sel] / 1000, row + jitter[sel], marker, ls="none", ms=8,
                color=color, mec=SURFACE, mew=1.5, alpha=0.9,
            )  # fmt: skip
        mean = float(np.sum(df.probability * df.cost_usd)) / 1000
        ax.plot([mean, mean], [row - 0.32, row + 0.32], color=INK, lw=2, solid_capstyle="butt")
        ax.annotate(f"mean {mean:.1f}", (mean, row + 0.34), ha="center", fontsize=8, color=INK_2)
    ax.set_yticks(range(len(keys)), [APPROACHES[k][0] for k in keys], color=INK)
    ax.set_xlabel("annual cost of the cheapest design, per future (k$ / year)", color=INK_2)
    ax.set_xlim(0, None)
    ax.set_ylim(-0.6, len(keys) - 0.3)
    handles = [
        Line2D([], [], marker="o", ls="none", color=INK_2, ms=7, label="6 h of grid per night"),
        Line2D([], [], marker="^", ls="none", color=INK_2, ms=7, label="no grid"),
        Line2D([], [], color=INK, lw=2, label="probability-weighted mean"),
    ]
    ax.legend(handles=handles, loc="lower right", frameon=False, fontsize=8.5, labelcolor=INK)
    ax.set_title(
        "Stress test: the same plan in 18 possible futures",
        loc="left", fontsize=12, fontweight="bold", color=INK,
    )  # fmt: skip
    fig.tight_layout()
    fig.savefig(ROOT / "docs" / "stress_test.png", dpi=130, facecolor=SURFACE)


def main() -> None:
    plt.rcParams.update({"font.size": 10, "text.color": INK})
    (ROOT / "docs").mkdir(exist_ok=True)
    pareto(pd.read_csv(ROOT / "results" / "pareto_fronts.csv"))
    stress(pd.read_csv(ROOT / "results" / "scenario_stress_test.csv"))
    print("saved docs/pareto_fronts.png and docs/stress_test.png")


if __name__ == "__main__":
    main()
