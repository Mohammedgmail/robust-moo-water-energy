"""Evaluate a fixed design in every scenario (stress-testing a plan)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pyomo.environ as pyo

from .data import Case
from .model import build_model, solve


def evaluate_design(case: Case, design: dict, alpha: float = 0.9) -> tuple[pd.DataFrame, dict]:
    """Fix the first-stage sizes and operate optimally in each scenario.

    The reliability target is *not* imposed here: any shortfall is trucked and
    reported, so designs that rely on optimistic assumptions show their gaps.
    Returns per-scenario results and summary statistics.
    """
    m = build_model(case, risk="expected", reliability=False, design=design)
    m.obj = pyo.Objective(expr=m.expected_cost)  # scenarios decouple once sizes are fixed
    solve(m)
    rows = []
    for s in case.scenarios:
        rows.append(
            {
                "scenario": s.name,
                "climate": s.climate,
                "diesel_price": s.diesel_price,
                "grid_hours": s.grid_hours,
                "probability": s.probability,
                "cost_usd": pyo.value(m.cost[s.name]),
                "co2_t": pyo.value(m.co2[s.name]),
                "unmet": pyo.value(m.unmet[s.name]),
            }
        )
    df = pd.DataFrame(rows)
    p = df["probability"].to_numpy()
    cost = df["cost_usd"].to_numpy()
    summary = {
        "expected_cost_usd": float(p @ cost),
        f"cvar{int(alpha * 100)}_cost_usd": cvar(cost, p, alpha),
        "worst_cost_usd": float(cost.max()),
        "expected_co2_t": float(p @ df["co2_t"].to_numpy()),
        "worst_unmet": float(df["unmet"].max()),
        "scenarios_missing_target": int(
            (df["unmet"] > case.params["water"]["max_unmet_fraction"] + 1e-9).sum()
        ),
    }
    return df, summary


def cvar(values, probabilities, alpha: float) -> float:
    """Conditional value-at-risk of a discrete distribution (mean of the worst 1-alpha tail)."""
    v = np.asarray(values, float)
    p = np.asarray(probabilities, float)
    order = np.argsort(v)[::-1]  # worst first
    v, p = v[order], p[order]
    tail = 1.0 - alpha
    taken, total = 0.0, 0.0
    for vi, pi in zip(v, p, strict=True):
        take = min(pi, tail - taken)
        if take <= 0:
            break
        total += take * vi
        taken += take
    return float(total / tail)
