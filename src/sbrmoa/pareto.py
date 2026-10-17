"""Pareto fronts with the augmented epsilon-constraint method (AUGMECON).

Objective 1 (minimised): the risk measure of annual cost (``m.cost_risk``).
Objective 2 (constrained, swept): expected CO2 emissions (``m.expected_co2``).

Following Mavrotas (2009), the payoff table is built with lexicographic
optimisation, and the epsilon constraint carries a slack variable that is
rewarded in the objective so that every returned point is Pareto-efficient
(not merely weakly efficient).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pyomo.environ as pyo

from .model import design_of, solve, solver


def _lexicographic(m, first, second, opt, rel_tol=1e-6) -> tuple[float, float]:
    """Minimise ``first``; then minimise ``second`` without worsening ``first``."""
    m.obj = pyo.Objective(expr=first)
    solve(m, opt)
    best = pyo.value(first)
    m.del_component(m.obj)
    m.lex_cap = pyo.Constraint(expr=first <= best + rel_tol * max(abs(best), 1.0))
    m.obj = pyo.Objective(expr=second)
    solve(m, opt)
    out = (pyo.value(m.cost_risk), pyo.value(m.expected_co2))
    m.del_component(m.obj)
    m.del_component(m.lex_cap)
    return out


def payoff_table(m, opt=None) -> pd.DataFrame:
    """Rows: the solution minimising each objective (lexicographically)."""
    opt = opt or solver()
    rows = {
        "min cost": _lexicographic(m, m.cost_objective, m.expected_co2, opt),
        "min CO2": _lexicographic(m, m.expected_co2, m.cost_objective, opt),
    }
    return pd.DataFrame(rows, index=["cost_risk", "expected_co2"]).T


def augmecon(m, n_points: int = 11, opt=None, delta: float = 1e-3) -> pd.DataFrame:
    """Sweep the CO2 cap from the min-cost solution's emissions down to the minimum.

    Returns one row per efficient point with both objectives, the per-scenario
    cost statistics and the first-stage design.
    """
    opt = opt or solver()
    pay = payoff_table(m, opt)
    co2_hi, co2_lo = pay.loc["min cost", "expected_co2"], pay.loc["min CO2", "expected_co2"]
    rng = max(co2_hi - co2_lo, 1e-9)

    m.eps = pyo.Param(mutable=True, initialize=co2_hi)
    m.eps_slack = pyo.Var(within=pyo.NonNegativeReals)
    m.eps_con = pyo.Constraint(expr=m.expected_co2 + m.eps_slack == m.eps)
    scale = pay["cost_risk"].min() / rng
    m.obj = pyo.Objective(expr=m.cost_objective - delta * scale * m.eps_slack)

    rows = []
    for k, eps in enumerate(np.linspace(co2_hi, co2_lo, n_points)):
        m.eps.set_value(float(eps))
        solve(m, opt)
        costs = np.array([pyo.value(m.cost[s]) for s in m.S])
        rows.append(
            {
                "point": k,
                "co2_cap_t": float(eps),
                "cost_risk_usd": pyo.value(m.cost_risk),
                "expected_cost_usd": pyo.value(m.expected_cost),
                "worst_cost_usd": float(costs.max()),
                "expected_co2_t": pyo.value(m.expected_co2),
                "max_unmet": max(pyo.value(m.unmet[s]) for s in m.S),
                **design_of(m),
            }
        )
    for c in ("obj", "eps_con", "eps_slack", "eps"):
        m.del_component(c)
    front = pd.DataFrame(rows)
    return front.drop_duplicates(subset=["cost_risk_usd", "expected_co2_t"]).reset_index(drop=True)
