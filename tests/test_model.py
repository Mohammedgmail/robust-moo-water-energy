import numpy as np
import pyomo.environ as pyo
import pytest

from sbrmoa.evaluate import cvar, evaluate_design
from sbrmoa.model import build_model, design_of, solve


def solved(case, risk, **kw):
    m = build_model(case, risk=risk, **kw)
    m.obj = pyo.Objective(expr=m.cost_objective)
    solve(m)
    return m


@pytest.fixture(scope="module")
def designs(small_case):
    return {r: design_of(solved(small_case, r)) for r in ("expected", "cvar", "worst")}


def test_water_balance_closes(small_case):
    """Over each cyclic day, pumped + trucked water equals demand exactly."""
    m = solved(small_case, "cvar")
    for s in m.S:
        for d in m.D:
            pumped = sum(pyo.value(m.q[s, d, t]) + pyo.value(m.short[s, d, t]) for t in m.T)
            assert pumped == pytest.approx(small_case.demand[s, d].sum(), rel=1e-6)


def test_reliability_target_holds_in_every_scenario(small_case):
    m = solved(small_case, "cvar")
    limit = small_case.params["water"]["max_unmet_fraction"]
    assert max(pyo.value(m.unmet[s]) for s in m.S) <= limit + 1e-7


def test_model_cvar_matches_discrete_cvar(small_case, designs):
    m = solved(small_case, "cvar")
    _, summary = evaluate_design(small_case, design_of(m))
    assert pyo.value(m.cost_risk) == pytest.approx(summary["cvar90_cost_usd"], rel=2e-3)


def test_each_design_is_best_at_its_own_risk_measure(small_case, designs):
    ev = {r: evaluate_design(small_case, d)[1] for r, d in designs.items()}
    tol = 1.003  # 0.1 % tie-break + solver tolerance
    assert ev["expected"]["expected_cost_usd"] <= ev["cvar"]["expected_cost_usd"] * tol
    assert ev["expected"]["expected_cost_usd"] <= ev["worst"]["expected_cost_usd"] * tol
    assert ev["worst"]["worst_cost_usd"] <= ev["cvar"]["worst_cost_usd"] * tol
    assert ev["worst"]["worst_cost_usd"] <= ev["expected"]["worst_cost_usd"] * tol
    assert ev["cvar"]["cvar90_cost_usd"] <= ev["expected"]["cvar90_cost_usd"] * tol


def test_nominal_plan_is_fragile(case):
    """A plan sized for the nominal future needs trucking when the grid is down."""
    nominal = design_of(solved(case.subset([case.nominal]), "expected"))
    df, _ = evaluate_design(case, nominal)
    no_grid = df[df.grid_hours == 0]
    assert (no_grid.unmet > 0.01).all()
    assert no_grid.cost_usd.min() > 2 * df.loc[df.scenario == case.nominal, "cost_usd"].iloc[0]


def test_no_equipment_means_trucking(small_case):
    df, summary = evaluate_design(small_case, {"pv": 0, "bat": 0, "dg": 0, "tank": 0})
    assert (df[df.grid_hours == 0].unmet > 0.999).all()  # nothing can pump without the grid
    assert summary["expected_co2_t"] >= 0


def test_fixed_design_is_respected(small_case):
    d = {"pv": 150.0, "bat": 40.0, "dg": 20.0, "tank": 100.0}
    m = build_model(small_case, design=d, reliability=False)
    m.obj = pyo.Objective(expr=m.expected_cost)
    solve(m)
    assert design_of(m) == pytest.approx(d)


def test_invalid_risk_measure(small_case):
    with pytest.raises(ValueError):
        build_model(small_case, risk="regret")


def test_discrete_cvar():
    v, p = [10, 20, 30, 40], [0.25] * 4
    assert cvar(v, p, 0.75) == pytest.approx(40)
    assert cvar(v, p, 0.5) == pytest.approx(35)
    assert cvar(v, p, 0.0) == pytest.approx(np.mean(v))
