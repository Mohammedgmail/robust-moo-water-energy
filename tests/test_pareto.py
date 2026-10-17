import numpy as np
import pytest

from sbrmoa.decision import knee_point, topsis
from sbrmoa.model import build_model
from sbrmoa.pareto import augmecon, payoff_table


@pytest.fixture(scope="module")
def front(small_case):
    return augmecon(build_model(small_case, risk="cvar"), n_points=6)


def test_front_spans_payoff_table(small_case, front):
    pay = payoff_table(build_model(small_case, risk="cvar"))
    # the cheapest point costs the payoff minimum and emits no more than the payoff says
    # (AUGMECON's slack reward may find an equally cheap point with lower emissions)
    cheapest = front.loc[front.cost_risk_usd.idxmin()]
    assert cheapest.cost_risk_usd == pytest.approx(pay.loc["min cost", "cost_risk"], rel=1e-3)
    assert cheapest.expected_co2_t <= pay.loc["min cost", "expected_co2"] + 1e-6
    assert front.expected_co2_t.min() == pytest.approx(pay.loc["min CO2", "expected_co2"], abs=1e-3)


def test_front_is_a_trade_off(front):
    f = front.sort_values("expected_co2_t")
    # lower emissions never come cheaper: cost is non-increasing in CO2
    assert (np.diff(f.cost_risk_usd.to_numpy()) <= 1e-6 * f.cost_risk_usd.max()).all()


def test_front_points_are_non_dominated(front):
    F = front[["cost_risk_usd", "expected_co2_t"]].to_numpy()
    for i in range(len(F)):
        for j in range(len(F)):
            if i != j:
                dominates = (F[j] <= F[i] + 1e-6).all() and (F[j] < F[i] - 1e-6).any()
                assert not dominates


def test_topsis_textbook_example():
    # two cost criteria; the balanced alternative wins with equal weights
    F = np.array([[1.0, 10.0], [5.0, 5.0], [10.0, 1.0]])
    best, score = topsis(F)
    assert best == 1 and score.shape == (3,)
    best_cost_only, _ = topsis(F, weights=[1, 0])
    assert best_cost_only == 0


def test_knee_point():
    f1 = [0, 1, 2, 6, 10]
    f2 = [10, 4, 2, 1, 0]
    assert knee_point(f1, f2) == 2
