import numpy as np
import pytest

from sbrmoa.data import crf, irradiance_profile, specific_energy


def test_scenario_set(case):
    assert len(case.scenarios) == 18
    assert np.isclose(sum(s.probability for s in case.scenarios), 1.0)
    assert case.nominal in {s.name for s in case.scenarios}
    assert len({s.name for s in case.scenarios}) == 18


def test_subset_renormalises(case, small_case):
    assert len(small_case.scenarios) == 4
    assert np.isclose(sum(s.probability for s in small_case.scenarios), 1.0)
    det = case.subset([case.nominal])
    assert det.scenarios[0].probability == pytest.approx(1.0)


def test_pv_yield_is_plausible_for_syria(case):
    s = case.scenarios[0].name
    annual = sum(case.day_weight[d] * case.pv_cf[s, d].sum() for d in case.days)
    assert 1300 < annual < 1800  # kWh per kWp per year


def test_hotter_climate_derates_pv(case):
    by_climate = {sc.climate: sc.name for sc in case.scenarios}
    cool, hot = by_climate["SSP1-2.6"], by_climate["SSP5-8.5"]
    assert case.pv_cf[hot, "summer"].sum() < case.pv_cf[cool, "summer"].sum()


def test_daily_demand_and_profile(case):
    s = case.scenarios[0]
    base = case.params["water"]["base_demand_m3_day"]
    factor = case.params["representative_days"]["summer"]["demand_factor"]
    expected = base * factor * (1 + s.demand_increase)
    assert case.demand[s.name, "summer"].sum() == pytest.approx(expected)


def test_specific_energy_formula():
    # rho g H / (eta 3.6e6): 100 m at 100 % efficiency is 0.2725 kWh/m3
    assert specific_energy(100, 1.0) == pytest.approx(0.2725, rel=1e-4)


def test_irradiance_zero_at_night():
    g = irradiance_profile(6.0, 18.0, 1000)
    assert g[:5].max() == 0 and g[19:].max() == 0 and g.max() <= 1000


def test_crf():
    assert crf(0.08, 20) == pytest.approx(0.101852, rel=1e-5)
