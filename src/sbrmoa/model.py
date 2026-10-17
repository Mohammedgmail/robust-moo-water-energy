"""Two-stage scenario-based LP of a solar-diesel-grid borehole water station.

First stage (decided now, identical in every scenario): PV (kWp), battery (kWh),
diesel generator (kW) and additional water-tank volume (m^3).

Second stage (recourse, per scenario / representative day / hour): pumping,
PV use and curtailment, diesel and grid supply, battery operation, tank level
and any shortfall covered by water trucking.

Per-scenario outcomes: annualised cost (USD/yr), CO2 (t/yr) and the fraction of
demand not met from the borehole. Robustness enters through the risk measure
applied to cost (expected value, CVaR or worst case) and through a reliability
constraint that must hold in every scenario.
"""

from __future__ import annotations

import pyomo.environ as pyo

from .data import HOURS, Case, crf

RISK_MEASURES = ("expected", "cvar", "worst")
TIE_BREAK = 1e-3


def build_model(
    case: Case,
    risk: str = "cvar",
    alpha: float = 0.9,
    reliability: bool = True,
    design: dict | None = None,
) -> pyo.ConcreteModel:
    """Build the model. ``design`` fixes the first-stage sizes (for evaluation)."""
    if risk not in RISK_MEASURES:
        raise ValueError(f"risk must be one of {RISK_MEASURES}")
    p = case.params
    tech, w, fin = p["technologies"], p["water"], p["finance"]
    pv, bat, dg, tank, grid = (tech[k] for k in ("pv", "battery", "diesel", "tank", "grid"))
    rate = fin["discount_rate"]

    m = pyo.ConcreteModel(name="sbrmoa_water_energy")
    m.S = pyo.Set(initialize=[s.name for s in case.scenarios], ordered=True)
    m.D = pyo.Set(initialize=case.days, ordered=True)
    m.T = pyo.RangeSet(0, HOURS - 1)
    prob = {s.name: s.probability for s in case.scenarios}
    scen = {s.name: s for s in case.scenarios}

    # ---------------------------------------------------------- first stage
    m.pv = pyo.Var(bounds=(0, pv["max_kw"]), doc="PV capacity (kWp)")
    m.bat = pyo.Var(bounds=(0, bat["max_kwh"]), doc="battery energy capacity (kWh)")
    m.dg = pyo.Var(bounds=(0, dg["max_kw"]), doc="diesel generator capacity (kW)")
    m.tank = pyo.Var(bounds=(0, tank["max_m3"]), doc="additional tank volume (m3)")
    if design is not None:
        for k in ("pv", "bat", "dg", "tank"):
            getattr(m, k).fix(design[k])

    # ---------------------------------------------------------- second stage
    idx = (m.S, m.D, m.T)
    nn = pyo.NonNegativeReals
    m.q = pyo.Var(*idx, bounds=(0, w["pump_max_flow_m3_h"]), doc="water pumped (m3/h)")
    m.pv_use = pyo.Var(*idx, within=nn)
    m.p_dg = pyo.Var(*idx, within=nn)
    m.p_grid = pyo.Var(*idx, within=nn)
    m.ch = pyo.Var(*idx, within=nn)
    m.dis = pyo.Var(*idx, within=nn)
    m.soc = pyo.Var(*idx, within=nn)
    m.level = pyo.Var(*idx, within=nn, doc="tank level at the start of hour t (m3)")
    m.short = pyo.Var(*idx, within=nn, doc="demand covered by trucking (m3/h)")
    m.ens = pyo.Var(*idx, within=nn, doc="auxiliary energy not served (kW), e.g. rental genset")

    eta = bat["round_trip_efficiency"] ** 0.5
    nxt = lambda t: (t + 1) % HOURS  # noqa: E731  (cyclic representative day)

    def rule(f):
        return pyo.Constraint(*idx, rule=f)

    m.pv_avail = rule(lambda m, s, d, t: m.pv_use[s, d, t] <= case.pv_cf[s, d][t] * m.pv)
    m.dg_cap = rule(lambda m, s, d, t: m.p_dg[s, d, t] <= m.dg)
    m.grid_cap = rule(
        lambda m, s, d, t: m.p_grid[s, d, t] <= grid["max_import_kw"] * case.grid_avail[s][t]
    )
    m.ch_cap = rule(lambda m, s, d, t: m.ch[s, d, t] <= bat["c_rate"] * m.bat)
    m.dis_cap = rule(lambda m, s, d, t: m.dis[s, d, t] <= bat["c_rate"] * m.bat)
    m.soc_max = rule(lambda m, s, d, t: m.soc[s, d, t] <= m.bat)
    m.soc_min = rule(lambda m, s, d, t: m.soc[s, d, t] >= (1 - bat["depth_of_discharge"]) * m.bat)
    m.soc_dyn = rule(
        lambda m, s, d, t: (
            m.soc[s, d, nxt(t)] == m.soc[s, d, t] + eta * m.ch[s, d, t] - m.dis[s, d, t] / eta
        )
    )
    m.energy = rule(
        lambda m, s, d, t: (
            m.pv_use[s, d, t] + m.p_dg[s, d, t] + m.p_grid[s, d, t] + m.dis[s, d, t]
            == case.sec[s] * m.q[s, d, t] + w["aux_load_kw"] - m.ens[s, d, t] + m.ch[s, d, t]
        )
    )
    m.ens_cap = rule(lambda m, s, d, t: m.ens[s, d, t] <= w["aux_load_kw"])
    m.water = rule(
        lambda m, s, d, t: (
            m.level[s, d, nxt(t)]
            == m.level[s, d, t] + m.q[s, d, t] - case.demand[s, d][t] + m.short[s, d, t]
        )
    )
    m.tank_cap = rule(lambda m, s, d, t: m.level[s, d, t] <= w["existing_tank_m3"] + m.tank)
    m.short_cap = rule(lambda m, s, d, t: m.short[s, d, t] <= case.demand[s, d][t])

    # ---------------------------------------------------------- outcomes
    def annual(tech_cost, size, life, om):
        return tech_cost * size * crf(rate, life) + om

    m.capex_annual = pyo.Expression(
        expr=annual(pv["capex_usd_per_kw"], m.pv, pv["life_years"], pv["om_usd_per_kw_yr"] * m.pv)
        + annual(
            bat["capex_usd_per_kwh"],
            m.bat,
            bat["life_years"],
            bat["om_fraction_yr"] * bat["capex_usd_per_kwh"] * m.bat,
        )
        + annual(dg["capex_usd_per_kw"], m.dg, dg["life_years"], 0.0)
        + annual(
            tank["capex_usd_per_m3"],
            m.tank,
            tank["life_years"],
            tank["om_fraction_yr"] * tank["capex_usd_per_m3"] * m.tank,
        )
    )

    def yearly(s, f):
        return sum(case.day_weight[d] * sum(f(s, d, t) for t in m.T) for d in m.D)

    m.opex = pyo.Expression(
        m.S,
        rule=lambda m, s: yearly(
            s,
            lambda s, d, t: (
                (scen[s].diesel_price * dg["fuel_l_per_kwh"] + dg["om_usd_per_kwh"])
                * m.p_dg[s, d, t]
                + grid["tariff_usd_per_kwh"] * m.p_grid[s, d, t]
                + w["trucking_cost_usd_m3"] * m.short[s, d, t]
                + w["unserved_energy_usd_kwh"] * m.ens[s, d, t]
            ),
        ),
    )
    m.cost = pyo.Expression(m.S, rule=lambda m, s: m.capex_annual + m.opex[s])
    m.co2 = pyo.Expression(
        m.S,
        rule=lambda m, s: (
            yearly(
                s,
                lambda s, d, t: (
                    dg["fuel_l_per_kwh"] * dg["co2_kg_per_l"] * m.p_dg[s, d, t]
                    + grid["co2_kg_per_kwh"] * m.p_grid[s, d, t]
                ),
            )
            / 1000.0
        ),
    )
    annual_demand = {s: sum(case.day_weight[d] * case.demand[s, d].sum() for d in case.days)
                     for s in m.S}  # fmt: skip
    m.unmet = pyo.Expression(
        m.S, rule=lambda m, s: yearly(s, lambda s, d, t: m.short[s, d, t]) / annual_demand[s]
    )
    if reliability:
        m.reliability = pyo.Constraint(m.S, rule=lambda m, s: m.unmet[s] <= w["max_unmet_fraction"])

    # ---------------------------------------------------------- risk measure on cost
    m.expected_cost = pyo.Expression(expr=sum(prob[s] * m.cost[s] for s in m.S))
    m.expected_co2 = pyo.Expression(expr=sum(prob[s] * m.co2[s] for s in m.S))
    if risk == "expected":
        m.cost_risk = pyo.Expression(expr=m.expected_cost)
    elif risk == "worst":
        m.theta = pyo.Var()
        m.worst_def = pyo.Constraint(m.S, rule=lambda m, s: m.theta >= m.cost[s])
        m.cost_risk = pyo.Expression(expr=m.theta)
    else:  # CVaR_alpha (Rockafellar & Uryasev, 2000): linear in the scenario costs
        m.var = pyo.Var()
        m.excess = pyo.Var(m.S, within=nn)
        m.excess_def = pyo.Constraint(m.S, rule=lambda m, s: m.excess[s] >= m.cost[s] - m.var)
        m.cost_risk = pyo.Expression(
            expr=m.var + sum(prob[s] * m.excess[s] for s in m.S) / (1 - alpha)
        )
    # CVaR and worst case only "see" the expensive scenarios, so on their own they leave the
    # recourse in the other scenarios arbitrary. A 0.1 % expected-cost tie-break makes every
    # scenario's dispatch cost-optimal without changing which risk is being minimised.
    tie = 0.0 if risk == "expected" else TIE_BREAK
    m.cost_objective = pyo.Expression(expr=m.cost_risk + tie * m.expected_cost)
    m.risk, m.alpha = risk, alpha
    return m


def solver():
    """HiGHS through Pyomo (``pip install highspy``)."""
    for name in ("appsi_highs", "highs"):
        opt = pyo.SolverFactory(name)
        try:
            if opt.available():
                return opt
        except Exception:  # noqa: BLE001  (interface not usable in this Pyomo version)
            continue
    raise RuntimeError("HiGHS not available: pip install highspy")


def solve(m: pyo.ConcreteModel, opt=None) -> None:
    """Solve and raise if the solver did not prove optimality."""
    opt = opt or solver()
    res = opt.solve(m, load_solutions=False)
    tc = res.solver.termination_condition
    if tc != pyo.TerminationCondition.optimal:
        raise RuntimeError(f"solver finished with {tc}")
    m.solutions.load_from(res)


def design_of(m: pyo.ConcreteModel) -> dict[str, float]:
    """First-stage sizes; solver round-off (e.g. -1e-12) is clipped to zero."""
    return {k: max(0.0, float(pyo.value(getattr(m, k)))) for k in ("pv", "bat", "dg", "tank")}
