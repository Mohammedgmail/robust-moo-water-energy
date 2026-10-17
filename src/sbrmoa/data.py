"""Case-study inputs: parameters, hourly representative days and the scenario set."""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import yaml

HOURS = 24
RHO_G = 1000 * 9.81  # water density x gravity (N/m^3)


@dataclass(frozen=True)
class Scenario:
    """One plausible future: climate pathway x diesel price x grid availability."""

    name: str
    probability: float
    climate: str
    diesel_price: float  # USD/L
    grid_hours: int  # hours per night with grid supply
    delta_t: float  # warming (deg C)
    demand_increase: float  # fraction
    head_m: float  # total dynamic head (m)


@dataclass
class Case:
    """Everything the optimisation model needs, already reduced to arrays."""

    params: dict
    days: list[str]
    day_weight: dict[str, float]
    scenarios: list[Scenario]
    demand: dict  # (scenario, day) -> np.ndarray[24] in m3/h
    pv_cf: dict  # (scenario, day) -> np.ndarray[24] kW per kWp
    grid_avail: dict  # scenario -> np.ndarray[24] of 0/1
    sec: dict  # scenario -> specific energy (kWh/m3)
    nominal: str = field(default="")

    def scenario(self, name: str) -> Scenario:
        return next(s for s in self.scenarios if s.name == name)

    def subset(self, names: list[str], renormalise: bool = True) -> Case:
        """A case restricted to some scenarios (e.g. the nominal one for a deterministic model)."""
        keep = [s for s in self.scenarios if s.name in names]
        total = sum(s.probability for s in keep)
        if renormalise:
            keep = [Scenario(**{**s.__dict__, "probability": s.probability / total}) for s in keep]
        names_set = {s.name for s in keep}
        return Case(
            params=self.params,
            days=self.days,
            day_weight=self.day_weight,
            scenarios=keep,
            demand={k: v for k, v in self.demand.items() if k[0] in names_set},
            pv_cf={k: v for k, v in self.pv_cf.items() if k[0] in names_set},
            grid_avail={k: v for k, v in self.grid_avail.items() if k in names_set},
            sec={k: v for k, v in self.sec.items() if k in names_set},
            nominal=self.nominal,
        )


def irradiance_profile(sunrise: float, sunset: float, peak: float) -> np.ndarray:
    """Hourly mean plane-of-array irradiance (W/m^2): a sharpened half-sine between sunrise and
    sunset, evaluated at hour mid-points."""
    t = np.arange(HOURS) + 0.5
    x = (t - sunrise) / (sunset - sunrise)
    day = (x > 0) & (x < 1)
    return np.where(day, peak * np.sin(np.pi * np.clip(x, 0, 1)) ** 1.5, 0.0)


def temperature_profile(t_mean: float, t_swing: float) -> np.ndarray:
    """Ambient temperature (deg C), minimum near 05:00 and maximum near 15:00."""
    t = np.arange(HOURS) + 0.5
    return t_mean + t_swing * np.sin(2 * np.pi * (t - 9.0) / 24.0)


def pv_capacity_factor(g: np.ndarray, t_amb: np.ndarray, pv: dict) -> np.ndarray:
    """kW output per kWp: irradiance, NOCT cell-temperature derating and system losses."""
    t_cell = t_amb + (pv["noct_c"] - 20.0) / 800.0 * g
    derate = 1.0 + pv["temp_coeff_per_c"] * (t_cell - 25.0)
    return np.clip(g / 1000.0 * derate * pv["system_efficiency"], 0.0, None)


def specific_energy(head_m: float, efficiency: float) -> float:
    """Pumping energy per cubic metre (kWh/m^3) = rho g H / (eta * 3.6e6)."""
    return RHO_G * head_m / (efficiency * 3.6e6)


def grid_availability(hours: int) -> np.ndarray:
    """Grid supply is live for the first ``hours`` hours after midnight."""
    a = np.zeros(HOURS)
    a[:hours] = 1.0
    return a


def load_case(path: str | Path) -> Case:
    with open(path, encoding="utf-8") as f:
        p = yaml.safe_load(f)

    w = p["water"]
    profile = np.asarray(w["demand_profile"], dtype=float)
    if len(profile) != HOURS or (profile < 0).any() or profile.sum() <= 0:
        raise ValueError("demand_profile must have 24 non-negative values")
    profile = profile / profile.sum()  # relative weights -> fractions of the day

    sc = p["scenarios"]
    scenarios = []
    for (cn, c), (pn, pr), (gn, g) in itertools.product(
        sc["climate"].items(), sc["diesel_price_usd_l"].items(), sc["grid_hours"].items()
    ):
        scenarios.append(
            Scenario(
                name=f"{cn}|fuel-{pn}|grid-{gn}",
                probability=c["probability"] * pr["probability"] * g["probability"],
                climate=cn,
                diesel_price=float(pr["value"]),
                grid_hours=int(g["value"]),
                delta_t=float(c["delta_t_c"]),
                demand_increase=float(c["demand_increase"]),
                head_m=float(w["base_head_m"] + c["head_increase_m"]),
            )
        )
    total = sum(s.probability for s in scenarios)
    scenarios = [
        Scenario(**{**s.__dict__, "probability": s.probability / total}) for s in scenarios
    ]

    days = list(p["representative_days"])
    demand, pv_cf = {}, {}
    for s in scenarios:
        for d in days:
            rd = p["representative_days"][d]
            daily = w["base_demand_m3_day"] * rd["demand_factor"] * (1 + s.demand_increase)
            demand[s.name, d] = daily * profile
            g = irradiance_profile(rd["sunrise"], rd["sunset"], rd["peak_irradiance"])
            t_amb = temperature_profile(rd["t_mean"] + s.delta_t, rd["t_swing"])
            pv_cf[s.name, d] = pv_capacity_factor(g, t_amb, p["technologies"]["pv"])

    nom = p["nominal_scenario"]
    nominal = f"{nom['climate']}|fuel-{nom['diesel_price_usd_l']}|grid-{nom['grid_hours']}"
    return Case(
        params=p,
        days=days,
        day_weight={d: float(p["representative_days"][d]["weight"]) for d in days},
        scenarios=scenarios,
        demand=demand,
        pv_cf=pv_cf,
        grid_avail={s.name: grid_availability(s.grid_hours) for s in scenarios},
        sec={s.name: specific_energy(s.head_m, w["pump_efficiency"]) for s in scenarios},
        nominal=nominal,
    )


def crf(rate: float, years: float) -> float:
    """Capital recovery factor: converts an up-front cost into an equal annual payment."""
    return rate * (1 + rate) ** years / ((1 + rate) ** years - 1)
