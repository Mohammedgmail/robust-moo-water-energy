from pathlib import Path

import pytest

from sbrmoa.data import load_case

DATA = Path(__file__).resolve().parents[1] / "data" / "nws_borehole_station.yaml"


@pytest.fixture(scope="session")
def case():
    return load_case(DATA)


@pytest.fixture(scope="session")
def small_case(case):
    """Four contrasting scenarios: mild/harsh climate x grid/no grid (mid diesel price)."""
    names = [
        f"{c}|fuel-mid|grid-{g}" for c in ("SSP1-2.6", "SSP5-8.5") for g in ("partial", "none")
    ]
    return case.subset(names)
