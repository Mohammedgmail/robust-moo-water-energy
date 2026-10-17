"""Run the full case study: four planning approaches, Pareto fronts, stress test.

    python scripts/run_case_study.py                      # default data file
    python scripts/run_case_study.py --points 15 --alpha 0.95

Writes CSV files to results/ and figures to docs/ (via plot_results.py).
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import pandas as pd

from sbrmoa.data import load_case
from sbrmoa.decision import topsis
from sbrmoa.evaluate import evaluate_design
from sbrmoa.model import build_model
from sbrmoa.pareto import augmecon

ROOT = Path(__file__).resolve().parents[1]
DESIGN = ["pv", "bat", "dg", "tank"]

APPROACHES = {  # label: (scenarios used for planning, risk measure)
    "deterministic": ("nominal", "expected"),
    "expected": ("all", "expected"),
    "cvar": ("all", "cvar"),
    "worst": ("all", "worst"),
}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--data", default=str(ROOT / "data" / "nws_borehole_station.yaml"))
    ap.add_argument("--points", type=int, default=11, help="points per Pareto front")
    ap.add_argument("--alpha", type=float, default=0.9, help="CVaR confidence level")
    args = ap.parse_args()

    case = load_case(args.data)
    cvar_col = f"cvar{int(args.alpha * 100)}_cost_usd"
    out = ROOT / "results"
    out.mkdir(exist_ok=True)
    t0 = time.time()

    fronts = []
    for label, (which, risk) in APPROACHES.items():
        planning = case.subset([case.nominal]) if which == "nominal" else case
        front = augmecon(build_model(planning, risk=risk, alpha=args.alpha), args.points)
        # stress-test every design on all scenarios with cost-optimal operation
        evals = [
            evaluate_design(case, r[DESIGN].to_dict(), args.alpha)[1] for _, r in front.iterrows()
        ]
        front = pd.concat([front, pd.DataFrame(evals).add_prefix("eval_")], axis=1)
        front.insert(0, "approach", label)
        fronts.append(front)
        print(f"{label:13s} {len(front)} efficient designs ({time.time() - t0:.0f} s)")
    fronts = pd.concat(fronts, ignore_index=True)

    # compromise on the CVaR front: TOPSIS with equal weights on risk-adjusted cost and CO2
    cv = fronts[fronts.approach == "cvar"]
    best, score = topsis(cv[[f"eval_{cvar_col}", "eval_expected_co2_t"]].to_numpy())
    fronts["topsis_score"] = float("nan")
    fronts.loc[cv.index, "topsis_score"] = score
    fronts["compromise"] = False
    fronts.loc[cv.index[best], "compromise"] = True
    fronts.to_csv(out / "pareto_fronts.csv", index=False)

    # head-to-head: each approach's cheapest design, plus the CVaR compromise
    picks = fronts.groupby("approach", sort=False).head(1).copy()
    picks["design"] = picks["approach"] + " (min cost)"
    comp = fronts[fronts.compromise].copy()
    comp["design"] = "cvar (TOPSIS compromise)"
    table = pd.concat([picks, comp])
    cols = ["design", *DESIGN, "eval_expected_cost_usd", f"eval_{cvar_col}", "eval_worst_cost_usd",
            "eval_expected_co2_t", "eval_worst_unmet", "eval_scenarios_missing_target"]  # fmt: skip
    table[cols].to_csv(out / "design_comparison.csv", index=False)
    print(table[cols].round(1).to_string(index=False))

    # per-scenario results for the cheapest deterministic and CVaR designs
    per = []
    for _, r in picks.iterrows():
        df, _ = evaluate_design(case, r[DESIGN].to_dict(), args.alpha)
        df.insert(0, "design", r["design"])
        per.append(df)
    pd.concat(per).to_csv(out / "scenario_stress_test.csv", index=False)
    print(f"done in {time.time() - t0:.0f} s -> results/")


if __name__ == "__main__":
    main()
