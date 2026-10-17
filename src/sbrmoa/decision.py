"""Choosing one design from a Pareto front."""

from __future__ import annotations

import numpy as np


def topsis(F: np.ndarray, weights=None, benefit=None) -> tuple[int, np.ndarray]:
    """TOPSIS (Hwang & Yoon, 1981). ``F`` is (alternatives x criteria).

    Criteria are costs (to minimise) unless flagged in ``benefit``. Returns the
    index of the best alternative and every alternative's closeness score.
    """
    F = np.asarray(F, dtype=float)
    n, k = F.shape
    w = np.full(k, 1.0 / k) if weights is None else np.asarray(weights, float) / np.sum(weights)
    benefit = np.zeros(k, bool) if benefit is None else np.asarray(benefit, bool)
    norm = np.linalg.norm(F, axis=0)
    V = F / np.where(norm == 0, 1.0, norm) * w
    ideal = np.where(benefit, V.max(0), V.min(0))
    anti = np.where(benefit, V.min(0), V.max(0))
    d_plus = np.linalg.norm(V - ideal, axis=1)
    d_minus = np.linalg.norm(V - anti, axis=1)
    score = d_minus / np.where(d_plus + d_minus == 0, 1.0, d_plus + d_minus)
    return int(np.argmax(score)), score


def knee_point(f1, f2) -> int:
    """Point of a 2-D front farthest from the line joining its extremes (both normalised)."""
    f1, f2 = np.asarray(f1, float), np.asarray(f2, float)
    x = (f1 - f1.min()) / max(np.ptp(f1), 1e-12)
    y = (f2 - f2.min()) / max(np.ptp(f2), 1e-12)
    ax, ay, bx, by = x[0], y[0], x[-1], y[-1]
    cross = (bx - ax) * (y - ay) - (by - ay) * (x - ax)
    d = np.abs(cross) / max(np.hypot(bx - ax, by - ay), 1e-12)
    return int(np.argmax(d))
