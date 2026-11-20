"""Build the abstraction filters (F1) for both benchmarks. For lane keeping, compare
with the same grid built from the exact model (the best any grid of that size can do)
and with a fine fixed-speed grid (close to the true maximal safe set).

Run: python experiments/02_build_filters.py   (writes results/f1_build.json)
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
from ddma import RectangularPartition, SymbolicAbstraction, safety

from sfilter.envs import LaneKeepingParams
from sfilter.filters import build_abstraction_filter
from sfilter.filters.quadrotor import build_quadrotor_filter

P = LaneKeepingParams()


class ExactIntervalModel:
    """Successor boxes from the true equations (interval arithmetic, exact on boxes)."""

    def __init__(self, partition, p=P):
        self.partition, self.n_x, self.m, self.p = partition, 3, 3, p

    def successor_bounds(self, idx, xl, xh, ul, uh):
        p = self.p
        c1 = np.stack([v * np.sin(e) for v in (xl[:, 2], xh[:, 2]) for e in (xl[:, 1], xh[:, 1])])
        c2 = np.stack([v * np.tan(d) for v in (xl[:, 2], xh[:, 2]) for d in (ul[:, 0], uh[:, 0])])
        k = p.dt * xh[:, 2] * p.kappa_max
        lo = np.stack([xl[:, 0] + p.dt * c1.min(0) - p.noise_y,
                       xl[:, 1] + p.dt * c2.min(0) / p.wheelbase - k - p.noise_psi, xl[:, 2]], 1)
        hi = np.stack([xh[:, 0] + p.dt * c1.max(0) + p.noise_y,
                       xh[:, 1] + p.dt * c2.max(0) / p.wheelbase + k + p.noise_psi, xh[:, 2]], 1)
        return lo, hi


def exact_coverage(cells, n_inputs, vband=(P.v_min, P.v_max)):
    Px = RectangularPartition.uniform([-P.e_y_max, -P.e_psi_max, vband[0]],
                                      [P.e_y_max, P.e_psi_max, vband[1]], cells)
    Pu = RectangularPartition.uniform([-P.delta_max], [P.delta_max], np.atleast_1d(n_inputs))
    abst = SymbolicAbstraction(Px, Pu, ExactIntervalModel(RectangularPartition.product(Px, Pu)))
    return safety(abst, np.ones(Px.n_cells, bool)).winning_fraction()


if __name__ == "__main__":
    t0 = time.perf_counter()
    f1 = build_abstraction_filter()
    s = dict(f1.stats, wall_time=time.perf_counter() - t0)
    s["exact_model_same_grid"] = exact_coverage(tuple(s["cells"]), s["n_inputs"])
    s["fine_grid_fixed_speed"] = {v: exact_coverage((300, 240, 1), 41, (v, v + 1e-4))
                                  for v in (15.0, 22.5, 29.9999)}
    t0 = time.perf_counter()
    q = build_quadrotor_filter()
    quad = {"lateral": q.filters[0].stats, "vertical": q.filters[1].stats,
            "wall_time": time.perf_counter() - t0}
    res = {"lane_keeping": s, "quadrotor": quad}
    print(json.dumps(res, indent=1))
    out = Path(__file__).resolve().parents[1] / "results" / "f1_build.json"
    out.write_text(json.dumps(res, indent=1))
