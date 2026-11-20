"""F1 for the quadrotor: one abstraction filter per channel, combined.

Lateral (y, vy, a_y) with input a_y,c; vertical (z, vz) with input a_z,c. Both are
learned from the same transitions (random states in the safe box, random commands and
gusts).
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from ..envs.quadrotor import QuadrotorParams, quadrotor_step
from .abstraction import ProductFilter, learn_abstraction_filter

__all__ = ["collect_quadrotor_transitions", "build_quadrotor_filter"]

LAT_FEATURES = [[1], [1, 2], [2, 3]]   # (y, vy, a_y, a_y,c)
VERT_FEATURES = [[1], [1, 2]]          # (z, vz, a_z,c)


def collect_quadrotor_transitions(n: int, p: QuadrotorParams, rng: np.random.Generator):
    """n transitions with states uniform in the safe box, commands and gusts uniform."""
    X = rng.uniform(p.lo, p.hi, (n, 5))
    U = rng.uniform(p.u_lo, p.u_hi, (n, 2))
    W = rng.uniform(-p.wind, p.wind, (n, 2))
    return X, U, quadrotor_step(X, U, p, W)


def build_quadrotor_filter(n_data: int = 2_000_000, lat_cells=(120, 240, 40),
                           vert_cells=(120, 80), n_inputs: int = 21,
                           p: Optional[QuadrotorParams] = None, seed: int = 0,
                           data=None) -> ProductFilter:
    p = p or QuadrotorParams()
    rng = np.random.default_rng(seed)
    X, U, Xn = data if data is not None else collect_quadrotor_transitions(n_data, p, rng)
    A_lo, A_hi, B_lo, B_hi, d_lo, d_hi = p.lat_bounds()
    lat = learn_abstraction_filter(X[:, :3], U[:, :1], Xn[:, :3], p.lo[:3], p.hi[:3],
                                   p.u_lo[:1], p.u_hi[:1], lat_cells, n_inputs,
                                   A_lo, A_hi, B_lo, B_hi, d_lo, d_hi, LAT_FEATURES)
    A_lo, A_hi, B_lo, B_hi, d_lo, d_hi = p.vert_bounds()
    vert = learn_abstraction_filter(X[:, 3:], U[:, 1:], Xn[:, 3:], p.lo[3:], p.hi[3:],
                                    p.u_lo[1:], p.u_hi[1:], vert_cells, n_inputs,
                                    A_lo, A_hi, B_lo, B_hi, d_lo, d_hi, VERT_FEATURES)
    return ProductFilter([lat, vert], states=[[0, 1, 2], [3, 4]], inputs=[[0], [1]])
