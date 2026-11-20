"""Can ddma's gridless envelopes (v0.9) shield lane keeping? A feasibility sweep.

``ddma.embedding_envelope`` certifies boxes that stay invariant for every input in a
constant input box. Lane keeping is an integrator chain (steering moves the heading,
the heading moves the offset), so no box of (e_y, e_psi) is invariant under a constant
input: at the corner (e_y high, e_psi > 0) every input lets e_y grow. The standard fix is
a fixed baseline loop

    z = (e_y + ell * e_psi, e_psi),   delta = -k1 z1 - (L / ell) z2 + r,

which makes the closed loop shrink boxes; the envelope then certifies boxes of z with a
box of residual inputs r. This script sweeps (ell, k1, r range, box size, speed band),
with and without curvature feed-forward, and reports the certified share of the lane
specification and the width of the residual input range.

Result (16 Nov 2026): without feed-forward no box is certified at all; with feed-forward
(curvature error <= 10 % of its bound) at most ~12 % of the specification, in the
25-30 m/s band only, with residual ranges of ~0.002 rad. The abstraction filter (F1)
certifies ~46 % at all speeds. Run: python experiments/00_envelope_feasibility.py
"""

from __future__ import annotations

import itertools
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
from ddma import BoxEnvelope, embedding_envelope

from sfilter.envs import LaneKeepingParams, lane_keeping_step

P0 = LaneKeepingParams()


def envelope(p, ell, k1, rmax, Z1, vband, kmax, n=2000, seed=0):
    L, dt = p.wheelbase, p.dt
    k2 = L / ell
    Z2 = min((p.delta_max - k1 * Z1 - rmax) / k2, p.e_psi_max)  # no saturation
    if Z2 <= 0:
        return None
    lo, hi = np.array([-Z1, -Z2, vband[0]]), np.array([Z1, Z2, vband[1]])
    rng = np.random.default_rng(seed)
    Z = rng.uniform(lo, hi, (n, 3))
    R = rng.uniform(-rmax, rmax, (n, 1))
    m = n // 2  # mirrored pairs: more pairs span boxes around the lane centre
    Z[m:2 * m, :2], R[m:2 * m] = -Z[:m, :2], -R[:m]
    X = np.stack([Z[:, 0] - ell * Z[:, 1], Z[:, 1], Z[:, 2]], 1)
    delta = -k1 * Z[:, 0] - k2 * Z[:, 1] + R[:, 0]
    kap = rng.uniform(-kmax, kmax, n)
    w = rng.uniform([-p.noise_y, -p.noise_psi], [p.noise_y, p.noise_psi], (n, 2))
    Xn = lane_keeping_step(X, delta, kap, p, w)
    Zn = np.stack([Xn[:, 0] + ell * Xn[:, 1], Xn[:, 1], Xn[:, 2]], 1)
    # Jacobian bounds of the closed loop (interval arithmetic on the domain)
    vl, vh = vband
    s2, tn = 1 / np.cos(p.delta_max) ** 2, np.tan(p.delta_max)
    c, sn = np.cos(Z2), np.sin(Z2)
    a_lo, a_hi = dt * vl / L, dt * vh * s2 / L
    tv = dt * sn + ell * dt * tn / L  # bound on |d z1+ / d v|
    jx_lo = np.array([[1 - ell * a_hi * k1, dt * vl * c - ell * a_hi * k2, -tv],
                      [-a_hi * k1, 1 - a_hi * k2, -dt * tn / L], [0, 0, 1]])
    jx_hi = np.array([[1 - ell * a_lo * k1, dt * vh - ell * a_lo * k2, tv],
                      [-a_lo * k1, 1 - a_lo * k2, dt * tn / L], [0, 0, 1]])
    ju_lo, ju_hi = np.array([[ell * a_lo], [a_lo], [0]]), np.array([[ell * a_hi], [a_hi], [0]])
    kd = dt * vh * kmax
    w_hi = np.array([ell * kd + p.noise_y + ell * p.noise_psi, kd + p.noise_psi, 0.0])
    env = embedding_envelope(Z, Zn, R, jx_lo, jx_hi, ju_lo, ju_hi, -w_hi, w_hi, lo, hi)
    keep = ((env.hi[:, 0] - ell * env.lo[:, 1] <= p.e_y_max)
            & (env.lo[:, 0] - ell * env.hi[:, 1] >= -p.e_y_max))
    env = BoxEnvelope(env.lo[keep], env.hi[keep], env.ulo[keep], env.uhi[keep],
                      env.pairs[keep], env.n_pairs)
    S = rng.uniform([-p.e_y_max, -p.e_psi_max, vl], [p.e_y_max, p.e_psi_max, vh], (20000, 3))
    Sz = np.stack([S[:, 0] + ell * S[:, 1], S[:, 1], S[:, 2]], 1)
    cov = float(env.contains(Sz).mean()) if env.n_boxes else 0.0
    rw = float((env.uhi - env.ulo).max()) if env.n_boxes else 0.0
    return dict(ell=ell, k1=k1, r_max=rmax, Z1=Z1, vband=vband, boxes=int(env.n_boxes),
                coverage=round(cov, 3), residual_width=round(rw, 4))


def sweep(p, kmax):
    out = []
    grid = itertools.product(([15, 30], [15, 20], [25, 30]), (3, 5, 8), (0.02, 0.04, 0.08),
                             (0.002, 0.005, 0.01), (0.4, 0.7))
    for vb, ell, k1, rmax, Z1 in grid:
        r = envelope(p, ell, k1, rmax, Z1, vb, kmax)
        if r is not None:
            out.append(r)
    return sorted(out, key=lambda r: -r["coverage"])


if __name__ == "__main__":
    res = {"no_feedforward": sweep(P0, P0.kappa_max)}
    p_ff = replace(P0, delta_max=P0.delta_max - np.arctan(P0.wheelbase * P0.kappa_max))
    res["feedforward_10pct_error"] = sweep(p_ff, 0.1 * P0.kappa_max)
    for k, v in res.items():
        print(f"{k}: best coverage {v[0]['coverage'] if v else 0}  {v[0] if v else ''}")
    out = Path(__file__).resolve().parents[1] / "results" / "envelope_feasibility.json"
    out.write_text(json.dumps({k: v[:10] for k, v in res.items()}, indent=1))
    print(f"saved {out}")
