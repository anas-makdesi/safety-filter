"""Lane keeping with a real car: a filter learned from 2 hours of RAV4 driving logs.

1. Real steps (speed, steering-wheel angle, yaw rate) from comma2k19, chunk 1 for
   learning, chunk 2 (other days) held out.
2. ddma.check_data: are all learning steps consistent with the stated assumptions (slope
   bounds of the steering response, yaw-rate error bound rho)? Pairwise, exact.
3. The filter (sfilter.real.lane): grid abstraction of the lane-keeping model whose
   heading change comes from the real steps.
4. Held-out test: for every held-out real step, is the real heading change inside the
   box the learned model predicts?
5. Closed loop on a simulated car fitted to the held-out logs (steady-state cornering,
   steering lag, the real residuals replayed in order): crude agents and a simple lane-centring
   driver, with and without the filter, on 20 held-out roads.

Run: python experiments/11_real_lane.py
(needs data/comma2k19_rav4_steps.npz, made from the raw download on first run:
python experiments/10_comma_download.py --chunks 1 2)
Writes results/real_lane.json.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
from ddma import check_data
from gymnasium.wrappers import RescaleAction

from sfilter.envs.roads import evaluation_roads
from sfilter.filters import SafetyFilterWrapper
from sfilter.filters.abstraction import DelayAwareFilter
from sfilter.real.comma import Twin, fit_twin, load_drives
from sfilter.real.lane import RealLaneEnv, RealLaneParams, build_real_lane_filter

ROOT = Path(__file__).resolve().parents[1]
STEPS = ROOT / "data" / "comma2k19_rav4_steps.npz"
P = RealLaneParams()


def real_steps():
    """Per-step arrays (v, sigma, r, drive id) for chunk 1 (learn) and chunk 2 (test),
    cached in data/comma2k19_rav4_steps.npz (about 2 MB, kept in git)."""
    if not STEPS.exists():
        out = {}
        for name, c in (("learn", 1), ("test", 2)):
            drives = load_drives(ROOT / "data" / "comma2k19_raw" / f"chunk{c}")
            out[f"{name}_v"] = np.concatenate([d.v for d in drives]).astype(np.float32)
            out[f"{name}_sigma"] = np.concatenate([d.sigma for d in drives]).astype(np.float32)
            out[f"{name}_r"] = np.concatenate([d.r for d in drives]).astype(np.float32)
            out[f"{name}_drive"] = np.concatenate([np.full(len(d.t), i, np.int32)
                                                   for i, d in enumerate(drives)])
        np.savez_compressed(STEPS, **out)
    d = np.load(STEPS)
    return {k: d[k].astype(float) if d[k].dtype == np.float32 else d[k] for k in d.files}


def in_band(d, name, sigma_max=None):
    v, s, r = d[f"{name}_v"], d[f"{name}_sigma"], d[f"{name}_r"]
    sm = P.sigma_max if sigma_max is None else sigma_max
    m = (v >= P.v_min) & (v <= P.v_max) & (np.abs(s) <= sm)
    return v[m], s[m], r[m], d[f"{name}_drive"][m]


def consistency(v, s, r):
    A_lo, A_hi, B_lo, B_hi = P.derivative_bounds()
    sl = np.array([[A_lo[1, 2], B_lo[1, 0]]])
    sh = np.array([[A_hi[1, 2], B_hi[1, 0]]])
    c = check_data(np.stack([v, s], 1), (P.dt * r)[:, None], sl, sh,
                   -P.dt * P.rho, P.dt * P.rho)
    return dict(ok=bool(c.ok), points=int(c.n_points), violating_pairs=int(c.violating[0]),
                rho_needed=float(c.needed["width"][0] / (2 * P.dt)))


def heldout_boxes(f, v, s, r):
    """Share of held-out real heading changes inside the learned model's box (the box
    of f + w, without the road-curvature part, which the logs do not contain)."""
    ab = f.controller.abstraction
    X, U, model = ab.X, ab.U, f.model
    x = np.stack([np.zeros_like(v), np.zeros_like(v), v], 1)
    qi = X.locate(x)
    ui = U.locate(s[:, None])
    xl, xh = X.cell_bounds(qi)
    ul, uh = U.cell_bounds(ui)
    lo, hi = model.successor_bounds(np.hstack([qi, ui]), xl, xh, ul, uh)
    curv = P.dt * P.v_max * P.kappa_max
    lo_p, hi_p = lo[:, 1] - xl[:, 1] + curv, hi[:, 1] - xh[:, 1] - curv
    y = P.dt * r
    inside = (y >= lo_p) & (y <= hi_p)
    return dict(steps=int(len(y)), inside=float(inside.mean()),
                mean_box_width_deg_per_s=float(np.rad2deg(np.mean(hi_p - lo_p)) / P.dt))


def lag_scan(d):
    """Steering-to-yaw-rate delay (in steps) with the smallest residual of the fit."""
    rms = {}
    for lag in range(5):
        v, s, r, ids = (d[f"learn_{k}"] for k in ("v", "sigma", "r", "drive"))
        same = ids[lag:] == ids[:len(ids) - lag]
        vv, ss, rr = v[lag:][same], s[:len(s) - lag][same], r[lag:][same]
        m = (vv >= P.v_min) & (vv <= P.v_max) & (np.abs(ss) <= 12)
        rms[lag] = float(np.sqrt(np.mean(fit_twin(vv[m], ss[m], rr[m]).residuals ** 2)))
    return min(rms, key=rms.get), rms


def driver(gain_y=6.0, gain_psi=60.0):
    """A plain lane-centring driver: steering-wheel angle [deg] from offset and heading."""
    def act(obs, env):
        ey, ep = env.state[:2]
        sigma = -gain_y * ey - gain_psi * ep + env.twin.sigma0
        return np.array([np.clip(sigma / P.sigma_max, -1, 1)])
    return act


def agents(seed=3):
    rng = np.random.default_rng(seed)
    return {"random": lambda o, e: rng.uniform(-1, 1, 1),
            "bang-bang": lambda o, e: np.array([1.0 if (e.t // 30) % 2 else -1.0]),
            "full-lock": lambda o, e: np.array([1.0]),
            "driver": driver()}


def closed_loop(twin: Twin, filt, n=20):
    roads = evaluation_roads(n, kappa_max=P.kappa_max)
    speeds = np.linspace(P.v_min, P.v_max - 1e-6, n)
    out = {}
    for name, ag in agents().items():
        base = RealLaneEnv(twin, P)
        env = base if filt is None else SafetyFilterWrapper(base, filt)
        env = RescaleAction(env, -1.0, 1.0)
        rows = []
        for i in range(n):
            obs, _ = env.reset(seed=950_000 + i, options=dict(road=roads[i], v=float(speeds[i]),
                                                                state=(0.0, 0.0)))
            done, ret, filtered, margin, t = False, 0.0, 0, np.inf, 0
            while not done:
                obs, rew, term, trunc, info = env.step(ag(obs, base))
                done, ret, t = term or trunc, ret + rew, t + 1
                filtered += int(info.get("correction", 0.0) > 1e-9)
                margin = min(margin, P.e_y_max - abs(base.state[0]))
            rows.append(dict(crash=bool(info["violation"]), ret=ret, steps=t,
                             filtered=filtered, margin=float(margin)))
        out[name] = dict(crashes=sum(r["crash"] for r in rows), episodes=n,
                         ret=float(np.mean([r["ret"] for r in rows])),
                         min_margin=float(min(r["margin"] for r in rows)),
                         filtered_share=float(sum(r["filtered"] for r in rows)
                                              / sum(r["steps"] for r in rows)))
        print("   ", name, out[name], flush=True)
    return out


if __name__ == "__main__":
    d = real_steps()
    res = {"params": {k: (list(v) if isinstance(v, tuple) else v) for k, v in asdict(P).items()}}
    v, s, r, _ = in_band(d, "learn")
    vt, st, rt, _ = in_band(d, "test")
    res["data"] = dict(learn_steps=int(len(v)), test_steps=int(len(vt)),
                       learn_minutes=len(v) * P.dt / 60, test_minutes=len(vt) * P.dt / 60,
                       learn_sigma_pct=[float(x) for x in np.percentile(s, [0.5, 99.5])],
                       share_beyond_12deg=float(np.mean(np.abs(s) > 12)))
    t0 = time.perf_counter()
    res["check_learn"] = consistency(v, s, r)
    res["check_test"] = consistency(vt, st, rt)
    res["check_seconds"] = time.perf_counter() - t0
    print("consistency", res["check_learn"], res["check_test"], flush=True)
    lag, rms = lag_scan(d)
    # the simulated car is fitted to the held-out chunk, so the closed-loop test uses a
    # car (and residuals) the filter never saw
    tw = fit_twin(vt, st, rt, lag_steps=lag)
    res["twin"] = dict(G=tw.G, K=tw.K, sigma0=tw.sigma0, lag_steps=lag, lag_rms=rms,
                       residual_pct=[float(x) for x in
                                     np.percentile(np.abs(tw.residuals), [50, 99, 99.9, 100])])
    print("twin", res["twin"], flush=True)
    t0 = time.perf_counter()
    f, _ = build_real_lane_filter(v, s, r, P)
    res["filter"] = dict(f.stats, wall=time.perf_counter() - t0,
                         contains_centre={f"{x:.0f}": f.contains([0, 0, x])
                                          for x in np.linspace(P.v_min + 0.01, P.v_max - 0.01, 5)})
    print("filter", res["filter"], flush=True)
    res["heldout"] = heldout_boxes(f, vt, st, rt)
    print("held-out", res["heldout"], flush=True)
    res["closed_loop"] = {}
    for name, filt in (("none", None), ("filter", f),
                       ("delay-aware filter", DelayAwareFilter(f, lag))):
        print(name, flush=True)
        res["closed_loop"][name] = closed_loop(tw, filt)
    (ROOT / "results" / "real_lane.json").write_text(json.dumps(res, indent=1))
