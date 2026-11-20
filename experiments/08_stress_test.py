"""Stress test: what happens when the assumptions behind the filter are wrong.

The filter's guarantee rests on stated assumptions (bounds on curvature or gusts, the
true state, the structure x+ = x + f(x, u) + w). Here each one is broken on purpose, on
the 20 held-out episodes of each benchmark:

  lane keeping  road curvature beyond its bound (x1.25 ... x3); noisy state given to the
                filter; steering applied one or two steps late
  quadrotor     gusts beyond their bound (x1.5 ... x3); noisy state given to the filter;
                slower tilt dynamics than assumed (lag 0.25 s, 0.4 s instead of 0.15 s)

Controllers: three crude agents that would crash without a filter (random, bang-bang /
chase, full lock) and the true-model MPC. Quadrotor: two filters, the nominal one and a
"robust" one built from the same data with twice the gust bound. Lane keeping: nominal
only (a larger curvature bound leaves no certified state at low speed).

For the sensor-noise and delay cases a third variant, "aware", is the nominal filter told
what is wrong: the noise bound (``state_error``: it allows only inputs that are safe for
every state the measurement could hide) or the delay (``DelayAwareFilter``: it predicts,
with its own model, where the car can be when the input takes effect).

ddma's assumption monitor (AssumptionMonitor) watches the transitions the filter sees,
seeded with the data the filter was learned from. For each episode we record whether it
raised an alarm and, if the car or drone crashed, whether the alarm came first.

Run: python experiments/08_stress_test.py [--bench lane|quad] [--quick]
Writes runs/stress/<bench>/*.json (one per case, restartable) and results/stress_test.json.
"""

from __future__ import annotations

import argparse
import json
import time
from collections import deque
from pathlib import Path

import gymnasium as gym
import numpy as np
from ddma import AssumptionMonitor
from gymnasium.wrappers import RescaleAction

from sfilter.controllers.mpc import LaneMPC, QuadMPC
from sfilter.controllers.rl import eval_options
from sfilter.envs import LaneKeepingEnv, LaneKeepingParams
from sfilter.envs.quadrotor import QuadrotorEnv, QuadrotorParams
from sfilter.envs.roads import evaluation_roads
from sfilter.filters import SafetyFilterWrapper, build_abstraction_filter
from sfilter.filters.abstraction import DelayAwareFilter, collect_transitions
from sfilter.filters.quadrotor import build_quadrotor_filter, collect_quadrotor_transitions

ROOT = Path(__file__).resolve().parents[1]
LP, QP = LaneKeepingParams(), QuadrotorParams()

# (name, changes). noise k: uniform sensor noise of NOISE[bench] * k on the filter's state
SCENARIOS = {
    "lane": [("nominal", {}),
             ("curvature x1.25", {"scale": 1.25}), ("curvature x1.5", {"scale": 1.5}),
             ("curvature x2", {"scale": 2.0}), ("curvature x3", {"scale": 3.0}),
             ("sensor noise S", {"noise": 1}), ("sensor noise M", {"noise": 3}),
             ("sensor noise L", {"noise": 10}),
             ("steering 1 step late", {"delay": 1}), ("steering 2 steps late", {"delay": 2})],
    "quad": [("nominal", {}),
             ("gusts x1.5", {"scale": 1.5}), ("gusts x2", {"scale": 2.0}),
             ("gusts x3", {"scale": 3.0}),
             ("sensor noise S", {"noise": 1}), ("sensor noise M", {"noise": 3}),
             ("sensor noise L", {"noise": 10}),
             ("tilt lag 0.25 s", {"tau": 0.25}), ("tilt lag 0.4 s", {"tau": 0.4})],
}
# noise level 1: lane (e_y 1 cm, e_psi 0.002 rad); quadrotor (1 cm, 2 cm/s, 0.05 m/s^2)
NOISE = {"lane": np.array([0.01, 0.002, 0.0]),
         "quad": np.array([0.01, 0.02, 0.05, 0.01, 0.02])}
AGENTS = {"lane": ("random", "bang-bang", "full-lock", "mpc"),
          "quad": ("random", "chase", "full-thrust", "mpc")}


class Delay(gym.Wrapper):
    """Applies each action ``k`` steps late (zero action before that)."""

    def __init__(self, env, k):
        super().__init__(env)
        self.k = k
        self.q = deque()

    def reset(self, **kw):
        self.q = deque([np.zeros(self.env.action_space.shape)] * self.k)
        return self.env.reset(**kw)

    def step(self, a):
        self.q.append(np.asarray(a, float))
        return self.env.step(self.q.popleft())


def build_filters(bench, quick):
    """Nominal and robust (2x disturbance bound) filters from the same data, plus the
    data for the monitors."""
    n = 2_000_000
    rng = np.random.default_rng(0)
    if bench == "lane":
        kw = {}
        data = collect_transitions(n, LP, rng)
        # no robust variant: the curvature bound enters with the top speed, and even
        # x1.25 leaves no certified state at 15 m/s (see the notes in the README)
        return {"nominal": build_abstraction_filter(data=data, **kw)}, data
    else:
        kw = {}
        data = collect_quadrotor_transitions(n, QP, rng)
        nominal = build_quadrotor_filter(data=data, **kw)
        robust = build_quadrotor_filter(data=data, p=QuadrotorParams(wind=2 * QP.wind), **kw)
    return {"nominal": nominal, "robust": robust}, data


MONITOR_SEED = 20_000   # monitor seeded with this many of the training transitions


def monitors(bench, filt, data):
    """One AssumptionMonitor per learned model, seeded with the first MONITOR_SEED
    training transitions (a random sample: with all 2 million, each check takes tens of
    ms; with 20,000 under 1 ms and it flags almost the same transitions); returns a
    function (x, u, x') -> True when the transition contradicts the assumptions."""
    X, U, Xn = (a[:MONITOR_SEED] for a in data)
    if bench == "lane":
        parts = [(filt.model, slice(0, 3), slice(0, 1))]
    else:
        parts = [(f.model, s, i) for f, s, i in
                 zip(filt.filters, (slice(0, 3), slice(3, 5)), (slice(0, 1), slice(1, 2)),
                     strict=True)]
    mons = [(AssumptionMonitor.for_model(m, np.hstack([X[:, s], U[:, i]]), Xn[:, s] - X[:, s]),
             s, i) for m, s, i in parts]

    def check(x, u, xn):
        bad = False
        for mon, s, i in mons:
            v = mon.observe(x[None, s], np.atleast_1d(u)[None, i], xn[None, s])
            bad |= len(v) > 0
        return bad
    return check


def make_case(bench, ch):
    """Environment (true dynamics) and held-out episodes for one scenario."""
    s = ch.get("scale", 1.0)
    if bench == "lane":
        env = LaneKeepingEnv(params=LaneKeepingParams(kappa_max=s * LP.kappa_max))
        eps = eval_options("lane")
        roads = evaluation_roads(len(eps), kappa_max=s * LP.kappa_max)
        for e, r in zip(eps, roads, strict=True):
            e["options"]["road"] = r
    else:
        env = QuadrotorEnv(params=QuadrotorParams(wind=s * QP.wind, tau=ch.get("tau", QP.tau)))
        eps = eval_options("quad")
    return env, eps


def agent(bench, name, seed):
    rng = np.random.default_rng(seed)
    if name == "mpc":
        return LaneMPC(LP) if bench == "lane" else QuadMPC(QP)
    if bench == "lane":
        return {"random": lambda o, e: rng.uniform(-1, 1, 1),
                "bang-bang": lambda o, e: np.array([1.0 if (e.t // 30) % 2 else -1.0]),
                "full-lock": lambda o, e: np.array([1.0])}[name]
    return {"random": lambda o, e: rng.uniform(-1, 1, 2),
            "chase": lambda o, e: np.sign(o[5:7]),
            "full-thrust": lambda o, e: np.array([1.0, 1.0])}[name]


def run_case(bench, filt, data, scen, ch, agent_name, use_filter=True, n_eps=None):
    env, eps = make_case(bench, ch)
    eps = eps[::len(eps) // n_eps] if n_eps else eps
    base = env
    rng = np.random.default_rng(7)
    noise = NOISE[bench] * ch.get("noise", 0)

    def measured(e):
        x = np.asarray(e.unwrapped.state, float)
        return x + rng.uniform(-noise, noise) if noise.any() else x.copy()

    inner = Delay(env, ch["delay"]) if ch.get("delay") else env
    if use_filter:
        inner = SafetyFilterWrapper(inner, filt, state_fn=measured)
    env = RescaleAction(inner, -1.0, 1.0)
    check = monitors(bench, filt, data) if use_filter else None
    ctrl = agent(bench, agent_name, 3)
    rows = []
    for ep in eps:
        obs, _ = env.reset(seed=ep["seed"], options=ep["options"])
        if hasattr(ctrl, "reset"):
            ctrl.reset(base)
        x_meas = measured(base)
        t, alarm, filtered, done = 0, None, 0, False
        margin = np.inf
        while not done:
            a = ctrl.act(obs, base) if hasattr(ctrl, "act") else ctrl(obs, base)
            obs, _, term, trunc, info = env.step(a)
            done = term or trunc
            t += 1
            if use_filter:
                filtered += int(info["correction"] > 1e-9)
                xn_meas = measured(base)
                if check(x_meas, info["safe_action"], xn_meas) and alarm is None:
                    alarm = t
                x_meas = xn_meas
            if bench == "lane":
                margin = min(margin, LP.e_y_max - abs(base.state[0]))
            else:
                x = base.state
                margin = min(margin, QP.y_max - abs(x[0]), x[3] - QP.z_min, QP.z_max - x[3])
        crash = bool(info["violation"])
        rows.append(dict(crash=crash, steps=t, alarm=alarm, filtered=filtered,
                         margin=float(margin)))
    return rows


def set_error(bench, filt, err):
    """Tell a filter the bound on the error of the state it is given (None: no error)."""
    if bench == "lane":
        filt.state_error = err
    else:
        filt.filters[0].state_error = None if err is None else err[:3]
        filt.filters[1].state_error = None if err is None else err[3:]


def summarize(rows):
    crashes = [r for r in rows if r["crash"]]
    lead = [r["steps"] - r["alarm"] for r in crashes if r["alarm"] is not None]
    return dict(episodes=len(rows), crashes=len(crashes),
                alarm_episodes=sum(r["alarm"] is not None for r in rows),
                crashes_alarmed_first=len(lead),
                lead_steps_median=float(np.median(lead)) if lead else None,
                lead_steps_min=int(min(lead)) if lead else None,
                min_margin=float(min(r["margin"] for r in rows)),
                filtered_share=float(sum(r["filtered"] for r in rows)
                                     / sum(r["steps"] for r in rows)))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", choices=("lane", "quad"), default=None)
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args()
    out_all = ROOT / "results" / "stress_test.json"
    res = json.loads(out_all.read_text()) if out_all.exists() and not a.quick else {}
    for bench in ([a.bench] if a.bench else ["lane", "quad"]):
        d = ROOT / "runs" / ("stress_quick" if a.quick else "stress") / bench
        d.mkdir(parents=True, exist_ok=True)
        filters, data = build_filters(bench, a.quick)
        res.setdefault(bench, {})["coverage"] = {
            k: (f.coverage if bench == "lane" else [g.coverage for g in f.filters])
            for k, f in filters.items()}
        scen = SCENARIOS[bench][:3] if a.quick else SCENARIOS[bench]
        cases = [(fn, s, ch, ag, True) for fn in filters for s, ch in scen
                 for ag in AGENTS[bench]]
        # baseline: the same scenarios without any filter (crude agents crash anyway)
        cases += [("none", s, ch, ag, False) for s, ch in scen for ag in AGENTS[bench]]
        # the nominal filter told the noise bound
        cases += [("aware", s, ch, ag, True) for s, ch in scen
                  if "noise" in ch or "delay" in ch for ag in AGENTS[bench]]
        for fn, s, ch, ag, use in cases:
            path = d / f"{fn}__{s.replace(' ', '_')}__{ag}.json"
            if not path.exists():
                t0 = time.perf_counter()
                filt = filters["nominal" if fn == "aware" else fn] if use else None
                if fn == "aware" and "noise" in ch:
                    set_error(bench, filt, NOISE[bench] * ch["noise"])
                if fn == "aware" and "delay" in ch:
                    filt = DelayAwareFilter(filt, ch["delay"])
                rows = run_case(bench, filt, data, s, ch, ag, use,
                                n_eps=4 if a.quick else None)
                if fn == "aware" and "noise" in ch:
                    set_error(bench, filt, None)
                path.write_text(json.dumps(rows))
                print(f"{time.strftime('%H:%M')} {bench} {fn:7s} {s:22s} {ag:11s} "
                      f"crashes {sum(r['crash'] for r in rows):2d}  "
                      f"({time.perf_counter() - t0:.0f} s)", flush=True)
            rows = json.loads(path.read_text())
            res[bench].setdefault(fn, {}).setdefault(s, {})[ag] = summarize(rows)
        if not a.quick:
            out_all.write_text(json.dumps(res, indent=1))
    if a.quick:
        print(json.dumps(res, indent=1)[:3000])
