"""Evaluation of any controller on the fixed held-out episodes, with or without a filter.

A controller is either a function ``obs -> action`` (a trained RL policy) or an object
with ``reset(env)`` and ``act(obs, env)`` (MPC, DeePC: they read the simulator's state).
Actions are scaled to [-1, 1].
"""

from __future__ import annotations

import time
from typing import Optional

import numpy as np

from .controllers.rl import TimedFilter, _margin, eval_options, make_env

__all__ = ["evaluate", "summarize"]


def evaluate(controller, benchmark: str, filt=None, n: int = 20, horizon: int = 600,
             record: bool = False, episodes: Optional[list] = None) -> dict:
    """Per-episode arrays: ``ret`` (return, at most ``horizon``), ``length``,
    ``violation``, ``filtered`` (steps changed by the filter), ``correction``, ``margin``
    (smallest distance to a position limit, metres); plus ``ctrl_ms`` (mean controller
    time per step) and ``filter_us`` (mean filter time per call).

    ``record=True`` adds ``traj``: per episode, the states, proposed and applied actions
    (physical units) and, for the quadrotor, the target positions.
    """
    tf = TimedFilter(filt) if filt is not None else None
    env = make_env(benchmark, tf, horizon=horizon)
    base = env.unwrapped
    has_act = hasattr(controller, "act")
    margins, traj, t_ctrl, n_steps = [], [], 0.0, 0
    for ep in episodes or eval_options(benchmark, n):
        obs, _ = env.reset(seed=ep["seed"], options=ep["options"])
        if has_act:
            controller.reset(base)
        done, m = False, np.inf
        rec = dict(x=[base.state.copy()], proposed=[], applied=[], target=[])
        while not done:
            t0 = time.perf_counter()
            a = controller.act(obs, base) if has_act else controller(obs)
            t_ctrl += time.perf_counter() - t0
            n_steps += 1
            if record and benchmark == "quad":
                rec["target"].append(base.target()[0])
            obs, _, term, trunc, info = env.step(a)
            m = min(m, _margin(base, benchmark))
            done = term or trunc
            applied = info["delta"] if benchmark == "lane" else info["action"]
            if hasattr(controller, "applied"):      # DeePC keeps a record of past inputs
                controller.applied(applied)
            if record:
                rec["x"].append(base.state.copy())
                rec["applied"].append(np.atleast_1d(applied))
                rec["proposed"].append(np.atleast_1d(info.get("proposed_action", applied)))
        margins.append(m)
        if record:
            traj.append({k: np.array(v) for k, v in rec.items()})
    out = env.arrays()
    out["margin"] = np.array(margins)
    out["ctrl_ms"] = 1e3 * t_ctrl / n_steps
    out["filter_us"] = 1e6 * tf.seconds / tf.calls if tf else 0.0
    if record:
        out["traj"] = traj
    return out


def summarize(res: dict) -> dict:
    """Numbers for the results table from :func:`evaluate`'s output."""
    steps = int(np.sum(res["length"]))
    return dict(episodes=int(len(res["ret"])),
                crashes=int(np.sum(res["violation"])),
                ret=float(np.mean(res["ret"])),
                min_margin=float(np.min(res["margin"])),
                filtered_share=float(np.sum(res["filtered"]) / steps),
                mean_correction=float(np.sum(res["correction"]) / steps),
                ctrl_ms=float(res.get("ctrl_ms", 0.0)),
                filter_us=float(res.get("filter_us", 0.0)))
