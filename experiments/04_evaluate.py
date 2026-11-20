"""Evaluation: one results file for the README, the figures and the website.

1. DeePC and the true-model MPC on the 20 held-out episodes of each benchmark, with and
   without the filter (run here; a few minutes).
2. The RL runs of 03_train_rl.py: violations during training (logs in runs/rl/), and
   every trained agent (runs/models/) on the held-out episodes with and without the
   filter: return, crashes, how much the filter steps in, and the reliance test
   (agents trained behind the filter, run with the filter removed).
3. Time per step: each controller alone, and the filter.

Run: python experiments/04_evaluate.py [--skip-controllers] [--skip-latency]
Writes results/evaluation.json (and results/rl_curves.json for the training curves).
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from sfilter.controllers.deepc import LaneDeePC, QuadDeePC
from sfilter.controllers.mpc import LaneMPC, QuadMPC
from sfilter.controllers.rl import make_env
from sfilter.envs import LaneKeepingParams
from sfilter.envs.quadrotor import QuadrotorParams
from sfilter.evaluation import evaluate, summarize
from sfilter.filters import build_abstraction_filter
from sfilter.filters.quadrotor import build_quadrotor_filter

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
BENCH = ("lane", "quad")
ALGOS = ("sac", "ppo")
CONDS = ("none", "filter", "penalty", "penalty5", "stop")
N_BINS = 60


def controllers(filters):
    out = {}
    make = {"lane": {"mpc": lambda: LaneMPC(LaneKeepingParams()),
                     "deepc": lambda: LaneDeePC(LaneKeepingParams())},
            "quad": {"mpc": lambda: QuadMPC(QuadrotorParams()),
                     "deepc": lambda: QuadDeePC(QuadrotorParams())}}
    for b in BENCH:
        for name, ctor in make[b].items():
            for cond, f in (("free", None), ("filtered", filters[b])):
                s = summarize(evaluate(ctor(), b, f))
                out[f"{b}/{name}/{cond}"] = s
                print(b, name, cond, s, flush=True)
    return out


def _ms(x):
    """mean and standard deviation over seeds"""
    x = np.asarray(x, float)
    return dict(mean=float(x.mean()), std=float(x.std()), values=[float(v) for v in x])


def rl_eval(filters):
    """Evaluate every trained agent on the held-out episodes, with and without the
    filter. Cached per model in runs/rl_eval/."""
    import torch
    from stable_baselines3 import PPO, SAC

    torch.set_num_threads(1)
    out_dir = ROOT / "runs" / "rl_eval"
    out_dir.mkdir(exist_ok=True)
    for path in sorted((ROOT / "runs" / "models").glob("*.zip")):
        out = out_dir / f"{path.stem}.npz"
        if out.exists():
            continue
        b, algo = path.stem.split("_")[:2]
        model = (SAC if algo == "sac" else PPO).load(path, device="cpu")

        def policy(obs, model=model):
            return model.predict(obs, deterministic=True)[0]

        data = {}
        for name, f in (("free", None), ("filtered", filters[b])):
            r = evaluate(policy, b, f)
            data.update({f"{name}_{k}": v for k, v in r.items()})
        np.savez(out, **data)
        print("evaluated", path.stem, flush=True)


SPAN = {"lane": 0.1, "quad": float(np.linalg.norm([6.0, 6.0]))}   # input range (norm)


def rl_results():
    """Aggregate the training logs (runs/rl/) and the evaluations (runs/rl_eval/) over seeds."""
    out, curves = {}, {}
    for b in BENCH:
        for algo in ALGOS:
            for cond in CONDS:
                files = sorted((ROOT / "runs" / "rl").glob(f"{b}_{algo}_{cond}_*.npz"))
                if not files:
                    continue
                runs = [np.load(f) for f in files]
                ev = [np.load(ROOT / "runs" / "rl_eval" / f.name) for f in files]
                steps = int(runs[0]["steps"])
                own, other = (("free", "filtered") if cond == "none" else ("filtered", "free"))

                def share(e, name):
                    return e[f"{name}_filtered"].sum() / e[f"{name}_length"].sum()

                r = dict(
                    seeds=len(runs), steps=steps,
                    train_violations=_ms([r["train_violation"].sum() for r in runs]),
                    # episodes ended by a large correction ("stop" condition only)
                    train_stops=_ms([r["train_stopped"].sum() if "train_stopped" in r else 0
                                     for r in runs]),
                    # mean change made by the filter per training step, % of the input range
                    train_correction_pct=_ms([100 * r["train_correction"].sum()
                                              / r["train_length"].sum() / SPAN[b] for r in runs]),
                    # as trained: without the filter for "none", behind it otherwise
                    eval_return=_ms([e[f"{own}_ret"].mean() for e in ev]),
                    eval_crashes=int(sum(e[f"{own}_violation"].sum() for e in ev)),
                    eval_filtered_share=_ms([share(e, own) for e in ev]),
                    eval_correction_pct=_ms([100 * e[f"{own}_correction"].sum()
                                             / e[f"{own}_length"].sum() / SPAN[b] for e in ev]),
                    eval_min_margin=float(min(e[f"{own}_margin"].min() for e in ev)),
                    eval_episodes=int(sum(len(e[f"{own}_ret"]) for e in ev)),
                    # the other deployment: filter removed (or added) after training
                    swap_return=_ms([e[f"{other}_ret"].mean() for e in ev]),
                    swap_crashes=int(sum(e[f"{other}_violation"].sum() for e in ev)),
                    swap_filtered_share=_ms([share(e, other) for e in ev]),
                    wall_min=_ms([float(r["wall"]) / 60 for r in runs]),
                    filter_us=_ms([1e6 * float(r["filter_seconds"]) / max(1, int(r["filter_calls"]))
                                   for r in runs]),
                    filter_time_share=_ms([float(r["filter_seconds"]) / float(r["wall"])
                                           for r in runs]))
                out[f"{b}/{algo}/{cond}"] = r
                # training curves on a common grid of environment steps
                edges = np.linspace(0, steps, N_BINS + 1)
                cum, ret = [], []
                for run in runs:
                    st = run["train_step"]
                    cum.append([int(run["train_violation"][st <= e].sum()) for e in edges[1:]])
                    idx = np.digitize(st, edges[1:-1])
                    ret.append([float(run["train_ret"][idx == k].mean()) if np.any(idx == k)
                                else None for k in range(N_BINS)])
                curves[f"{b}/{algo}/{cond}"] = dict(step=edges[1:].tolist(),
                                                    cum_violations=cum, ret=ret)
    return out, curves


def latency(filters, n=5000):
    """Time per step on one thread: each RL policy alone, and the filter (median, p99)."""
    import torch
    from stable_baselines3 import PPO, SAC

    torch.set_num_threads(1)
    out = {}
    for b in BENCH:
        env = make_env(b, None)
        obs, _ = env.reset(seed=0)
        base = env.unwrapped
        rng = np.random.default_rng(0)
        X, U = [], []
        for _ in range(n):            # states and proposed inputs along a random drive
            a = rng.uniform(-1, 1, env.action_space.shape)
            X.append(base.state.copy())
            U.append(env.action_space.low * 0 + a * base.action_space.high)
            obs, _, term, trunc, _ = env.step(0.3 * a)
            if term or trunc:
                obs, _ = env.reset()
        f = filters[b]
        f.filter(X[0], U[0])
        ts = []
        for x, u in zip(X, U, strict=True):
            t0 = time.perf_counter()
            f.filter(x, u)
            ts.append(time.perf_counter() - t0)
        out[f"{b}/filter_us"] = dict(median=1e6 * float(np.median(ts)),
                                     p99=1e6 * float(np.percentile(ts, 99)))
        for algo, cls in (("sac", SAC), ("ppo", PPO)):
            path = ROOT / "runs" / "models" / f"{b}_{algo}_filter_0.zip"
            if not path.exists():
                continue
            model = cls.load(path, device="cpu")
            ts = []
            for _ in range(n):
                t0 = time.perf_counter()
                model.predict(obs, deterministic=True)
                ts.append(time.perf_counter() - t0)
            out[f"{b}/{algo}_policy_us"] = dict(median=1e6 * float(np.median(ts)),
                                                p99=1e6 * float(np.percentile(ts, 99)))
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-controllers", action="store_true")
    ap.add_argument("--skip-latency", action="store_true")
    a = ap.parse_args()
    path = RES / "evaluation.json"
    res = json.loads(path.read_text()) if path.exists() else {}
    filters = {"lane": build_abstraction_filter(), "quad": build_quadrotor_filter()}
    res["coverage"] = {"lane": filters["lane"].coverage,
                       "quad_lateral": filters["quad"].filters[0].coverage,
                       "quad_vertical": filters["quad"].filters[1].coverage}
    if not a.skip_controllers:
        res["controllers"] = controllers(filters)
    rl_eval(filters)
    rl, curves = rl_results()
    # merge: runs/ is not kept in git, so a later session may evaluate only new runs
    res["rl"] = {**res.get("rl", {}), **rl}
    cpath = RES / "rl_curves.json"
    curves = {**(json.loads(cpath.read_text()) if cpath.exists() else {}), **curves}
    if not a.skip_latency:
        res["latency"] = latency(filters)
    path.write_text(json.dumps(res, indent=1))
    cpath.write_text(json.dumps(curves))
    for k, v in res["rl"].items():
        print(f"{k:18s} train viol {v['train_violations']['mean']:6.1f}  return "
              f"{v['eval_return']['mean']:6.1f} ± {v['eval_return']['std']:4.1f}  crashes "
              f"{v['eval_crashes']}/{v['eval_episodes']}  filter steps "
              f"{v['eval_filtered_share']['mean']:.1%}  swapped: return "
              f"{v['swap_return']['mean']:6.1f} crashes {v['swap_crashes']}")
