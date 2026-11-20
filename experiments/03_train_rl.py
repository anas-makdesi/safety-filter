"""Train SAC and PPO on both benchmarks, in three conditions, five seeds each.

Conditions:
  none      no safety filter (the agent can crash while it learns)
  filter    behind the abstraction filter
  penalty   behind the filter, and each correction by the filter costs reward
  penalty5  the same with a five times larger penalty
  stop      behind the filter; a correction larger than 5 % of the input range ends the
            episode with the crash reward (the system stays safe: the filter's input is
            applied; only the agent is told it would have crashed)
The first three are the default; the last two (follow-up of 19 Nov, against the reliance
on the filter) are run with --conditions penalty5 stop --bench quad.

Each run writes runs/rl/<benchmark>_<algo>_<condition>_<seed>.npz (training log, timing,
and the trained agent's results on the 20 held-out episodes, with and without the filter)
and the model in runs/models/. Finished runs are skipped, so the script can be restarted.

Run: python experiments/03_train_rl.py [--workers 2] [--quick] [--limit N]
     [--conditions ...] [--bench ...] [--algos ...]
"""

from __future__ import annotations

import argparse
import itertools
import multiprocessing as mp
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
STEPS = {"sac": 60_000, "ppo": 300_000}
CONDITIONS = ("none", "filter", "penalty", "penalty5", "stop")
DEFAULT = ("none", "filter", "penalty")
PENALTY = {"penalty": 1.0, "penalty5": 5.0}   # reward per step for a full-range correction
STOP = 0.05       # "stop": share of the input range above which a correction ends the episode
SEEDS = range(5)
_filters = {}


def _init():
    from sfilter.filters import build_abstraction_filter
    from sfilter.filters.quadrotor import build_quadrotor_filter
    _filters["lane"] = build_abstraction_filter()
    _filters["quad"] = build_quadrotor_filter()


def run(job):
    bench, algo, cond, seed, steps, out = job
    from sfilter.controllers.rl import TimedFilter, evaluate, train

    filt = None if cond == "none" else TimedFilter(_filters[bench])
    t0 = time.perf_counter()
    model, env = train(algo, bench, filt, steps, seed, penalty=PENALTY.get(cond, 0.0),
                       end_on_correction=STOP if cond == "stop" else None)
    wall = time.perf_counter() - t0
    data = {f"train_{k}": v for k, v in env.arrays().items()}
    data.update(wall=wall, steps=steps,
                filter_seconds=filt.seconds if filt else 0.0,
                filter_calls=filt.calls if filt else 0)

    def policy(obs):
        return model.predict(obs, deterministic=True)[0]

    for name, f in (("free", None), ("filtered", _filters[bench])):
        for k, v in evaluate(policy, bench, f).items():
            data[f"eval_{name}_{k}"] = v
    model.save(ROOT / "runs" / "models" / out.stem)
    tmp = out.with_suffix(".tmp.npz")
    np.savez(tmp, **data)
    tmp.rename(out)
    return f"{out.stem}: {wall / 60:.1f} min, training violations " \
           f"{int(data['train_violation'].sum())}"


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--quick", action="store_true", help="tiny budgets (smoke test)")
    ap.add_argument("--limit", type=int, default=0, help="stop after this many runs")
    ap.add_argument("--conditions", nargs="+", default=list(DEFAULT), choices=CONDITIONS)
    ap.add_argument("--bench", nargs="+", default=["lane", "quad"], choices=["lane", "quad"])
    ap.add_argument("--algos", nargs="+", default=["ppo", "sac"], choices=["ppo", "sac"])
    a = ap.parse_args()
    d = ROOT / "runs" / ("rl_quick" if a.quick else "rl")
    d.mkdir(parents=True, exist_ok=True)
    (ROOT / "runs" / "models").mkdir(exist_ok=True)
    jobs = []
    for seed, algo, bench, cond in itertools.product(
            range(1) if a.quick else SEEDS, a.algos, a.bench, a.conditions):
        out = d / f"{bench}_{algo}_{cond}_{seed}.npz"
        if not out.exists():
            jobs.append((bench, algo, cond, seed, 3000 if a.quick else STEPS[algo], out))
    print(f"{len(jobs)} runs to do", flush=True)
    if a.limit:
        jobs = jobs[:a.limit]
    with mp.get_context("spawn").Pool(a.workers, initializer=_init) as pool:
        for msg in pool.imap_unordered(run, jobs):
            print(time.strftime("%H:%M"), msg, flush=True)
