"""DeePC as a practitioner might set it up, with and without the safety filter.

The default DeePC of 04_evaluate.py never left the lane or hit a wall, so the filter
had nothing to do. DeePC's safety rests on its data and its tuning, and nothing
guarantees it. Here a few common choices are varied (fixed before running; every variant
is reported):

  default        the settings of 04_evaluate.py
  short data     4 s (lane) / 6 s (quadrotor) of recorded data instead of 15 / 20 s
  noisy data     measurement noise of 5 % of the limits in the recorded outputs
  stiff limits   output-limit slack weight 1e4 instead of 1e2
  no back-off    quadrotor only: walls without the 10 cm margin

Each variant runs on the 20 held-out episodes alone and behind the filter.

Run: python experiments/13_deepc_variants.py [--bench lane|quad]
Writes runs/deepc/<bench>__<variant>__<free|filtered>.json and results/deepc_variants.json.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from sfilter.controllers.deepc import LaneDeePC, QuadDeePC
from sfilter.envs import LaneKeepingParams
from sfilter.envs.quadrotor import QuadrotorParams
from sfilter.evaluation import evaluate, summarize
from sfilter.filters import build_abstraction_filter
from sfilter.filters.quadrotor import build_quadrotor_filter

ROOT = Path(__file__).resolve().parents[1]
VARIANTS = {
    "lane": {"default": {}, "short data": dict(T=80), "noisy data": dict(data_noise=0.05),
             "stiff limits": dict(lam_y=1e4)},
    "quad": {"default": {}, "short data": dict(T=120), "noisy data": dict(data_noise=0.05),
             "stiff limits": dict(lam_y=1e4), "no back-off": dict(margin=0.0)},
}


def make(bench, kw):
    if bench == "lane":
        return LaneDeePC(LaneKeepingParams(), **kw)
    return QuadDeePC(QuadrotorParams(), **kw)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", choices=("lane", "quad"), nargs="+", default=["lane", "quad"])
    a = ap.parse_args()
    d = ROOT / "runs" / "deepc"
    d.mkdir(parents=True, exist_ok=True)
    out_path = ROOT / "results" / "deepc_variants.json"
    for bench in a.bench:
        filt = build_abstraction_filter() if bench == "lane" else build_quadrotor_filter()
        for name, kw in VARIANTS[bench].items():
            for cond, f in (("free", None), ("filtered", filt)):
                path = d / f"{bench}__{name.replace(' ', '_')}__{cond}.json"
                if path.exists():
                    continue
                t0 = time.perf_counter()
                s = summarize(evaluate(make(bench, kw), bench, f))
                path.write_text(json.dumps(s))
                print(f"{time.strftime('%H:%M')} {bench} {name:13s} {cond:8s} crashes "
                      f"{s['crashes']:2d}/20  return {s['ret']:6.1f}  filter steps "
                      f"{100 * s['filtered_share']:4.1f} %  ({time.perf_counter() - t0:.0f} s)",
                      flush=True)
    res = {}
    for bench, vs in VARIANTS.items():
        for name in vs:
            for cond in ("free", "filtered"):
                path = d / f"{bench}__{name.replace(' ', '_')}__{cond}.json"
                if path.exists():
                    res.setdefault(bench, {}).setdefault(name, {})[cond] = \
                        json.loads(path.read_text())
    out_path.write_text(json.dumps(res, indent=1))
