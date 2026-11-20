"""Print the README's result tables (Markdown) from results/evaluation.json."""

from __future__ import annotations

import json
from pathlib import Path

R = json.loads((Path(__file__).resolve().parents[1] / "results" / "evaluation.json").read_text())
RL, C, L = R["rl"], R["controllers"], R["latency"]
BENCH = {"lane": "Lane keeping", "quad": "Quadrotor"}
ALGO = {"sac": "SAC", "ppo": "PPO"}


def rng(d, fmt="{:.0f}"):
    v = d["values"]
    return f"{fmt.format(d['mean'])} ({fmt.format(min(v))}–{fmt.format(max(v))})"


print("### Crashes while learning (mean over 5 seeds, range in brackets)\n")
print("| Benchmark | Agent | Training steps | No filter | Safety filter | Filter + penalty |")
print("|---|---|---:|---:|---:|---:|")
for b in BENCH:
    for a in ALGO:
        r = [RL[f"{b}/{a}/{c}"] for c in ("none", "filter", "penalty")]
        print(f"| {BENCH[b]} | {ALGO[a]} | {r[0]['steps']:,} | {rng(r[0]['train_violations'])} | "
              f"{r[1]['train_violations']['mean']:.0f} | {r[2]['train_violations']['mean']:.0f} |")

print("\n### After training: 20 held-out episodes of 30 s (return: 600 = perfect)\n")
print("| Benchmark | Controller | Alone: crashes | Alone: return | With filter: crashes | "
      "With filter: return | Steps the filter changed |")
print("|---|---|---:|---:|---:|---:|---:|")
for b in BENCH:
    for a in ALGO:
        n, f = RL[f"{b}/{a}/none"], RL[f"{b}/{a}/filter"]
        print(f"| {BENCH[b]} | {ALGO[a]} (5 seeds) | {n['eval_crashes']} / {n['eval_episodes']} | "
              f"{n['eval_return']['mean']:.0f} | {f['eval_crashes']} / {f['eval_episodes']} | "
              f"{f['eval_return']['mean']:.0f} | {100 * f['eval_filtered_share']['mean']:.0f} % |")
    for name, lab in (("deepc", "DeePC"), ("mpc", "MPC, true model")):
        n, f = C[f"{b}/{name}/free"], C[f"{b}/{name}/filtered"]
        print(f"| {BENCH[b]} | {lab} | {n['crashes']} / {n['episodes']} | {n['ret']:.1f} | "
              f"{f['crashes']} / {f['episodes']} | {f['ret']:.1f} | "
              f"{100 * f['filtered_share']:.1f} % |")

print("\n### Does the agent come to rely on the filter? (100 held-out episodes per cell)\n")
print("| Benchmark | Agent | Trained without filter, run without | ... run with the filter | "
      "Trained behind filter, filter removed | Trained with penalty, filter removed |")
print("|---|---|---:|---:|---:|---:|")
for b in BENCH:
    for a in ALGO:
        n, f, p = (RL[f"{b}/{a}/{c}"] for c in ("none", "filter", "penalty"))
        print(f"| {BENCH[b]} | {ALGO[a]} | {n['eval_crashes']} crashes | {n['swap_crashes']} | "
              f"{f['swap_crashes']} | {p['swap_crashes']} |")

new = [(b, a) for b in BENCH for a in ALGO if f"{b}/{a}/stop" in RL]
if new:
    print("\n### Against the reliance (follow-up): two stronger training signals "
          "(100 held-out episodes per cell)\n")
    print("| Benchmark | Agent | Condition | Return behind filter | Steps the filter changed | "
          "Filter removed: crashes | Filter removed: return | Episodes stopped in training |")
    print("|---|---|---|---:|---:|---:|---:|---:|")
    for b, a in new:
        for c, lab in (("filter", "filter"), ("penalty", "filter + penalty"),
                       ("penalty5", "filter + 5x penalty"),
                       ("stop", "filter + correction ends episode")):
            r = RL[f"{b}/{a}/{c}"]
            print(f"| {BENCH[b]} | {ALGO[a]} | {lab} | {r['eval_return']['mean']:.0f} | "
                  f"{100 * r['eval_filtered_share']['mean']:.1f} % | {r['swap_crashes']} | "
                  f"{r['swap_return']['mean']:.0f} | "
                  f"{r.get('train_stops', {'mean': 0})['mean']:.0f} |")

print("\n### Return and filter activity of the three training conditions (as trained)\n")
print("| Benchmark | Agent | Condition | Return | Steps the filter changed | "
      "Mean change (% of input range) |")
print("|---|---|---|---:|---:|---:|")
for b in BENCH:
    for a in ALGO:
        for c, lab in (("none", "no filter"), ("filter", "filter"),
                       ("penalty", "filter + penalty")):
            r = RL[f"{b}/{a}/{c}"]
            ret, sh = r["eval_return"], 100 * r["eval_filtered_share"]["mean"]
            print(f"| {BENCH[b]} | {ALGO[a]} | {lab} | {ret['mean']:.0f} ± {ret['std']:.0f} | "
                  f"{sh:.1f} % | {r['eval_correction_pct']['mean']:.2f} |")

print("\n### Time per step\n")
for k, v in L.items():
    print(f"- {k}: median {v['median']:.0f} µs, p99 {v['p99']:.0f} µs")
for k, v in C.items():
    print(f"- {k}: controller {v['ctrl_ms']:.1f} ms, filter {v['filter_us']:.0f} µs")
for k, v in RL.items():
    if not k.endswith("none"):
        print(f"- training {k}: filter {v['filter_us']['mean']:.0f} µs/call, "
              f"{100 * v['filter_time_share']['mean']:.0f} % of training time, "
              f"{v['wall_min']['mean']:.1f} min/run")
