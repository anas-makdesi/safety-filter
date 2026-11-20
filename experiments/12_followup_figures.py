"""Figures and tables for the follow-ups: stress test and real-car lane keeping.

Run: python experiments/12_followup_figures.py
  -> figures/stress_test.png, figures/real_lane.png, and the README tables on stdout
(needs results/stress_test.json and results/real_lane.json)
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "figures"
INK, MUTED, GRID = "#0B1220", "#6B7280", "#E4E4E0"
C = {"none": "#FB923C", "nominal": "#2a9d8f", "aware": "#5B5FC7", "robust": "#5B5FC7"}
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.spines.top": False,
                     "axes.spines.right": False, "axes.edgecolor": "#9CA3AF",
                     "axes.labelcolor": MUTED, "xtick.color": MUTED, "ytick.color": MUTED,
                     "axes.titlecolor": INK})
CRUDE = {"lane": ("random", "bang-bang", "full-lock"),
         "quad": ("random", "chase", "full-thrust")}


def crude_crashes(res, bench, filt, scen):
    """Crashes of the three crude agents together (of 60 episodes), or None."""
    d = res[bench].get(filt, {}).get(scen)
    if d is None:
        return None
    return sum(d[a]["crashes"] for a in CRUDE[bench])


def stress_figure(res):
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.4), gridspec_kw=dict(width_ratios=[10, 9]))
    for ax, bench, title in ((axes[0], "lane", "Lane keeping"), (axes[1], "quad", "Quadrotor")):
        scen = [s for s in res[bench]["nominal"]]
        other = "robust" if bench == "quad" else "aware"
        series = [("nominal", "filter as built")]
        if bench == "quad":
            series.append(("robust", "filter built for 2x the gusts"))
        series.append(("aware", "filter told the noise / delay"))
        w = 0.8 / len(series)
        for k, (f, lab) in enumerate(series):
            vals = [crude_crashes(res, bench, f, s) for s in scen]
            x = np.arange(len(scen)) + (k - (len(series) - 1) / 2) * w
            col = C["nominal"] if f == "nominal" else ("#5B5FC7" if f == "aware" else "#A78BFA")
            ax.bar([xi for xi, v in zip(x, vals, strict=True) if v is not None],
                   [v for v in vals if v is not None], w * 0.92, color=col, label=lab)
            for xi, v in zip(x, vals, strict=True):
                if v is not None:
                    ax.text(xi, v + 1, f"{v}", ha="center", va="bottom", fontsize=8, color=INK)
        del other
        short = {"steering 1 step late": "late\nsteering:\n1 step",
                 "steering 2 steps late": "late\nsteering:\n2 steps"}
        ax.set_xticks(np.arange(len(scen)),
                      [short.get(s, s.replace(" ", "\n", 1)) for s in scen], fontsize=8.5)
        ax.set_ylim(0, 66)
        ax.set_ylabel("crashes of 60 episodes")
        ax.set_title(f"{title}: crude agents behind the filter, assumptions broken",
                     loc="left", fontsize=11)
        ax.grid(axis="y", color=GRID, lw=0.8)
        ax.set_axisbelow(True)
        ax.legend(frameon=False, fontsize=9, loc="upper left")
    fig.tight_layout()
    fig.savefig(FIG / "stress_test.png", dpi=150)
    plt.close(fig)


def real_figure(real):
    from sfilter.real.lane import RealLaneParams, build_real_lane_filter

    d = np.load(ROOT / "data" / "comma2k19_rav4_steps.npz")
    P = RealLaneParams()
    v, s, r = (d[f"learn_{k}"].astype(float) for k in ("v", "sigma", "r"))
    m = (v >= P.v_min) & (v <= P.v_max) & (np.abs(s) <= P.sigma_max)
    f, _ = build_real_lane_filter(v[m], s[m], r[m], P)
    ab = f.controller.abstraction
    X, U, model = ab.X, ab.U, f.model
    vq = 24.0
    sig = np.linspace(-P.sigma_max + 1e-6, P.sigma_max - 1e-6, 400)
    x = np.tile([0.0, 0.0, vq], (len(sig), 1))
    qi, ui = X.locate(x), U.locate(sig[:, None])
    xl, xh = X.cell_bounds(qi)
    ul, uh = U.cell_bounds(ui)
    lo, hi = model.successor_bounds(np.hstack([qi, ui]), xl, xh, ul, uh)
    curv = P.dt * P.v_max * P.kappa_max
    lo_r = np.rad2deg((lo[:, 1] - xl[:, 1] + curv) / P.dt)
    hi_r = np.rad2deg((hi[:, 1] - xh[:, 1] - curv) / P.dt)
    vt, st, rt = (d[f"test_{k}"].astype(float) for k in ("v", "sigma", "r"))
    mt = (np.abs(vt - vq) <= 0.5) & (np.abs(st) <= P.sigma_max)

    fig, axes = plt.subplots(1, 2, figsize=(14, 4.6), gridspec_kw=dict(width_ratios=[6, 5]))
    ax = axes[0]
    ax.fill_between(sig, lo_r, hi_r, color="#2a9d8f", alpha=0.25, lw=0,
                    label="what the learned model guarantees")
    ax.scatter(st[mt], np.rad2deg(rt[mt]), s=2, color=INK, alpha=0.35,
               label="held-out real steps (other days)")
    lim = real["data"]["learn_sigma_pct"]
    ax.axvspan(-P.sigma_max, lim[0], color="#F3F4F6", zorder=0)
    ax.axvspan(lim[1], P.sigma_max, color="#F3F4F6", zorder=0)
    ax.text(lim[1] + 0.5, ax.get_ylim()[0] * 0.85, "beyond 99 % of\nthe logs: slope\nbounds only",
            fontsize=8.5, color=MUTED)
    ax.set_xlabel("steering-wheel angle (deg)")
    ax.set_ylabel("yaw rate (deg/s)")
    ax.set_title(f"Toyota RAV4 at {vq:.0f} ± 0.5 m/s: steering response", loc="left",
                 fontsize=11)
    ax.legend(frameon=False, fontsize=9, loc="upper left", markerscale=4)
    ax.grid(color=GRID, lw=0.8)
    ax.set_axisbelow(True)

    ax = axes[1]
    cl = real["closed_loop"]
    agents = list(cl["none"])
    series = (("none", "no filter", C["none"]), ("filter", "filter, lag ignored", "#9CA3AF"),
              ("delay-aware filter", "filter, lag accounted for", C["nominal"]))
    w = 0.26
    for k, (key, lab, col) in enumerate(series):
        vals = [cl[key][a]["crashes"] for a in agents]
        xs = np.arange(len(agents)) + (k - 1) * w
        ax.bar(xs, vals, w * 0.92, color=col, label=lab)
        for xi, val in zip(xs, vals, strict=True):
            ax.text(xi, val + 0.3, f"{val}", ha="center", va="bottom", fontsize=9, color=INK)
    ax.set_xticks(np.arange(len(agents)), agents)
    ax.set_ylim(0, 23)
    ax.set_ylabel("lane departures of 20 drives")
    ax.set_title("Closed loop on a car fitted to the held-out logs", loc="left", fontsize=11)
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, fontsize=9, loc="upper right")
    fig.tight_layout()
    fig.savefig(FIG / "real_lane.png", dpi=150)
    plt.close(fig)


def tables(res, real):
    print("### Stress test: crashes of the three crude agents together (of 60), MPC (of 20)\n")
    for bench in ("lane", "quad"):
        cols = ["nominal"] + (["robust"] if bench == "quad" else []) + ["aware"]
        print(f"**{bench}**\n")
        print("| Scenario | No filter | " + " | ".join(cols) + " | MPC alone | MPC + filter |"
              " Monitor alarm before every crash |")
        print("|---|" + "---:|" * (len(cols) + 4))
        for s, d in res[bench]["nominal"].items():
            vals = [crude_crashes(res, bench, c, s) for c in cols]
            none = crude_crashes(res, bench, "none", s)
            first = all(d[a]["crashes_alarmed_first"] == d[a]["crashes"] for a in d)
            fa = sum(d[a]["alarm_episodes"] for a in d)
            print(f"| {s} | {none} | " + " | ".join("–" if v is None else str(v) for v in vals)
                  + f" | {res[bench]['none'][s]['mpc']['crashes']} | {d['mpc']['crashes']} | "
                  f"{'yes' if first else 'no'} (alarms in {fa} of 80) |")
        print()
    print("### Real data\n")
    print(json.dumps({k: real[k] for k in ("data", "check_learn", "check_test", "twin",
                                           "heldout")}, indent=1))
    for k, v in real["closed_loop"].items():
        print(k, {a: (x["crashes"], round(x["ret"]), round(100 * x["filtered_share"], 1))
                  for a, x in v.items()})


if __name__ == "__main__":
    res = json.loads((ROOT / "results" / "stress_test.json").read_text())
    real = json.loads((ROOT / "results" / "real_lane.json").read_text())
    stress_figure(res)
    real_figure(real)
    tables(res, real)
