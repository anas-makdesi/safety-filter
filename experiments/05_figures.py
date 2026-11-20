"""Figures from results/evaluation.json and results/rl_curves.json.

Run: python experiments/05_figures.py   -> figures/training_curves.png,
figures/reliance.png, figures/hero.png
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
# one colour per training condition, the same in every figure
COND = {"none": ("#FB923C", "no filter"),
        "filter": ("#2a9d8f", "safety filter"),
        "penalty": ("#5B5FC7", "filter + penalty"),
        "penalty5": ("#A78BFA", "filter + 5x penalty"),
        "stop": ("#0B1220", "filter + correction ends episode")}
BENCH = {"lane": "Lane keeping", "quad": "Quadrotor"}
ALGO = {"sac": "SAC", "ppo": "PPO"}
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.spines.top": False,
                     "axes.spines.right": False, "axes.edgecolor": "#9CA3AF",
                     "axes.labelcolor": MUTED, "xtick.color": MUTED, "ytick.color": MUTED,
                     "axes.titlecolor": INK})


def _band(ax, x, rows, color, label_text):
    """Mean over seeds (line) and the range over seeds (band)."""
    a = np.array([[np.nan if v is None else v for v in r] for r in rows], float)
    mean = np.nanmean(a, 0)
    ax.fill_between(x, np.nanmin(a, 0), np.nanmax(a, 0), color=color, alpha=0.18, lw=0)
    ax.plot(x, mean, color=color, lw=2)
    return mean


def training_curves(curves):
    fig, axes = plt.subplots(2, 4, figsize=(14, 6.2))
    for i, b in enumerate(BENCH):
        for j, algo in enumerate(ALGO):
            ax_v, ax_r = axes[i, 2 * j], axes[i, 2 * j + 1]
            for cond, (color, _name) in COND.items():
                c = curves.get(f"{b}/{algo}/{cond}")
                if c is None:
                    continue
                x = np.array(c["step"]) / 1000
                v = _band(ax_v, x, c["cum_violations"], color, None)
                _band(ax_r, x, c["ret"], color, None)
                ax_v.annotate(f"{v[-1]:.0f}", (x[-1], v[-1]), xytext=(4, 0),
                              textcoords="offset points", va="center", color=INK, fontsize=9)
            ax_v.set_title(f"{BENCH[b]}, {ALGO[algo]}: crashes so far", loc="left", fontsize=10.5)
            ax_r.set_title(f"{BENCH[b]}, {ALGO[algo]}: return per episode", loc="left",
                           fontsize=10.5)
            ax_r.set_ylim(-60, 620)
            for ax in (ax_v, ax_r):
                ax.grid(axis="y", color=GRID, lw=0.8)
                ax.set_axisbelow(True)
                ax.margins(x=0.02)
                if i == 1:
                    ax.set_xlabel("training steps (thousands)")
    present = [c for c in COND if any(k.endswith("/" + c) for k in curves)]
    handles = [plt.Line2D([], [], color=COND[c][0], lw=2.5) for c in present]
    fig.legend(handles, [COND[c][1] for c in present], loc="lower center", ncol=len(present),
               frameon=False, fontsize=10)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(FIG / "training_curves.png", dpi=150)
    plt.close(fig)


def reliance(rl):
    """Crashes of the trained agents on the held-out episodes when no filter is present."""
    fig, ax = plt.subplots(figsize=(10, 4.2))
    groups = [(b, a) for b in BENCH for a in ALGO]
    conds = [c for c in COND if any(f"{b}/{a}/{c}" in rl for b, a in groups)]
    w = 0.8 / len(conds)
    for k, cond in enumerate(conds):
        color, name = COND[cond]
        xs, vals = [], []
        for g, (b, a) in enumerate(groups):
            r = rl.get(f"{b}/{a}/{cond}")
            if r is None:
                continue
            n = r["eval_crashes"] if cond == "none" else r["swap_crashes"]
            xs.append(g + (k - (len(conds) - 1) / 2) * w)
            vals.append(100.0 * n / r["eval_episodes"])
        ax.bar(xs, vals, w * 0.92, color=color, label=f"trained with: {name}")
        for xi, v in zip(xs, vals, strict=True):
            ax.text(xi, v + 1.5, f"{v:.0f}", ha="center", va="bottom", color=INK, fontsize=8.5)
    ax.set_xticks(np.arange(len(groups)), [f"{BENCH[b]}\n{ALGO[a]}" for b, a in groups])
    ax.set_ylabel("episodes with a crash (%)")
    ax.set_ylim(0, 110)
    ax.set_title("Trained agents run without any filter", loc="left", fontsize=11)
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.2),
              fontsize=9)
    fig.tight_layout()
    fig.savefig(FIG / "reliance.png", dpi=150)
    plt.close(fig)


def hero(curves):
    """Wide figure for the website: crashes during training, the four agent/benchmark pairs."""
    fig, axes = plt.subplots(1, 4, figsize=(14, 4.6))
    k = 0
    for b in BENCH:
        for algo in ALGO:
            ax = axes[k]
            k += 1
            for cond in ("none", "filter"):
                c = curves.get(f"{b}/{algo}/{cond}")
                if c is None:
                    continue
                color, name = COND[cond]
                x = np.array(c["step"]) / 1000
                v = _band(ax, x, c["cum_violations"], color, None)
                ax.annotate(f"{v[-1]:.0f}", (x[-1], v[-1]), xytext=(4, 0),
                            textcoords="offset points", va="center", color=INK, fontsize=11)
            ax.set_title(f"{BENCH[b]}, {ALGO[algo]}", loc="left", fontsize=13)
            ax.set_xlabel("training steps (thousands)")
            ax.grid(axis="y", color=GRID, lw=0.8)
            ax.set_axisbelow(True)
            ax.margins(x=0.03)
    axes[0].set_ylabel("crashes so far")
    handles = [plt.Line2D([], [], color=COND[c][0], lw=2.5) for c in ("none", "filter")]
    fig.legend(handles, ["learning without a filter", "learning behind the safety filter"],
               loc="lower center", ncol=2, frameon=False, fontsize=12)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.savefig(FIG / "hero.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    FIG.mkdir(exist_ok=True)
    res = json.loads((ROOT / "results" / "evaluation.json").read_text())
    curves = json.loads((ROOT / "results" / "rl_curves.json").read_text())
    training_curves(curves)
    reliance(res["rl"])
    hero(curves)
    print("wrote figures to", FIG)
