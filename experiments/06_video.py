"""Two short videos (GIF): an agent early in its training, without and with the filter.

The same young agent (PPO after a few thousand steps, still exploring) runs the same
episode twice, with the same random numbers: once alone, once behind the safety filter.

Run: python experiments/06_video.py   -> figures/lane_keeping.gif, figures/quadrotor.gif
"""

from __future__ import annotations

from pathlib import Path

import imageio.v2 as imageio
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import Polygon, Rectangle  # noqa: E402

from sfilter.controllers.rl import train  # noqa: E402
from sfilter.envs import LaneKeepingParams  # noqa: E402
from sfilter.envs.quadrotor import QuadrotorParams  # noqa: E402
from sfilter.envs.roads import showcase_road  # noqa: E402
from sfilter.evaluation import evaluate  # noqa: E402
from sfilter.filters import build_abstraction_filter  # noqa: E402
from sfilter.filters.quadrotor import build_quadrotor_filter  # noqa: E402

FIG = Path(__file__).resolve().parents[1] / "figures"
INK, TEAL, ORANGE, MUTED, RED = "#0B1220", "#2a9d8f", "#FB923C", "#6B7280", "#C2410C"
PAPER, LINE = "#F7F7F5", "#D4D4CF"
YOUNG_STEPS = 6000
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10})


def young_agent(benchmark):
    """A PPO agent after YOUNG_STEPS steps of training without the filter. It acts with
    its exploration noise switched on, as it does while it learns."""
    model, _ = train("ppo", benchmark, None, YOUNG_STEPS, seed=0)
    rng = np.random.default_rng(0)
    noise = {}

    class Agent:
        # the same noise sequence in both runs, so they differ only by the filter
        def reset(self, env):
            self.t = 0

        def act(self, obs, env):
            mean = model.predict(obs, deterministic=True)[0]
            std = float(np.exp(model.policy.log_std.detach().numpy()).mean())
            if self.t not in noise:
                noise[self.t] = rng.standard_normal(mean.shape)
            self.t += 1
            return np.clip(mean + std * noise[self.t - 1], -1, 1)

    return Agent()


def frames_to_gif(draw, n, path, fig, step=2, hold=15):
    frames = []
    for k in list(range(0, n, step)) + [n - 1] * hold:
        draw(k)
        fig.canvas.draw()
        frames.append(np.asarray(fig.canvas.buffer_rgba())[..., :3].copy())
    imageio.mimsave(path, frames, duration=0.05 * step, loop=0)
    plt.close(fig)
    print("wrote", path, f"{path.stat().st_size / 1e6:.1f} MB")


# ------------------------------------------------------------------ lane keeping
def lane_video():
    p = LaneKeepingParams()
    filt = build_abstraction_filter()
    agent = young_agent("lane")
    ep = [dict(seed=7, options=dict(road=showcase_road(), v=25.0, state=(0.0, 0.0)))]
    runs = [evaluate(agent, "lane", f, record=True, episodes=ep)["traj"][0]
            for f in (None, filt)]
    n = max(len(r["x"]) for r in runs)
    half_lane, car_l, car_w = 1.75, 4.5, 1.8
    fig, axes = plt.subplots(2, 1, figsize=(7.2, 3.3), dpi=100)
    fig.patch.set_facecolor(PAPER)
    fig.subplots_adjust(left=0.03, right=0.97, top=0.9, bottom=0.04, hspace=0.5)
    art = []
    for ax, r, title, col in zip(axes, runs, ("Without the filter", "With the safety filter"),
                                 (ORANGE, TEAL), strict=True):
        s = np.concatenate([[0], np.cumsum(p.dt * r["x"][1:, 2] * np.cos(r["x"][1:, 1]))])
        ax.set_facecolor("#E9EAE6")
        ax.add_patch(Rectangle((-1e4, -half_lane), 2e4, 2 * half_lane, color="#3A4150", lw=0))
        for y in (-half_lane, half_lane):
            ax.axhline(y, color="white", lw=2.5)
        ax.set_ylim(-2.75, 2.75)
        ax.set_aspect("equal")                 # true proportions
        ax.set_xticks([])
        ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_visible(False)
        ax.set_title(title, loc="left", color=INK, fontsize=11, fontweight="bold")
        marks = [ax.plot([], [], color="#9AA3B2", lw=2)[0] for _ in range(6)]
        car = Polygon(np.zeros((4, 2)), closed=True, color=col, lw=0, zorder=3)
        glass = Polygon(np.zeros((4, 2)), closed=True, color=INK, lw=0, zorder=4)
        ax.add_patch(car)
        ax.add_patch(glass)
        note = ax.text(0.99, 1.04, "", transform=ax.transAxes, ha="right", va="bottom",
                       color=INK, fontsize=10)
        art.append((ax, r, s, marks, car, glass, note))

    def draw(k):
        for ax, r, s, marks, car, glass, note in art:
            last = len(r["x"]) - 1
            i = min(k, last)
            ey, ep_, sk = r["x"][i, 0], r["x"][i, 1], s[i]
            ax.set_xlim(sk - 10, sk + 30)
            start = 10 * np.floor((sk - 10) / 10)
            for j, mk in enumerate(marks):          # dashed centre marks every 10 m
                mk.set_data([start + 10 * j, start + 10 * j + 4], [-half_lane - 0.55] * 2)
            c, sn = np.cos(ep_), np.sin(ep_)
            body = np.array([[-car_l / 2, -car_w / 2], [car_l / 2, -car_w / 2],
                             [car_l / 2, car_w / 2], [-car_l / 2, car_w / 2]])
            rot = np.array([[c, sn], [-sn, c]])
            car.set_xy(body @ rot + [sk, ey])
            front = np.array([[0.5, -0.7], [1.3, -0.7], [1.3, 0.7], [0.5, 0.7]])
            glass.set_xy(front @ rot + [sk, ey])
            j = min(i, len(r["applied"]) - 1)
            changed = abs(r["applied"][j, 0] - r["proposed"][j, 0]) > 1e-12
            if k > last and last < 600:
                note.set_text(f"left the lane after {last * p.dt:.1f} s")
                note.set_color(RED)
            elif changed:
                note.set_text("filter corrects the steering")
                note.set_color(INK)
            else:
                note.set_text(f"{i * p.dt:4.1f} s")
                note.set_color(MUTED)

    frames_to_gif(draw, n, FIG / "lane_keeping.gif", fig)
    return runs


# --------------------------------------------------------------------- quadrotor
def quad_video():
    p = QuadrotorParams()
    filt = build_quadrotor_filter()
    agent = young_agent("quad")
    ep = [dict(seed=11, options=dict(state=(0.0, 0.0, 0.0, 1.5, 0.0)))]
    runs = [evaluate(agent, "quad", f, record=True, episodes=ep)["traj"][0]
            for f in (None, filt)]
    n = max(len(r["x"]) for r in runs)
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 4.0), dpi=100)
    fig.patch.set_facecolor(PAPER)
    fig.subplots_adjust(left=0.03, right=0.97, top=0.88, bottom=0.08, wspace=0.08)
    art = []
    for ax, r, title, col in zip(axes, runs, ("Without the filter", "With the safety filter"),
                                 (ORANGE, TEAL), strict=True):
        ax.set_facecolor("#3A4150")
        ax.add_patch(Rectangle((-p.y_max, p.z_min), 2 * p.y_max, p.z_max - p.z_min,
                               color="white", lw=0))
        ax.set_xlim(-1.7, 1.7)
        ax.set_ylim(0.0, 3.0)
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_visible(False)
        ax.set_title(title, loc="left", color=INK, fontsize=11, fontweight="bold")
        trail, = ax.plot([], [], color=col, lw=1.5, alpha=0.6)
        body, = ax.plot([], [], color=col, lw=5, solid_capstyle="round", zorder=4)
        tgt, = ax.plot([], [], "o", mfc="none", mec=INK, mew=1.8, ms=11, zorder=3)
        note = ax.text(0.5, -0.06, "", transform=ax.transAxes, ha="center", va="top",
                       color=INK, fontsize=10)
        art.append((r, trail, body, tgt, note))

    def draw(k):
        for r, trail, body, tgt, note in art:
            last = len(r["x"]) - 1
            i = min(k, last)
            y, z, ay = r["x"][i, 0], r["x"][i, 3], r["x"][i, 2]
            th = np.arctan2(ay, 9.81)                 # tilt that gives this acceleration
            dx, dz = 0.16 * np.cos(th), -0.16 * np.sin(th)
            body.set_data([y - dx, y + dx], [z - dz, z + dz])
            trail.set_data(r["x"][max(0, i - 40):i + 1, 0], r["x"][max(0, i - 40):i + 1, 3])
            j = min(i, len(r["target"]) - 1)
            tgt.set_data([r["target"][j, 0]], [r["target"][j, 1]])
            changed = np.abs(r["applied"][j] - r["proposed"][j]).max() > 1e-12
            if k > last and last < 600:
                note.set_text(f"left the corridor after {last * p.dt:.1f} s")
                note.set_color(RED)
            elif changed:
                note.set_text("filter corrects the command")
                note.set_color(INK)
            else:
                note.set_text(f"{i * p.dt:4.1f} s   (circle: target)")
                note.set_color(MUTED)

    frames_to_gif(draw, n, FIG / "quadrotor.gif", fig)
    return runs


if __name__ == "__main__":
    FIG.mkdir(exist_ok=True)
    for name, r in (("lane", lane_video()), ("quad", quad_video())):
        print(name, "steps without / with the filter:", len(r[0]["x"]) - 1, len(r[1]["x"]) - 1)
