"""Lane keeping: a kinematic bicycle in road-aligned (Frenet) coordinates.

State x = (e_y, e_psi, v): lateral offset from the lane centre [m], heading error
[rad], speed [m/s] (constant during an episode). Input: front-wheel steering angle
delta [rad]. Disturbance: road curvature kappa [1/m], bounded by kappa_max.

    e_y+   = e_y   + dt * v * sin(e_psi)                          + w_y
    e_psi+ = e_psi + dt * v * (tan(delta) / L - kappa)            + w_psi
    v+     = v

(small-curvature form of the kinematic bicycle in road coordinates; w = small process
noise). For |e_psi| < pi/2 the map is increasing in e_y, e_psi and delta and decreasing
in kappa; it is not monotone in v. The safety filters only see sampled transitions of
this map plus the assumptions in :class:`LaneKeepingParams` (bounds on curvature, noise
and on the partial derivatives), never the equations.

Safety specification: |e_y| <= e_y_max (the car stays in its lane) and
|e_psi| <= e_psi_max.

:class:`LaneKeepingEnv` is the Gymnasium task: drive along a random road with bounded
curvature, as close to the lane centre and as smoothly as possible. Leaving the
specification ends the episode with a large penalty (crashing never pays: the per-step
reward is non-negative).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from .roads import Road, random_road

__all__ = ["LaneKeepingParams", "lane_keeping_step", "LaneKeepingEnv"]


@dataclass(frozen=True)
class LaneKeepingParams:
    dt: float = 0.05            # sampling time [s]
    wheelbase: float = 2.7      # L [m]
    delta_max: float = 0.05     # steering limit [rad] (road-wheel angle)
    kappa_max: float = 0.004    # road curvature bound [1/m] (radius >= 250 m)
    v_min: float = 15.0         # speed range [m/s]
    v_max: float = 30.0
    e_y_max: float = 0.85       # 3.5 m lane, 1.8 m wide car
    e_psi_max: float = 0.2      # heading error bound [rad]
    noise_y: float = 0.002      # process noise bounds (uniform) [m], [rad]
    noise_psi: float = 0.0005

    @property
    def safe_lo(self) -> np.ndarray:
        return np.array([-self.e_y_max, -self.e_psi_max, self.v_min])

    @property
    def safe_hi(self) -> np.ndarray:
        return np.array([self.e_y_max, self.e_psi_max, self.v_max])

    def derivative_bounds(self):
        """Bounds on the partial derivatives of the unknown increments f(x, u), where
        x+ = x + f(x, u) + w. Rows: (e_y, e_psi, v); columns of A: (e_y, e_psi, v);
        column of B: delta. Valid on the safe box and the input range; they are the
        only model knowledge the abstraction filter uses."""
        dt, L, dm = self.dt, self.wheelbase, self.delta_max
        sp = np.sin(self.e_psi_max)
        A_lo = np.array([[0.0, dt * self.v_min * np.cos(self.e_psi_max), -dt * sp],
                         [0.0, 0.0, -dt * np.tan(dm) / L],
                         [0.0, 0.0, 0.0]])
        A_hi = np.array([[0.0, dt * self.v_max, dt * sp],
                         [0.0, 0.0, dt * np.tan(dm) / L],
                         [0.0, 0.0, 0.0]])
        B_lo = np.array([[0.0], [dt * self.v_min / L], [0.0]])
        B_hi = np.array([[0.0], [dt * self.v_max / (L * np.cos(dm) ** 2)], [0.0]])
        return A_lo, A_hi, B_lo, B_hi

    def disturbance_bounds(self):
        """Bounds on the additive disturbance per state coordinate (curvature + noise)."""
        k = self.dt * self.v_max * self.kappa_max
        d = np.array([self.noise_y, k + self.noise_psi, 0.0])
        return -d, d


def lane_keeping_step(X, delta, kappa, p: LaneKeepingParams, noise=None) -> np.ndarray:
    """Successor states (vectorized over rows of X). ``noise``: (N, 2) or None."""
    X = np.atleast_2d(np.asarray(X, float))
    delta = np.asarray(delta, float).reshape(-1)
    kappa = np.asarray(kappa, float).reshape(-1)
    ey, ep, v = X[:, 0], X[:, 1], X[:, 2]
    out = np.empty_like(X)
    out[:, 0] = ey + p.dt * v * np.sin(ep)
    out[:, 1] = ep + p.dt * v * (np.tan(delta) / p.wheelbase - kappa)
    out[:, 2] = v
    if noise is not None:
        out[:, :2] += noise
    return out


class LaneKeepingEnv(gym.Env):
    """Lane centring on a random road with bounded curvature.

    Observation (scaled to about [-1, 1]): e_y, e_psi, v, curvature now and 0.5 s / 1 s
    ahead (a camera preview: the agent may use it, the filters do not), last steering.
    Action: steering angle delta in [-delta_max, delta_max] (physical units; wrap with
    ``gymnasium.wrappers.RescaleAction`` for agents that act in [-1, 1]).

    Reward per step: 1 - (e_y / e_y_max)^2 - 0.1 (e_psi / e_psi_max)^2
    - 0.5 (delta change / delta_max)^2, at least 0. Leaving the specification: -50 and
    the episode ends (``info["violation"] = True``).

    ``options`` of :meth:`reset`: ``road`` (a :class:`Road`), ``v`` (speed),
    ``state`` (initial (e_y, e_psi), speed taken from ``v`` or drawn).
    """

    metadata = {"render_modes": []}

    def __init__(self, params: Optional[LaneKeepingParams] = None, horizon: int = 600,
                 init_e_y: float = 0.3, init_e_psi: float = 0.05, road_length: float = 1500.0):
        self.p = params or LaneKeepingParams()
        self.horizon = int(horizon)
        self.init_e_y, self.init_e_psi = init_e_y, init_e_psi
        self.road_length = road_length
        dm = self.p.delta_max
        self.action_space = spaces.Box(-dm, dm, shape=(1,), dtype=np.float32)
        self.observation_space = spaces.Box(-5.0, 5.0, shape=(7,), dtype=np.float32)
        self.state = np.zeros(3)
        self.road: Optional[Road] = None
        self.s = 0.0
        self.t = 0
        self.last_delta = 0.0

    # --------------------------------------------------------------------- helpers
    def kappa(self, ahead: float = 0.0) -> float:
        return float(self.road.curvature(self.s + ahead))

    def _obs(self):
        p, (ey, ep, v) = self.p, self.state
        km = p.kappa_max
        return np.array([ey / p.e_y_max, ep / p.e_psi_max,
                         (v - 0.5 * (p.v_min + p.v_max)) / (0.5 * (p.v_max - p.v_min)),
                         self.kappa() / km, self.kappa(0.5 * v) / km, self.kappa(v) / km,
                         self.last_delta / p.delta_max], dtype=np.float32)

    def in_spec(self, x=None) -> bool:
        x = self.state if x is None else x
        return bool(abs(x[0]) <= self.p.e_y_max and abs(x[1]) <= self.p.e_psi_max)

    # ------------------------------------------------------------------------ API
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        options = options or {}
        p = self.p
        self.road = options.get("road") or random_road(self.np_random, self.road_length,
                                                       p.kappa_max)
        v = options.get("v", self.np_random.uniform(p.v_min, p.v_max))
        if "state" in options:
            ey, ep = options["state"][:2]
        else:
            ey = self.np_random.uniform(-self.init_e_y, self.init_e_y)
            ep = self.np_random.uniform(-self.init_e_psi, self.init_e_psi)
        self.state = np.array([ey, ep, v], float)
        self.s, self.t, self.last_delta = 0.0, 0, 0.0
        return self._obs(), {"state": self.state.copy()}

    def step(self, action):
        p = self.p
        delta = float(np.clip(np.asarray(action, float).reshape(-1)[0], -p.delta_max,
                              p.delta_max))
        kappa = self.kappa()
        w = self.np_random.uniform([-p.noise_y, -p.noise_psi], [p.noise_y, p.noise_psi])
        self.state = lane_keeping_step(self.state, delta, kappa, p, w[None])[0]
        self.s += p.dt * self.state[2] * np.cos(self.state[1])
        self.t += 1
        ddelta, self.last_delta = delta - self.last_delta, delta
        violation = not self.in_spec()
        if violation:
            reward = -50.0
        else:
            ey, ep = self.state[0] / p.e_y_max, self.state[1] / p.e_psi_max
            reward = max(0.0, 1.0 - ey**2 - 0.1 * ep**2 - 0.5 * (ddelta / p.delta_max) ** 2)
        truncated = self.t >= self.horizon and not violation
        info = {"state": self.state.copy(), "delta": delta, "kappa": kappa,
                "violation": violation}
        return self._obs(), float(reward), violation, truncated, info
