"""Quadrotor in a corridor (lateral-vertical plane), cascaded control.

As in PX4-style autopilots, the position-level controller (here the learned agent)
outputs acceleration setpoints; the attitude and thrust loops underneath realise them.
The lateral acceleration follows its setpoint with a first-order lag (the time the
vehicle needs to tilt); vertical acceleration acts directly (thrust is fast).
Aerodynamic drag is linear plus quadratic; wind gusts act on both accelerations.

    lateral   y+  = y  + dt vy
              vy+ = vy + dt (a_y - c1 vy - c2 vy |vy|) + dt w_y
              ay+ = a_y + dt (a_y,c - a_y) / tau
    vertical  z+  = z  + dt vz
              vz+ = vz + dt (a_z,c - c1 vz - c2 vz |vz|) + dt w_z

State x = (y, vy, a_y, z, vz); input u = (a_y,c, a_z,c), both bounded by a_max (the
tilt and thrust limits). Specification: stay in the corridor |y| <= y_max, between floor
and ceiling, with speed limits.

The channels are decoupled, so each gets its own filter (lateral 3-D, vertical 2-D).

:class:`QuadrotorEnv`: follow a moving target that sometimes leaves the corridor (the
agent is tempted to follow it into the wall). Reward per step 1 - distance^2 (metres),
at least 0; leaving the specification: -50 and the episode ends.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import gymnasium as gym
import numpy as np
from gymnasium import spaces

__all__ = ["QuadrotorParams", "quadrotor_step", "QuadrotorEnv"]


@dataclass(frozen=True)
class QuadrotorParams:
    dt: float = 0.05
    tau: float = 0.15          # lateral acceleration lag (tilt dynamics) [s]
    c1: float = 0.1            # linear drag [1/s]
    c2: float = 0.1            # quadratic drag [1/m]
    ay_max: float = 3.0        # lateral acceleration limit (tilt ~17 deg) [m/s^2]
    az_max: float = 3.0        # vertical acceleration limit [m/s^2]
    wind: float = 0.5          # gust bound on each acceleration [m/s^2]
    y_max: float = 1.0         # corridor half-width [m]
    z_min: float = 0.3         # floor margin [m]
    z_max: float = 2.7         # ceiling [m]
    vy_max: float = 3.0
    vz_max: float = 2.0

    @property
    def lo(self):
        return np.array([-self.y_max, -self.vy_max, -self.ay_max, self.z_min, -self.vz_max])

    @property
    def hi(self):
        return np.array([self.y_max, self.vy_max, self.ay_max, self.z_max, self.vz_max])

    @property
    def u_lo(self):
        return np.array([-self.ay_max, -self.az_max])

    @property
    def u_hi(self):
        return np.array([self.ay_max, self.az_max])

    def _drag_slope(self, vmax):
        # d/dv of -(c1 v + c2 v|v|) lies in [-(c1 + 2 c2 vmax), -c1]
        return -(self.c1 + 2 * self.c2 * vmax), -self.c1

    def lat_bounds(self):
        """Lateral channel (y, vy, a_y), input a_y,c: derivative and disturbance bounds."""
        dt = self.dt
        s_lo, s_hi = self._drag_slope(self.vy_max)
        A_lo = np.array([[0, dt, 0], [0, dt * s_lo, dt], [0, 0, -dt / self.tau]])
        A_hi = np.array([[0, dt, 0], [0, dt * s_hi, dt], [0, 0, -dt / self.tau]])
        B = np.array([[0.0], [0.0], [dt / self.tau]])
        w = dt * self.wind
        return A_lo, A_hi, B, B.copy(), np.array([0, -w, 0]), np.array([0, w, 0])

    def vert_bounds(self):
        """Vertical channel (z, vz), input a_z,c."""
        dt = self.dt
        s_lo, s_hi = self._drag_slope(self.vz_max)
        A_lo = np.array([[0, dt], [0, dt * s_lo]])
        A_hi = np.array([[0, dt], [0, dt * s_hi]])
        B = np.array([[0.0], [dt]])
        w = dt * self.wind
        return A_lo, A_hi, B, B.copy(), np.array([0, -w]), np.array([0, w])


def quadrotor_step(X, U, p: QuadrotorParams, wind=None) -> np.ndarray:
    """Successor states (rows of X = (y, vy, a_y, z, vz), U = (a_y,c, a_z,c)).
    ``wind``: (N, 2) gust accelerations (lateral, vertical) or None."""
    X, U = np.atleast_2d(np.asarray(X, float)), np.atleast_2d(np.asarray(U, float))
    y, vy, ay, z, vz = X.T
    ayc, azc = U.T
    wy, wz = (np.zeros(len(X)), np.zeros(len(X))) if wind is None else np.asarray(wind).T
    dt, c1, c2 = p.dt, p.c1, p.c2
    return np.stack([y + dt * vy,
                     vy + dt * (ay - c1 * vy - c2 * vy * np.abs(vy) + wy),
                     ay + dt * (ayc - ay) / p.tau,
                     z + dt * vz,
                     vz + dt * (azc - c1 * vz - c2 * vz * np.abs(vz) + wz)], axis=1)


class QuadrotorEnv(gym.Env):
    """Target following in a corridor.

    Observation: state (scaled), target position relative to the drone, target velocity.
    Action: (a_y,c, a_z,c) in physical units. The target moves on a smooth random curve
    whose lateral amplitude reaches beyond the walls.
    """

    metadata = {"render_modes": []}

    def __init__(self, params: Optional[QuadrotorParams] = None, horizon: int = 600,
                 target_amp=(1.4, 1.0)):
        self.p = params or QuadrotorParams()
        self.horizon = int(horizon)
        self.target_amp = np.asarray(target_amp, float)
        p = self.p
        self.action_space = spaces.Box(p.u_lo.astype(np.float32), p.u_hi.astype(np.float32))
        self.observation_space = spaces.Box(-10.0, 10.0, shape=(9,), dtype=np.float32)
        self.state = np.zeros(5)
        self.t = 0
        self._freq = np.ones((2, 2))
        self._phase = np.zeros((2, 2))

    def target(self, t=None):
        """Target position and velocity at step t: sum of two sines per axis."""
        t = (self.t if t is None else t) * self.p.dt
        zc = 0.5 * (self.p.z_min + self.p.z_max)
        arg = self._freq * t + self._phase
        pos = 0.5 * self.target_amp * np.sin(arg).sum(1) + np.array([0.0, zc])
        vel = 0.5 * self.target_amp * (self._freq * np.cos(arg)).sum(1)
        return pos, vel

    def _obs(self):
        p, x = self.p, self.state
        tp, tv = self.target()
        rel = tp - x[[0, 3]]
        zc, zh = 0.5 * (p.z_min + p.z_max), 0.5 * (p.z_max - p.z_min)
        return np.array([x[0] / p.y_max, x[1] / p.vy_max, x[2] / p.ay_max, (x[3] - zc) / zh,
                         x[4] / p.vz_max, rel[0], rel[1], tv[0], tv[1]], dtype=np.float32)

    def in_spec(self, x=None) -> bool:
        x = self.state if x is None else x
        return bool(np.all(x >= self.p.lo) and np.all(x <= self.p.hi))

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        options = options or {}
        rng = self.np_random
        self._freq = rng.uniform(0.2, 0.9, (2, 2))
        self._phase = rng.uniform(0, 2 * np.pi, (2, 2))
        if "state" in options:
            self.state = np.asarray(options["state"], float).copy()
        else:
            self.state = np.array([rng.uniform(-0.3, 0.3), rng.uniform(-0.3, 0.3), 0.0,
                                   rng.uniform(1.2, 1.8), rng.uniform(-0.3, 0.3)])
        self.t = 0
        return self._obs(), {"state": self.state.copy()}

    def step(self, action):
        p = self.p
        u = np.clip(np.asarray(action, float).reshape(-1)[:2], p.u_lo, p.u_hi)
        w = self.np_random.uniform(-p.wind, p.wind, 2)
        self.state = quadrotor_step(self.state, u, p, w[None])[0]
        self.t += 1
        violation = not self.in_spec()
        if violation:
            reward = -50.0
        else:
            d2 = float(np.sum((self.target()[0] - self.state[[0, 3]]) ** 2))
            reward = max(0.0, 1.0 - d2)
        truncated = self.t >= self.horizon and not violation
        return self._obs(), reward, violation, truncated, {
            "state": self.state.copy(), "action": u, "violation": violation}
