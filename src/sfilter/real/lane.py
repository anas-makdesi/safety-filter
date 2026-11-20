"""Lane keeping with a real car's steering response (RAV4 logs from comma2k19).

Same lane geometry as :mod:`sfilter.envs.lane_keeping`, but the input is the
steering-wheel angle sigma [deg] as logged on the CAN bus, and the heading change per step
comes from the real car:

    e_y+   = e_y   + dt * v * sin(e_psi)                       (exact kinematics)
    e_psi+ = e_psi + f(v, sigma)  - dt * v * kappa  + w        (f: the real car)
    v+     = v

f is never written down. The filter learns it from the logged triples (v, sigma,
dt * yaw rate), with stated assumptions:

* bounds on df/dsigma and df/dv, from a family of steady-state cornering models
  r = v (sigma - sigma0) / (G (1 + K v^2)) with G in [G_lo, G_hi], K in [K_lo, K_hi]
  (wide ranges around the fit; checked against the logs with ddma.check_data);
* |w| <= dt * rho: what the steady-state picture misses (road bank, wind, the car's own
  yaw dynamics, sensor noise), with rho from the logs;
* road curvature |kappa| <= kappa_max (motorway curves).

Transitions for the filter: each real step (v, sigma, dpsi) is paired with lane offsets
and heading errors drawn in the safe box and a curvature drawn in its bound; e_y+
follows from the kinematics. So the vehicle-dependent part is real data, the geometry is
exact.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from ..envs.roads import Road, random_road
from ..filters.abstraction import learn_abstraction_filter
from .comma import DT, Twin

__all__ = ["RealLaneParams", "real_transitions", "build_real_lane_filter", "RealLaneEnv"]

FEATURES = [[1, 2], [2, 3], [2]]     # (e_y, e_psi, v, sigma), as in lane_keeping


@dataclass(frozen=True)
class RealLaneParams:
    dt: float = DT
    v_min: float = 20.0           # speed band [m/s] (72-100 km/h)
    v_max: float = 28.0
    sigma_max: float = 25.0       # steering-wheel angle limit used by the filter [deg];
                                  # 99 % of the logged steps in this speed band lie in
                                  # -17...+21 deg; beyond, the slope bounds extrapolate
    kappa_max: float = 0.0005     # road curvature bound [1/m] (radius >= 2 km)
    rho: float = 0.03             # yaw-rate error bound beyond the steady-state picture
    e_y_max: float = 0.85
    e_psi_max: float = 0.1
    noise_y: float = 0.002
    G: tuple = (50.0, 62.0)       # steering ratio x wheelbase [m]
    K: tuple = (0.0, 0.0012)      # understeer factor [s^2/m^2]
    sigma0_max: float = 0.5       # |steering offset| [deg]

    @property
    def safe_lo(self):
        return np.array([-self.e_y_max, -self.e_psi_max, self.v_min])

    @property
    def safe_hi(self):
        return np.array([self.e_y_max, self.e_psi_max, self.v_max])

    def derivative_bounds(self):
        """df/dx, df/du bounds on the safe box and input range (rows e_y, e_psi, v)."""
        dt, c = self.dt, np.pi / 180
        sp = np.sin(self.e_psi_max)
        v = np.linspace(self.v_min, self.v_max, 200)
        gain = v / (1 + self.K[1] * v**2)          # smallest v / (1 + K v^2) at K_hi
        b_lo = dt * c * gain.min() / self.G[1]
        b_hi = dt * c * self.v_max / self.G[0]
        a = dt * c * (self.sigma_max + self.sigma0_max) / self.G[0]   # |d/dv| <= this
        A_lo = np.array([[0.0, dt * self.v_min * np.cos(self.e_psi_max), -dt * sp],
                         [0.0, 0.0, -a], [0.0, 0.0, 0.0]])
        A_hi = np.array([[0.0, dt * self.v_max, dt * sp],
                         [0.0, 0.0, a], [0.0, 0.0, 0.0]])
        B_lo = np.array([[0.0], [b_lo], [0.0]])
        B_hi = np.array([[0.0], [b_hi], [0.0]])
        return A_lo, A_hi, B_lo, B_hi

    def disturbance_bounds(self):
        d = np.array([self.noise_y,
                      self.dt * (self.v_max * self.kappa_max + self.rho), 0.0])
        return -d, d


def real_transitions(v, sigma, r, p: RealLaneParams, rng, copies: int = 8):
    """Filter training data from real steps (v, sigma, yaw rate r): each real step is
    used ``copies`` times with random (e_y, e_psi), curvature and lateral noise."""
    m = (v >= p.v_min) & (v <= p.v_max) & (np.abs(sigma) <= p.sigma_max)
    v, sigma, r = (np.tile(a[m], copies) for a in (v, sigma, r))
    n = len(v)
    ey = rng.uniform(-p.e_y_max, p.e_y_max, n)
    ep = rng.uniform(-p.e_psi_max, p.e_psi_max, n)
    kappa = rng.uniform(-p.kappa_max, p.kappa_max, n)
    X = np.stack([ey, ep, v], 1)
    U = sigma[:, None]
    Xn = np.stack([ey + p.dt * v * np.sin(ep) + rng.uniform(-p.noise_y, p.noise_y, n),
                   ep + p.dt * r - p.dt * v * kappa, v], 1)
    return X, U, Xn


def build_real_lane_filter(v, sigma, r, p: Optional[RealLaneParams] = None,
                           cells=(160, 100, 16), n_inputs: int = 26, copies: int = 8,
                           seed: int = 0):
    """Abstraction filter of the real-car lane-keeping model, from real steps."""
    p = p or RealLaneParams()
    X, U, Xn = real_transitions(v, sigma, r, p, np.random.default_rng(seed), copies)
    d_lo, d_hi = p.disturbance_bounds()
    f = learn_abstraction_filter(X, U, Xn, p.safe_lo, p.safe_hi, [-p.sigma_max],
                                 [p.sigma_max], cells, n_inputs, *p.derivative_bounds(),
                                 d_lo, d_hi, FEATURES)
    f.stats["n_real_steps"] = len(X) // copies
    return f, (X, U, Xn)


class RealLaneEnv(gym.Env):
    """Closed-loop test bed: lane keeping with the :class:`~sfilter.real.comma.Twin`
    fitted to the logs (steady-state cornering + steering lag + resampled real
    residuals). Action: steering-wheel angle [deg] in [-sigma_max, sigma_max].
    Observation, reward and episode end as in :class:`sfilter.envs.LaneKeepingEnv`."""

    metadata = {"render_modes": []}

    def __init__(self, twin: Twin, params: Optional[RealLaneParams] = None,
                 horizon: int = 600):
        self.twin, self.p, self.horizon = twin, params or RealLaneParams(), int(horizon)
        sm = self.p.sigma_max
        self.action_space = spaces.Box(-sm, sm, shape=(1,), dtype=np.float32)
        self.observation_space = spaces.Box(-5.0, 5.0, shape=(7,), dtype=np.float32)
        self.state = np.zeros(3)
        self.road: Optional[Road] = None
        self.s, self.t, self.last_delta = 0.0, 0, 0.0
        self._queue: list = []

    def kappa(self, ahead: float = 0.0) -> float:
        return float(self.road.curvature(self.s + ahead))

    def _obs(self):
        p, (ey, ep, v) = self.p, self.state
        km = p.kappa_max
        return np.array([ey / p.e_y_max, ep / p.e_psi_max,
                         (v - 0.5 * (p.v_min + p.v_max)) / (0.5 * (p.v_max - p.v_min)),
                         self.kappa() / km, self.kappa(0.5 * v) / km, self.kappa(v) / km,
                         self.last_delta / p.sigma_max], dtype=np.float32)

    def in_spec(self, x=None) -> bool:
        x = self.state if x is None else x
        return bool(abs(x[0]) <= self.p.e_y_max and abs(x[1]) <= self.p.e_psi_max)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        options = options or {}
        p = self.p
        self.road = options.get("road") or random_road(self.np_random, 1500.0, p.kappa_max)
        v = options.get("v", self.np_random.uniform(p.v_min, p.v_max))
        ey, ep = options.get("state", (self.np_random.uniform(-0.3, 0.3),
                                       self.np_random.uniform(-0.02, 0.02)))[:2]
        self.state = np.array([ey, ep, v], float)
        self.s, self.t, self.last_delta = 0.0, 0, 0.0
        self._queue = [self.twin.sigma0] * self.twin.lag_steps
        self._res = self.np_random.integers(len(self.twin.residuals))
        return self._obs(), {"state": self.state.copy()}

    def step(self, action):
        p, tw = self.p, self.twin
        sigma = float(np.clip(np.asarray(action, float).reshape(-1)[0], -p.sigma_max,
                              p.sigma_max))
        self._queue.append(sigma)
        applied = self._queue.pop(0)
        kappa = self.kappa()
        ey, ep, v = self.state
        # residuals replayed in their logged order from a random start (they are
        # correlated in time: road bank, wind)
        w = float(tw.residuals[self._res % len(tw.residuals)])
        self._res += 1
        r = tw.yaw_rate(v, applied) + w
        self.state = np.array([ey + p.dt * v * np.sin(ep)
                               + self.np_random.uniform(-p.noise_y, p.noise_y),
                               ep + p.dt * (r - v * kappa), v])
        self.s += p.dt * v * np.cos(self.state[1])
        self.t += 1
        dd, self.last_delta = sigma - self.last_delta, sigma
        violation = not self.in_spec()
        if violation:
            reward = -50.0
        else:
            y, e = self.state[0] / p.e_y_max, self.state[1] / p.e_psi_max
            reward = max(0.0, 1.0 - y**2 - 0.1 * e**2 - 0.5 * (dd / p.sigma_max) ** 2)
        truncated = self.t >= self.horizon and not violation
        return self._obs(), float(reward), violation, truncated, {
            "state": self.state.copy(), "delta": sigma, "kappa": kappa,
            "violation": violation}
