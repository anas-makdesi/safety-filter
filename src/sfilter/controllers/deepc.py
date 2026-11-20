"""DeePC: data-enabled predictive control (Coulson, Lygeros, Doerfler, 2019).

DeePC is predictive control without a model. It records one trajectory of inputs and
outputs of the system, and uses it directly as the predictor: for a linear system, every
trajectory of length L is a combination of the length-L windows cut from a sufficiently
rich recorded one (Willems' fundamental lemma). Stack those windows as columns of a
matrix H. Split its rows into "past" (the last ``t_ini`` steps: they pin down the current
state) and "future" (the next ``horizon`` steps). At every step solve

    minimise    tracking cost of the future outputs + input cost
                + lam_g |g|^2 + lam_s |sigma|^2
    subject to  H_past   g = (recent inputs, recent outputs + sigma)
                H_future g = (future inputs, future outputs)
                input limits, soft output limits

and apply the first future input. The two regularisers make it work with noise and mild
nonlinearity: ``lam_g`` keeps the combination g small (do not trust exotic combinations
of noisy data), and the slack sigma lets the recent outputs be matched approximately.

To keep the problem small and fast, the window matrix H is replaced by its ``rank``
dominant directions (singular value decomposition, H ~ U S V'): g then has ``rank``
entries instead of one per window, and the weak directions, which are mostly noise, are
dropped (Zhang, Zheng, Li, 2023; Breschi, Chiuso, Formentin, 2023).

Signals that are measured but not chosen (road curvature) enter as extra input channels
whose future values are given (``n_dist`` channels, placed after the controlled ones).

:class:`DeePC` is the generic controller on scaled signals. :class:`LaneDeePC` and
:class:`QuadDeePC` collect their data and connect it to the two benchmarks.
"""

from __future__ import annotations

import cvxpy as cp
import numpy as np

from ..envs.lane_keeping import lane_keeping_step
from ..envs.quadrotor import quadrotor_step
from ..envs.roads import Road

SOLVER = cp.OSQP

__all__ = ["DeePC", "LaneDeePC", "QuadDeePC", "hankel"]


def hankel(w: np.ndarray, depth: int) -> np.ndarray:
    """Windows of length ``depth`` of the signal w (T, q), as columns: (depth*q, T-depth+1)."""
    T = len(w)
    return np.stack([w[i:i + depth].reshape(-1) for i in range(T - depth + 1)], axis=1)


class DeePC:
    """Regularised DeePC on recorded data U (T, m) and Y (T, p).

    The first ``m - n_dist`` input channels are controlled, the rest are measured
    disturbances with known future values. ``q``: output weights (p,); ``r``: weight on
    the controlled inputs; ``r_delta``: weight on their change from one step to the next;
    ``u_lo``, ``u_hi``: input limits; ``y_lo``, ``y_hi``: soft output limits (or None).
    """

    def __init__(self, U, Y, t_ini: int, horizon: int, q, r=0.0, r_delta=0.0,
                 u_lo=None, u_hi=None, y_lo=None, y_hi=None, n_dist: int = 0,
                 lam_g: float = 1.0, lam_s: float = 1e2, lam_y: float = 1e2,
                 rank: int | None = None):
        U, Y = np.atleast_2d(U), np.atleast_2d(Y)
        m, p, N = U.shape[1], Y.shape[1], horizon
        mc = m - n_dist
        self.t_ini, self.N, self.m, self.mc, self.p = t_ini, N, m, mc, p
        HU, HY = hankel(U, t_ini + N), hankel(Y, t_ini + N)
        if rank is not None and rank < HU.shape[1]:
            H = np.vstack([HU, HY])
            Us, S, _ = np.linalg.svd(H, full_matrices=False)
            H = Us[:, :rank] * S[:rank]          # same column space, dominant part only
            HU, HY = H[:len(HU)], H[len(HU):]
        Up, Uf = HU[:t_ini * m], HU[t_ini * m:]
        Yp, Yf = HY[:t_ini * p], HY[t_ini * p:]
        g = cp.Variable(HU.shape[1])
        sigma = cp.Variable(t_ini * p)
        self.u_ini = cp.Parameter(t_ini * m)
        self.y_ini = cp.Parameter(t_ini * p)
        self.ref = cp.Parameter((N, p))
        self.u_prev = cp.Parameter(mc)
        uf = cp.reshape(Uf @ g, (N, m), order="C")      # future inputs, one row per step
        yf = cp.reshape(Yf @ g, (N, p), order="C")
        u = uf[:, :mc]
        cons = [Up @ g == self.u_ini, Yp @ g == self.y_ini + sigma]
        if n_dist:
            self.d_future = cp.Parameter((N, n_dist))
            cons.append(uf[:, mc:] == self.d_future)
        if u_lo is not None:
            cons += [u >= np.asarray(u_lo)[None], u <= np.asarray(u_hi)[None]]
        cost = (cp.sum_squares(cp.multiply(yf - self.ref, np.sqrt(np.asarray(q, float))[None]))
                + lam_g * cp.sum_squares(g) + lam_s * cp.sum_squares(sigma))
        if r:
            cost += r * cp.sum_squares(u)
        if r_delta:
            cost += r_delta * (cp.sum_squares(u[0] - self.u_prev)
                               + cp.sum_squares(u[1:] - u[:-1]))
        if y_lo is not None:
            s = cp.Variable(N, nonneg=True)
            cons += [yf >= np.asarray(y_lo)[None] - s[:, None],
                     yf <= np.asarray(y_hi)[None] + s[:, None]]
            cost += lam_y * cp.sum(s)
        self.u = u
        self.prob = cp.Problem(cp.Minimize(cost), cons)
        self.past_u = np.zeros((t_ini, m))
        self.past_y = np.zeros((t_ini, p))

    def start(self, u0, y0):
        """Start of an episode: assume the system has been resting at (u0, y0)."""
        self.past_u[:] = u0
        self.past_y[:] = y0

    def observe(self, u, y):
        """Record the input applied (all channels) and the output that followed it."""
        self.past_u = np.vstack([self.past_u[1:], u])
        self.past_y = np.vstack([self.past_y[1:], y])

    def solve(self, ref, d_future=None):
        """First controlled input of the optimal plan (None if the solver failed)."""
        self.u_ini.value = self.past_u.reshape(-1)
        self.y_ini.value = self.past_y.reshape(-1)
        self.ref.value = np.broadcast_to(ref, (self.N, self.p))
        self.u_prev.value = self.past_u[-1, :self.mc]
        if d_future is not None:
            self.d_future.value = d_future
        self.prob.solve(solver=SOLVER, warm_start=True)
        return None if self.u.value is None else self.u.value[0]


# ------------------------------------------------------------------ lane keeping
class LaneDeePC:
    """DeePC for lane keeping. Outputs (e_y, e_psi), input steering, measured
    disturbance: road curvature (with its preview, like the camera of the RL agents).

    The car's response depends on its speed, so at the start of each drive the
    controller uses ``T`` steps (15 s by default) of data recorded at that speed: a
    weak lane-centring loop plus random steering, on a test track whose curvature
    changes at random (so the effect of curvature can be identified). All signals are
    scaled by their limits.
    """

    def __init__(self, params, T: int = 300, t_ini: int = 4, horizon: int = 20,
                 lam_g: float = 1.0, rank: int | None = 60, seed: int = 0,
                 lam_y: float = 1e2, data_noise: float = 0.0):
        self.p, self.T, self.t_ini, self.N = params, T, t_ini, horizon
        self.lam_g, self.rank, self.lam_y, self.data_noise = lam_g, rank, lam_y, data_noise
        self.rng = np.random.default_rng(seed)
        self.scale_y = np.array([params.e_y_max, params.e_psi_max])
        self.ctrl = None

    def collect(self, v: float):
        """Recorded data at speed v: inputs (steering, curvature) and outputs (e_y, e_psi)."""
        p, rng, T = self.p, self.rng, self.T
        ds = p.dt * v
        s_pts = np.arange(0.0, (T + 2) * ds, 4 * ds)       # curvature changes every 4 steps
        road = Road(s_pts, rng.uniform(-p.kappa_max, p.kappa_max, len(s_pts)))
        x, s = np.array([0.0, 0.0, v]), 0.0
        U, Y = np.zeros((T, 2)), np.zeros((T, 2))
        for t in range(T):
            delta = np.clip(-0.01 * x[0] - 0.2 * x[1] + rng.uniform(-0.015, 0.015),
                            -p.delta_max, p.delta_max)
            kappa = float(road.curvature(s))
            w = rng.uniform([-p.noise_y, -p.noise_psi], [p.noise_y, p.noise_psi])
            x = lane_keeping_step(x, delta, kappa, p, w[None])[0]
            s += ds * np.cos(x[1])
            U[t] = delta / p.delta_max, kappa / p.kappa_max
            Y[t] = x[:2] / self.scale_y
        if self.data_noise:          # measurement noise in the recorded outputs (scaled)
            Y = Y + rng.uniform(-self.data_noise, self.data_noise, Y.shape)
        return U, Y

    def reset(self, env):
        p = self.p
        U, Y = self.collect(float(env.state[2]))
        # weights = the environment's reward terms (signals are already scaled)
        self.ctrl = DeePC(U, Y, self.t_ini, self.N, q=[1.0, 0.1], r_delta=0.5,
                          u_lo=[-1.0], u_hi=[1.0], y_lo=[-1.0, -1.0], y_hi=[1.0, 1.0],
                          n_dist=1, lam_g=self.lam_g, rank=min(self.rank or 10**9,
                                                               self.T - self.t_ini - self.N),
                          lam_y=self.lam_y)
        self.ctrl.start([0.0, env.kappa() / p.kappa_max], env.state[:2] / self.scale_y)
        self._last = None

    def act(self, obs, env):
        p, c = self.p, self.ctrl
        if self._last is not None:           # what the last step did
            c.observe(self._last, env.state[:2] / self.scale_y)
        v = env.state[2]
        ahead = env.s + p.dt * v * np.arange(self.N)
        kap = env.road.curvature(ahead) / p.kappa_max
        u = c.solve(np.zeros(2), kap[:, None])
        u = 0.0 if u is None else float(np.clip(u[0], -1.0, 1.0))
        self._last = np.array([u, kap[0]])
        return np.array([u])

    def applied(self, a):
        """Tell the controller which input was really applied (after a safety filter)."""
        self._last[0] = float(np.asarray(a).reshape(-1)[0]) / self.p.delta_max


# --------------------------------------------------------------------- quadrotor
class QuadDeePC:
    """DeePC for the quadrotor. Outputs: the state (y, vy, a_y, z, vz); inputs: the two
    acceleration setpoints. Data: ``T`` steps (20 s by default) of hovering with a weak
    position loop plus random commands, with gusts. The target's path over the horizon
    is the reference for (y, z). The walls are soft output limits with a ``margin``."""

    def __init__(self, params, T: int = 400, t_ini: int = 2, horizon: int = 25,
                 lam_g: float = 1.0, margin: float = 0.1, rank: int | None = 80,
                 seed: int = 0, lam_y: float = 1e2, data_noise: float = 0.0):
        p = params
        self.p, self.N = p, horizon
        rng = np.random.default_rng(seed)
        self.mid, self.half = 0.5 * (p.lo + p.hi), 0.5 * (p.hi - p.lo)
        x = np.array([0.0, 0.0, 0.0, self.mid[3], 0.0])
        U, Y = np.zeros((T, 2)), np.zeros((T, 5))
        for t in range(T):
            u = np.array([-2.0 * x[0] - 2.0 * x[1], -2.0 * (x[3] - self.mid[3]) - 2.0 * x[4]])
            u = np.clip(u + rng.uniform(-2.0, 2.0, 2), p.u_lo, p.u_hi)
            x = quadrotor_step(x, u, p, rng.uniform(-p.wind, p.wind, (1, 2)))[0]
            U[t], Y[t] = u / p.u_hi, (x - self.mid) / self.half
        if data_noise:               # measurement noise in the recorded outputs (scaled)
            Y = Y + rng.uniform(-data_noise, data_noise, Y.shape)
        lim = np.ones(5)
        lim[[0, 3]] -= margin / self.half[[0, 3]]
        self.ctrl = DeePC(U, Y, t_ini, horizon, q=[1.0, 0, 0, 1.0, 0], r=1e-3,
                          u_lo=[-1.0, -1.0], u_hi=[1.0, 1.0], y_lo=-lim, y_hi=lim,
                          lam_g=lam_g, lam_y=lam_y,
                          rank=None if rank is None else min(rank, T - t_ini - horizon))
        # cost in metres, as in the reward: scale the two position weights
        self.q_scale = self.half[[0, 3]]

    def _y(self, x):
        return (x - self.mid) / self.half

    def reset(self, env):
        self.ctrl.start(np.zeros(2), self._y(env.state))
        self._last = None

    def act(self, obs, env):
        c = self.ctrl
        if self._last is not None:
            c.observe(self._last, self._y(env.state))
        ref = np.zeros((self.N, 5))
        tgt = np.array([env.target(env.t + k + 1)[0] for k in range(self.N)])
        ref[:, [0, 3]] = (tgt - self.mid[[0, 3]]) / self.half[[0, 3]]
        u = c.solve(ref)
        u = np.zeros(2) if u is None else np.clip(u, -1.0, 1.0)
        self._last = u.copy()
        return u

    def applied(self, a):
        self._last = np.asarray(a, float).reshape(-1) / self.p.u_hi
