"""Reference controllers: model predictive control with the true model.

These are not learned. They know the equations, the parameters and what lies ahead (the
road curvature, the target's path), and they optimise the same quantity the environments
reward. They show what performance is possible, so the cost of the safety filter and the
quality of the learned controllers can be measured against something.

Both solve a small quadratic program at every step (CVXPY + OSQP). The dynamics are
linearised: for lane keeping around small angles (error below 1 % inside the
specification), for the quadrotor the quadratic drag is linearised around the current
speed at every step. Unknown noise and gusts are not modelled; state limits are soft
constraints with an optional back-off ``margin``.

A controller has ``reset(env)`` (start of an episode) and ``act(obs, env)``, which returns
the action scaled to [-1, 1]; ``env`` is the simulator (true state, road, target).
"""

from __future__ import annotations

import cvxpy as cp
import numpy as np

SOLVER = cp.CLARABEL

__all__ = ["LaneMPC", "QuadMPC"]


class LaneMPC:
    """Lane keeping. Cost = the environment's reward terms over ``horizon`` steps:
    (e_y / e_y_max)^2 + 0.1 (e_psi / e_psi_max)^2 + 0.5 (steering change / delta_max)^2."""

    def __init__(self, params, horizon: int = 20, margin: float = 0.0):
        p, N = params, horizon
        self.p, self.N = p, N
        x = cp.Variable((N + 1, 2))          # predicted (e_y, e_psi)
        u = cp.Variable(N)                   # steering
        s = cp.Variable(N, nonneg=True)      # slack of the soft state limits
        self.x0, self.u_prev = cp.Parameter(2), cp.Parameter()
        self.a = cp.Parameter(nonneg=True)   # dt * v
        self.b = cp.Parameter(nonneg=True)   # dt * v / L
        self.c = cp.Parameter(N)             # dt * v * kappa along the horizon
        cons = [x[0] == self.x0, cp.abs(u) <= p.delta_max]
        cost = 0
        for k in range(N):
            cons += [x[k + 1, 0] == x[k, 0] + self.a * x[k, 1],
                     x[k + 1, 1] == x[k, 1] + self.b * u[k] - self.c[k],
                     cp.abs(x[k + 1, 0]) <= p.e_y_max - margin + s[k],
                     cp.abs(x[k + 1, 1]) <= p.e_psi_max + s[k]]
            du = u[k] - (self.u_prev if k == 0 else u[k - 1])
            cost += (cp.square(x[k + 1, 0] / p.e_y_max)
                     + 0.1 * cp.square(x[k + 1, 1] / p.e_psi_max)
                     + 0.5 * cp.square(du / p.delta_max) + 1e3 * s[k])
        self.u = u
        self.prob = cp.Problem(cp.Minimize(cost), cons)

    def reset(self, env):
        pass

    def act(self, obs, env):
        p = self.p
        ey, ep, v = env.state
        self.x0.value = np.array([ey, ep])
        self.u_prev.value = env.last_delta
        self.a.value = p.dt * v
        self.b.value = p.dt * v / p.wheelbase
        ahead = env.s + p.dt * v * np.arange(self.N)
        self.c.value = p.dt * v * env.road.curvature(ahead)
        self.prob.solve(solver=SOLVER, warm_start=True)
        delta = 0.0 if self.u.value is None else float(self.u.value[0])
        return np.array([np.clip(delta / p.delta_max, -1.0, 1.0)])


class QuadMPC:
    """Quadrotor. Cost = squared distance to the target over ``horizon`` steps (the
    environment's reward), plus a small weight on the commands. ``margin``: back-off of
    the position limits [m], which leaves room for the gusts the model ignores."""

    def __init__(self, params, horizon: int = 25, margin: float = 0.1):
        p, N = params, horizon
        self.p, self.N = p, N
        x = cp.Variable((N + 1, 5))          # (y, vy, a_y, z, vz)
        u = cp.Variable((N, 2))              # (a_y,c, a_z,c)
        s = cp.Variable(N, nonneg=True)
        self.x0 = cp.Parameter(5)
        self.ref = cp.Parameter((N, 2))      # target (y, z) along the horizon
        self.al = cp.Parameter(2)            # drag linearisation: v+ = al v + dt a + be
        self.be = cp.Parameter(2)
        lo, hi = p.lo.copy(), p.hi.copy()
        lo[[0, 3]] += margin
        hi[[0, 3]] -= margin
        dt = p.dt
        cons = [x[0] == self.x0, u >= p.u_lo[None], u <= p.u_hi[None]]
        cost = 0
        for k in range(N):
            cons += [x[k + 1, 0] == x[k, 0] + dt * x[k, 1],
                     x[k + 1, 1] == self.al[0] * x[k, 1] + dt * x[k, 2] + self.be[0],
                     x[k + 1, 2] == x[k, 2] + dt / p.tau * (u[k, 0] - x[k, 2]),
                     x[k + 1, 3] == x[k, 3] + dt * x[k, 4],
                     x[k + 1, 4] == self.al[1] * x[k, 4] + dt * u[k, 1] + self.be[1],
                     x[k + 1] >= lo - s[k], x[k + 1] <= hi + s[k]]
            cost += (cp.sum_squares(x[k + 1, [0, 3]] - self.ref[k])
                     + 1e-3 * cp.sum_squares(u[k]) + 1e3 * s[k])
        self.u = u
        self.prob = cp.Problem(cp.Minimize(cost), cons)

    def reset(self, env):
        pass

    def act(self, obs, env):
        p = self.p
        x = env.state
        v0 = x[[1, 4]]
        d1 = p.c1 + 2 * p.c2 * np.abs(v0)            # slope of the drag at v0
        d0 = p.c1 * v0 + p.c2 * v0 * np.abs(v0)      # drag at v0
        self.x0.value = x
        self.al.value = 1 - p.dt * d1
        self.be.value = p.dt * (d1 * v0 - d0)
        self.ref.value = np.array([env.target(env.t + k + 1)[0] for k in range(self.N)])
        self.prob.solve(solver=SOLVER, warm_start=True)
        a = np.zeros(2) if self.u.value is None else self.u.value[0]
        mid, half = 0.5 * (p.u_lo + p.u_hi), 0.5 * (p.u_hi - p.u_lo)
        return np.clip((a - mid) / half, -1.0, 1.0)
