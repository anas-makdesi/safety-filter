"""F1: safety filters from symbolic abstractions learned from data (ddma).

Pipeline: sampled transitions -> bounded-derivative model of the unknown increments
(ddma ``BoundedDerivativeModel``) -> symbolic abstraction on a grid of the safe box ->
maximal controlled invariant set (``ddma.safety``) -> filter: in each grid cell, the
inputs whose successors provably stay in the invariant set.

What a filter knows about the system: x+ = x + f(x, u) + w (a discrete-time
integrator), which state/input coordinates each increment depends on, bounds on the
partial derivatives of f and on w, and the data. Never the equations.

:func:`learn_abstraction_filter` is the generic builder; :func:`build_abstraction_filter`
applies it to lane keeping (the quadrotor builder is in :mod:`sfilter.filters.quadrotor`).
:class:`ProductFilter` combines filters of decoupled channels (e.g. lateral and vertical).
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np
from ddma import (
    BoundedDerivativeModel,
    Controller,
    RectangularPartition,
    SymbolicAbstraction,
    affine_bounds,
    safety,
)

from ..envs.lane_keeping import LaneKeepingParams, lane_keeping_step

__all__ = ["AbstractionFilter", "DelayAwareFilter", "ProductFilter",
           "learn_abstraction_filter", "collect_transitions", "build_abstraction_filter"]

# lane keeping: which (x, u) coordinates each increment depends on: (e_y, e_psi, v, delta)
FEATURES = [[1, 2], [2, 3], [2]]


@dataclass
class AbstractionFilter:
    """:class:`SafetyFilter` backed by a ddma :class:`Controller`.

    ``state_error``: optional bound on the error of the state given to :meth:`filter`
    (one entry per state coordinate; e.g. sensor noise). When set, the filter only
    allows inputs that are admissible in *every* certified grid cell the box
    [x - state_error, x + state_error] touches, so the guarantee survives a measurement
    error up to that bound. If no input is admissible in all of them, it falls back to
    the cell of x itself (and counts the event in ``robust_fallbacks``).
    """

    controller: Controller
    stats: dict = field(default_factory=dict)
    model: object = None     # the learned ddma model (e.g. for an assumption monitor)
    state_error: Optional[np.ndarray] = None
    robust_fallbacks: int = 0

    def contains(self, x) -> bool:
        return self.controller.input_cells(np.asarray(x, float)).size > 0

    def _robust_cells(self, x):
        """Input cells admissible in every certified cell of the error box around x."""
        ctrl = self.controller
        X = ctrl.abstraction.X
        e = np.asarray(self.state_error, float)
        lo = np.clip(x - e, X.lo, X.hi - 1e-12 * (X.hi - X.lo))
        hi = np.clip(x + e, X.lo, X.hi - 1e-12 * (X.hi - X.lo))
        i0, i1 = X.locate(np.stack([lo, hi]))
        grids = np.meshgrid(*[np.arange(a, b + 1) for a, b in zip(i0, i1, strict=True)],
                            indexing="ij")
        q = np.ravel_multi_index(tuple(g.ravel() for g in grids), X.shape)
        q = q[ctrl.domain[q]]
        if q.size == 0:
            return None
        rows = ctrl.allowed[q] if ctrl.allowed is not None else \
            np.array([ctrl.allowed_row(k) for k in q])
        return np.flatnonzero(rows.all(axis=0))

    def filter(self, x, u) -> Optional[np.ndarray]:
        x, u = np.asarray(x, float), np.asarray(u, float)
        if self.state_error is None:
            return self.controller.filter(x, u)
        return self.filter_box(x - self.state_error, x + self.state_error, u)

    def filter_box(self, lo, hi, u) -> Optional[np.ndarray]:
        """The input closest to u that is admissible for every state in [lo, hi]."""
        x = 0.5 * (np.asarray(lo, float) + np.asarray(hi, float))
        u = np.asarray(u, float)
        saved, self.state_error = self.state_error, 0.5 * (np.asarray(hi) - np.asarray(lo))
        try:
            ps = self._robust_cells(x)
        finally:
            self.state_error = saved
        if ps is None or ps.size == 0:
            self.robust_fallbacks += 1
            return self.controller.filter(x, u)
        U = self.controller.abstraction.U
        lo, hi = U.cell_bounds(np.stack(np.unravel_index(ps, U.shape), axis=1))
        cand = np.clip(u.ravel(), lo, hi - 1e-9 * (hi - lo))
        return cand[np.argmin(np.linalg.norm(cand - u.ravel(), axis=1))]

    @property
    def coverage(self) -> float:
        """Share of the safe box certified (fraction of grid cells)."""
        return self.controller.winning_fraction()


class DelayAwareFilter:
    """An :class:`AbstractionFilter` for a system whose inputs act ``delay`` steps late.

    The inputs already sent but not yet applied are known (the filter sent them). At each
    step the filter predicts, with its own learned model (successor boxes, disturbance
    included), the box of states in which the system can be when the new input takes
    effect, and returns the input closest to the proposal that is admissible for every
    state of that box. Call :meth:`reset` at the start of an episode (the wrapper does).
    """

    def __init__(self, base: AbstractionFilter, delay: int, idle=None):
        self.base, self.delay = base, int(delay)
        U = base.controller.abstraction.U
        self.idle = np.zeros(U.ndim) if idle is None else np.atleast_1d(idle)
        self.reset()

    def reset(self):
        self.pending = [self.idle.copy() for _ in range(self.delay)]

    def contains(self, x) -> bool:
        return self.base.contains(x)

    @property
    def coverage(self) -> float:
        return self.base.coverage

    @property
    def model(self):
        return self.base.model

    def _predict(self, lo, hi, u):
        """Box of successors of every state in [lo, hi] under input u."""
        ab = self.base.controller.abstraction
        X, U, model = ab.X, ab.U, self.base.model
        eps = 1e-12 * (X.hi - X.lo)
        lo, hi = np.clip(lo, X.lo, X.hi - eps), np.clip(hi, X.lo, X.hi - eps)
        first, last = X.box_to_cells(lo[None], hi[None])
        grids = np.meshgrid(*[np.arange(a, b + 1) for a, b in zip(first[0], last[0], strict=True)],
                            indexing="ij")
        qi = np.stack([g.ravel() for g in grids], 1)
        ui = np.repeat(U.locate(np.clip(u, U.lo, U.hi - 1e-12 * (U.hi - U.lo))[None]),
                       len(qi), 0)
        xl, xh = X.cell_bounds(qi)
        ul, uh = U.cell_bounds(ui)
        slo, shi = model.successor_bounds(np.hstack([qi, ui]), xl, xh, ul, uh)
        return slo.min(0), shi.max(0)

    def filter(self, x, u) -> Optional[np.ndarray]:
        x = np.asarray(x, float)
        lo, hi = x.copy(), x.copy()
        for p in self.pending:
            lo, hi = self._predict(lo, hi, p)
        out = self.base.filter_box(lo, hi, u)
        if out is None:
            out = np.asarray(u, float).ravel()
        if self.delay:
            self.pending = self.pending[1:] + [np.asarray(out, float).ravel()]
        return out


@dataclass
class ProductFilter:
    """Filters of decoupled channels side by side: channel k sees the state coordinates
    ``states[k]`` and filters the input coordinates ``inputs[k]``. The coupling between
    channels must be covered by each channel's disturbance bounds."""

    filters: Sequence
    states: Sequence[Sequence[int]]
    inputs: Sequence[Sequence[int]]

    def contains(self, x) -> bool:
        x = np.asarray(x, float)
        return all(f.contains(x[list(s)])
                   for f, s in zip(self.filters, self.states, strict=True))

    def filter(self, x, u) -> Optional[np.ndarray]:
        x, u = np.asarray(x, float), np.asarray(u, float).copy()
        for f, s, i in zip(self.filters, self.states, self.inputs, strict=True):
            v = f.filter(x[list(s)], u[list(i)])
            if v is None:
                return None
            u[list(i)] = v
        return u

    @property
    def coverage(self) -> float:
        """Share of the product safe box certified (product of the channel shares)."""
        return float(np.prod([f.coverage for f in self.filters]))


def _identity(X, U):
    return np.asarray(X, float)


def learn_abstraction_filter(X, U, Xn, safe_lo, safe_hi, u_lo, u_hi, cells, n_inputs,
                             A_lo, A_hi, B_lo, B_hi, d_lo, d_hi, features,
                             method: str = "growth",
                             local: Optional[str] = "slab") -> AbstractionFilter:
    """Abstraction filter of x+ = x + f(x, u) + w from transitions (X, U, Xn).

    ``cells`` / ``n_inputs``: grid of the safe box [safe_lo, safe_hi] and of the input box
    [u_lo, u_hi]. ``A_*``, ``B_*``: bounds on df/dx and df/du on these boxes; ``d_*``:
    bounds on w; ``features[i]``: the (x, u) coordinates increment i depends on.
    ``method`` / ``local``: see ``ddma.BoundedDerivativeModel.from_data``.
    """
    safe_lo, safe_hi = np.asarray(safe_lo, float), np.asarray(safe_hi, float)
    u_lo = np.atleast_1d(np.asarray(u_lo, float))
    u_hi = np.atleast_1d(np.asarray(u_hi, float))
    n, m = len(safe_lo), len(u_lo)
    cells = [int(c) for c in np.broadcast_to(cells, (n,))]
    n_inputs = [int(c) for c in np.broadcast_to(n_inputs, (m,))]
    Px = RectangularPartition.uniform(safe_lo, safe_hi, cells)
    Pu = RectangularPartition.uniform(u_lo, u_hi, n_inputs)
    P = RectangularPartition.product(Px, Pu)
    t0 = time.perf_counter()
    model = BoundedDerivativeModel.from_data(
        X, U, Xn, P, A_lo, A_hi, B_lo, B_hi, d_lo, d_hi,
        g=_identity, g_bounds=affine_bounds(np.eye(n), np.zeros((n, m))),
        features=features, method=method,
        local=local if method in ("growth", "lin_growth") else None)
    t1 = time.perf_counter()
    abst = SymbolicAbstraction(Px, Pu, model)
    t2 = time.perf_counter()
    ctrl = safety(abst, np.ones(Px.n_cells, bool))
    t3 = time.perf_counter()
    stats = dict(n_data=len(X), cells=cells, n_inputs=n_inputs, method=method, local=local,
                 t_model=t1 - t0, t_abstraction=t2 - t1, t_synthesis=t3 - t2,
                 coverage=ctrl.winning_fraction(),
                 memory_mb=(ctrl.allowed.nbytes if ctrl.allowed is not None else 0) / 2**20)
    return AbstractionFilter(ctrl, stats, model)


def collect_transitions(n: int, p: LaneKeepingParams, rng: np.random.Generator):
    """n lane-keeping transitions from states and inputs drawn uniformly in the safe box
    and input range, with curvature and noise drawn uniformly in their bounds."""
    X = rng.uniform(p.safe_lo, p.safe_hi, (n, 3))
    U = rng.uniform(-p.delta_max, p.delta_max, (n, 1))
    kappa = rng.uniform(-p.kappa_max, p.kappa_max, n)
    w = rng.uniform([-p.noise_y, -p.noise_psi], [p.noise_y, p.noise_psi], (n, 2))
    return X, U, lane_keeping_step(X, U[:, 0], kappa, p, w)


def build_abstraction_filter(n_data: int = 2_000_000, cells=(160, 120, 24),
                             n_inputs: int = 21, p: Optional[LaneKeepingParams] = None,
                             method: str = "growth", local: Optional[str] = "slab",
                             seed: int = 0, data=None) -> AbstractionFilter:
    """Lane-keeping filter: learn from ``n_data`` transitions (or ``data=(X, U, Xn)``),
    abstraction on ``cells`` = (e_y, e_psi, v) cells x ``n_inputs`` steering cells,
    maximal controlled invariant set of the lane-keeping specification."""
    p = p or LaneKeepingParams()
    rng = np.random.default_rng(seed)
    X, U, Xn = data if data is not None else collect_transitions(n_data, p, rng)
    d_lo, d_hi = p.disturbance_bounds()
    return learn_abstraction_filter(X, U, Xn, p.safe_lo, p.safe_hi, [-p.delta_max],
                                    [p.delta_max], cells, n_inputs, *p.derivative_bounds(),
                                    d_lo, d_hi, FEATURES, method=method, local=local)
