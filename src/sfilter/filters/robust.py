"""Safety filters with guarantees under measurement error and a known input delay.

Prototype for ddma (see the note `robust-filters-math` in the project). Works on any ddma
``SymbolicAbstraction`` on a uniform grid.

Measurement error. The filter sees y with |y - x| <= eps (per coordinate). Let
m_i = ceil(eps_i / h_i) (h: cell widths) and N(q) the cells whose index differs from q's
by at most m. If x is in cell q_x and y_c = clip(y) is in cell q_y, then q_x is in
N(q_y). The *measurement-robust* controlled invariant set W is the greatest fixed point of

    A_W(q, p)  = [Post(q, p) inside W]                       (ordinary safety check)
    R_W(q_y, p) = AND over q' in N(q_y) and in W of A_W(q', p)  (safe for every hidden state)
    W          <- W and { q : every q_y in N(q) has some p with R_W(q_y, p) }

Guarantee: if x_0 is in W and the error bound holds, the filter that returns an input of a
cell p with R_W(cell(clip(y_t)), p) always has a choice, and x_t stays in W for ever.
(Induction: x_t in W, so cell(x_t) is one of the cells R_W checks; the chosen input keeps
x_{t+1} in W; and since cell(clip(y_{t+1})) is within m of cell(x_{t+1}) in W, it has an
input.) No fallback is ever needed.

Known input delay of d steps. The inputs already sent are known. Starting from the exact
cell of the measured state, the filter propagates a box of cells through the pending
inputs with the abstraction's successor boxes. A bound r_d (in cells) on the half-width of
that box holds whatever the inputs, computed once from the abstraction:

    r_0 = 0,  r_{k+1} = max over q, p of half-width of the hull of Post(q', p), q' in N_{r_k}(q)

Then the delayed system is handled exactly as a measurement error of r_d cells around the
centre of the predicted box: same fixed point with m = r_d, same guarantee, provided the
d inputs already in the pipe at the start keep the state in W (e.g. the system starts
inside W at rest and the idle input keeps it there for d steps).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
from ddma.synthesis import _bad_in_boxes, _summed_area
from scipy.ndimage import maximum_filter, minimum_filter

__all__ = ["pair_table", "robust_safety", "delay_radius", "RobustFilter"]


def pair_table(abst, chunk: int = 2_000_000):
    """enabled (Q, P), first / last (Q, P, n): successor boxes of every (state, input) pair."""
    Q, P, n = abst.n_states, abst.n_inputs, abst.X.ndim
    en = np.zeros((Q, P), bool)
    first = np.zeros((Q, P, n), np.int32)
    last = np.zeros((Q, P, n), np.int32)
    qq, pp = np.meshgrid(np.arange(Q), np.arange(P), indexing="ij")
    qq, pp = qq.ravel(), pp.ravel()
    for s in range(0, len(qq), chunk):
        e, f, l_ = abst.pair_boxes(qq[s:s + chunk], pp[s:s + chunk])
        en.ravel()[s:s + chunk] = e
        first.reshape(-1, n)[s:s + chunk] = f
        last.reshape(-1, n)[s:s + chunk] = l_
    return en, first, last


def _window(m, P=None):
    size = tuple(2 * int(k) + 1 for k in m)
    return size if P is None else size + (1,)


def robust_safety(abst, safe, m, table=None, max_iter: int = 10_000):
    """Measurement-robust maximal controlled invariant set for an error of ``m`` cells
    per coordinate. Returns (W (Q,) bool, R (Q, P) bool, iterations)."""
    shape, Q, P = abst.X.shape, abst.n_states, abst.n_inputs
    en, first, last = table if table is not None else pair_table(abst)
    m = np.broadcast_to(np.asarray(m, int), (len(shape),))
    W = np.asarray(safe, bool).ravel().copy()
    for it in range(1, max_iter + 1):
        sat = _summed_area(abst, W)
        A = en & (_bad_in_boxes(sat, first, last) == 0)             # Post(q, p) in W
        B = (A | ~W[:, None]).reshape(shape + (P,))                  # cells outside W: no say
        R = minimum_filter(B, size=_window(m, P), mode="constant", cval=True).reshape(Q, P)
        ok = R.any(axis=1).reshape(shape)                            # observed cell has a choice
        new = W & minimum_filter(ok, size=_window(m), mode="constant", cval=True).ravel()
        if np.array_equal(new, W):
            return W, R, it
        W = new
    raise RuntimeError("no fixed point")


def delay_radius(abst, d: int, table=None):
    """Bounds r_1..r_d (cells per coordinate) on the half-width of the predicted box after
    k pending inputs, starting from the exact cell of the measured state."""
    shape, P, n = abst.X.shape, abst.n_inputs, abst.X.ndim
    en, first, last = table if table is not None else pair_table(abst)
    big = np.iinfo(np.int32).max
    f = np.where(en[..., None], first, big).reshape(shape + (P, n))
    l_ = np.where(en[..., None], last, -1).reshape(shape + (P, n))
    r = np.zeros(n, int)
    out = []
    for _ in range(d):
        size = _window(r) + (1, 1)
        lo = minimum_filter(f, size=size, mode="constant", cval=big)
        hi = maximum_filter(l_, size=size, mode="constant", cval=-1)
        ok = hi >= 0
        half = np.where(ok, np.ceil((hi - lo) / 2), 0).reshape(-1, n)
        r = half.max(axis=0).astype(int)
        out.append(r.copy())
    return out


@dataclass
class RobustFilter:
    """Safety filter from :func:`robust_safety`: measured state with error up to ``m``
    cells (``delay`` = 0), or exact state with inputs acting ``delay`` steps late
    (``m`` = :func:`delay_radius`'s last bound)."""

    abst: object
    W: np.ndarray
    R: np.ndarray
    m: np.ndarray
    delay: int = 0
    table: Optional[tuple] = None
    idle: Optional[np.ndarray] = None

    def __post_init__(self):
        self.reset()

    def reset(self):
        U = self.abst.U
        idle = np.zeros(U.ndim) if self.idle is None else np.atleast_1d(self.idle)
        self.pending = [idle.copy() for _ in range(self.delay)]

    @property
    def coverage(self) -> float:
        return float(self.W.mean())

    def _cell(self, x):
        X = self.abst.X
        x = np.clip(np.asarray(x, float), X.lo, X.hi - 1e-12 * (X.hi - X.lo))
        return X.locate(x[None])[0]

    def contains(self, x) -> bool:
        X = self.abst.X
        if not X.contains(np.atleast_2d(x))[0]:
            return False
        return bool(self.W[np.ravel_multi_index(tuple(self._cell(x)), X.shape)])

    def _center(self, x):
        """Index of the cell whose R row to use: the measured cell, or the centre of the
        box predicted through the pending inputs."""
        idx = self._cell(x)
        if not self.delay:
            return idx
        en, first, last = self.table
        X, U = self.abst.X, self.abst.U
        lo, hi = idx.copy(), idx.copy()
        for u in self.pending:
            p = int(np.ravel_multi_index(tuple(U.locate(np.clip(u, U.lo, U.hi - 1e-12
                                                                 * (U.hi - U.lo))[None])[0]),
                                         U.shape))
            grids = np.meshgrid(*[np.arange(a, b + 1) for a, b in zip(lo, hi, strict=True)],
                                indexing="ij")
            q = np.ravel_multi_index(tuple(g.ravel() for g in grids), X.shape)
            q = q[en[q, p]]
            if q.size == 0:
                return idx
            lo, hi = first[q, p].min(0), last[q, p].max(0)
        return (lo + hi) // 2

    def filter(self, x, u):
        X, U = self.abst.X, self.abst.U
        c = self._center(x)
        ps = np.flatnonzero(self.R[np.ravel_multi_index(tuple(c), X.shape)])
        if ps.size == 0:
            out = np.asarray(u, float).ravel()       # outside the robust set: no promise
        else:
            lo, hi = U.cell_bounds(np.stack(np.unravel_index(ps, U.shape), axis=1))
            u = np.asarray(u, float).ravel()
            cand = np.clip(u, lo, hi - 1e-9 * (hi - lo))
            out = cand[np.argmin(np.linalg.norm(cand - u, axis=1))]
        if self.delay:
            self.pending = self.pending[1:] + [out]
        return out
