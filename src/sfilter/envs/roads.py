"""Random roads with bounded curvature (clothoid transitions between arcs).

A road is its curvature profile kappa(s) along the arc length s, piecewise linear:
straight or constant-curvature sections joined by clothoid ramps (curvature changing
linearly with s). Curvature always stays in [-kappa_max, kappa_max], which is the only
thing the safety filters assume about the road.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = ["Road", "random_road", "evaluation_roads", "showcase_road"]


@dataclass
class Road:
    """Curvature profile kappa(s): linear interpolation between breakpoints (s_k, kappa_k)."""

    s: np.ndarray
    kappa: np.ndarray

    @property
    def length(self) -> float:
        return float(self.s[-1])

    def curvature(self, s) -> np.ndarray:
        """kappa at arc length(s) s (constant beyond the end of the road)."""
        return np.interp(s, self.s, self.kappa)

    def centerline(self, ds: float = 0.5):
        """(x, y, heading) of the lane centre, sampled every ds metres (for plots)."""
        s = np.arange(0.0, self.length + ds, ds)
        k = self.curvature(s)
        th = np.concatenate([[0.0], np.cumsum(0.5 * (k[1:] + k[:-1]) * ds)])
        x = np.concatenate([[0.0], np.cumsum(np.cos(th[:-1]) * ds)])
        y = np.concatenate([[0.0], np.cumsum(np.sin(th[:-1]) * ds)])
        return s, x, y, th


def random_road(rng: np.random.Generator, length: float = 1500.0, kappa_max: float = 0.004,
                section=(40.0, 200.0), ramp=(30.0, 120.0), p_straight: float = 0.3) -> Road:
    """A random road: sections of constant curvature joined by clothoid ramps.

    Each section has a curvature drawn uniformly in [-kappa_max, kappa_max] (or 0 with
    probability ``p_straight``) and a length drawn in ``section``; consecutive sections
    are joined by ramps with lengths drawn in ``ramp``.
    """
    s_pts, k_pts = [0.0], [0.0]
    s = 0.0
    while s < length:
        k = 0.0 if rng.random() < p_straight else rng.uniform(-kappa_max, kappa_max)
        s += rng.uniform(*ramp)
        s_pts.append(s)
        k_pts.append(k)
        s += rng.uniform(*section)
        s_pts.append(s)
        k_pts.append(k)
    return Road(np.array(s_pts), np.clip(np.array(k_pts), -kappa_max, kappa_max))


def evaluation_roads(n: int = 20, kappa_max: float = 0.004, seed: int = 12345, **kw) -> list:
    """The fixed set of held-out roads used for evaluation (never used in training)."""
    rng = np.random.default_rng(seed)
    return [random_road(rng, kappa_max=kappa_max, **kw) for _ in range(n)]


def showcase_road(kappa_max: float = 0.004) -> Road:
    """A demanding road for figures and videos: an S-bend at full curvature, quick ramps."""
    s = np.array([0, 60, 90, 250, 290, 450, 480, 560, 590, 700, 730, 900], float)
    k = kappa_max * np.array([0, 0, 1, 1, -1, -1, 0, 0, 1, 1, -1, -1], float)
    return Road(s, k)
