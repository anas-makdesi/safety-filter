"""Real driving data: the steering response of a Toyota RAV4, from comma2k19.

comma2k19 (Schafer, Santana, Haden, Biasini, "A Commute in Data: The comma2k19 Dataset",
2018; MIT licence; https://github.com/commaai/comma2k19) holds 33 hours of highway
commute in California, logged from the cars' CAN bus and a phone-grade IMU. Chunks 1-2
are a Toyota RAV4. We use, at a 0.05 s grid:

* ``v``: speed from CAN [m/s];
* ``sigma``: steering-wheel angle from CAN [deg] (the input; no steering ratio assumed);
* ``r``: yaw rate from the gyroscope [rad/s] (the device's z axis, sign flipped: the
  correlation with v * sigma is above 0.95), with isolated spikes removed by a running
  median over 5 samples (0.25 s). Without it, single-sample gyro glitches (up to
  0.2 rad/s) make the data inconsistent with any useful disturbance bound.

The only vehicle-specific part of the lane-keeping model is the heading change per step,
``dt * r``, as a function of speed and steering. That is what the real data provide; the
lane geometry (how heading error moves the car sideways) is exact kinematics.

Download first: ``python experiments/10_comma_download.py --chunks 1 2``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

__all__ = ["DT", "Drive", "load_drives", "steps", "fit_twin", "Twin"]

DT = 0.05


@dataclass
class Drive:
    """One one-minute segment on the 0.05 s grid."""

    name: str
    t: np.ndarray
    v: np.ndarray
    sigma: np.ndarray
    r: np.ndarray


def _load(path: Path, dt: float = DT, margin: float = 0.5, median: int = 5):
    d = np.load(path)
    tg, ig = np.unique(d["gyro_t"], return_index=True)
    ts, iss = np.unique(d["steer_t"], return_index=True)
    tv, iv = np.unique(d["speed_t"], return_index=True)
    t0 = max(tg[0], ts[0], tv[0]) + margin
    t1 = min(tg[-1], ts[-1], tv[-1]) - margin
    if t1 - t0 < 5:
        return None
    # largest gap in any stream: segments with holes are skipped
    if max(np.diff(tg).max(), np.diff(ts).max(), np.diff(tv).max()) > 0.25:
        return None
    t = np.arange(t0, t1, dt)
    # yaw rate at the middle of each step: the heading change over [t, t + dt] / dt
    r = np.interp(t + 0.5 * dt, tg, -d["gyro"][ig, 2])
    if median > 1:
        from scipy.ndimage import median_filter
        r = median_filter(r, size=median, mode="nearest")
    return Drive(path.stem, t, np.interp(t, tv, d["speed"][iv, 0]),
                 np.interp(t, ts, d["steer"][iss]), r)


def load_drives(folder, median: int = 5) -> list[Drive]:
    """All usable segments in ``folder`` (files written by 10_comma_download.py)."""
    out = []
    for p in sorted(Path(folder).glob("*.npz")):
        dr = _load(p, median=median)
        if dr is not None:
            out.append(dr)
    return out


def steps(drives, v_lo: float, v_hi: float):
    """Pooled per-step samples (v, sigma, r) with v_lo <= v <= v_hi."""
    v = np.concatenate([d.v for d in drives])
    s = np.concatenate([d.sigma for d in drives])
    r = np.concatenate([d.r for d in drives])
    m = (v >= v_lo) & (v <= v_hi)
    return v[m], s[m], r[m]


@dataclass
class Twin:
    """A simple vehicle fitted to the logs, used only to *simulate* closed loops:
    r = v * (sigma - sigma0) * (pi / 180) / (G * (1 + K v^2)), steering acting ``lag``
    seconds late, plus residuals resampled from the real fit.

    G is steering ratio times wheelbase [m], K the understeer factor [s^2/m^2]. The
    filter never sees this model: it is built from the logs themselves."""

    G: float
    K: float
    sigma0: float
    lag_steps: int
    residuals: np.ndarray

    def yaw_rate(self, v, sigma):
        return v * np.deg2rad(sigma - self.sigma0) / (self.G * (1 + self.K * v**2))


def fit_twin(v, s, r, lag_steps: int = 2) -> Twin:
    """Least squares for 1/G, K and the steering offset on pooled samples (grid search
    over K, linear fit for the rest), residuals kept for resampling."""
    best = None
    for K in np.linspace(0.0, 0.003, 61):
        X = np.stack([v * np.deg2rad(s) / (1 + K * v**2), v / (1 + K * v**2)], 1)
        coef, *_ = np.linalg.lstsq(X, r, rcond=None)
        res = r - X @ coef
        if best is None or res @ res < best[0]:
            best = (res @ res, K, coef, res)
    _, K, (a, b), res = best
    G = 1.0 / a
    sigma0 = -np.rad2deg(b / a)
    return Twin(G=float(G), K=float(K), sigma0=float(sigma0), lag_steps=lag_steps,
                residuals=res.astype(np.float32))
