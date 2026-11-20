"""Tests for the follow-ups of 19 Nov: noise-aware and delay-aware filtering, the
"end on correction" training signal, and the real-car lane-keeping filter."""

from collections import deque

import gymnasium as gym
import numpy as np
import pytest

from sfilter.controllers.rl import EndOnCorrection, make_env
from sfilter.envs import LaneKeepingEnv, LaneKeepingParams
from sfilter.filters import SafetyFilterWrapper, build_abstraction_filter
from sfilter.filters.abstraction import DelayAwareFilter
from sfilter.real.comma import Twin, fit_twin
from sfilter.real.lane import RealLaneEnv, RealLaneParams, build_real_lane_filter

P = LaneKeepingParams()


@pytest.fixture(scope="module")
def filt():
    return build_abstraction_filter(300_000, cells=(80, 60, 8), seed=0)


class Delay(gym.Wrapper):
    def __init__(self, env, k):
        super().__init__(env)
        self.k, self.q = k, deque()

    def reset(self, **kw):
        self.q = deque([np.zeros(1)] * self.k)
        return self.env.reset(**kw)

    def step(self, a):
        self.q.append(np.asarray(a, float))
        return self.env.step(self.q.popleft())


def violations(env, policy, episodes=3):
    n = 0
    for ep in range(episodes):
        env.reset(seed=ep)
        while True:
            _, _, term, trunc, info = env.step(policy(env))
            if term or trunc:
                n += info["violation"]
                break
    return n


def bang_bang(env):
    return [P.delta_max if (env.unwrapped.t // 30) % 2 else -P.delta_max]


def test_noise_aware_filter_survives_sensor_noise(filt):
    noise = np.array([0.03, 0.006, 0.0])
    rng = np.random.default_rng(0)

    def measured(e):
        return e.unwrapped.state + rng.uniform(-noise, noise)

    filt.state_error = noise
    try:
        env = SafetyFilterWrapper(LaneKeepingEnv(horizon=300), filt, state_fn=measured)
        assert violations(env, bang_bang) == 0
    finally:
        filt.state_error = None


def test_delay_aware_filter_survives_a_late_actuator(filt):
    env = SafetyFilterWrapper(Delay(LaneKeepingEnv(horizon=300), 1),
                              DelayAwareFilter(filt, 1))
    assert violations(env, lambda e: [P.delta_max]) == 0


def test_filter_box_is_more_cautious_than_a_point(filt):
    x = np.array([0.0, 0.0, 22.0])
    u_point = filt.filter(x, np.array([P.delta_max]))
    u_box = filt.filter_box(x - [0.2, 0.05, 0.0], x + [0.2, 0.05, 0.0],
                            np.array([P.delta_max]))
    assert u_box[0] <= u_point[0]


def test_end_on_correction_ends_the_episode_and_stays_safe(filt):
    env = make_env("lane", filt, end_on_correction=0.05)
    env.reset(seed=1)
    for _ in range(600):
        _, r, term, trunc, info = env.step(np.array([1.0]))
        assert env.unwrapped.in_spec()
        if term or trunc:
            break
    assert term and info["stopped"] and r == -50.0 and not info["violation"]
    assert env.rows[-1]["stopped"]
    assert isinstance(env.env.env, EndOnCorrection)


def synthetic_logs(n=60_000, seed=0):
    """Steps from a known car (G = 55 m, K = 6e-4) with bounded yaw-rate noise."""
    rng = np.random.default_rng(seed)
    v = rng.uniform(20, 28, n)
    s = np.clip(rng.normal(0, 5, n), -12, 12)
    r = v * np.deg2rad(s) / (55 * (1 + 6e-4 * v**2)) + rng.uniform(-0.01, 0.01, n)
    return v, s, r


def test_twin_fit_recovers_a_known_car():
    tw = fit_twin(*synthetic_logs())
    assert tw.G == pytest.approx(55, rel=0.05) and tw.K == pytest.approx(6e-4, abs=2e-4)


def test_real_lane_filter_from_logs_keeps_the_car_in_lane():
    v, s, r = synthetic_logs()
    p = RealLaneParams(rho=0.012)
    f, _ = build_real_lane_filter(v, s, r, p)   # coarser grids certify nothing
    assert f.coverage > 0.1
    tw = Twin(G=55.0, K=6e-4, sigma0=0.0, lag_steps=0, residuals=np.zeros(10, np.float32))
    env = SafetyFilterWrapper(RealLaneEnv(tw, p, horizon=300), f)
    assert violations(env, lambda e: [p.sigma_max]) == 0


def test_robust_table_matches_brute_force(filt):
    from sfilter.filters.robust import pair_table, robust_safety
    c = filt.controller
    ab = c.abstraction
    W, R, _ = robust_safety(ab, c.domain, [2, 1, 0], pair_table(ab))
    assert W.sum() > 0
    rng = np.random.default_rng(0)
    shape = ab.X.shape
    if not np.array_equal(W, c.domain):
        pytest.skip("robust set smaller than the nominal one on this coarse grid")
    A = c.allowed
    for q in rng.choice(np.flatnonzero(W), 50, replace=False):
        idx = np.array(np.unravel_index(q, shape))
        lo, hi = np.maximum(idx - [2, 1, 0], 0), np.minimum(idx + [2, 1, 0], np.array(shape) - 1)
        g = np.meshgrid(*[np.arange(a, b + 1) for a, b in zip(lo, hi, strict=True)],
                        indexing="ij")
        qs = np.ravel_multi_index(tuple(x.ravel() for x in g), shape)
        qs = qs[W[qs]]
        assert np.array_equal(A[qs].all(0), R[q])


def test_robust_filter_with_noise_and_delay_never_crashes(filt):
    from sfilter.filters.robust import RobustFilter, delay_radius, pair_table, robust_safety
    c = filt.controller
    ab = c.abstraction
    tab = pair_table(ab)
    W, R, _ = robust_safety(ab, c.domain, [3, 2, 0], tab)
    noise = np.array([0.03, 0.006, 0.0])
    rng = np.random.default_rng(1)
    env = SafetyFilterWrapper(LaneKeepingEnv(horizon=300), RobustFilter(ab, W, R, [3, 2, 0]),
                              state_fn=lambda e: e.unwrapped.state
                              + rng.uniform(-noise, noise))
    assert violations(env, bang_bang) == 0
    r = delay_radius(ab, 1, tab)[-1]
    W, R, _ = robust_safety(ab, c.domain, r, tab)
    if W.sum() == 0:
        pytest.skip("nothing certified with this delay on the coarse test grid")
    env = SafetyFilterWrapper(Delay(LaneKeepingEnv(horizon=300), 1),
                              RobustFilter(ab, W, R, r, delay=1, table=tab))
    assert violations(env, lambda e: [P.delta_max]) == 0
