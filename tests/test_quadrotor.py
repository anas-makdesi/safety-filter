import numpy as np
import pytest

from sfilter.envs.quadrotor import QuadrotorEnv, QuadrotorParams, quadrotor_step
from sfilter.filters import SafetyFilter, SafetyFilterWrapper
from sfilter.filters.quadrotor import build_quadrotor_filter

P = QuadrotorParams()


def test_assumptions_hold_on_samples():
    """Derivative bounds of each channel's increments and gust bounds."""
    rng = np.random.default_rng(0)
    n, h = 20000, 1e-6
    X = rng.uniform(P.lo, P.hi, (n, 5))
    U = rng.uniform(P.u_lo, P.u_hi, (n, 2))
    f0 = quadrotor_step(X, U, P) - X
    for (A_lo, A_hi, B_lo, B_hi, *_), sx, su in ((P.lat_bounds(), [0, 1, 2], [0]),
                                                 (P.vert_bounds(), [3, 4], [1])):
        for jj, j in enumerate(sx):
            dX = X.copy()
            dX[:, j] += h
            d = (quadrotor_step(dX, U, P) - dX - f0)[:, sx] / h
            assert np.all(d >= A_lo[:, jj] - 1e-6) and np.all(d <= A_hi[:, jj] + 1e-6)
        for jj, j in enumerate(su):
            dU = U.copy()
            dU[:, j] += h
            d = (quadrotor_step(X, dU, P) - X - f0)[:, sx] / h
            assert np.all(d >= B_lo[:, jj] - 1e-6) and np.all(d <= B_hi[:, jj] + 1e-6)
    W = rng.uniform(-P.wind, P.wind, (n, 2))
    w = quadrotor_step(X, U, P, W) - quadrotor_step(X, U, P)
    d_hi = np.concatenate([P.lat_bounds()[5], P.vert_bounds()[5]])
    assert np.all(np.abs(w) <= d_hi + 1e-12)


def test_env_ends_on_wall_hit():
    env = QuadrotorEnv()
    env.reset(seed=0)
    for _ in range(200):
        _, r, term, _, info = env.step([P.ay_max, 0.0])
        if term:
            break
    assert term and info["violation"] and r == -50.0


@pytest.fixture(scope="module")
def filt():
    return build_quadrotor_filter(300_000, lat_cells=(60, 200, 20), vert_cells=(60, 40))


def test_is_a_safety_filter(filt):
    assert isinstance(filt, SafetyFilter)
    assert all(0.1 < f.coverage < 1.0 for f in filt.filters)


@pytest.mark.parametrize("name", ["random", "chase", "full"])
def test_never_hits_the_walls_with_filter(filt, name):
    rng = np.random.default_rng(1)
    policy = {"random": lambda o: rng.uniform(P.u_lo, P.u_hi),
              "chase": lambda o: P.u_hi * np.sign(o[5:7]),
              "full": lambda o: P.u_hi}[name]
    env = SafetyFilterWrapper(QuadrotorEnv(horizon=300), filt)
    for ep in range(3):
        obs, _ = env.reset(seed=ep)
        while True:
            obs, _, term, trunc, info = env.step(policy(obs))
            assert not info["violation"]
            if term or trunc:
                break
