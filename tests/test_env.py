import numpy as np

from sfilter.envs import LaneKeepingEnv, LaneKeepingParams, lane_keeping_step, random_road
from sfilter.envs.roads import evaluation_roads, showcase_road

P = LaneKeepingParams()


def test_roads_respect_curvature_bound():
    rng = np.random.default_rng(0)
    for road in [random_road(rng) for _ in range(20)] + evaluation_roads(5) + [showcase_road()]:
        s = np.linspace(0, road.length, 5000)
        assert np.all(np.abs(road.curvature(s)) <= P.kappa_max + 1e-15)


def test_evaluation_roads_are_fixed():
    a, b = evaluation_roads(3), evaluation_roads(3)
    assert all(np.array_equal(r.kappa, q.kappa) for r, q in zip(a, b, strict=True))


def test_step_matches_equations():
    x = np.array([[0.1, 0.05, 20.0]])
    y = lane_keeping_step(x, 0.02, 0.001, P)[0]
    assert np.isclose(y[0], 0.1 + P.dt * 20 * np.sin(0.05))
    assert np.isclose(y[1], 0.05 + P.dt * 20 * (np.tan(0.02) / P.wheelbase - 0.001))
    assert y[2] == 20.0


def test_assumptions_hold_on_samples():
    """Monotonicity, derivative bounds and disturbance bounds that the filters assume."""
    rng = np.random.default_rng(1)
    n = 20000
    X = rng.uniform(P.safe_lo, P.safe_hi, (n, 3))
    U = rng.uniform(-P.delta_max, P.delta_max, n)
    A_lo, A_hi, B_lo, B_hi = P.derivative_bounds()
    h = 1e-6
    f0 = lane_keeping_step(X, U, 0.0, P) - X
    for j in range(3):
        dX = X.copy()
        dX[:, j] += h
        d = (lane_keeping_step(dX, U, 0.0, P) - dX - f0) / h
        assert np.all(d >= A_lo[:, j] - 1e-6) and np.all(d <= A_hi[:, j] + 1e-6)
    d = (lane_keeping_step(X, U + h, 0.0, P) - X - f0) / h
    assert np.all(d >= B_lo[:, 0] - 1e-6) and np.all(d <= B_hi[:, 0] + 1e-6)
    # monotone in e_y, e_psi, delta; decreasing in curvature
    assert np.all(A_lo[:2, :2] >= 0) and np.all(B_lo >= 0)
    k = rng.uniform(-P.kappa_max, P.kappa_max, n)
    w_lo, w_hi = P.disturbance_bounds()
    w = lane_keeping_step(X, U, k, P) - lane_keeping_step(X, U, 0.0, P)
    assert np.all(w >= w_lo - 1e-12) and np.all(w <= w_hi + 1e-12)


def test_episode_ends_on_departure_with_penalty():
    env = LaneKeepingEnv()
    env.reset(seed=0, options={"state": (0.8, 0.1), "v": 30.0})
    for _ in range(100):
        _, r, term, trunc, info = env.step([P.delta_max])
        if term:
            break
    # full left steering cannot stop it in time from there
    assert term and info["violation"] and r == -50.0


def test_reward_in_range_and_truncation():
    env = LaneKeepingEnv(horizon=50)
    env.reset(seed=1, options={"state": (0.0, 0.0), "v": 20.0, "road": showcase_road()})
    for _ in range(50):
        _, r, term, trunc, _ = env.step([0.0])
        assert 0.0 <= r <= 1.0 and not term
    assert trunc
