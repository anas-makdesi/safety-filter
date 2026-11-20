"""Controllers and evaluation: logging, DeePC on a known system, MPC, a tiny RL run."""

import numpy as np
import pytest

from sfilter.controllers.rl import TimedFilter, eval_options, make_env
from sfilter.envs import LaneKeepingParams
from sfilter.evaluation import evaluate, summarize
from sfilter.filters import build_abstraction_filter

P = LaneKeepingParams()


@pytest.fixture(scope="module")
def filt():
    return build_abstraction_filter(300_000, cells=(80, 60, 8), seed=0)


def test_episode_log_counts_crashes_and_filter_steps(filt):
    rng = np.random.default_rng(0)
    for f, crashes_expected in ((None, True), (TimedFilter(filt), False)):
        env = make_env("lane", f, horizon=200)
        env.reset(seed=0)
        for _ in range(3000):
            _, _, term, trunc, _ = env.step(rng.choice([-1.0, 1.0], 1))
            if term or trunc:
                env.reset()
        a = env.arrays()
        assert (a["violation"].sum() > 0) == crashes_expected
        assert a["length"].sum() <= 3000
        if f is not None:
            assert a["filtered"].sum() > 0 and f.calls == 3000 and f.seconds > 0


def test_penalty_is_not_counted_in_the_logged_return(filt):
    rets = []
    for penalty in (0.0, 5.0):
        env = make_env("lane", filt, penalty=penalty, horizon=100)
        env.reset(seed=3)
        rng = np.random.default_rng(1)
        done = False
        while not done:
            _, _, term, trunc, _ = env.step(rng.choice([-1.0, 1.0], 1))
            done = term or trunc
        rets.append(env.arrays()["ret"][0])
    assert rets[0] == pytest.approx(rets[1])


def test_held_out_episodes_are_fixed():
    a, b = eval_options("lane"), eval_options("lane")
    assert len(a) == 20 and all(x["seed"] == y["seed"] for x, y in zip(a, b, strict=True))
    assert np.array_equal(a[3]["options"]["road"].kappa, b[3]["options"]["road"].kappa)


def test_deepc_predicts_a_linear_system_exactly():
    cp = pytest.importorskip("cvxpy")  # noqa: F841
    from sfilter.controllers.deepc import DeePC, hankel

    rng = np.random.default_rng(0)
    A, B = np.array([[1.0, 0.1], [0.0, 1.0]]), np.array([0.005, 0.1])   # double integrator
    x, U, Y = np.zeros(2), [], []
    for _ in range(120):
        u = rng.uniform(-1, 1) - 0.5 * x[0] - 0.5 * x[1]
        x = A @ x + B * u
        U.append([u])
        Y.append(x.copy())
    assert hankel(np.array(U), 5).shape == (5, 116)
    c = DeePC(np.array(U), np.array(Y), t_ini=2, horizon=15, q=[1.0, 0.1], r=1e-3,
              u_lo=[-1.0], u_hi=[1.0], lam_g=1e-6)
    x = np.array([0.5, 0.0])
    c.start([0.0], x)
    for _ in range(60):                     # drive the position to the reference 0
        u = c.solve(np.zeros(2))
        x = A @ x + B * u[0]
        c.observe(u, x)
    assert abs(x[0]) < 0.02 and abs(x[1]) < 0.05


def test_mpc_keeps_the_lane_and_the_filter_leaves_it_alone(filt):
    pytest.importorskip("cvxpy")
    from sfilter.controllers.mpc import LaneMPC

    eps = eval_options("lane")[8:10]
    free = summarize(evaluate(LaneMPC(P), "lane", None, horizon=150, episodes=eps))
    safe = summarize(evaluate(LaneMPC(P), "lane", filt, horizon=150, episodes=eps))
    assert free["crashes"] == 0 and free["ret"] > 149
    assert safe["crashes"] == 0 and safe["ret"] == pytest.approx(free["ret"], abs=0.5)


def test_rl_training_behind_the_filter_never_crashes(filt):
    pytest.importorskip("stable_baselines3")
    from sfilter.controllers.rl import train

    _, env = train("ppo", "lane", filt, steps=1024, seed=0, n_steps=512)
    a = env.arrays()
    assert a["violation"].sum() == 0 and len(a["ret"]) > 0
    _, env = train("ppo", "lane", None, steps=1024, seed=0, n_steps=512)
    assert env.arrays()["violation"].sum() > 0
