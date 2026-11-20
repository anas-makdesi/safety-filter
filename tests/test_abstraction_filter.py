import numpy as np
import pytest

from sfilter.envs import LaneKeepingEnv, LaneKeepingParams
from sfilter.filters import SafetyFilter, SafetyFilterWrapper, build_abstraction_filter

P = LaneKeepingParams()


@pytest.fixture(scope="module")
def filt():
    return build_abstraction_filter(300_000, cells=(80, 60, 8), seed=0)


def run(env, policy, episodes=4):
    violations = 0
    for ep in range(episodes):
        obs, _ = env.reset(seed=ep)
        while True:
            obs, _, term, trunc, info = env.step(policy(env, obs))
            if term or trunc:
                violations += info["violation"]
                break
    return violations


def policies(seed=0):
    rng = np.random.default_rng(seed)
    return {
        "random": lambda env, o: rng.uniform(-P.delta_max, P.delta_max, 1),
        "bang-bang": lambda env, o: [P.delta_max if (env.unwrapped.t // 30) % 2 else -P.delta_max],
        "full-left": lambda env, o: [P.delta_max],
    }


def test_is_a_safety_filter(filt):
    assert isinstance(filt, SafetyFilter)
    assert 0.1 < filt.coverage < 1.0


@pytest.mark.parametrize("name", ["random", "bang-bang", "full-left"])
def test_never_leaves_the_lane_with_filter(filt, name):
    env = SafetyFilterWrapper(LaneKeepingEnv(horizon=300), filt)
    assert run(env, policies()[name]) == 0


def test_same_agents_leave_the_lane_without_filter():
    for name in ("random", "full-left"):
        assert run(LaneKeepingEnv(horizon=300), policies()[name]) > 0


def test_safe_actions_pass_unchanged_and_outside_is_none(filt):
    x = np.array([0.0, 0.0, 22.0])
    assert filt.contains(x)
    u = filt.filter(x, np.array([0.0]))
    assert np.array_equal(u, [0.0])
    assert filt.filter(np.array([0.84, 0.19, 30.0]), np.array([0.0])) is None


def test_wrapper_reports_corrections(filt):
    env = SafetyFilterWrapper(LaneKeepingEnv(horizon=100), filt)
    env.reset(seed=3)
    n = 0
    for _ in range(100):
        _, _, term, trunc, info = env.step([P.delta_max])
        n += info["filtered"]
        assert info["correction"] == pytest.approx(abs(info["safe_action"][0] - P.delta_max))
        if term or trunc:
            break
    assert n > 0 and env.n_filtered == n
