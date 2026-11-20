"""Reinforcement learning on the two benchmarks, with or without a safety filter.

The agents come from Stable-Baselines3 (SAC and PPO). From the agent's point of view the
filter is part of the environment: the agent proposes an action, the filter may change
it, and the agent is never told directly (it only sees the next observation and the
reward). So the same training code runs with and without the filter.

    env = make_env("lane", filt=None)        # plain environment
    env = make_env("lane", filt=my_filter)   # same, behind the safety filter

:class:`EpisodeLog` records what happens during training, episode by episode (return,
crash or not, how often the filter stepped in, time spent in the filter).
:func:`train` runs one training run and returns the model and the log.
:func:`evaluate` runs a trained agent on fixed held-out episodes.
"""

from __future__ import annotations

import time
from typing import Optional

import gymnasium as gym
import numpy as np
from gymnasium.wrappers import RescaleAction

from ..envs import LaneKeepingEnv
from ..envs.quadrotor import QuadrotorEnv
from ..envs.roads import evaluation_roads
from ..filters.base import SafetyFilter, SafetyFilterWrapper

__all__ = ["BENCHMARKS", "TimedFilter", "EpisodeLog", "EndOnCorrection", "make_env", "train",
           "evaluate", "eval_options"]

BENCHMARKS = ("lane", "quad")


class TimedFilter:
    """A filter that also measures the time spent in its own calls."""

    def __init__(self, filt: SafetyFilter):
        self.filt = filt
        self.seconds = 0.0
        self.calls = 0

    def contains(self, x) -> bool:
        return self.filt.contains(x)

    def filter(self, x, u):
        t0 = time.perf_counter()
        out = self.filt.filter(x, u)
        self.seconds += time.perf_counter() - t0
        self.calls += 1
        return out


class EpisodeLog(gym.Wrapper):
    """Records one row per finished episode in :attr:`rows`:

    ``step`` (environment steps so far), ``ret`` (sum of the task rewards, without any
    intervention penalty, so that runs with and without a penalty are comparable),
    ``length``, ``violation`` (the episode ended by leaving the specification),
    ``filtered`` (steps where the filter changed the action), ``correction`` (sum of the
    changes, physical units), ``stopped`` (the episode was ended by
    :class:`EndOnCorrection`).
    """

    def __init__(self, env, penalty_rate: float = 0.0):
        super().__init__(env)
        self.penalty_rate = float(penalty_rate)
        # a change below one millionth of the input range is rounding, not an intervention
        # (e.g. a float32 action a hair outside the input limits)
        space = env.unwrapped.action_space
        self.tol = 1e-6 * float(np.linalg.norm(space.high - space.low))
        self.rows: list[dict] = []
        self.total_steps = 0
        self._new()

    def _new(self):
        self._ret, self._len, self._filtered, self._corr = 0.0, 0, 0, 0.0

    def reset(self, **kwargs):
        self._new()
        return self.env.reset(**kwargs)

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.total_steps += 1
        self._ret += float(reward) + self.penalty_rate * float(info.get("correction", 0.0))
        self._len += 1
        corr = float(info.get("correction", 0.0))
        self._filtered += int(corr > self.tol)
        self._corr += corr
        if terminated or truncated:
            self.rows.append(dict(step=self.total_steps, ret=self._ret, length=self._len,
                                  violation=bool(info.get("violation", False)),
                                  filtered=self._filtered, correction=self._corr,
                                  stopped=bool(info.get("stopped", False))))
        return obs, reward, terminated, truncated, info

    def arrays(self) -> dict:
        keys = ("step", "ret", "length", "violation", "filtered", "correction", "stopped")
        return {k: np.array([r[k] for r in self.rows]) for k in keys}


class EndOnCorrection(gym.Wrapper):
    """Training signal against relying on the filter: a correction larger than
    ``threshold`` (a share of the input range) ends the episode with the crash reward.
    The filter's safe input is still applied, so the system stays safe; only the agent
    is told "you would have crashed". ``info["stopped"]`` marks these episode ends."""

    def __init__(self, env, threshold: float, reward: float = -50.0):
        super().__init__(env)
        space = env.unwrapped.action_space
        self.limit = threshold * float(np.linalg.norm(space.high - space.low))
        self.reward = float(reward)

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        stop = float(info.get("correction", 0.0)) > self.limit
        if stop and not terminated:
            reward, terminated, truncated = self.reward, True, False
        return obs, reward, terminated, truncated, dict(info, stopped=stop)


def make_env(benchmark: str, filt: Optional[SafetyFilter] = None, penalty: float = 0.0,
             horizon: int = 600, end_on_correction: Optional[float] = None) -> EpisodeLog:
    """The benchmark environment with actions in [-1, 1], optionally behind ``filt``.

    Order of the layers, from the agent down to the simulator:
    EpisodeLog -> RescaleAction ([-1, 1] to physical units) -> SafetyFilterWrapper ->
    environment. The filter therefore works in physical units on the true state.
    ``penalty``: reward penalty per unit of correction, in units of the action range
    (so 1.0 means: a correction across the whole range costs 1 per step).
    ``end_on_correction``: if set, a correction larger than this share of the input
    range ends the episode with the crash reward (:class:`EndOnCorrection`).
    """
    if benchmark == "lane":
        env = LaneKeepingEnv(horizon=horizon)
    elif benchmark == "quad":
        env = QuadrotorEnv(horizon=horizon)
    else:
        raise ValueError(f"unknown benchmark {benchmark!r}")
    rate = 0.0
    if filt is not None:
        rate = penalty / float(np.linalg.norm(env.action_space.high - env.action_space.low))
        env = SafetyFilterWrapper(env, filt, penalty=rate)
        if end_on_correction is not None:
            env = EndOnCorrection(env, end_on_correction)
    return EpisodeLog(RescaleAction(env, -1.0, 1.0), rate)


def train(algo: str, benchmark: str, filt: Optional[SafetyFilter], steps: int, seed: int,
          penalty: float = 0.0, end_on_correction: Optional[float] = None, **algo_kwargs):
    """One training run. Returns (model, env) where ``env.rows`` is the episode log."""
    import torch
    from stable_baselines3 import PPO, SAC

    torch.set_num_threads(1)
    env = make_env(benchmark, filt, penalty, end_on_correction=end_on_correction)
    env.reset(seed=seed)
    if algo == "sac":
        kw = dict(learning_starts=2000, buffer_size=steps, batch_size=256)
        kw.update(algo_kwargs)
        model = SAC("MlpPolicy", env, seed=seed, device="cpu", verbose=0, **kw)
    elif algo == "ppo":
        kw = dict(n_steps=2048, batch_size=128, n_epochs=10, gamma=0.99)
        kw.update(algo_kwargs)
        model = PPO("MlpPolicy", env, seed=seed, device="cpu", verbose=0, **kw)
    else:
        raise ValueError(f"unknown algorithm {algo!r}")
    model.learn(total_timesteps=steps)
    return model, env


def eval_options(benchmark: str, n: int = 20) -> list[dict]:
    """The fixed held-out evaluation episodes: (seed, reset options) for each.

    Lane keeping: the 20 evaluation roads, speeds spread over the range, starting at the
    lane centre. Quadrotor: 20 fixed target trajectories (seeds never used in training).
    """
    if benchmark == "lane":
        roads = evaluation_roads(n)
        v = np.linspace(15.0, 30.0, n)
        return [dict(seed=900_000 + i, options=dict(road=roads[i], v=float(v[i]),
                                                    state=(0.0, 0.0))) for i in range(n)]
    return [dict(seed=900_000 + i, options=dict(state=(0.0, 0.0, 0.0, 1.5, 0.0)))
            for i in range(n)]


def evaluate(policy, benchmark: str, filt: Optional[SafetyFilter] = None, n: int = 20,
             horizon: int = 600) -> dict:
    """Run ``policy(obs) -> action in [-1, 1]`` on the held-out episodes, without
    exploration noise. Returns per-episode arrays plus the minimum margin to the edge of
    the specification (``margin``; negative = left it)."""
    env = make_env(benchmark, filt, horizon=horizon)
    base = env.unwrapped
    margins = []
    for ep in eval_options(benchmark, n):
        obs, _ = env.reset(seed=ep["seed"], options=ep["options"])
        done, m = False, np.inf
        while not done:
            obs, _, term, trunc, _ = env.step(policy(obs))
            m = min(m, _margin(base, benchmark))
            done = term or trunc
        margins.append(m)
    out = env.arrays()
    out["margin"] = np.array(margins)
    return out


def _margin(base, benchmark: str) -> float:
    """Distance from the position limits (metres): lane edge, or walls / floor / ceiling."""
    x, p = base.state, base.p
    if benchmark == "lane":
        return float(p.e_y_max - abs(x[0]))
    return float(min(p.y_max - abs(x[0]), x[3] - p.z_min, p.z_max - x[3]))
