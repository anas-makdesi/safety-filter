"""A common interface for safety filters, and a Gymnasium wrapper that applies one.

A safety filter has two methods:

* ``filter(x, u)``: the certified input closest to ``u`` at state ``x`` (``u`` itself when
  it is already safe), or None when ``x`` is outside the filter's certified set;
* ``contains(x)``: whether ``x`` is inside the certified set.

As long as the state starts inside the certified set and the assumptions behind the
filter hold, every input it returns keeps the state inside, forever.
"""

from __future__ import annotations

from typing import Callable, Optional, Protocol, runtime_checkable

import gymnasium as gym
import numpy as np

__all__ = ["SafetyFilter", "SafetyFilterWrapper"]


@runtime_checkable
class SafetyFilter(Protocol):
    def filter(self, x, u) -> Optional[np.ndarray]: ...

    def contains(self, x) -> bool: ...


class SafetyFilterWrapper(gym.Wrapper):
    """Puts a :class:`SafetyFilter` between an agent and an environment.

    Every action goes through ``filt.filter(x, action)`` with the environment's true state
    x (``state_fn(env)``, default ``env.unwrapped.state``). ``info`` gains
    ``proposed_action``, ``safe_action``, ``filtered`` and ``correction``; the counters
    :attr:`n_steps` and :attr:`n_filtered` run over the wrapper's lifetime.

    ``penalty``: reward penalty per unit of correction (default 0).
    ``reset_in_set``: redraw the initial state until it lies in the certified set
    (default True), so the guarantee holds from the first step.
    ``outside``: "pass" (default: apply the action unchanged) or "raise" when the state
    is outside the certified set.
    """

    def __init__(self, env, filt: SafetyFilter, state_fn: Optional[Callable] = None,
                 penalty: float = 0.0, reset_in_set: bool = True, outside: str = "pass"):
        super().__init__(env)
        if outside not in ("pass", "raise"):
            raise ValueError("outside must be 'pass' or 'raise'")
        self.filt = filt
        self.state_fn = state_fn or (lambda e: np.asarray(e.unwrapped.state, float))
        self.penalty = float(penalty)
        self.reset_in_set = bool(reset_in_set)
        self.outside = outside
        self.n_steps = 0
        self.n_filtered = 0

    def reset(self, **kwargs):
        if hasattr(self.filt, "reset"):          # filters with memory (DelayAwareFilter)
            self.filt.reset()
        obs, info = self.env.reset(**kwargs)
        if self.reset_in_set:
            kwargs.pop("seed", None)
            for _ in range(1000):
                if self.filt.contains(self.state_fn(self.env)):
                    break
                obs, info = self.env.reset(**kwargs)
            else:
                raise RuntimeError("no initial state in the certified set after 1000 resets")
        return obs, info

    def step(self, action):
        u = np.asarray(action, dtype=float).reshape(-1)
        s = self.filt.filter(self.state_fn(self.env), u)
        if s is None:
            if self.outside == "raise":
                raise RuntimeError("state outside the certified set")
            s = u
        s = np.asarray(s, float).reshape(-1)
        corr = float(np.linalg.norm(s - u))
        # float64 on purpose: rounding to float32 could move the input across the edge
        # of a certified input cell
        obs, reward, terminated, truncated, info = self.env.step(s)
        self.n_steps += 1
        self.n_filtered += int(corr > 0)
        if self.penalty:
            reward -= self.penalty * corr
        info = dict(info, proposed_action=u, safe_action=s, filtered=corr > 0,
                    correction=corr)
        return obs, reward, terminated, truncated, info
