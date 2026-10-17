"""A modern interface to DeepGroebner's optimized environment."""

from numbers import Integral

import numpy as np

from deepgroebner.wrapped import CLeadMonomialsEnv


class BuchbergerEnv:
    """Translate the Cython interface without changing algebra or row order.

    Parameters
    ----------
    ideal_dist : str
        Distribution string understood by the optimized DeepGroebner generator.
    k : int
        Number of leading monomials per polynomial in each pair row.
    elimination : str
        Only ``'gebauermoeller'`` is supported by the optimized environment.
    rewards : str
        Only ``'additions'`` is supported (reward is -1 minus reduction steps).
    sort_input, sort_reducers : bool
        Forwarded unchanged to the optimized environment.
    max_episode_steps : int or None
        Optional adapter time limit. Natural termination takes precedence.

    Notes
    -----
    Observations are the original int32 NumPy arrays, including terminal arrays
    of shape (0, d). Row i always denotes underlying action i. Reset resamples
    ideals with no initial pairs, as in DeepGroebner. This is a Gymnasium-like
    interface, not a Gymnasium Env subclass. Seeding affects the ideal generator,
    not PyTorch sampling. Fixed-seed streams need not agree across C++ platforms.
    """

    def __init__(self, ideal_dist: str = "3-20-10-weighted", *, k: int = 1,
                 elimination: str = "gebauermoeller", rewards: str = "additions",
                 sort_input: bool = False, sort_reducers: bool = True,
                 max_episode_steps: int | None = None) -> None:
        if elimination != "gebauermoeller" or rewards != "additions":
            raise ValueError("CLeadMonomialsEnv supports only gebauermoeller and additions")
        if isinstance(k, bool) or not isinstance(k, Integral) or k < 1:
            raise ValueError("k must be a positive integer")
        if max_episode_steps is not None and (
            isinstance(max_episode_steps, bool)
            or not isinstance(max_episode_steps, Integral) or max_episode_steps < 1
        ):
            raise ValueError("max_episode_steps must be a positive integer or None")
        self._env = CLeadMonomialsEnv(ideal_dist, sort_input=sort_input,
                                     sort_reducers=sort_reducers, k=k)
        self.max_episode_steps = max_episode_steps
        self._steps = 0
        self._num_actions = 0
        self._active = False

    def reset(self, *, seed: int | None = None) -> tuple[np.ndarray, dict]:
        """Start an episode; None continues the generator's existing RNG stream."""
        if seed is not None:
            if (isinstance(seed, bool) or not isinstance(seed, Integral)
                    or not 0 <= seed <= np.iinfo(np.int32).max):
                raise ValueError("seed must be an integer in [0, 2**31 - 1]")
            self._env.seed(int(seed))
        observation = self._env.reset()
        self._steps = 0
        self._num_actions = len(observation)
        self._active = self._num_actions > 0
        return observation, {"num_actions": self._num_actions}

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict]:
        """Apply the exact row index; reject unsafe native out-of-range accesses."""
        if not self._active:
            raise RuntimeError("reset is required before stepping a new or finished episode")
        if (isinstance(action, bool) or not isinstance(action, Integral)
                or not 0 <= action < self._num_actions):
            raise ValueError("action must be an integer index of an available pair")
        observation, reward, done, info = self._env.step(int(action))
        self._steps += 1
        self._num_actions = len(observation)
        terminated = bool(done)
        truncated = (not terminated and self.max_episode_steps is not None
                     and self._steps >= self.max_episode_steps)
        self._active = not (terminated or truncated)
        return observation, reward, terminated, truncated, {**info, "num_actions": self._num_actions}

    def value(self, strategy: str = "degree", gamma: float = 0.99) -> float:
        """Estimate remaining discounted return using the native rollout heuristic.

        Random/sample strategies use the native RNG behavior, which is not fully
        controlled by reset(seed). This is not a learned value network.
        """
        if strategy not in {"first", "degree", "normal", "sugar", "random", "sample"}:
            raise ValueError("unsupported native value strategy")
        if not np.isfinite(gamma) or not 0 <= gamma <= 1:
            raise ValueError("gamma must be finite and in [0, 1]")
        if self._num_actions == 0 and self._steps == 0:
            raise RuntimeError("reset is required before value")
        return self._env.value(strategy=strategy, gamma=gamma)
