from __future__ import annotations

from pathlib import Path

import gymnasium as gym
import numpy as np
import pytest

import boxoban  # noqa: F401


REGISTERED_ENV_IDS = [
    "Boxoban-medium-train-v0",
    "Boxoban-medium-valid-v0",
    "Boxoban-unfiltered-train-v0",
    "Boxoban-unfiltered-valid-v0",
    "Boxoban-unfiltered-test-v0",
    "Boxoban-hard-v0",
]


@pytest.mark.parametrize("env_id", REGISTERED_ENV_IDS)
def test_registered_envs_can_be_created(env_id: str, mini_boxoban_root: Path) -> None:
    env = gym.make(
        env_id,
        level_root=str(mini_boxoban_root),
        disable_env_checker=True,
    )
    obs, _ = env.reset(seed=0)
    assert obs.shape == (80, 80, 3)
    env.close()


@pytest.mark.parametrize("mode", ["sync", "async"])
def test_make_vec_sync_async(mode: str, mini_boxoban_root: Path) -> None:
    kwargs = {
        "num_envs": 4,
        "vectorization_mode": mode,
        "level_root": str(mini_boxoban_root),
    }

    try:
        vec_env = gym.make_vec(
            "Boxoban-medium-train-v0",
            disable_env_checker=True,
            **kwargs,
        )
    except TypeError:
        vec_env = gym.make_vec("Boxoban-medium-train-v0", **kwargs)

    obs, _ = vec_env.reset(seed=0)
    assert obs.shape == (4, 80, 80, 3)

    actions = np.array([0, 1, 2, 3], dtype=np.int64)
    obs, reward, terminated, truncated, _ = vec_env.step(actions)

    assert obs.shape == (4, 80, 80, 3)
    assert reward.shape == (4,)
    assert terminated.shape == (4,)
    assert truncated.shape == (4,)

    vec_env.close()
