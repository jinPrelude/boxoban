from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from boxoban.colors import PLAYER
from boxoban.env import BoxobanEnv


_DEFAULT_OBS_SIZE = 80
_PIXEL_SIZE = _DEFAULT_OBS_SIZE // 10


def _player_position(obs: np.ndarray) -> tuple[int, int]:
    """Return the grid-level (y, x) position of the player."""
    hits = np.argwhere(np.all(obs == PLAYER, axis=-1))
    assert hits.shape[0] >= 1
    return int(hits[0, 0]) // _PIXEL_SIZE, int(hits[0, 1]) // _PIXEL_SIZE


def test_observation_space(mini_boxoban_root: Path) -> None:
    env = BoxobanEnv(level_set="medium", split="train", level_root=str(mini_boxoban_root))
    obs, info = env.reset(seed=0, options={"level_idx": 0})

    assert obs.shape == (_DEFAULT_OBS_SIZE, _DEFAULT_OBS_SIZE, 3)
    assert obs.dtype == np.uint8
    assert env.observation_space.contains(obs)
    assert info["level_idx"] == 0
    env.close()


def test_wall_collision_has_only_step_penalty(mini_boxoban_root: Path) -> None:
    env = BoxobanEnv(level_set="medium", split="train", level_root=str(mini_boxoban_root))
    obs, _ = env.reset(seed=0, options={"level_idx": 0})
    assert _player_position(obs) == (1, 1)

    obs, reward, terminated, truncated, info = env.step(0)

    assert _player_position(obs) == (1, 1)
    assert reward == pytest.approx(-0.1)
    assert not terminated
    assert not truncated
    assert info["boxes_on_target"] == 0
    assert info["steps"] == 1
    env.close()


def test_push_box_on_and_off_goal_rewards(mini_boxoban_root: Path) -> None:
    env = BoxobanEnv(level_set="medium", split="train", level_root=str(mini_boxoban_root))
    env.reset(seed=0, options={"level_idx": 1})

    _, reward_on, terminated, truncated, info = env.step(3)
    assert reward_on == pytest.approx(0.9)
    assert info["boxes_on_target"] == 1
    assert not terminated
    assert not truncated

    _, reward_off, terminated, truncated, info = env.step(3)
    assert reward_off == pytest.approx(-1.1)
    assert info["boxes_on_target"] == 0
    assert not terminated
    assert not truncated
    env.close()


def test_solve_and_truncate_flags(mini_boxoban_root: Path) -> None:
    solve_env = BoxobanEnv(level_set="hard", level_root=str(mini_boxoban_root), max_steps=120)
    solve_env.reset(seed=0, options={"level_idx": 0})

    _, reward, terminated, truncated, info = solve_env.step(0)
    assert reward == pytest.approx(10.9)
    assert terminated
    assert not truncated
    assert info["is_success"] is True
    solve_env.close()

    trunc_env = BoxobanEnv(
        level_set="medium",
        split="train",
        level_root=str(mini_boxoban_root),
        max_steps=1,
    )
    trunc_env.reset(seed=0, options={"level_idx": 0})
    _, _, terminated, truncated, _ = trunc_env.step(0)
    assert not terminated
    assert truncated
    trunc_env.close()


def test_reset_level_idx_and_seed_reproducibility(mini_boxoban_root: Path) -> None:
    env = BoxobanEnv(level_set="hard", level_root=str(mini_boxoban_root))
    _, info = env.reset(options={"level_idx": 1})
    assert info["level_idx"] == 1
    env.close()

    env1 = BoxobanEnv(level_set="medium", split="train", level_root=str(mini_boxoban_root))
    env2 = BoxobanEnv(level_set="medium", split="train", level_root=str(mini_boxoban_root))

    sequence1 = []
    sequence2 = []

    _, info = env1.reset(seed=123)
    sequence1.append(info["level_idx"])
    _, info = env2.reset(seed=123)
    sequence2.append(info["level_idx"])

    for _ in range(10):
        _, info = env1.reset()
        sequence1.append(info["level_idx"])
        _, info = env2.reset()
        sequence2.append(info["level_idx"])

    assert sequence1 == sequence2
    env1.close()
    env2.close()


def test_fixed_level_idx(mini_boxoban_root: Path) -> None:
    env = BoxobanEnv(
        level_set="medium",
        split="train",
        level_root=str(mini_boxoban_root),
        fixed_level_idx=1,
    )
    for _ in range(3):
        _, info = env.reset()
        assert info["level_idx"] == 1
    env.close()

    with pytest.raises(ValueError):
        BoxobanEnv(
            level_set="medium",
            split="train",
            level_root=str(mini_boxoban_root),
            fixed_level_idx=5,
        )


def test_noop_action_space(mini_boxoban_root: Path) -> None:
    env = BoxobanEnv(level_set="medium", split="train", level_root=str(mini_boxoban_root), noop_action=True)
    assert env.action_space.n == 5
    env.close()


def test_noop_action_does_not_move(mini_boxoban_root: Path) -> None:
    env = BoxobanEnv(level_set="medium", split="train", level_root=str(mini_boxoban_root), noop_action=True)
    obs, _ = env.reset(seed=0, options={"level_idx": 0})
    assert _player_position(obs) == (1, 1)

    obs2, reward, terminated, truncated, info = env.step(0)

    assert _player_position(obs2) == (1, 1)
    assert np.array_equal(obs, obs2)
    assert reward == pytest.approx(-0.1)
    assert not terminated
    assert not truncated
    assert info["steps"] == 1
    env.close()


def test_noop_movement_actions(mini_boxoban_root: Path) -> None:
    # Action 1 = up on wall_collision level (player at 1,1, wall above)
    env = BoxobanEnv(level_set="medium", split="train", level_root=str(mini_boxoban_root), noop_action=True)
    env.reset(seed=0, options={"level_idx": 0})

    _, reward, terminated, truncated, info = env.step(1)  # up -> wall
    assert reward == pytest.approx(-0.1)
    assert not terminated
    assert not truncated

    env.close()

    # Action 4 = right on push_on_off_goal level (pushes box onto goal)
    env2 = BoxobanEnv(level_set="medium", split="train", level_root=str(mini_boxoban_root), noop_action=True)
    env2.reset(seed=0, options={"level_idx": 1})

    _, reward_on, terminated, truncated, info = env2.step(4)  # right
    assert reward_on == pytest.approx(0.9)
    assert info["boxes_on_target"] == 1
    assert not terminated
    assert not truncated
    env2.close()


def test_noop_truncation(mini_boxoban_root: Path) -> None:
    env = BoxobanEnv(
        level_set="medium",
        split="train",
        level_root=str(mini_boxoban_root),
        max_steps=1,
        noop_action=True,
    )
    env.reset(seed=0, options={"level_idx": 0})
    _, _, terminated, truncated, _ = env.step(0)  # noop
    assert not terminated
    assert truncated
    env.close()


def test_noop_invalid_action(mini_boxoban_root: Path) -> None:
    env = BoxobanEnv(level_set="medium", split="train", level_root=str(mini_boxoban_root), noop_action=True)
    env.reset(seed=0, options={"level_idx": 0})

    with pytest.raises(ValueError):
        env.step(5)
    with pytest.raises(ValueError):
        env.step(-1)
    env.close()
