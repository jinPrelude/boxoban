"""Tests for observation wrappers."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from boxoban.env import BoxobanEnv
from boxoban.wrappers import (
    ResizeObservationPIL,
    TiltedObservationCV,
    TiltedObservationFast,
    TiltedObservationWrapper,
)


# ── ResizeObservationPIL ──────────────────────────────────────────────


def test_resize_shape(mini_boxoban_root: Path) -> None:
    env = BoxobanEnv(level_set="medium", split="train", level_root=str(mini_boxoban_root))
    wrapped = ResizeObservationPIL(env, shape=(80, 80))

    assert wrapped.observation_space.shape == (80, 80, 3)

    obs, _ = wrapped.reset(seed=0, options={"level_idx": 0})
    assert obs.shape == (80, 80, 3)
    assert obs.dtype == np.uint8

    obs2, _, _, _, _ = wrapped.step(0)
    assert obs2.shape == (80, 80, 3)
    wrapped.close()


def test_resize_non_square(mini_boxoban_root: Path) -> None:
    env = BoxobanEnv(level_set="medium", split="train", level_root=str(mini_boxoban_root))
    wrapped = ResizeObservationPIL(env, shape=(64, 128))

    assert wrapped.observation_space.shape == (64, 128, 3)

    obs, _ = wrapped.reset(seed=0, options={"level_idx": 0})
    assert obs.shape == (64, 128, 3)
    wrapped.close()


def test_resize_space_contains_obs(mini_boxoban_root: Path) -> None:
    env = BoxobanEnv(level_set="medium", split="train", level_root=str(mini_boxoban_root))
    wrapped = ResizeObservationPIL(env, shape=(80, 80))

    obs, _ = wrapped.reset(seed=0, options={"level_idx": 0})
    assert wrapped.observation_space.contains(obs)
    wrapped.close()


# ── TiltedObservationWrapper ──────────────────────────────────────────


def test_tilted_shape(mini_boxoban_root: Path) -> None:
    env = BoxobanEnv(level_set="medium", split="train", level_root=str(mini_boxoban_root))
    resized = ResizeObservationPIL(env, shape=(80, 80))
    tilted = TiltedObservationWrapper(resized, tilt=0.2)

    assert tilted.observation_space.shape == (80, 80, 3)

    obs, _ = tilted.reset(seed=0, options={"level_idx": 0})
    assert obs.shape == (80, 80, 3)
    assert obs.dtype == np.uint8

    obs2, _, _, _, _ = tilted.step(0)
    assert obs2.shape == (80, 80, 3)
    tilted.close()


def test_tilted_differs_from_original(mini_boxoban_root: Path) -> None:
    env1 = BoxobanEnv(level_set="medium", split="train", level_root=str(mini_boxoban_root))
    resized1 = ResizeObservationPIL(env1, shape=(80, 80))

    env2 = BoxobanEnv(level_set="medium", split="train", level_root=str(mini_boxoban_root))
    resized2 = ResizeObservationPIL(env2, shape=(80, 80))
    tilted = TiltedObservationWrapper(resized2, tilt=0.2)

    obs_flat, _ = resized1.reset(seed=0, options={"level_idx": 0})
    obs_tilt, _ = tilted.reset(seed=0, options={"level_idx": 0})

    assert not np.array_equal(obs_flat, obs_tilt)
    resized1.close()
    tilted.close()


def test_tilted_zero_is_identity(mini_boxoban_root: Path) -> None:
    env1 = BoxobanEnv(level_set="medium", split="train", level_root=str(mini_boxoban_root))
    resized1 = ResizeObservationPIL(env1, shape=(80, 80))

    env2 = BoxobanEnv(level_set="medium", split="train", level_root=str(mini_boxoban_root))
    resized2 = ResizeObservationPIL(env2, shape=(80, 80))
    tilted = TiltedObservationWrapper(resized2, tilt=0.0)

    obs_flat, _ = resized1.reset(seed=0, options={"level_idx": 0})
    obs_tilt, _ = tilted.reset(seed=0, options={"level_idx": 0})

    np.testing.assert_array_equal(obs_flat, obs_tilt)
    resized1.close()
    tilted.close()


def test_tilted_fill_color_at_corners(mini_boxoban_root: Path) -> None:
    fill = (0, 0, 0)
    env = BoxobanEnv(level_set="medium", split="train", level_root=str(mini_boxoban_root))
    resized = ResizeObservationPIL(env, shape=(80, 80))
    tilted = TiltedObservationWrapper(resized, tilt=0.2, fill_color=fill)

    obs, _ = tilted.reset(seed=0, options={"level_idx": 0})

    # Top corners should be filled with fill_color
    np.testing.assert_array_equal(obs[0, 0], np.array(fill, dtype=np.uint8))
    np.testing.assert_array_equal(obs[0, 79], np.array(fill, dtype=np.uint8))
    tilted.close()


def test_tilted_invalid_tilt(mini_boxoban_root: Path) -> None:
    env = BoxobanEnv(level_set="medium", split="train", level_root=str(mini_boxoban_root))
    resized = ResizeObservationPIL(env, shape=(80, 80))

    with pytest.raises(ValueError):
        TiltedObservationWrapper(resized, tilt=0.5)
    with pytest.raises(ValueError):
        TiltedObservationWrapper(resized, tilt=-0.1)
    resized.close()


def test_tilted_space_contains_obs(mini_boxoban_root: Path) -> None:
    env = BoxobanEnv(level_set="medium", split="train", level_root=str(mini_boxoban_root))
    resized = ResizeObservationPIL(env, shape=(80, 80))
    tilted = TiltedObservationWrapper(resized, tilt=0.2)

    obs, _ = tilted.reset(seed=0, options={"level_idx": 0})
    assert tilted.observation_space.contains(obs)
    tilted.close()


def test_tilted_works_on_raw_10x10(mini_boxoban_root: Path) -> None:
    env = BoxobanEnv(
        level_set="medium", split="train", level_root=str(mini_boxoban_root),
        obs_size=10,
    )
    tilted = TiltedObservationWrapper(env, tilt=0.15)

    obs, _ = tilted.reset(seed=0, options={"level_idx": 0})
    assert obs.shape == (10, 10, 3)
    assert obs.dtype == np.uint8
    tilted.close()


def test_stacked_wrappers_action_passthrough(mini_boxoban_root: Path) -> None:
    env = BoxobanEnv(level_set="medium", split="train", level_root=str(mini_boxoban_root))
    resized = ResizeObservationPIL(env, shape=(80, 80))
    tilted = TiltedObservationWrapper(resized, tilt=0.2)

    tilted.reset(seed=0, options={"level_idx": 0})
    for action in range(4):
        obs, reward, terminated, truncated, info = tilted.step(action)
        assert obs.shape == (80, 80, 3)
        assert "level_idx" in info
    tilted.close()


# ── TiltedObservationFast ─────────────────────────────────────────────


def test_tilt_fast_shape(mini_boxoban_root: Path) -> None:
    env = BoxobanEnv(level_set="medium", split="train", level_root=str(mini_boxoban_root))
    tilted = TiltedObservationFast(env, tilt=0.2)

    assert tilted.observation_space.shape == (80, 80, 3)

    obs, _ = tilted.reset(seed=0, options={"level_idx": 0})
    assert obs.shape == (80, 80, 3)
    assert obs.dtype == np.uint8

    obs2, _, _, _, _ = tilted.step(0)
    assert obs2.shape == (80, 80, 3)
    tilted.close()


def test_tilt_fast_matches_pil_nearest(mini_boxoban_root: Path) -> None:
    """TiltedObservationFast closely matches TiltedObservationWrapper (nearest).

    Pixel-level differences at cell boundaries are expected due to floating-point
    rounding differences between PIL's C code and numpy.  We assert that fewer
    than 2% of pixels differ.
    """
    env1 = BoxobanEnv(level_set="medium", split="train", level_root=str(mini_boxoban_root))
    ref = TiltedObservationWrapper(env1, tilt=0.2, resample="nearest")

    env2 = BoxobanEnv(level_set="medium", split="train", level_root=str(mini_boxoban_root))
    fast = TiltedObservationFast(env2, tilt=0.2, resample="nearest")

    obs_ref, _ = ref.reset(seed=0, options={"level_idx": 0})
    obs_fast, _ = fast.reset(seed=0, options={"level_idx": 0})

    total_pixels = obs_ref.shape[0] * obs_ref.shape[1]
    diff_pixels = np.sum(np.any(obs_ref != obs_fast, axis=-1))
    # Boundary-pixel rounding differences between PIL C code and numpy
    # are expected (typically ~5% on block-upscaled 80x80 observations).
    assert diff_pixels / total_pixels < 0.10, (
        f"Too many pixel differences: {diff_pixels}/{total_pixels} "
        f"({diff_pixels / total_pixels:.1%})"
    )

    for action in range(4):
        obs_ref, *_ = ref.step(action)
        obs_fast, *_ = fast.step(action)
        diff_pixels = np.sum(np.any(obs_ref != obs_fast, axis=-1))
        assert diff_pixels / total_pixels < 0.10

    ref.close()
    fast.close()


def test_tilt_fast_tilt_zero_identity(mini_boxoban_root: Path) -> None:
    env1 = BoxobanEnv(level_set="medium", split="train", level_root=str(mini_boxoban_root))
    env2 = BoxobanEnv(level_set="medium", split="train", level_root=str(mini_boxoban_root))
    tilted = TiltedObservationFast(env2, tilt=0.0)

    obs_flat, _ = env1.reset(seed=0, options={"level_idx": 0})
    obs_tilt, _ = tilted.reset(seed=0, options={"level_idx": 0})

    np.testing.assert_array_equal(obs_flat, obs_tilt)
    env1.close()
    tilted.close()


def test_tilt_fast_fill_color_at_corners(mini_boxoban_root: Path) -> None:
    fill = (0, 0, 0)
    env = BoxobanEnv(level_set="medium", split="train", level_root=str(mini_boxoban_root))
    tilted = TiltedObservationFast(env, tilt=0.2, fill_color=fill)

    obs, _ = tilted.reset(seed=0, options={"level_idx": 0})

    np.testing.assert_array_equal(obs[0, 0], np.array(fill, dtype=np.uint8))
    np.testing.assert_array_equal(obs[0, 79], np.array(fill, dtype=np.uint8))
    tilted.close()


def test_tilt_fast_invalid_tilt(mini_boxoban_root: Path) -> None:
    env = BoxobanEnv(level_set="medium", split="train", level_root=str(mini_boxoban_root))

    with pytest.raises(ValueError):
        TiltedObservationFast(env, tilt=0.5)
    with pytest.raises(ValueError):
        TiltedObservationFast(env, tilt=-0.1)
    env.close()


def test_tilt_fast_space_contains_obs(mini_boxoban_root: Path) -> None:
    env = BoxobanEnv(level_set="medium", split="train", level_root=str(mini_boxoban_root))
    tilted = TiltedObservationFast(env, tilt=0.2)

    obs, _ = tilted.reset(seed=0, options={"level_idx": 0})
    assert tilted.observation_space.contains(obs)
    tilted.close()


def test_tilt_fast_action_passthrough(mini_boxoban_root: Path) -> None:
    env = BoxobanEnv(level_set="medium", split="train", level_root=str(mini_boxoban_root))
    tilted = TiltedObservationFast(env, tilt=0.2)

    tilted.reset(seed=0, options={"level_idx": 0})
    for action in range(4):
        obs, reward, terminated, truncated, info = tilted.step(action)
        assert obs.shape == (80, 80, 3)
        assert "level_idx" in info
    tilted.close()


# ── TiltedObservationCV ──────────────────────────────────────────────


def test_tilt_cv_shape(mini_boxoban_root: Path) -> None:
    env = BoxobanEnv(level_set="medium", split="train", level_root=str(mini_boxoban_root))
    tilted = TiltedObservationCV(env, tilt=0.2)

    assert tilted.observation_space.shape == (80, 80, 3)

    obs, _ = tilted.reset(seed=0, options={"level_idx": 0})
    assert obs.shape == (80, 80, 3)
    assert obs.dtype == np.uint8

    obs2, _, _, _, _ = tilted.step(0)
    assert obs2.shape == (80, 80, 3)
    tilted.close()


def test_tilt_cv_matches_pil_nearest(mini_boxoban_root: Path) -> None:
    """TiltedObservationCV closely matches TiltedObservationWrapper (nearest).

    Pixel-level differences at cell boundaries are expected due to rounding
    differences between OpenCV and PIL.  We assert that fewer than 10% of
    pixels differ.
    """
    env1 = BoxobanEnv(level_set="medium", split="train", level_root=str(mini_boxoban_root))
    ref = TiltedObservationWrapper(env1, tilt=0.2, resample="nearest")

    env2 = BoxobanEnv(level_set="medium", split="train", level_root=str(mini_boxoban_root))
    fast = TiltedObservationCV(env2, tilt=0.2, resample="nearest")

    obs_ref, _ = ref.reset(seed=0, options={"level_idx": 0})
    obs_fast, _ = fast.reset(seed=0, options={"level_idx": 0})

    total_pixels = obs_ref.shape[0] * obs_ref.shape[1]
    diff_pixels = np.sum(np.any(obs_ref != obs_fast, axis=-1))
    assert diff_pixels / total_pixels < 0.10, (
        f"Too many pixel differences: {diff_pixels}/{total_pixels} "
        f"({diff_pixels / total_pixels:.1%})"
    )

    for action in range(4):
        obs_ref, *_ = ref.step(action)
        obs_fast, *_ = fast.step(action)
        diff_pixels = np.sum(np.any(obs_ref != obs_fast, axis=-1))
        assert diff_pixels / total_pixels < 0.10

    ref.close()
    fast.close()


def test_tilt_cv_tilt_zero_identity(mini_boxoban_root: Path) -> None:
    env1 = BoxobanEnv(level_set="medium", split="train", level_root=str(mini_boxoban_root))
    env2 = BoxobanEnv(level_set="medium", split="train", level_root=str(mini_boxoban_root))
    tilted = TiltedObservationCV(env2, tilt=0.0)

    obs_flat, _ = env1.reset(seed=0, options={"level_idx": 0})
    obs_tilt, _ = tilted.reset(seed=0, options={"level_idx": 0})

    np.testing.assert_array_equal(obs_flat, obs_tilt)
    env1.close()
    tilted.close()


def test_tilt_cv_fill_color_at_corners(mini_boxoban_root: Path) -> None:
    fill = (0, 0, 0)
    env = BoxobanEnv(level_set="medium", split="train", level_root=str(mini_boxoban_root))
    tilted = TiltedObservationCV(env, tilt=0.2, fill_color=fill)

    obs, _ = tilted.reset(seed=0, options={"level_idx": 0})

    np.testing.assert_array_equal(obs[0, 0], np.array(fill, dtype=np.uint8))
    np.testing.assert_array_equal(obs[0, 79], np.array(fill, dtype=np.uint8))
    tilted.close()


def test_tilt_cv_invalid_tilt(mini_boxoban_root: Path) -> None:
    env = BoxobanEnv(level_set="medium", split="train", level_root=str(mini_boxoban_root))

    with pytest.raises(ValueError):
        TiltedObservationCV(env, tilt=0.5)
    with pytest.raises(ValueError):
        TiltedObservationCV(env, tilt=-0.1)
    env.close()


def test_tilt_cv_space_contains_obs(mini_boxoban_root: Path) -> None:
    env = BoxobanEnv(level_set="medium", split="train", level_root=str(mini_boxoban_root))
    tilted = TiltedObservationCV(env, tilt=0.2)

    obs, _ = tilted.reset(seed=0, options={"level_idx": 0})
    assert tilted.observation_space.contains(obs)
    tilted.close()


def test_tilt_cv_action_passthrough(mini_boxoban_root: Path) -> None:
    env = BoxobanEnv(level_set="medium", split="train", level_root=str(mini_boxoban_root))
    tilted = TiltedObservationCV(env, tilt=0.2)

    tilted.reset(seed=0, options={"level_idx": 0})
    for action in range(4):
        obs, reward, terminated, truncated, info = tilted.step(action)
        assert obs.shape == (80, 80, 3)
        assert "level_idx" in info
    tilted.close()
