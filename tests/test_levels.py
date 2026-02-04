from __future__ import annotations

from pathlib import Path

import pytest

from boxoban.levels import count_levels, resolve_level_root


def test_mini_dataset_counts(mini_boxoban_root: Path) -> None:
    assert count_levels("hard", level_root=mini_boxoban_root) == 2
    assert count_levels("medium", split="train", level_root=mini_boxoban_root) == 2
    assert count_levels("medium", split="valid", level_root=mini_boxoban_root) == 1
    assert count_levels("unfiltered", split="train", level_root=mini_boxoban_root) == 2
    assert count_levels("unfiltered", split="valid", level_root=mini_boxoban_root) == 1
    assert count_levels("unfiltered", split="test", level_root=mini_boxoban_root) == 1


def test_resolve_level_root_from_env_var(monkeypatch: pytest.MonkeyPatch, mini_boxoban_root: Path) -> None:
    monkeypatch.setenv("BOXOBAN_LEVELS_DIR", str(mini_boxoban_root))
    assert resolve_level_root() == mini_boxoban_root.resolve()


def test_official_dataset_counts_if_available() -> None:
    root = Path(__file__).resolve().parents[1] / "boxoban-levels"
    if not (root / "hard").is_dir():
        pytest.skip("official boxoban-levels directory not found")

    assert count_levels("hard", level_root=root) == 3332
    assert count_levels("medium", split="train", level_root=root) == 450000
    assert count_levels("medium", split="valid", level_root=root) == 50000
    assert count_levels("unfiltered", split="train", level_root=root) == 900000
    assert count_levels("unfiltered", split="valid", level_root=root) == 100000
    assert count_levels("unfiltered", split="test", level_root=root) == 1000
