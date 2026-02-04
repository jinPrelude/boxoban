from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np

GRID_SIZE = 10

_ALLOWED_LEVEL_SETS = {"hard", "medium", "unfiltered"}
_ALLOWED_SPLITS = {
    "medium": {"train", "valid"},
    "unfiltered": {"train", "valid", "test"},
}

_CACHE: dict[tuple[str, str, str | None], "LevelCollection"] = {}


@dataclass(frozen=True)
class LevelCollection:
    level_set: str
    split: str | None
    root: Path
    walls: np.ndarray
    goals: np.ndarray
    initial_boxes: np.ndarray
    initial_players: np.ndarray
    goal_counts: np.ndarray

    @property
    def num_levels(self) -> int:
        return int(self.initial_players.shape[0])


def resolve_level_root(level_root: str | os.PathLike[str] | None = None) -> Path:
    explicit = _normalize_root_candidate(level_root)
    if explicit is not None:
        return explicit

    env_root = _normalize_root_candidate(os.environ.get("BOXOBAN_LEVELS_DIR"))
    if env_root is not None:
        return env_root

    module_path = Path(__file__).resolve()
    candidates = (
        Path.cwd(),
        module_path.parents[2],
        module_path.parents[3],
    )
    for candidate in candidates:
        normalized = _normalize_root_candidate(candidate)
        if normalized is not None:
            return normalized

    raise FileNotFoundError(
        "Unable to find boxoban levels. Pass level_root=..., or set BOXOBAN_LEVELS_DIR."
    )


def count_levels(
    level_set: str,
    split: str | None = None,
    level_root: str | os.PathLike[str] | None = None,
) -> int:
    root = resolve_level_root(level_root)
    level_dir = _resolve_level_dir(root, level_set, split)
    return _count_levels_in_dir(level_dir)


def get_level_collection(
    level_set: str,
    split: str | None = None,
    level_root: str | os.PathLike[str] | None = None,
) -> LevelCollection:
    root = resolve_level_root(level_root)
    key = (str(root), level_set, split)
    cached = _CACHE.get(key)
    if cached is not None:
        return cached

    level_dir = _resolve_level_dir(root, level_set, split)
    total_levels = _count_levels_in_dir(level_dir)
    if total_levels <= 0:
        raise ValueError(f"No levels found in {level_dir}")

    walls = np.empty((total_levels, GRID_SIZE, GRID_SIZE), dtype=np.bool_)
    goals = np.empty((total_levels, GRID_SIZE, GRID_SIZE), dtype=np.bool_)
    initial_boxes = np.empty((total_levels, GRID_SIZE, GRID_SIZE), dtype=np.bool_)
    initial_players = np.empty((total_levels, 2), dtype=np.int8)

    idx = 0
    for level_file in sorted(level_dir.glob("*.txt")):
        lines = level_file.read_text(encoding="ascii").splitlines()
        cursor = 0
        while cursor < len(lines):
            line = lines[cursor]
            if not line:
                cursor += 1
                continue
            if not line.startswith(";"):
                raise ValueError(
                    f"Malformed level header in {level_file}:{cursor + 1}: {line!r}"
                )

            grid_start = cursor + 1
            grid_end = grid_start + GRID_SIZE
            if grid_end > len(lines):
                raise ValueError(f"Truncated level in {level_file}:{cursor + 1}")

            grid = lines[grid_start:grid_end]
            _parse_grid(
                grid,
                walls[idx],
                goals[idx],
                initial_boxes[idx],
                initial_players[idx],
                level_file,
                grid_start,
            )
            idx += 1
            cursor = grid_end
            if cursor < len(lines) and not lines[cursor]:
                cursor += 1

    if idx != total_levels:
        raise RuntimeError(f"Parsed {idx} levels but expected {total_levels} in {level_dir}")

    goal_counts = goals.reshape(total_levels, -1).sum(axis=1, dtype=np.int16)

    collection = LevelCollection(
        level_set=level_set,
        split=split,
        root=root,
        walls=walls,
        goals=goals,
        initial_boxes=initial_boxes,
        initial_players=initial_players,
        goal_counts=goal_counts,
    )
    _CACHE[key] = collection
    return collection


def _normalize_root_candidate(candidate: str | os.PathLike[str] | None) -> Path | None:
    if candidate is None:
        return None
    path = Path(candidate).expanduser().resolve()
    if _is_dataset_root(path):
        return path

    nested = path / "boxoban-levels"
    if _is_dataset_root(nested):
        return nested
    return None


def _is_dataset_root(path: Path) -> bool:
    return (
        path.is_dir()
        and (path / "hard").is_dir()
        and (path / "medium").is_dir()
        and (path / "unfiltered").is_dir()
    )


def _resolve_level_dir(root: Path, level_set: str, split: str | None) -> Path:
    if level_set not in _ALLOWED_LEVEL_SETS:
        raise ValueError(f"Unsupported level_set={level_set!r}")

    if level_set == "hard":
        if split not in (None, ""):
            raise ValueError("hard does not use split. Use split=None.")
        level_dir = root / "hard"
    else:
        if split is None:
            raise ValueError(f"split is required for level_set={level_set!r}")
        allowed_splits = _ALLOWED_SPLITS[level_set]
        if split not in allowed_splits:
            raise ValueError(
                f"Unsupported split={split!r} for level_set={level_set!r}; "
                f"allowed={sorted(allowed_splits)}"
            )
        level_dir = root / level_set / split

    if not level_dir.is_dir():
        raise FileNotFoundError(f"Missing level directory: {level_dir}")
    return level_dir


def _count_levels_in_dir(level_dir: Path) -> int:
    total = 0
    for level_file in sorted(level_dir.glob("*.txt")):
        with level_file.open("r", encoding="ascii") as handle:
            for line in handle:
                if line.startswith(";"):
                    total += 1
    return total


def _parse_grid(
    rows: list[str],
    walls: np.ndarray,
    goals: np.ndarray,
    boxes: np.ndarray,
    player: np.ndarray,
    level_file: Path,
    line_offset: int,
) -> None:
    if len(rows) != GRID_SIZE:
        raise ValueError(f"Expected {GRID_SIZE} rows in {level_file}:{line_offset}")

    walls.fill(False)
    goals.fill(False)
    boxes.fill(False)

    player_count = 0
    for y, row in enumerate(rows):
        if len(row) != GRID_SIZE:
            raise ValueError(
                f"Expected width={GRID_SIZE} in {level_file}:{line_offset + y + 1}, "
                f"got {len(row)}"
            )
        for x, token in enumerate(row):
            if token == "#":
                walls[y, x] = True
            elif token == ".":
                goals[y, x] = True
            elif token == "$":
                boxes[y, x] = True
            elif token == "@":
                player[0] = y
                player[1] = x
                player_count += 1
            elif token == " ":
                continue
            else:
                raise ValueError(
                    f"Unsupported token {token!r} in {level_file}:{line_offset + y + 1}"
                )

    if player_count != 1:
        raise ValueError(
            f"Expected exactly one player in {level_file}:{line_offset + 1}, "
            f"found {player_count}"
        )
