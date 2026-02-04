from __future__ import annotations

from pathlib import Path

import pytest


def _write_level_file(path: Path, levels: list[list[str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="ascii") as handle:
        for idx, grid in enumerate(levels):
            handle.write(f"; {idx}\n")
            for row in grid:
                if len(row) != 10:
                    raise ValueError(f"row must be width=10, got {row!r}")
                handle.write(f"{row}\n")
            handle.write("\n")


@pytest.fixture()
def mini_boxoban_root(tmp_path: Path) -> Path:
    wall_collision = [
        "##########",
        "#@  .    #",
        "# $      #",
        "# .      #",
        "# $      #",
        "# .      #",
        "# $      #",
        "# .      #",
        "#   $    #",
        "##########",
    ]
    push_on_off_goal = [
        "##########",
        "#        #",
        "#  @$.   #",
        "#   .    #",
        "#   $    #",
        "#   .    #",
        "#   $    #",
        "#   .    #",
        "#   $    #",
        "##########",
    ]
    solve_one_push = [
        "##########",
        "#   .    #",
        "#   $    #",
        "#   @    #",
        "#        #",
        "#        #",
        "#        #",
        "#        #",
        "#        #",
        "##########",
    ]

    root = tmp_path / "boxoban-levels"

    _write_level_file(root / "hard" / "000.txt", [solve_one_push, wall_collision])
    _write_level_file(root / "medium" / "train" / "000.txt", [wall_collision, push_on_off_goal])
    _write_level_file(root / "medium" / "valid" / "000.txt", [push_on_off_goal])
    _write_level_file(root / "unfiltered" / "train" / "000.txt", [wall_collision, push_on_off_goal])
    _write_level_file(root / "unfiltered" / "valid" / "000.txt", [wall_collision])
    _write_level_file(root / "unfiltered" / "test" / "000.txt", [push_on_off_goal])

    return root
