from __future__ import annotations

import os
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from .colors import BACKGROUND, BOX, BOX_ON_GOAL, GOAL, PLAYER, WALL
from .levels import GRID_SIZE, get_level_collection

_ACTION_TO_DELTA = (
    (-1, 0),
    (1, 0),
    (0, -1),
    (0, 1),
)


class BoxobanEnv(gym.Env[np.ndarray, int]):
    metadata = {"render_modes": ["rgb_array"], "render_fps": 20}

    def __init__(
        self,
        level_set: str,
        split: str | None = None,
        level_root: str | os.PathLike[str] | None = None,
        fixed_level_idx: int | None = None,
        max_steps: int = 120,
        step_penalty: float = -0.1,
        box_on_target_reward: float = 1.0,
        box_off_target_penalty: float = -1.0,
        solve_reward: float = 10.0,
        seed: int | None = None,
        render_mode: str | None = None,
    ) -> None:
        super().__init__()
        if render_mode not in (None, "rgb_array"):
            raise ValueError("render_mode must be None or 'rgb_array'")
        if max_steps <= 0:
            raise ValueError("max_steps must be > 0")

        self.render_mode = render_mode
        self.max_steps = int(max_steps)
        self.step_penalty = float(step_penalty)
        self.box_on_target_reward = float(box_on_target_reward)
        self.box_off_target_penalty = float(box_off_target_penalty)
        self.solve_reward = float(solve_reward)

        self._levels = get_level_collection(
            level_set=level_set,
            split=split,
            level_root=level_root,
        )
        if fixed_level_idx is not None:
            fixed_level_idx = int(fixed_level_idx)
            if fixed_level_idx < 0 or fixed_level_idx >= self._levels.num_levels:
                raise ValueError(
                    f"fixed_level_idx must be in [0, {self._levels.num_levels - 1}], "
                    f"got {fixed_level_idx}"
                )
        self._fixed_level_idx = fixed_level_idx
        self._sampling_mode = "random" if split == "train" else "sequential"
        self._seed_on_first_reset = seed

        self.observation_space = spaces.Box(
            low=0,
            high=255,
            shape=(GRID_SIZE, GRID_SIZE, 3),
            dtype=np.uint8,
        )
        self.action_space = spaces.Discrete(4)

        self._walls = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.bool_)
        self._goals = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.bool_)
        self._boxes = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.bool_)
        self._overlap = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.bool_)
        self._player = np.zeros(2, dtype=np.int8)
        self._obs = np.empty((GRID_SIZE, GRID_SIZE, 3), dtype=np.uint8)

        self._level_idx = -1
        self._next_level_idx = 0
        self._steps = 0
        self._boxes_on_target = 0
        self._target_count = 0

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        if seed is None and self._seed_on_first_reset is not None:
            seed = self._seed_on_first_reset
            self._seed_on_first_reset = None
        super().reset(seed=seed)

        level_idx = self._select_level_idx(options)
        self._level_idx = level_idx

        self._walls = self._levels.walls[level_idx]
        self._goals = self._levels.goals[level_idx]
        np.copyto(self._boxes, self._levels.initial_boxes[level_idx])
        self._player[:] = self._levels.initial_players[level_idx]

        self._steps = 0
        self._target_count = int(self._levels.goal_counts[level_idx])
        np.logical_and(self._boxes, self._goals, out=self._overlap)
        self._boxes_on_target = int(np.count_nonzero(self._overlap))

        self._render_full_observation()
        return self._obs, self._info(is_success=False)

    def step(
        self,
        action: int,
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        action_int = int(action)
        if action_int < 0 or action_int >= 4:
            raise ValueError(f"Action must be in [0, 3], got {action!r}")

        reward = self.step_penalty
        dy, dx = _ACTION_TO_DELTA[action_int]

        old_py = int(self._player[0])
        old_px = int(self._player[1])

        pushed = False
        box_from_y = box_from_x = box_to_y = box_to_x = -1

        new_py = old_py + dy
        new_px = old_px + dx

        if 0 <= new_py < GRID_SIZE and 0 <= new_px < GRID_SIZE:
            if not self._walls[new_py, new_px]:
                if self._boxes[new_py, new_px]:
                    beyond_y = new_py + dy
                    beyond_x = new_px + dx
                    if (
                        0 <= beyond_y < GRID_SIZE
                        and 0 <= beyond_x < GRID_SIZE
                        and not self._walls[beyond_y, beyond_x]
                        and not self._boxes[beyond_y, beyond_x]
                    ):
                        pushed = True
                        box_from_y, box_from_x = new_py, new_px
                        box_to_y, box_to_x = beyond_y, beyond_x

                        was_on_goal = bool(self._goals[box_from_y, box_from_x])
                        now_on_goal = bool(self._goals[box_to_y, box_to_x])

                        self._boxes[box_from_y, box_from_x] = False
                        self._boxes[box_to_y, box_to_x] = True

                        if now_on_goal and not was_on_goal:
                            reward += self.box_on_target_reward
                            self._boxes_on_target += 1
                        elif was_on_goal and not now_on_goal:
                            reward += self.box_off_target_penalty
                            self._boxes_on_target -= 1

                        self._player[0] = new_py
                        self._player[1] = new_px
                else:
                    self._player[0] = new_py
                    self._player[1] = new_px

        self._steps += 1
        is_success = self._boxes_on_target == self._target_count
        if is_success:
            reward += self.solve_reward

        terminated = bool(is_success)
        truncated = self._steps >= self.max_steps

        self._paint_cell(old_py, old_px)
        if pushed:
            self._paint_cell(box_from_y, box_from_x)
            self._paint_cell(box_to_y, box_to_x)
        self._paint_cell(int(self._player[0]), int(self._player[1]))

        return self._obs, reward, terminated, truncated, self._info(is_success=is_success)

    def render(self) -> np.ndarray | None:
        if self.render_mode == "rgb_array":
            return self._obs.copy()
        return None

    def close(self) -> None:
        return None

    @property
    def num_levels(self) -> int:
        return self._levels.num_levels

    def _select_level_idx(self, options: dict[str, Any] | None) -> int:
        if self._fixed_level_idx is not None:
            return self._fixed_level_idx
        if options is not None and "level_idx" in options:
            level_idx = int(options["level_idx"])
            if level_idx < 0 or level_idx >= self._levels.num_levels:
                raise ValueError(
                    f"level_idx must be in [0, {self._levels.num_levels - 1}], got {level_idx}"
                )
            return level_idx

        if self._sampling_mode == "random":
            return int(self.np_random.integers(0, self._levels.num_levels))

        level_idx = self._next_level_idx
        self._next_level_idx = (self._next_level_idx + 1) % self._levels.num_levels
        return level_idx

    def _render_full_observation(self) -> None:
        self._obs[:, :] = BACKGROUND
        self._obs[self._walls] = WALL
        self._obs[self._goals] = GOAL
        self._obs[self._boxes] = BOX
        np.logical_and(self._boxes, self._goals, out=self._overlap)
        self._obs[self._overlap] = BOX_ON_GOAL
        self._obs[int(self._player[0]), int(self._player[1])] = PLAYER

    def _paint_cell(self, y: int, x: int) -> None:
        if self._walls[y, x]:
            self._obs[y, x] = WALL
            return

        if int(self._player[0]) == y and int(self._player[1]) == x:
            self._obs[y, x] = PLAYER
            return

        if self._boxes[y, x]:
            if self._goals[y, x]:
                self._obs[y, x] = BOX_ON_GOAL
            else:
                self._obs[y, x] = BOX
            return

        if self._goals[y, x]:
            self._obs[y, x] = GOAL
        else:
            self._obs[y, x] = BACKGROUND

    def _info(self, *, is_success: bool) -> dict[str, Any]:
        return {
            "level_idx": self._level_idx,
            "boxes_on_target": self._boxes_on_target,
            "is_success": is_success,
            "steps": self._steps,
        }
