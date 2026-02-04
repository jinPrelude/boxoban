from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent
SRC_DIR = REPO_ROOT / "src"
if SRC_DIR.is_dir():
    sys.path.insert(0, str(SRC_DIR))

import boxoban  # noqa: F401 - import registers Gymnasium env IDs

KEY_TO_ACTION = {
    "up": 0,
    "down": 1,
    "left": 2,
    "right": 3,
}


def env_id_for(level_set: str, split: str | None) -> str:
    if level_set in ("easy", "hard"):
        return f"Boxoban-{level_set}-v0"
    return f"Boxoban-{level_set}-{split}-v0"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Play Boxoban manually with pygame.",
    )
    parser.add_argument(
        "--level-set",
        default="hard",
        choices=("easy", "hard", "medium", "unfiltered"),
        help="Level set to load.",
    )
    parser.add_argument(
        "--split",
        default=None,
        choices=("train", "valid", "test"),
        help="Split to load (required for medium/unfiltered).",
    )
    parser.add_argument(
        "--level-root",
        default=None,
        help="Path to boxoban-levels root directory.",
    )
    parser.add_argument(
        "--level-idx",
        type=int,
        default=0,
        help="Initial level index.",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=120,
        help="Maximum steps before truncation.",
    )
    parser.add_argument(
        "--tile-size",
        type=int,
        default=64,
        help="Pixels per tile.",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=60,
        help="Render loop FPS.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Seed used for the first reset.",
    )
    return parser


def action_from_key(key: int, pygame_module: Any) -> int | None:
    if key in (pygame_module.K_UP, pygame_module.K_w):
        return KEY_TO_ACTION["up"]
    if key in (pygame_module.K_DOWN, pygame_module.K_s):
        return KEY_TO_ACTION["down"]
    if key in (pygame_module.K_LEFT, pygame_module.K_a):
        return KEY_TO_ACTION["left"]
    if key in (pygame_module.K_RIGHT, pygame_module.K_d):
        return KEY_TO_ACTION["right"]
    return None


def to_surface(frame: np.ndarray, tile_size: int, pygame_module: Any) -> Any:
    board_h, board_w, _ = frame.shape
    base = pygame_module.surfarray.make_surface(np.transpose(frame, (1, 0, 2)))
    target_size = (board_w * tile_size, board_h * tile_size)
    return pygame_module.transform.scale(base, target_size)


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    if args.level_set not in ("easy", "hard") and args.split is None:
        args.split = "valid"
    if args.level_set in ("easy", "hard") and args.split is not None:
        parser.error(f"{args.level_set} does not use --split. Remove --split.")
    if args.level_idx < 0:
        parser.error("--level-idx must be >= 0")
    if args.tile_size <= 0:
        parser.error("--tile-size must be > 0")
    if args.max_steps <= 0:
        parser.error("--max-steps must be > 0")
    if args.fps <= 0:
        parser.error("--fps must be > 0")

    try:
        import pygame
    except ImportError as exc:
        print("pygame is required. Install it with: pip install pygame", file=sys.stderr)
        return 1

    env = gym.make(
        env_id_for(args.level_set, args.split),
        level_root=args.level_root,
        max_steps=args.max_steps,
        render_mode="rgb_array",
        seed=args.seed,
        disable_env_checker=True,
    )

    pygame.init()
    pygame.display.set_caption("Boxoban Player")

    hud_height = 96
    board_pixels = 10 * args.tile_size
    screen = pygame.display.set_mode((board_pixels, board_pixels + hud_height))
    font = pygame.font.Font(None, 30)
    small_font = pygame.font.Font(None, 24)
    clock = pygame.time.Clock()

    current_level_idx = args.level_idx
    num_levels = int(env.unwrapped._levels.num_levels)
    obs, info = env.reset(options={"level_idx": current_level_idx})

    done = False
    terminated = False
    truncated = False
    last_reward = 0.0

    try:
        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                    break

                if event.type != pygame.KEYDOWN:
                    continue

                if event.key in (pygame.K_ESCAPE, pygame.K_q):
                    running = False
                    break

                if event.key == pygame.K_r:
                    obs, info = env.reset(options={"level_idx": current_level_idx})
                    done = False
                    terminated = False
                    truncated = False
                    last_reward = 0.0
                    continue

                if event.key == pygame.K_n:
                    current_level_idx = (current_level_idx + 1) % num_levels
                    obs, info = env.reset(options={"level_idx": current_level_idx})
                    done = False
                    terminated = False
                    truncated = False
                    last_reward = 0.0
                    continue

                if done:
                    continue

                action = action_from_key(event.key, pygame)
                if action is None:
                    continue

                obs, last_reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated

            frame = to_surface(obs, args.tile_size, pygame)

            screen.fill((20, 22, 28))
            screen.blit(frame, (0, 0))
            pygame.draw.rect(screen, (30, 33, 40), (0, board_pixels, board_pixels, hud_height))

            status = "playing"
            if terminated:
                status = "solved"
            elif truncated:
                status = "truncated"

            title = (
                f"Level {info['level_idx']}/{num_levels - 1}  "
                f"Step {info['steps']}  "
                f"Boxes {info['boxes_on_target']}  "
                f"Reward {last_reward:+.1f}  "
                f"Status {status}"
            )
            controls = "Move: arrows/WASD | R: restart | N: next level | Q/ESC: quit"

            screen.blit(font.render(title, True, (236, 239, 244)), (10, board_pixels + 10))
            screen.blit(
                small_font.render(controls, True, (200, 205, 215)),
                (10, board_pixels + 52),
            )

            pygame.display.flip()
            clock.tick(args.fps)
    finally:
        env.close()
        pygame.quit()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
