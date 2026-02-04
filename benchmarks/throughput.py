from __future__ import annotations

import argparse
import time

import gymnasium as gym
import numpy as np

import boxoban  # noqa: F401


def benchmark_single(env_id: str, level_root: str | None, steps: int) -> float:
    env = gym.make(
        env_id,
        level_root=level_root,
        disable_env_checker=True,
    )
    env.reset(seed=0)
    rng = np.random.default_rng(0)

    start = time.perf_counter()
    for _ in range(steps):
        action = int(rng.integers(0, 4))
        _, _, terminated, truncated, _ = env.step(action)
        if terminated or truncated:
            env.reset()
    elapsed = time.perf_counter() - start
    env.close()
    return steps / elapsed


def benchmark_vector(
    env_id: str,
    level_root: str | None,
    steps: int,
    num_envs: int,
    mode: str,
) -> float:
    try:
        vec_env = gym.make_vec(
            env_id,
            num_envs=num_envs,
            vectorization_mode=mode,
            level_root=level_root,
            disable_env_checker=True,
        )
    except TypeError:
        vec_env = gym.make_vec(
            env_id,
            num_envs=num_envs,
            vectorization_mode=mode,
            level_root=level_root,
        )

    vec_env.reset(seed=0)
    rng = np.random.default_rng(0)

    start = time.perf_counter()
    for _ in range(steps):
        actions = rng.integers(0, 4, size=num_envs, dtype=np.int64)
        vec_env.step(actions)
    elapsed = time.perf_counter() - start

    vec_env.close()
    return (num_envs * steps) / elapsed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark Boxoban environment throughput")
    parser.add_argument("--env-id", default="Boxoban-medium-train-v0")
    parser.add_argument("--num-envs", type=int, default=128)
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--level-root", type=str, default=None)
    parser.add_argument(
        "--skip-async",
        action="store_true",
        help="Skip async vector benchmark",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    single_sps = benchmark_single(args.env_id, args.level_root, args.steps)
    sync_sps = benchmark_vector(
        args.env_id,
        args.level_root,
        args.steps,
        args.num_envs,
        mode="sync",
    )

    print(f"single env-steps/s: {single_sps:,.0f}")
    print(f"sync   env-steps/s: {sync_sps:,.0f}")

    if not args.skip_async:
        async_sps = benchmark_vector(
            args.env_id,
            args.level_root,
            args.steps,
            args.num_envs,
            mode="async",
        )
        print(f"async  env-steps/s: {async_sps:,.0f}")


if __name__ == "__main__":
    main()
