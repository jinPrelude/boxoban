from __future__ import annotations

from gymnasium.envs.registration import register, registry

_ENV_SPECS: tuple[tuple[str, str, dict[str, str | None]], ...] = (
    ("Boxoban-medium-train-v0", "boxoban.env:BoxobanEnv", {"level_set": "medium", "split": "train"}),
    ("Boxoban-medium-valid-v0", "boxoban.env:BoxobanEnv", {"level_set": "medium", "split": "valid"}),
    ("Boxoban-unfiltered-train-v0", "boxoban.env:BoxobanEnv", {"level_set": "unfiltered", "split": "train"}),
    ("Boxoban-unfiltered-valid-v0", "boxoban.env:BoxobanEnv", {"level_set": "unfiltered", "split": "valid"}),
    ("Boxoban-unfiltered-test-v0", "boxoban.env:BoxobanEnv", {"level_set": "unfiltered", "split": "test"}),
    ("Boxoban-hard-v0", "boxoban.env:BoxobanEnv", {"level_set": "hard", "split": None}),
)


def register_envs() -> None:
    for env_id, entry_point, kwargs in _ENV_SPECS:
        if env_id in registry:
            continue
        register(
            id=env_id,
            entry_point=entry_point,
            kwargs=kwargs,
            max_episode_steps=120,
        )
