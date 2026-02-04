from __future__ import annotations

from .env import BoxobanEnv
from .registration import register_envs

register_envs()

__all__ = ["BoxobanEnv", "register_envs"]
