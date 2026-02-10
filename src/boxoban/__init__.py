from __future__ import annotations

from .env import BoxobanEnv
from .registration import register_envs
from .wrappers import TiltedObservationWrapper

register_envs()

__all__ = [
    "BoxobanEnv",
    "TiltedObservationWrapper",
    "register_envs",
]
