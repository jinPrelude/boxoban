"""Observation wrappers for Boxoban environments."""

from __future__ import annotations


import gymnasium as gym
import numpy as np
from gymnasium import spaces

__all__ = [
    "TiltedObservationWrapper",
]


class TiltedObservationWrapper(gym.ObservationWrapper):
    """Apply a perspective transform simulating a tilted camera using OpenCV.

    The top edge of the image is narrowed while the bottom stays unchanged,
    producing a trapezoid that gives a 3-D perspective feel. The entire
    transform runs in a single C++ call with SIMD acceleration.

    Parameters
    ----------
    env:
        Environment whose observations are ``HxWx3`` uint8 images.
    tilt:
        Fraction of the image width to inset on each side at the top.
        ``0.0`` = identity (no tilt), must be ``< 0.5``.
    fill_color:
        RGB colour for areas outside the trapezoid.
        Defaults to the Boxoban background ``(23, 26, 32)``.
    resample:
        ``"nearest"`` or ``"bilinear"``.  Default ``"nearest"``.
    """

    def __init__(
        self,
        env: gym.Env,
        tilt: float = 0.2,
        fill_color: tuple[int, int, int] = (23, 26, 32),
        resample: str = "nearest",
    ) -> None:
        super().__init__(env)
        assert isinstance(env.observation_space, spaces.Box)
        assert len(env.observation_space.shape) == 3

        try:
            import cv2
        except ImportError as exc:
            raise ImportError(
                "opencv-python is required for TiltedObservationWrapper. "
                "Install it with: pip install opencv-python"
            ) from exc

        if resample not in ("nearest", "bilinear"):
            raise ValueError(
                f"resample must be 'nearest' or 'bilinear', got {resample!r}"
            )
        if not (0.0 <= tilt < 0.5):
            raise ValueError(f"tilt must be in [0.0, 0.5), got {tilt}")

        self._tilt = tilt
        self._is_identity = tilt == 0.0
        self._cv2 = cv2

        H, W = env.observation_space.shape[:2]
        self._dsize = (W, H)

        self.observation_space = spaces.Box(
            low=0, high=255,
            shape=env.observation_space.shape,
            dtype=np.uint8,
        )

        if self._is_identity:
            return

        inset = float(W * tilt)
        src = np.array(
            [[0, 0], [W-1, 0], [W-1, H-1], [0, H-1]],
            dtype=np.float32,
        )
        dst = np.array(
            [[inset, 0], [W-1 - inset, 0], [W-1, H-1], [0, H-1]],
            dtype=np.float32,
        )
        # Inverse matrix (dst → src): for each destination pixel, look up
        # the source pixel directly — avoids per-frame matrix inversion.
        self._M = cv2.getPerspectiveTransform(dst, src)
        self._flags = (
            (cv2.INTER_NEAREST if resample == "nearest" else cv2.INTER_LINEAR)
            | cv2.WARP_INVERSE_MAP
        )
        self._border_value = fill_color

    def observation(self, observation: np.ndarray) -> np.ndarray:
        if self._is_identity:
            return observation
        return self._cv2.warpPerspective(
            observation,
            self._M,
            self._dsize,
            flags=self._flags,
            borderMode=self._cv2.BORDER_CONSTANT,
            borderValue=self._border_value,
        )
