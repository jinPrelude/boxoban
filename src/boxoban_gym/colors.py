from __future__ import annotations

import numpy as np

BACKGROUND = np.array((23, 26, 32), dtype=np.uint8)
WALL = np.array((86, 91, 98), dtype=np.uint8)
PLAYER = np.array((52, 152, 219), dtype=np.uint8)
BOX = np.array((236, 191, 54), dtype=np.uint8)
GOAL = np.array((88, 181, 99), dtype=np.uint8)
BOX_ON_GOAL = np.array((250, 223, 121), dtype=np.uint8)
