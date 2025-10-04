"""Screen capture stub.
Implement actual capture with mss or similar.
"""
from typing import Tuple
import numpy as np


def grab_frame(roi: Tuple[int, int, int, int]) -> np.ndarray:
    """Return a dummy RGB frame ndarray (H,W,3) uint8 for now."""
    h, w = roi[3], roi[2]
    return np.zeros((h, w, 3), dtype=np.uint8)
