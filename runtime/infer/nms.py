"""Non-Maximum Suppression (NMS) placeholder.
Provide a simple CPU NMS implementation later.
"""
from __future__ import annotations
import numpy as np
from typing import List


def nms(boxes: np.ndarray, scores: np.ndarray, iou_thresh: float) -> List[int]:
    if len(boxes) == 0:
        return []
    # Placeholder greedy selection: return all indices
    return list(range(len(boxes)))
