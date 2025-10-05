"""OpenCV-based full-screen selection overlay to choose a capture region.

This module provides select_capture_area_cv(sct, screen_width, screen_height)
which returns a monitor dict {top,left,width,height} or None.
It does not persist configuration; callers can save if desired.
"""
from __future__ import annotations

import time
import cv2
import numpy as np

# State for rectangle selection
_drawing = False
_rect_start = None
_rect_end = None
_selection_mode = False
_DISPLAY_SCALE = 1.0


def _mouse_callback(event, x, y, flags, param):
    global _drawing, _rect_start, _rect_end

    if not _selection_mode:
        return

    if event == cv2.EVENT_LBUTTONDOWN:
        if not _drawing:
            _rect_start = (x, y)
            _rect_end = None
            _drawing = True
        else:
            _rect_end = (x, y)
            _drawing = False
    elif event == cv2.EVENT_MOUSEMOVE:
        if _drawing and _rect_start is not None:
            _rect_end = (x, y)


def select_capture_area_cv(sct, screen_width: int, screen_height: int):
    """Open a full-screen OpenCV overlay to select a rectangular region.

    Returns: dict(top,left,width,height) or None
    """
    global _selection_mode, _rect_start, _rect_end, _drawing, _DISPLAY_SCALE

    cv2.destroyAllWindows()
    cv2.waitKey(1)
    time.sleep(0.2)

    _selection_mode = True
    _rect_start = None
    _rect_end = None
    _drawing = False

    # Screenshot background of primary monitor
    full_monitor = sct.monitors[1]
    screenshot = np.array(sct.grab(full_monitor))
    screenshot = cv2.cvtColor(screenshot, cv2.COLOR_BGRA2BGR)

    cv2.namedWindow("Selection Overlay", cv2.WINDOW_NORMAL)
    cv2.setWindowProperty("Selection Overlay", cv2.WND_PROP_TOPMOST, 1)
    cv2.moveWindow("Selection Overlay", 0, 0)
    cv2.resizeWindow("Selection Overlay", screen_width, screen_height)
    cv2.setWindowProperty("Selection Overlay", cv2.WND_PROP_ASPECT_RATIO, cv2.WINDOW_FREERATIO)
    cv2.setMouseCallback("Selection Overlay", _mouse_callback)

    cv2.waitKey(50)

    print("\nSelection Mode (OpenCV):")
    print("- Click and drag to select area")
    print("- Press 's' to confirm selection")
    print("- Press 'c' or Esc to cancel")

    while True:
        overlay = screenshot.copy()

        cv2.putText(overlay, "SELECTION MODE (cv2)", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 255), 3)
        cv2.putText(overlay, "Click-drag, then press 's' to confirm", (50, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

        if _rect_start:
            x1, y1 = _rect_start
            cv2.circle(overlay, (x1, y1), 6, (0, 255, 0), -1)

        if _rect_start and _rect_end:
            x1, y1 = _rect_start
            x2, y2 = _rect_end
            left = min(x1, x2)
            top = min(y1, y2)
            right = max(x1, x2)
            bottom = max(y1, y2)
            cv2.rectangle(overlay, (left, top), (right, bottom), (0, 255, 0), 3)
            width = right - left
            height = bottom - top
            cv2.putText(overlay, f"Area: {width}x{height} at ({left},{top})", (left, max(30, top - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        cv2.imshow("Selection Overlay", overlay)

        x, y, w, h = cv2.getWindowImageRect("Selection Overlay")
        _DISPLAY_SCALE = w / float(screen_width or 1)

        key = cv2.waitKey(50) & 0xFF
        if key in (ord('s'), ord('S')):
            if _rect_start and _rect_end:
                x1, y1 = _rect_start
                x2, y2 = _rect_end
                left = min(x1, x2) / _DISPLAY_SCALE
                top = min(y1, y2) / _DISPLAY_SCALE
                width = abs(x2 - x1) / _DISPLAY_SCALE
                height = abs(y2 - y1) / _DISPLAY_SCALE
                if width > 10 and height > 10:
                    cv2.destroyWindow("Selection Overlay")
                    _selection_mode = False
                    return {"top": top, "left": left, "width": width, "height": height}
                else:
                    print("Selection too small; select a larger area.")
            else:
                print("No selection yet; click and drag first.")
        elif key in (ord('c'), ord('C'), 27):
            cv2.destroyWindow("Selection Overlay")
            _selection_mode = False
            return None
