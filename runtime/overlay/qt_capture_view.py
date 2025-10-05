from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import mss
import numpy as np

from PyQt5 import QtCore, QtGui, QtWidgets
import sys
# Ensure project root is on sys.path when running this file directly
_THIS = Path(__file__).resolve()
_ROOT = _THIS.parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
from runtime.capture.cv_selection import select_capture_area_cv


CONFIG_PATH = Path("capture_config.json")


@dataclass
class MonitorRect:
    top: int
    left: int
    width: int
    height: int

    @classmethod
    def from_mapping(cls, m: dict) -> "MonitorRect":
        return cls(int(round(m["top"])), int(round(m["left"])), int(round(m["width"])), int(round(m["height"])))

    def as_dict(self) -> dict:
        return {"top": self.top, "left": self.left, "width": self.width, "height": self.height}


def load_capture_config() -> Optional[MonitorRect]:
    if CONFIG_PATH.exists():
        try:
            data = json.loads(CONFIG_PATH.read_text())
            mon = data.get("monitor")
            if mon:
                return MonitorRect.from_mapping(mon)
        except Exception:
            pass
    return None


def save_capture_config(monitor: MonitorRect) -> None:
    payload = {"monitor": monitor.as_dict()}
    CONFIG_PATH.write_text(json.dumps(payload, indent=2))


class SelectionOverlay(QtWidgets.QWidget):
    """Fullscreen selection overlay using QRubberBand.

    Emits a 'regionSelected' signal with a MonitorRect in screen coordinates.
    """

    regionSelected = QtCore.pyqtSignal(MonitorRect)

    def __init__(self, screen: QtGui.QScreen, parent=None):
        super().__init__(parent)
        self.setWindowFlags(
            QtCore.Qt.FramelessWindowHint
            | QtCore.Qt.WindowStaysOnTopHint
            | QtCore.Qt.Tool  # avoid taskbar/dock focus
        )
        self.setAttribute(QtCore.Qt.WA_TranslucentBackground)
        self.setCursor(QtCore.Qt.CrossCursor)

        # Cover the given screen
        geo = screen.geometry()
        self.setGeometry(geo)

        self._rubber = QtWidgets.QRubberBand(QtWidgets.QRubberBand.Rectangle, self)
        self._origin = QtCore.QPoint()

        # Instruction label
        self._label = QtWidgets.QLabel(
            "Click and drag to select area. Enter: save, Esc: cancel",
            self,
        )
        self._label.setStyleSheet(
            "color:white; background:rgba(0,0,0,120); padding:6px; border-radius:4px;"
        )
        self._label.move(20, 20)

    def paintEvent(self, event: QtGui.QPaintEvent) -> None:
        # Dim the background
        p = QtGui.QPainter(self)
        p.fillRect(self.rect(), QtGui.QColor(0, 0, 0, 80))
        p.end()

    def mousePressEvent(self, e: QtGui.QMouseEvent) -> None:
        if e.button() == QtCore.Qt.LeftButton:
            self._origin = e.pos()
            self._rubber.setGeometry(QtCore.QRect(self._origin, QtCore.QSize()))
            self._rubber.show()

    def mouseMoveEvent(self, e: QtGui.QMouseEvent) -> None:
        if not self._rubber.isVisible():
            return
        rect = QtCore.QRect(self._origin, e.pos()).normalized()
        self._rubber.setGeometry(rect)

    def keyPressEvent(self, e: QtGui.QKeyEvent) -> None:
        if e.key() in (QtCore.Qt.Key_Escape,):
            self._rubber.hide()
            self.close()
            return
        if e.key() in (QtCore.Qt.Key_Return, QtCore.Qt.Key_Enter):
            self._emit_selection_and_close()

    def mouseReleaseEvent(self, e: QtGui.QMouseEvent) -> None:
        if e.button() == QtCore.Qt.LeftButton:
            # Keep selection visible; wait for Enter to confirm
            pass

    def _emit_selection_and_close(self):
        rect = self._rubber.geometry()
        if rect.width() > 10 and rect.height() > 10:
            geo = self.geometry()
            # Convert widget-local to screen coords
            top = geo.top() + rect.top()
            left = geo.left() + rect.left()
            mon = MonitorRect(top=top, left=left, width=rect.width(), height=rect.height())
            self.regionSelected.emit(mon)
        self._rubber.hide()
        self.close()


class CaptureWindow(QtWidgets.QWidget):
    """Displays captured screen region with a timer. Keyboard shortcuts:
    - R: open selection overlay
    - S: save screenshot
    - Q/Esc: quit
    """

    def __init__(self, initial_monitor: Optional[MonitorRect] = None, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Screen Capture")
        self.resize(900, 700)
        # Ensure the window can receive keyboard focus
        self.setFocusPolicy(QtCore.Qt.StrongFocus)

        self._image_label = QtWidgets.QLabel()
        self._image_label.setAlignment(QtCore.Qt.AlignCenter)
        self._image_label.setMinimumSize(320, 240)
        self._image_label.setStyleSheet("background:#111; border:1px solid #333;")
        # Don't steal focus from the main window for key handling
        self._image_label.setFocusPolicy(QtCore.Qt.NoFocus)

        self._info_label = QtWidgets.QLabel()
        self._info_label.setStyleSheet("color:#ddd;")

        # Action buttons as a visible fallback for keyboard shortcuts
        self._btn_row = QtWidgets.QHBoxLayout()
        self._btn_select = QtWidgets.QPushButton("Select Region (R)")
        self._btn_shot = QtWidgets.QPushButton("Screenshot (S)")
        self._btn_quit = QtWidgets.QPushButton("Quit (Q)")
        for b in (self._btn_select, self._btn_shot, self._btn_quit):
            b.setFocusPolicy(QtCore.Qt.NoFocus)
        self._btn_row.addWidget(self._btn_select)
        self._btn_row.addWidget(self._btn_shot)
        self._btn_row.addStretch(1)
        self._btn_row.addWidget(self._btn_quit)

        layout = QtWidgets.QVBoxLayout(self)
        layout.addWidget(self._image_label, 1)
        layout.addWidget(self._info_label)
        layout.addLayout(self._btn_row)

        self._sct = mss.mss()
        self._monitor = initial_monitor or MonitorRect(100, 100, 800, 600)

        self._timer = QtCore.QTimer(self)
        self._timer.setInterval(1000 // 30)
        self._timer.timeout.connect(self._on_timer)

        self._last_ts = time.time()
        self._frame_count = 0

        self._update_info()
        self._make_actions()
        self._wire_buttons()
        self._timer.start()


    # ----- capture -----
    def _grab_frame(self) -> Optional[np.ndarray]:
        mon = {
            "top": int(self._monitor.top),
            "left": int(self._monitor.left),
            "width": int(self._monitor.width),
            "height": int(self._monitor.height),
        }
        try:
            raw = self._sct.grab(mon)
            frame = np.asarray(raw)  # BGRA
            frame = frame[:, :, :3]  # BGR
            return frame
        except Exception:
            return None

    def _on_timer(self):
        frame = self._grab_frame()
        if frame is None:
            self._info_label.setText("Capture failed. Adjust region with 'R'.")
            return

        # Compute FPS
        self._frame_count += 1
        now = time.time()
        dt = now - self._last_ts
        if dt >= 1.0:
            fps = self._frame_count / dt
            self._info_label.setText(
                f"Area: {self._monitor.width}x{self._monitor.height}  |  FPS: {fps:.1f}  |  R: select  S: screenshot  Q/Esc: quit"
            )
            self._frame_count = 0
            self._last_ts = now

        # Convert BGR (numpy) -> QImage (RGB)
        rgb = frame[:, :, ::-1].copy()  # BGR to RGB
        h, w, _ = rgb.shape
        qimg = QtGui.QImage(rgb.data, w, h, 3 * w, QtGui.QImage.Format_RGB888)
        pix = QtGui.QPixmap.fromImage(qimg)

        # Fit into label while keeping aspect ratio
        scaled = pix.scaled(self._image_label.size(), QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation)
        self._image_label.setPixmap(scaled)

    # ----- actions -----
    def _update_info(self):
        self._info_label.setText(
            f"Area: {self._monitor.width}x{self._monitor.height}  |  R: select  S: screenshot  Q/Esc: quit"
        )

    # Key handling is provided via QActions to avoid ambiguity

    def showEvent(self, e: QtGui.QShowEvent) -> None:
        super().showEvent(e)
        # Ensure window is active and receives keys
        self.raise_()
        self.activateWindow()
        self.setFocus(QtCore.Qt.ActiveWindowFocusReason)

    # Removed duplicate QShortcuts; using QActions exclusively

    def _wire_buttons(self):
        self._btn_select.clicked.connect(self.open_selection_overlay)
        self._btn_shot.clicked.connect(self.save_screenshot)
        self._btn_quit.clicked.connect(self.close)

    def _make_actions(self):
        # Use QActions with a single shortcut each for reliable key handling
        def add_action(text, shortcut_seq, slot):
            act = QtWidgets.QAction(text, self)
            act.setShortcut(QtGui.QKeySequence(shortcut_seq))
            act.setShortcutContext(QtCore.Qt.WidgetWithChildrenShortcut)
            act.triggered.connect(slot)
            self.addAction(act)
            return act

        self._act_select = add_action("Select Region", "R", self.open_selection_overlay)
        self._act_shot = add_action("Screenshot", "S", self.save_screenshot)
        self._act_quit = add_action("Quit", "Q", self.close)
        self._act_esc = add_action("Quit", "Esc", self.close)

    def open_selection_overlay(self):
        # Use the OpenCV-based selection overlay for compatibility with existing workflow
        # Temporarily hide this window and pause updates to avoid appearing in selection
        was_running = self._timer.isActive()
        if was_running:
            self._timer.stop()
        self.hide()

        # Ensure hide is flushed to the compositor before taking any screenshots
        QtWidgets.QApplication.processEvents(QtCore.QEventLoop.AllEvents, 100)
        time.sleep(0.25)
        QtWidgets.QApplication.processEvents(QtCore.QEventLoop.AllEvents, 100)

        try:
            # Get primary screen size as base (matches mss primary monitor index 1)
            screen = self.windowHandle().screen() if self.windowHandle() else QtWidgets.QApplication.primaryScreen()
            size = screen.geometry().size()
            screen_width, screen_height = size.width(), size.height()
            # mss object used by this window
            region = select_capture_area_cv(self._sct, screen_width, screen_height)
        finally:
            # Restore window and resume updates
            self.show()
            self.raise_()
            self.activateWindow()
            self.setFocus(QtCore.Qt.ActiveWindowFocusReason)
            if was_running:
                self._timer.start()

        if region:
            rect = MonitorRect(
                int(round(region["top"])),
                int(round(region["left"])),
                int(round(region["width"])),
                int(round(region["height"]))
            )
            self._on_region_selected(rect)

    def _on_region_selected(self, rect: MonitorRect):
        # Update capture region and persist
        self._monitor = rect
        save_capture_config(rect)
        self._update_info()

    def save_screenshot(self):
        frame = self._grab_frame()
        if frame is None:
            return
        # Save with Qt to avoid cv2 dependency here
        rgb = frame[:, :, ::-1].copy()
        h, w, _ = rgb.shape
        qimg = QtGui.QImage(rgb.data, w, h, 3 * w, QtGui.QImage.Format_RGB888)
        ts = int(time.time())
        path = Path(f"screenshot_{ts}.png")
        qimg.save(str(path))


def run_qt_capture():
    import sys

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)
    mon = load_capture_config()
    w = CaptureWindow(mon)
    w.show()
    w.raise_(); w.activateWindow(); w.setFocus()
    return app.exec_()


class _GlobalKeyFilter(QtCore.QObject):
    """Global key event filter to trigger actions even if a child widget has focus."""
    def __init__(self, window: CaptureWindow):
        super().__init__(window)
        self._w = window

    def eventFilter(self, obj, event):
        if event.type() == QtCore.QEvent.KeyPress:
            key = event.key()
            if key in (QtCore.Qt.Key_Q, QtCore.Qt.Key_Escape):
                self._w.close()
                return True
            if key in (QtCore.Qt.Key_R,):
                self._w.open_selection_overlay()
                return True
            if key in (QtCore.Qt.Key_S,):
                self._w.save_screenshot()
                return True
            # Lowercase variants for some platforms
            if key in (QtCore.Qt.Key_R, QtCore.Qt.Key_S, QtCore.Qt.Key_Q):
                # already handled above, keep for clarity
                return False
        return False
