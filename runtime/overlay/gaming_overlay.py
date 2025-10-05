"""
Beautiful Gaming Overlay for EdgeRoyaleAnalytics
A professional gaming overlay that displays real-time card detection data.
"""
import sys
import time
import threading
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

try:
    from PyQt5.QtWidgets import (QApplication, QWidget, QVBoxLayout, QHBoxLayout, 
                                QLabel, QPushButton, QFrame, QGraphicsDropShadowEffect)
    from PyQt5.QtCore import Qt, QTimer, pyqtSignal, QThread
    from PyQt5.QtGui import (QFont, QPalette, QColor, QPainter, QPen, QBrush, 
                            QLinearGradient, QPixmap, QFontMetrics)
except ImportError:
    print("PyQt5 not installed. Run: pip install PyQt5")
    sys.exit(1)

# Add project root to path for imports
project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from runtime.infer.onnx_engine import CardDetectionEngine
from runtime.capture.screen import capture_single_frame
from runtime.overlay.qt_capture_view import load_capture_config, MonitorRect
from runtime.gamestate.state import GameState

@dataclass
class OverlayStats:
    """Statistics displayed on overlay."""
    fps: float = 0.0
    inference_time: float = 0.0
    detections_count: int = 0
    total_frames: int = 0
    session_time: float = 0.0

class InferenceThread(QThread):
    """Background thread for running inference."""
    detections_ready = pyqtSignal(list)  # Signal when new detections are ready
    stats_ready = pyqtSignal(object)      # Signal when new stats are ready
    
    def __init__(self):
        super().__init__()
        self.running = False
        self.engine = None
        self.game_state = GameState()
        self.confidence_threshold = 0.5
        
        # Performance tracking
        self.frame_times = []
        self.session_start = time.time()
        self.total_frames = 0
        
    def initialize_engine(self):
        """Initialize the inference engine."""
        try:
            self.engine = CardDetectionEngine()
            return True
        except Exception as e:
            print(f"Failed to initialize inference engine: {e}")
            return False
    
    def set_confidence(self, confidence: float):
        """Update confidence threshold."""
        self.confidence_threshold = max(0.1, min(0.9, confidence))
    
    def run(self):
        """Main inference loop."""
        if not self.initialize_engine():
            return
            
        self.running = True
        
        while self.running:
            try:
                # Capture frame
                frame = capture_single_frame()
                if frame is None:
                    self.msleep(33)  # ~30 FPS fallback
                    continue
                
                # Run inference
                start_time = time.time()
                detections = self.engine.predict(frame, self.confidence_threshold)
                inference_time = (time.time() - start_time) * 1000
                
                # Update game state
                self.game_state.ingest_detections(detections)
                
                # Track performance
                self.total_frames += 1
                current_time = time.time()
                self.frame_times.append(current_time)
                
                # Keep only last second of frame times for FPS calculation
                self.frame_times = [t for t in self.frame_times if current_time - t < 1.0]
                
                # Calculate stats
                stats = OverlayStats(
                    fps=len(self.frame_times),
                    inference_time=inference_time,
                    detections_count=len(detections),
                    total_frames=self.total_frames,
                    session_time=current_time - self.session_start
                )
                
                # Emit signals
                self.detections_ready.emit(detections)
                self.stats_ready.emit(stats)
                
                # Small delay to prevent CPU overload
                self.msleep(10)
                
            except Exception as e:
                print(f"Inference thread error: {e}")
                self.msleep(100)
    
    def stop(self):
        """Stop the inference thread."""
        self.running = False

class DetectionCard(QFrame):
    """Widget showing individual card detection."""
    
    def __init__(self, card_name: str, confidence: float):
        super().__init__()
        self.card_name = card_name
        self.confidence = confidence
        self.setup_ui()
    
    def setup_ui(self):
        """Setup the card detection UI."""
        self.setFixedSize(200, 60)
        self.setStyleSheet("""
            QFrame {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                    stop:0 rgba(40, 44, 52, 0.95),
                    stop:1 rgba(50, 54, 62, 0.95));
                border: 2px solid rgba(76, 175, 80, 0.8);
                border-radius: 8px;
                margin: 2px;
            }
        """)
        
        # Add drop shadow
        shadow = QGraphicsDropShadowEffect()
        shadow.setBlurRadius(10)
        shadow.setColor(QColor(0, 0, 0, 150))
        shadow.setOffset(2, 2)
        self.setGraphicsEffect(shadow)
        
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 4, 8, 4)
        
        # Card name
        name_label = QLabel(self.card_name)
        name_label.setStyleSheet("""
            QLabel {
                color: #4CAF50;
                font-weight: bold;
                font-size: 14px;
                background: transparent;
                border: none;
            }
        """)
        
        # Confidence
        conf_text = f"{self.confidence:.1%}"
        conf_label = QLabel(conf_text)
        conf_label.setStyleSheet("""
            QLabel {
                color: #FFF;
                font-size: 12px;
                background: transparent;
                border: none;
            }
        """)
        
        layout.addWidget(name_label)
        layout.addWidget(conf_label)

class StatsPanel(QFrame):
    """Panel showing performance statistics."""
    
    def __init__(self):
        super().__init__()
        self.setup_ui()
    
    def setup_ui(self):
        """Setup the stats panel UI."""
        self.setFixedSize(220, 120)
        self.setStyleSheet("""
            QFrame {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                    stop:0 rgba(30, 34, 42, 0.95),
                    stop:1 rgba(40, 44, 52, 0.95));
                border: 2px solid rgba(33, 150, 243, 0.8);
                border-radius: 10px;
                margin: 4px;
            }
        """)
        
        # Add drop shadow
        shadow = QGraphicsDropShadowEffect()
        shadow.setBlurRadius(15)
        shadow.setColor(QColor(0, 0, 0, 180))
        shadow.setOffset(3, 3)
        self.setGraphicsEffect(shadow)
        
        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 8, 12, 8)
        
        # Title
        title = QLabel("ERA Stats")
        title.setStyleSheet("""
            QLabel {
                color: #2196F3;
                font-weight: bold;
                font-size: 16px;
                background: transparent;
                border: none;
            }
        """)
        layout.addWidget(title)
        
        # Stats labels
        self.fps_label = QLabel("FPS: --")
        self.inference_label = QLabel("Inference: --ms")
        self.detections_label = QLabel("Detections: 0")
        self.session_label = QLabel("Session: 0s")
        
        for label in [self.fps_label, self.inference_label, self.detections_label, self.session_label]:
            label.setStyleSheet("""
                QLabel {
                    color: #FFF;
                    font-size: 11px;
                    background: transparent;
                    border: none;
                    margin: 1px 0;
                }
            """)
            layout.addWidget(label)
    
    def update_stats(self, stats: OverlayStats):
        """Update the displayed statistics."""
        # FPS with color coding
        fps_color = "#4CAF50" if stats.fps >= 25 else "#FF9800" if stats.fps >= 15 else "#F44336"
        self.fps_label.setText(f"FPS: <span style='color: {fps_color}'>{stats.fps:.1f}</span>")
        
        # Inference time with color coding
        inf_color = "#4CAF50" if stats.inference_time <= 30 else "#FF9800" if stats.inference_time <= 60 else "#F44336"
        self.inference_label.setText(f"Inference: <span style='color: {inf_color}'>{stats.inference_time:.1f}ms</span>")
        
        # Detections
        self.detections_label.setText(f"Detections: <span style='color: #4CAF50'>{stats.detections_count}</span>")
        
        # Session time
        session_min = int(stats.session_time // 60)
        session_sec = int(stats.session_time % 60)
        self.session_label.setText(f"Session: {session_min:02d}:{session_sec:02d}")

class ControlPanel(QFrame):
    """Control panel with buttons and settings."""
    
    confidence_changed = pyqtSignal(float)
    
    def __init__(self):
        super().__init__()
        self.confidence = 0.5
        self.setup_ui()
    
    def setup_ui(self):
        """Setup the control panel UI."""
        self.setFixedSize(220, 80)
        self.setStyleSheet("""
            QFrame {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                    stop:0 rgba(30, 34, 42, 0.95),
                    stop:1 rgba(40, 44, 52, 0.95));
                border: 2px solid rgba(156, 39, 176, 0.8);
                border-radius: 10px;
                margin: 4px;
            }
        """)
        
        # Add drop shadow
        shadow = QGraphicsDropShadowEffect()
        shadow.setBlurRadius(15)
        shadow.setColor(QColor(0, 0, 0, 180))
        shadow.setOffset(3, 3)
        self.setGraphicsEffect(shadow)
        
        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 8, 12, 8)
        
        # Title
        title = QLabel("Controls")
        title.setStyleSheet("""
            QLabel {
                color: #9C27B0;
                font-weight: bold;
                font-size: 14px;
                background: transparent;
                border: none;
            }
        """)
        layout.addWidget(title)
        
        # Buttons layout
        btn_layout = QHBoxLayout()
        
        # Confidence buttons
        conf_down = QPushButton("-")
        conf_down.setFixedSize(30, 25)
        conf_down.clicked.connect(lambda: self.adjust_confidence(-0.1))
        
        self.conf_label = QLabel(f"{self.confidence:.1f}")
        self.conf_label.setAlignment(Qt.AlignCenter)
        self.conf_label.setFixedWidth(40)
        
        conf_up = QPushButton("+")
        conf_up.setFixedSize(30, 25)
        conf_up.clicked.connect(lambda: self.adjust_confidence(0.1))
        
        for btn in [conf_down, conf_up]:
            btn.setStyleSheet("""
                QPushButton {
                    background: rgba(156, 39, 176, 0.8);
                    color: white;
                    border: none;
                    border-radius: 4px;
                    font-weight: bold;
                }
                QPushButton:hover {
                    background: rgba(156, 39, 176, 1.0);
                }
                QPushButton:pressed {
                    background: rgba(136, 19, 156, 1.0);
                }
            """)
        
        self.conf_label.setStyleSheet("""
            QLabel {
                color: #FFF;
                font-size: 12px;
                background: transparent;
                border: none;
            }
        """)
        
        btn_layout.addWidget(QLabel("Conf:"))
        btn_layout.addWidget(conf_down)
        btn_layout.addWidget(self.conf_label)
        btn_layout.addWidget(conf_up)
        btn_layout.addStretch()
        
        layout.addLayout(btn_layout)
    
    def adjust_confidence(self, delta: float):
        """Adjust confidence threshold."""
        self.confidence = max(0.1, min(0.9, self.confidence + delta))
        self.conf_label.setText(f"{self.confidence:.1f}")
        self.confidence_changed.emit(self.confidence)

class GamingOverlay(QWidget):
    """Main gaming overlay widget."""
    
    def __init__(self):
        super().__init__()
        self.capture_config = None
        self.inference_thread = None
        self.detection_cards = []
        
        self.setup_ui()
        self.setup_inference()
        
    def setup_ui(self):
        """Setup the main overlay UI."""
        # Load capture configuration
        self.capture_config = load_capture_config()
        if not self.capture_config:
            print("No capture configuration found. Please run the capture tool first.")
            return
        
        # Window properties
        self.setWindowTitle("EdgeRoyaleAnalytics - Gaming Overlay")
        self.setWindowFlags(Qt.WindowStaysOnTopHint | Qt.FramelessWindowHint)
        self.setAttribute(Qt.WA_TranslucentBackground)
        
        # Position over capture area
        self.setGeometry(
            self.capture_config.left - 10,
            self.capture_config.top - 10,
            self.capture_config.width + 20,
            self.capture_config.height + 20
        )
        
        # Main layout
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(10, 10, 10, 10)
        
        # Top panel (stats and controls)
        top_panel = QHBoxLayout()
        
        self.stats_panel = StatsPanel()
        self.control_panel = ControlPanel()
        self.control_panel.confidence_changed.connect(self.on_confidence_changed)
        
        top_panel.addWidget(self.stats_panel)
        top_panel.addStretch()
        top_panel.addWidget(self.control_panel)
        
        main_layout.addLayout(top_panel)
        main_layout.addStretch()
        
        # Detection cards area
        self.detections_layout = QHBoxLayout()
        main_layout.addLayout(self.detections_layout)
        
        # Add title with ERA branding
        self.add_title()
        
    def add_title(self):
        """Add ERA title to overlay."""
        title_label = QLabel("ERA - EdgeRoyaleAnalytics")
        title_label.setStyleSheet("""
            QLabel {
                color: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #4CAF50,
                    stop:0.5 #2196F3,
                    stop:1 #9C27B0);
                font-size: 18px;
                font-weight: bold;
                background: rgba(20, 20, 20, 0.8);
                border: 2px solid rgba(255, 255, 255, 0.3);
                border-radius: 8px;
                padding: 8px 16px;
                margin: 4px;
            }
        """)
        title_label.setAlignment(Qt.AlignCenter)
        
        # Add to top of layout
        self.layout().insertWidget(0, title_label)
    
    def setup_inference(self):
        """Setup the inference thread."""
        self.inference_thread = InferenceThread()
        self.inference_thread.detections_ready.connect(self.update_detections)
        self.inference_thread.stats_ready.connect(self.update_stats)
        
    def start_inference(self):
        """Start the inference engine."""
        if self.inference_thread and not self.inference_thread.isRunning():
            self.inference_thread.start()
    
    def stop_inference(self):
        """Stop the inference engine."""
        if self.inference_thread and self.inference_thread.isRunning():
            self.inference_thread.stop()
            self.inference_thread.wait(3000)  # Wait up to 3 seconds
    
    def update_detections(self, detections: List[Dict[str, Any]]):
        """Update the detection cards display."""
        # Clear existing cards
        for card in self.detection_cards:
            card.deleteLater()
        self.detection_cards.clear()
        
        # Add new detection cards
        for detection in detections[:4]:  # Limit to 4 cards to avoid clutter
            card = DetectionCard(detection['class_name'], detection['confidence'])
            self.detection_cards.append(card)
            self.detections_layout.addWidget(card)
        
        # Add stretch to keep cards aligned
        self.detections_layout.addStretch()
    
    def update_stats(self, stats: OverlayStats):
        """Update the statistics display."""
        self.stats_panel.update_stats(stats)
    
    def on_confidence_changed(self, confidence: float):
        """Handle confidence threshold change."""
        if self.inference_thread:
            self.inference_thread.set_confidence(confidence)
    
    def keyPressEvent(self, event):
        """Handle key press events."""
        if event.key() == Qt.Key_Escape:
            self.close()
        elif event.key() == Qt.Key_F11:
            # Toggle fullscreen
            if self.isFullScreen():
                self.showNormal()
            else:
                self.showFullScreen()
        super().keyPressEvent(event)
    
    def closeEvent(self, event):
        """Handle window close event."""
        self.stop_inference()
        event.accept()

class OverlayManager:
    """Manager for the gaming overlay system."""
    
    def __init__(self):
        self.app = None
        self.overlay = None
    
    def start(self):
        """Start the gaming overlay."""
        # Check prerequisites
        if not self.check_prerequisites():
            return False
        
        # Create QApplication if needed
        if not QApplication.instance():
            self.app = QApplication(sys.argv)
        else:
            self.app = QApplication.instance()
        
        # Create and show overlay
        self.overlay = GamingOverlay()
        self.overlay.show()
        
        # Start inference
        self.overlay.start_inference()
        
        print("🎮 Gaming Overlay Started!")
        print("📋 Controls:")
        print("   ESC = Close overlay")
        print("   F11 = Toggle fullscreen")
        print("   +/- = Adjust confidence")
        
        # Run the application
        try:
            if self.app:
                return self.app.exec_()
            return 0
        except KeyboardInterrupt:
            print("\n🛑 Overlay stopped by user")
            return 0
    
    def check_prerequisites(self):
        """Check if all prerequisites are met."""
        # Check if model exists
        model_path = Path("model/weights/best.onnx")
        if not model_path.exists():
            print(f"❌ Model not found: {model_path}")
            print("📝 Export your model first with: python scripts/export_model.py")
            return False
        
        # Check if capture config exists
        config = load_capture_config()
        if not config:
            print("❌ No capture area configured")
            print("📝 Run the capture tool first: python runtime/capture/screen.py")
            print("📝 Use 'R' to select your game area, then close the window")
            return False
        
        print("✅ All prerequisites met!")
        return True

def main():
    """Main function to start the gaming overlay."""
    print("🎮 EdgeRoyaleAnalytics - Gaming Overlay")
    print("=" * 50)
    
    manager = OverlayManager()
    return manager.start()

if __name__ == "__main__":
    sys.exit(main())