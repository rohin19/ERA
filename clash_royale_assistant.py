"""
Production-Ready Clash Royale Analytics System
Clean architecture with proper separation of concerns
"""
import sys
import time
import threading
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
from enum import Enum
import json

# Add project root to path
project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

try:
    from PyQt5.QtWidgets import (QApplication, QWidget, QVBoxLayout, QHBoxLayout, 
                                QLabel, QFrame, QProgressBar, QListWidget, QListWidgetItem)
    from PyQt5.QtCore import Qt, QTimer, pyqtSignal, QObject, QThread
    from PyQt5.QtGui import QFont, QColor, QPalette, QPainter, QPen, QBrush, QImage, QPixmap
    import numpy as np
    import cv2
    import mss
    
    from runtime.overlay.qt_capture_view import MonitorRect, load_capture_config
    from runtime.infer.onnx_engine import CardDetectionEngine
    from runtime.gamestate.state import GameState
    
    DEPENDENCIES_OK = True
except ImportError as e:
    print(f"❌ Dependencies missing: {e}")
    print("📝 Install: pip install PyQt5 opencv-python mss numpy onnxruntime")
    DEPENDENCIES_OK = False

class GamePhase(Enum):
    """Game phases for tracking"""
    LOADING = "loading"
    PLAYING = "playing"
    OVERTIME = "overtime"
    ENDED = "ended"

@dataclass
class OpponentState:
    """Track opponent's state"""
    deck_cards: List[str] = field(default_factory=list)
    current_hand: List[str] = field(default_factory=list)
    next_card: Optional[str] = None
    elixir: float = 5.0
    plays: List[Dict] = field(default_factory=list)
    last_play_time: float = 0.0

@dataclass
class GameAnalytics:
    """Complete game analytics state"""
    phase: GamePhase = GamePhase.LOADING
    opponent: OpponentState = field(default_factory=OpponentState)
    game_time: float = 0.0
    total_detections: int = 0
    detection_confidence: float = 0.0

class CaptureEngine(QThread):
    """Handles screen capture and inference"""
    
    # Signals for data pipeline
    frame_captured = pyqtSignal(np.ndarray)  # Raw frame
    detections_ready = pyqtSignal(list)      # Card detections
    analytics_updated = pyqtSignal(object)   # Game analytics
    
    def __init__(self, capture_config: MonitorRect, model_path: str):
        super().__init__()
        self.capture_config = capture_config
        self.model_path = model_path
        self.running = False
        
        # Initialize components
        self.inference_engine = None
        self.game_state = GameState()
        self.analytics = GameAnalytics()
        
        # Performance tracking
        self.fps = 0.0
        self.frame_count = 0
        self.last_fps_time = time.time()
        
    def initialize(self) -> bool:
        """Initialize the capture engine"""
        try:
            self.inference_engine = CardDetectionEngine(self.model_path)
            print("✅ Inference engine initialized")
            return True
        except Exception as e:
            print(f"❌ Failed to initialize inference engine: {e}")
            return False
    
    def run(self):
        """Main capture and inference loop"""
        if not self.initialize():
            return
            
        self.running = True
        print("🚀 Capture engine started")
        
        with mss.mss() as sct:
            monitor_config = {
                "top": int(self.capture_config.top),
                "left": int(self.capture_config.left),
                "width": int(self.capture_config.width),
                "height": int(self.capture_config.height)
            }
            
            while self.running:
                try:
                    # 1. CAPTURE: Get frame from game
                    screenshot = sct.grab(monitor_config)
                    frame = np.array(screenshot)
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
                    self.frame_captured.emit(frame)
                    
                    # 2. INFERENCE: Detect cards
                    detections = self.inference_engine.predict(frame, confidence_threshold=0.5)
                    self.detections_ready.emit(detections)
                    
                    # 3. ANALYTICS: Update game state
                    self._update_analytics(detections)
                    self.analytics_updated.emit(self.analytics)
                    
                    # Update FPS
                    self._update_fps()
                    
                    # Control frame rate (30 FPS target)
                    self.msleep(33)
                    
                except Exception as e:
                    print(f"❌ Capture loop error: {e}")
                    self.msleep(100)
    
    def _update_analytics(self, detections: List[Dict]):
        """Update game analytics with new detections"""
        current_time = time.time()
        
        # Update detection stats
        self.analytics.total_detections += len(detections)
        if detections:
            avg_confidence = sum(d['confidence'] for d in detections) / len(detections)
            self.analytics.detection_confidence = avg_confidence
        
        # Update opponent tracking
        for detection in detections:
            card_name = detection['class_name']
            confidence = detection['confidence']
            
            # Track opponent's deck
            if card_name not in self.analytics.opponent.deck_cards:
                self.analytics.opponent.deck_cards.append(card_name)
                print(f"🎴 New card discovered: {card_name}")
            
            # Estimate elixir based on plays
            self._estimate_opponent_elixir(card_name, current_time)
            
            # Track plays
            play_record = {
                'card': card_name,
                'confidence': confidence,
                'timestamp': current_time,
                'game_time': self.analytics.game_time
            }
            self.analytics.opponent.plays.append(play_record)
            self.analytics.opponent.last_play_time = current_time
        
        # Update game time
        self.analytics.game_time += 0.033  # ~30 FPS
    
    def _estimate_opponent_elixir(self, card_name: str, timestamp: float):
        """Estimate opponent's elixir based on card plays"""
        # Card costs (simplified)
        card_costs = {
            "Baby Dragon": 4, "Bomber": 2, "Dart Goblin": 3, "Giant": 5,
            "Hog Rider": 4, "Knight": 3, "Mini Pekka": 4, "Valkyrie": 4
        }
        
        cost = card_costs.get(card_name, 3)  # Default cost
        
        # Deduct elixir cost
        self.analytics.opponent.elixir = max(0, self.analytics.opponent.elixir - cost)
        
        # Elixir regenerates over time
        time_since_last = timestamp - self.analytics.opponent.last_play_time
        elixir_regen = min(time_since_last * 1.4, 10 - self.analytics.opponent.elixir)  # 1.4 elixir/sec
        self.analytics.opponent.elixir = min(10, self.analytics.opponent.elixir + elixir_regen)
    
    def _update_fps(self):
        """Update FPS counter"""
        self.frame_count += 1
        current_time = time.time()
        if current_time - self.last_fps_time >= 1.0:
            self.fps = self.frame_count / (current_time - self.last_fps_time)
            self.frame_count = 0
            self.last_fps_time = current_time
    
    def stop(self):
        """Stop the capture engine"""
        self.running = False
        self.wait()

class GameFeedWidget(QWidget):
    """Displays the live game feed"""
    
    def __init__(self, width: int, height: int):
        super().__init__()
        self.setFixedSize(width, height)
        self.setStyleSheet("background-color: black; border: 2px solid #333;")
        
        self.current_frame = None
        self.detections = []
        
    def update_frame(self, frame: np.ndarray):
        """Update the displayed frame"""
        self.current_frame = frame
        self.update()
    
    def update_detections(self, detections: List[Dict]):
        """Update detection overlays"""
        self.detections = detections
        self.update()
    
    def paintEvent(self, event):
        """Paint the game feed with overlays"""
        painter = QPainter(self)
        
        if self.current_frame is not None:
            # Convert frame to Qt format and display
            height, width, channel = self.current_frame.shape
            bytes_per_line = 3 * width
            q_image = QImage(self.current_frame.data, width, height, 
                           bytes_per_line, QImage.Format_RGB888).rgbSwapped()
            
            # Scale to widget size
            scaled_pixmap = QPixmap.fromImage(q_image).scaled(
                self.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
            )
            painter.drawPixmap(0, 0, scaled_pixmap)
            
            # Draw detection overlays
            self._draw_detections(painter)
    
    def _draw_detections(self, painter: QPainter):
        """Draw detection bounding boxes"""
        if not self.detections:
            return
            
        # Scale factor for detection coordinates
        scale_x = self.width() / 640  # Model input size
        scale_y = self.height() / 640
        
        painter.setPen(QPen(QColor(0, 255, 0), 2))  # Green boxes
        
        for detection in self.detections:
            bbox = detection['bbox']
            x1 = int(bbox[0] * scale_x)
            y1 = int(bbox[1] * scale_y)
            x2 = int(bbox[2] * scale_x)
            y2 = int(bbox[3] * scale_y)
            
            # Draw bounding box
            painter.drawRect(x1, y1, x2-x1, y2-y1)
            
            # Draw label
            label = f"{detection['class_name']} ({detection['confidence']:.0%})"
            painter.setPen(QPen(QColor(255, 255, 255)))
            painter.drawText(x1, y1-5, label)

class AnalyticsPanel(QFrame):
    """Analytics and opponent tracking panel"""
    
    def __init__(self):
        super().__init__()
        self.setFixedWidth(300)
        self.setStyleSheet("""
            QFrame {
                background-color: #2b2b2b;
                border-radius: 8px;
                padding: 10px;
            }
            QLabel {
                color: white;
                font-size: 12px;
            }
        """)
        
        self.setup_ui()
        
    def setup_ui(self):
        """Setup the analytics UI"""
        layout = QVBoxLayout(self)
        
        # Title
        title = QLabel("🎮 Clash Royale Analytics")
        title.setFont(QFont("Arial", 14, QFont.Bold))
        title.setStyleSheet("color: #4CAF50; margin-bottom: 10px;")
        layout.addWidget(title)
        
        # Opponent Elixir
        self.elixir_label = QLabel("⚡ Opponent Elixir: 5.0/10")
        self.elixir_bar = QProgressBar()
        self.elixir_bar.setRange(0, 100)
        self.elixir_bar.setValue(50)
        self.elixir_bar.setStyleSheet("""
            QProgressBar {
                border: 2px solid grey;
                border-radius: 5px;
                text-align: center;
                color: white;
            }
            QProgressBar::chunk {
                background-color: #4CAF50;
                border-radius: 3px;
            }
        """)
        layout.addWidget(self.elixir_label)
        layout.addWidget(self.elixir_bar)
        
        # Opponent Deck
        deck_label = QLabel("🎴 Opponent Deck:")
        deck_label.setFont(QFont("Arial", 12, QFont.Bold))
        layout.addWidget(deck_label)
        
        self.deck_list = QListWidget()
        self.deck_list.setMaximumHeight(120)
        self.deck_list.setStyleSheet("""
            QListWidget {
                background-color: #1e1e1e;
                border: 1px solid #555;
                border-radius: 4px;
                color: white;
            }
            QListWidgetItem {
                padding: 4px;
                border-bottom: 1px solid #333;
            }
        """)
        layout.addWidget(self.deck_list)
        
        # Recent Plays
        plays_label = QLabel("📋 Recent Plays:")
        plays_label.setFont(QFont("Arial", 12, QFont.Bold))
        layout.addWidget(plays_label)
        
        self.plays_list = QListWidget()
        self.plays_list.setMaximumHeight(100)
        self.plays_list.setStyleSheet(self.deck_list.styleSheet())
        layout.addWidget(self.plays_list)
        
        # Stats
        self.stats_label = QLabel("📊 Detections: 0 | Confidence: 0%")
        layout.addWidget(self.stats_label)
        
        layout.addStretch()
    
    def update_analytics(self, analytics: GameAnalytics):
        """Update the analytics display"""
        # Update elixir
        elixir = analytics.opponent.elixir
        self.elixir_label.setText(f"⚡ Opponent Elixir: {elixir:.1f}/10")
        self.elixir_bar.setValue(int(elixir * 10))
        
        # Update elixir bar color based on level
        if elixir >= 7:
            color = "#FF5722"  # Red (dangerous)
        elif elixir >= 4:
            color = "#FF9800"  # Orange (moderate)
        else:
            color = "#4CAF50"  # Green (safe)
            
        self.elixir_bar.setStyleSheet(f"""
            QProgressBar {{
                border: 2px solid grey;
                border-radius: 5px;
                text-align: center;
                color: white;
            }}
            QProgressBar::chunk {{
                background-color: {color};
                border-radius: 3px;
            }}
        """)
        
        # Update deck
        self.deck_list.clear()
        for card in analytics.opponent.deck_cards:
            item = QListWidgetItem(f"🎴 {card}")
            self.deck_list.addItem(item)
        
        # Update recent plays
        self.plays_list.clear()
        recent_plays = analytics.opponent.plays[-5:]  # Last 5 plays
        for play in reversed(recent_plays):
            time_str = f"{play['game_time']:.1f}s"
            confidence_str = f"{play['confidence']:.0%}"
            item = QListWidgetItem(f"⏰ {time_str}: {play['card']} ({confidence_str})")
            self.plays_list.addItem(item)
        
        # Update stats
        self.stats_label.setText(
            f"📊 Detections: {analytics.total_detections} | "
            f"Confidence: {analytics.detection_confidence:.0%}"
        )

class ClashRoyaleAssistant(QWidget):
    """Main application window"""
    
    def __init__(self):
        super().__init__()
        self.capture_engine = None
        self.setup_ui()
        self.setup_capture()
        
    def setup_ui(self):
        """Setup the main UI"""
        self.setWindowTitle("🎮 Clash Royale Analytics Assistant")
        self.setGeometry(100, 100, 1200, 800)
        self.setStyleSheet("background-color: #1e1e1e;")
        
        # Main layout
        main_layout = QHBoxLayout(self)
        
        # Game feed (left side)
        self.game_feed = GameFeedWidget(800, 600)
        main_layout.addWidget(self.game_feed)
        
        # Analytics panel (right side)
        self.analytics_panel = AnalyticsPanel()
        main_layout.addWidget(self.analytics_panel)
        
    def setup_capture(self):
        """Setup the capture engine"""
        # Load capture configuration
        capture_config = load_capture_config()
        if not capture_config:
            self.show_error("No capture area configured. Run screen capture tool first.")
            return
            
        # Check if model exists
        model_path = Path("model/weights/best.onnx")
        if not model_path.exists():
            self.show_error(f"Model not found: {model_path}")
            return
            
        # Create capture engine
        self.capture_engine = CaptureEngine(capture_config, str(model_path))
        
        # Connect signals
        self.capture_engine.frame_captured.connect(self.game_feed.update_frame)
        self.capture_engine.detections_ready.connect(self.game_feed.update_detections)
        self.capture_engine.analytics_updated.connect(self.analytics_panel.update_analytics)
        
        # Start capture
        self.capture_engine.start()
        print("✅ Capture engine started")
    
    def show_error(self, message: str):
        """Show error message"""
        error_label = QLabel(f"❌ {message}")
        error_label.setStyleSheet("color: red; font-size: 14px; margin: 20px;")
        error_label.setAlignment(Qt.AlignCenter)
        
        layout = QVBoxLayout(self)
        layout.addWidget(error_label)
    
    def closeEvent(self, event):
        """Handle window close"""
        if self.capture_engine:
            self.capture_engine.stop()
        event.accept()

def main():
    """Start the Clash Royale Assistant"""
    if not DEPENDENCIES_OK:
        return 1
        
    print("🎮 Clash Royale Analytics Assistant")
    print("=" * 50)
    
    app = QApplication(sys.argv)
    assistant = ClashRoyaleAssistant()
    assistant.show()
    
    return app.exec_()

if __name__ == "__main__":
    sys.exit(main())