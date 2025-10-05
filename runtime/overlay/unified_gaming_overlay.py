"""Complete integrated gaming system: Capture → Inference → Game State → Overlay Display"""
import sys
import time
import cv2
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional
import mss

# Add project root to path for imports
project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Import all components
try:
    from runtime.overlay.qt_capture_view import MonitorRect, load_capture_config
    from runtime.infer.onnx_engine import CardDetectionEngine
    from runtime.gamestate.gamestate import GameState
    IMPORTS_SUCCESSFUL = True
except ImportError as e:
    print(f"Import error: {e}")
    # Fallback to simpler game state if advanced one not available
    try:
        from runtime.gamestate.state import GameState
        from runtime.overlay.qt_capture_view import MonitorRect, load_capture_config
        from runtime.infer.onnx_engine import CardDetectionEngine
        IMPORTS_SUCCESSFUL = True
    except ImportError as e2:
        print(f"Fallback import also failed: {e2}")
        print("❌ Cannot import required modules")
        IMPORTS_SUCCESSFUL = False
        
        # Define minimal fallbacks to prevent crashes
        class MonitorRect:
            def __init__(self, top=0, left=0, width=640, height=480):
                self.top = top
                self.left = left
                self.width = width
                self.height = height
        
        def load_capture_config():
            return None
            
        class GameState:
            def __init__(self):
                self.elixir = 5.0
                self.plays = []
                self.last_detections = []
            
            def ingest_detections(self, detections):
                self.last_detections = detections
            
            def snapshot(self):
                return {
                    'elixir': self.elixir,
                    'plays': self.plays,
                    'last_detections': self.last_detections
                }
        
        class CardDetectionEngine:
            def __init__(self, model_path):
                raise ImportError("CardDetectionEngine not available")
            
            def predict(self, frame, confidence_threshold=0.4):
                return []

class IntegratedGamingSystem:
    """Complete system: Gameplay → Capture → Inference → Game State → Display"""
    
    def __init__(self, model_path: str = "model/weights/best.onnx"):
        # Load capture configuration
        self.monitor_rect = load_capture_config()
        if not self.monitor_rect:
            raise ValueError("No capture area configured. Run: python runtime/capture/screen.py")
        
        # Initialize all components
        self.engine = CardDetectionEngine(model_path)
        self.game_state = GameState()
        
        # Display setup
        self.window_name = "🎮 ERA - Live Gaming Feed"
        
        # Performance tracking
        self.frame_count = 0
        self.last_fps_time = time.time()
        self.current_fps = 0.0
        
        print(f"✅ Gaming system initialized")
        print(f"📺 Capture area: {self.monitor_rect.width}x{self.monitor_rect.height}")
        print(f"🎯 Model: {model_path}")
    
    def start(self):
        """Start the complete gaming system."""
        print("🚀 Starting ERA Gaming System...")
        print("📊 Data Pipeline: Gameplay → Capture → Inference → Game State → Display")
        print("🎮 Controls: 'q' = quit, 's' = screenshot, 'r' = reset game state")
        
        # Create OpenCV window
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.window_name, self.monitor_rect.width, self.monitor_rect.height)
        
        # Main processing loop
        try:
            with mss.mss() as sct:
                # Configure screen capture area
                monitor_config = {
                    "top": int(self.monitor_rect.top),
                    "left": int(self.monitor_rect.left),
                    "width": int(self.monitor_rect.width),
                    "height": int(self.monitor_rect.height)
                }
                
                while True:
                    # 1. CAPTURE: Get frame from gameplay
                    screenshot = sct.grab(monitor_config)
                    frame = np.array(screenshot)
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
                    
                    # 2. INFERENCE: Detect cards in frame
                    detections = self.engine.predict(frame, confidence_threshold=0.4)
                    
                    # 3. GAME STATE: Update with detections
                    self.game_state.ingest_detections(detections)
                    game_snapshot = self.game_state.snapshot()
                    
                    # 4. DISPLAY: Create overlay with game state info
                    display_frame = self._create_overlay_display(frame, detections, game_snapshot)
                    
                    # 5. SHOW: Display the complete result
                    cv2.imshow(self.window_name, display_frame)
                    
                    # Update FPS
                    self._update_fps()
                    
                    # Handle user input
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q') or cv2.getWindowProperty(self.window_name, cv2.WND_PROP_VISIBLE) < 1:
                        break
                    elif key == ord('s'):
                        self._save_screenshot(display_frame)
                    elif key == ord('r'):
                        self._reset_game_state()
                    elif key == ord('g'):
                        self._toggle_game_tracking()
                        
        except KeyboardInterrupt:
            print("\n🛑 Interrupted by user")
        finally:
            cv2.destroyAllWindows()
        
        return 0
    
    def _create_overlay_display(self, frame: np.ndarray, detections: List[Dict], game_state: Dict) -> np.ndarray:
        """Create the complete overlay display with game state information."""
        # Start with the original frame
        display_frame = frame.copy()
        
        # Draw detection boxes (from inference)
        display_frame = self._draw_detections(display_frame, detections)
        
        # Draw game state overlay (the important part!)
        display_frame = self._draw_game_state_overlay(display_frame, game_state)
        
        # Draw system info
        display_frame = self._draw_system_info(display_frame)
        
        return display_frame
    
    def _draw_detections(self, frame: np.ndarray, detections: List[Dict]) -> np.ndarray:
        """Draw detection bounding boxes on frame."""
        for detection in detections:
            # Scale coordinates from model space (640x640) to frame space
            bbox = detection['bbox']
            scale_x = frame.shape[1] / 640
            scale_y = frame.shape[0] / 640
            
            x1 = int(bbox[0] * scale_x)
            y1 = int(bbox[1] * scale_y)
            x2 = int(bbox[2] * scale_x)
            y2 = int(bbox[3] * scale_y)
            
            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Draw label
            label = f"{detection['class_name']}: {detection['confidence']:.0%}"
            cv2.putText(frame, label, (x1, y1 - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        return frame
    
    def _draw_game_state_overlay(self, frame: np.ndarray, game_state: Dict) -> np.ndarray:
        """Draw game state information overlay - THIS IS THE KEY PART!"""
        # Panel background
        panel_height = 150
        panel_width = 300
        overlay = frame.copy()
        
        # Draw semi-transparent background panel
        cv2.rectangle(overlay, (10, 10), (panel_width, panel_height), (0, 0, 0), -1)
        cv2.addWeighted(frame, 0.7, overlay, 0.3, 0, frame)
        
        # Game state information
        y_offset = 35
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.7
        color = (255, 255, 255)
        thickness = 2
        
        # 1. Game Status
        elixir = game_state.get('elixir', 0.0)
        status_text = f"⚡ Elixir: {elixir:.1f}/10"
        elixir_color = self._get_elixir_color(elixir)
        cv2.putText(frame, status_text, (20, y_offset), font, font_scale, elixir_color, thickness)
        y_offset += 30
        
        # 2. Detection Count
        detection_count = len(game_state.get('last_detections', []))
        detections_text = f"🎴 Detections: {detection_count}"
        cv2.putText(frame, detections_text, (20, y_offset), font, 0.6, color, 2)
        y_offset += 25
        
        # 3. Recent Plays
        plays = game_state.get('plays', [])
        if plays:
            plays_text = f"📋 Total Plays: {len(plays)}"
            cv2.putText(frame, plays_text, (20, y_offset), font, 0.6, color, 2)
            y_offset += 25
            
            # Show last play
            if plays:
                last_play = plays[-1]
                last_play_text = f"Last: {last_play['card']}"
                cv2.putText(frame, last_play_text, (20, y_offset), font, 0.5, (100, 255, 255), 2)
        else:
            plays_text = "📋 No plays yet"
            cv2.putText(frame, plays_text, (20, y_offset), font, 0.6, (128, 128, 128), 2)
        
        # Draw elixir bar
        self._draw_elixir_bar(frame, elixir)
        
        return frame
    
    def _draw_elixir_bar(self, frame: np.ndarray, elixir: float):
        """Draw elixir bar visualization."""
        # Bar dimensions
        bar_x = frame.shape[1] - 220
        bar_y = 20
        bar_width = 200
        bar_height = 20
        
        # Background bar
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), (50, 50, 50), -1)
        
        # Elixir fill
        fill_width = int((elixir / 10.0) * bar_width)
        fill_color = self._get_elixir_color(elixir)
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + fill_width, bar_y + bar_height), fill_color, -1)
        
        # Border
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), (255, 255, 255), 2)
        
        # Text
        cv2.putText(frame, f"{elixir:.1f}", (bar_x + bar_width + 10, bar_y + 15), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    def _get_elixir_color(self, elixir: float) -> tuple:
        """Get color based on elixir level."""
        if elixir >= 8:
            return (255, 0, 255)  # Magenta (full)
        elif elixir >= 5:
            return (0, 255, 0)    # Green (good)
        elif elixir >= 2:
            return (0, 255, 255)  # Yellow (low)
        else:
            return (0, 0, 255)    # Red (critical)
    
    def _draw_system_info(self, frame: np.ndarray) -> np.ndarray:
        """Draw system performance info."""
        # FPS counter
        fps_text = f"FPS: {self.current_fps:.1f}"
        cv2.putText(frame, fps_text, (10, frame.shape[0] - 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Controls
        controls_text = "Controls: Q=quit, S=screenshot, R=reset, G=toggle"
        cv2.putText(frame, controls_text, (10, frame.shape[0] - 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        return frame
    
    def _update_fps(self):
        """Update FPS counter."""
        self.frame_count += 1
        current_time = time.time()
        if current_time - self.last_fps_time >= 1.0:
            self.current_fps = self.frame_count / (current_time - self.last_fps_time)
            self.frame_count = 0
            self.last_fps_time = current_time
    
    def _save_screenshot(self, frame: np.ndarray):
        """Save screenshot with timestamp."""
        timestamp = int(time.time())
        filename = f"era_screenshot_{timestamp}.png"
        cv2.imwrite(filename, frame)
        print(f"📸 Screenshot saved: {filename}")
    
    def _reset_game_state(self):
        """Reset the game state."""
        # Reset the game state object
        self.game_state = GameState()
        print("🔄 Game state reset")
    
    def _toggle_game_tracking(self):
        """Toggle game state tracking on/off."""
        # For now just print, could implement pause/resume logic
        print("🎮 Game tracking toggle (feature can be expanded)")

def main():
    """Start the complete integrated gaming system."""
    print("🎮 ERA - Complete Integrated Gaming System")
    print("=" * 50)
    
    # Check if imports were successful
    if not IMPORTS_SUCCESSFUL:
        print("❌ Required modules not available")
        print("📝 Make sure you're running from the project root directory")
        print("📝 Try: cd to ERA directory first, then run the script")
        return 1
    
    # Check if model exists
    model_path = Path("model/weights/best.onnx")
    if not model_path.exists():
        print(f"❌ Model not found: {model_path}")
        print("📝 Export your model first: python scripts/export_model.py")
        return 1
    
    # Check if capture area is configured
    config = load_capture_config()
    if not config:
        print("❌ No capture area configured")
        print("📝 Run: python runtime/capture/screen.py")
        print("📝 Press 'R' to select your game area")
        return 1
    
    try:
        # Start the integrated system
        system = IntegratedGamingSystem()
        return system.start()
        
    except Exception as e:
        print(f"❌ System failed: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())
    """Background thread that captures frames and runs inference."""
    
    frame_ready = pyqtSignal(np.ndarray, list, dict, float)  # frame, detections, game_state, fps
    
    def __init__(self, monitor_rect: MonitorRect, model_path: str):
        super().__init__()
        self.monitor_rect = monitor_rect
        self.engine = CardDetectionEngine(model_path)
        self.game_state = GameState()
        self.running = False
        
        # Performance tracking
        self.frame_count = 0
        self.last_fps_time = time.time()
        
    def run(self):
        """Main processing loop."""
        self.running = True
        
        with mss.mss() as sct:
            monitor_config = {
                "top": int(self.monitor_rect.top),
                "left": int(self.monitor_rect.left),
                "width": int(self.monitor_rect.width),
                "height": int(self.monitor_rect.height)
            }
            
            while self.running:
                try:
                    # Capture frame
                    screenshot = sct.grab(monitor_config)
                    frame = np.array(screenshot)
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
                    
                    # Run inference
                    detections = self.engine.predict(frame, confidence_threshold=0.4)
                    
                    # Update game state
                    self.game_state.ingest_detections(detections)
                    game_snapshot = self.game_state.snapshot()
                    
                    # Calculate FPS
                    self.frame_count += 1
                    current_time = time.time()
                    if current_time - self.last_fps_time >= 1.0:
                        fps = self.frame_count / (current_time - self.last_fps_time)
                        self.frame_count = 0
                        self.last_fps_time = current_time
                    else:
                        fps = 0.0
                    
                    # Emit results
                    self.frame_ready.emit(frame, detections, game_snapshot, fps)
                    
                    # Control frame rate (~30 FPS)
                    self.msleep(33)
                    
                except Exception as e:
                    print(f"Frame processing error: {e}")
                    self.msleep(100)
    
    def stop(self):
        """Stop the processing thread."""
        self.running = False
        self.wait()

class GameOverlay(QtWidgets.QWidget):
    """Transparent overlay that shows over the OpenCV window."""
    
    def __init__(self, cv_window_name: str = "Game Feed"):
        super().__init__()
        
        self.cv_window_name = cv_window_name
        self.detections = []
        self.game_state = {}
        self.fps = 0.0
        
        # Make transparent overlay
        self.setWindowFlags(
            QtCore.Qt.FramelessWindowHint |
            QtCore.Qt.WindowStaysOnTopHint |
            QtCore.Qt.Tool
        )
        self.setAttribute(QtCore.Qt.WA_TranslucentBackground)
        
        # Card colors
        self.card_colors = {
            "Baby Dragon": QtGui.QColor(255, 100, 100, 200),
            "Bomber": QtGui.QColor(255, 165, 0, 200),
            "Dart Goblin": QtGui.QColor(100, 255, 100, 200),
            "Giant": QtGui.QColor(100, 150, 255, 200),
            "Hog Rider": QtGui.QColor(255, 255, 100, 200),
            "Knight": QtGui.QColor(200, 100, 255, 200),
            "Mini Pekka": QtGui.QColor(100, 255, 255, 200),
            "Valkyrie": QtGui.QColor(255, 255, 255, 200),
        }
        
        # Timer to update overlay position
        self.position_timer = QTimer()
        self.position_timer.timeout.connect(self.update_position)
        self.position_timer.start(100)  # Update position every 100ms
        
    def update_position(self):
        """Update overlay position to match OpenCV window."""
        try:
            # Get OpenCV window position and size
            cv_rect = cv2.getWindowImageRect(self.cv_window_name)
            if cv_rect and len(cv_rect) == 4:
                x, y, w, h = cv_rect
                if w > 0 and h > 0:
                    self.setGeometry(x, y, w, h)
        except:
            pass  # OpenCV window might not exist yet
    
    def update_data(self, detections: List[Dict], game_state: Dict, fps: float):
        """Update overlay with new data."""
        self.detections = detections
        self.game_state = game_state
        self.fps = fps
        self.update()
    
    def paintEvent(self, event):
        """Draw the overlay content."""
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.Antialiasing)
        
        # Draw FPS
        self._draw_fps_panel(painter)
        
        # Draw detection info
        self._draw_detection_info(painter)
        
        # Draw game state info
        if self.game_state:
            self._draw_game_info(painter)
        
        painter.end()
    
    def _draw_fps_panel(self, painter):
        """Draw FPS counter."""
        panel_rect = QtCore.QRect(10, 10, 120, 40)
        
        # Background
        painter.setBrush(QtGui.QBrush(QtGui.QColor(0, 0, 0, 150)))
        painter.setPen(QtCore.Qt.NoPen)
        painter.drawRoundedRect(panel_rect, 8, 8)
        
        # FPS text
        font = QtGui.QFont("Arial", 12, QtGui.QFont.Bold)
        painter.setFont(font)
        painter.setPen(QtGui.QPen(QtCore.Qt.white))
        painter.drawText(panel_rect, QtCore.Qt.AlignCenter, f"FPS: {self.fps:.1f}")
    
    def _draw_detection_info(self, painter):
        """Draw detection count and confidence."""
        if not self.detections:
            return
            
        y_offset = 60
        for i, detection in enumerate(self.detections[:5]):  # Show top 5
            card_name = detection['class_name']
            confidence = detection['confidence']
            color = self.card_colors.get(card_name, QtGui.QColor(255, 255, 255))
            
            # Background
            text_rect = QtCore.QRect(10, y_offset, 200, 25)
            painter.setBrush(QtGui.QBrush(QtGui.QColor(0, 0, 0, 150)))
            painter.setPen(QtCore.Qt.NoPen)
            painter.drawRoundedRect(text_rect, 5, 5)
            
            # Text
            font = QtGui.QFont("Arial", 10, QtGui.QFont.Bold)
            painter.setFont(font)
            painter.setPen(QtGui.QPen(color))
            painter.drawText(text_rect, QtCore.Qt.AlignCenter, f"{card_name}: {confidence:.0%}")
            
            y_offset += 30
    
    def _draw_game_info(self, painter):
        """Draw game state information."""
        info_rect = QtCore.QRect(self.width() - 200, 10, 180, 100)
        
        # Background
        painter.setBrush(QtGui.QBrush(QtGui.QColor(0, 0, 0, 150)))
        painter.setPen(QtCore.Qt.NoPen)
        painter.drawRoundedRect(info_rect, 8, 8)
        
        # Game info text
        font = QtGui.QFont("Arial", 10, QtGui.QFont.Bold)
        painter.setFont(font)
        painter.setPen(QtGui.QPen(QtCore.Qt.white))
        
        y = 30
        if 'elixir' in self.game_state:
            painter.drawText(self.width() - 190, y, f"Elixir: {self.game_state['elixir']:.1f}")
            y += 20
        
        if 'plays' in self.game_state and self.game_state['plays']:
            painter.drawText(self.width() - 190, y, f"Plays: {len(self.game_state['plays'])}")

class UnifiedGamingSystem:
    """Complete gaming system with OpenCV display + PyQt overlay."""
    
    def __init__(self, model_path: str = "model/weights/best.onnx"):
        # Load capture configuration
        self.monitor_rect = load_capture_config()
        if not self.monitor_rect:
            raise ValueError("No capture area configured. Run screen capture tool first.")
        
        self.model_path = model_path
        self.app = None
        self.overlay = None
        self.processor = None
        self.cv_window_name = "ERA Gaming Feed"
        
        print(f"Gaming system initialized with area: {self.monitor_rect.width}x{self.monitor_rect.height}")
    
    def start(self):
        """Start the unified gaming system."""
        # Create Qt application for overlay
        self.app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)
        
        # Create transparent overlay
        self.overlay = GameOverlay(self.cv_window_name)
        self.overlay.show()
        
        # Create frame processor
        self.processor = FrameProcessor(self.monitor_rect, self.model_path)
        self.processor.frame_ready.connect(self._on_frame_ready)
        self.processor.start()
        
        # Create OpenCV window
        cv2.namedWindow(self.cv_window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.cv_window_name, self.monitor_rect.width, self.monitor_rect.height)
        
        print("Unified Gaming System Started!")
        print("OpenCV window shows the game feed")
        print("PyQt overlay shows detections and game data")
        print("Controls: 'q' = quit, 'f' = toggle fullscreen")
        
        # Main loop
        try:
            while True:
                # Process Qt events
                self.app.processEvents()
                
                # Check for OpenCV window events
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q') or cv2.getWindowProperty(self.cv_window_name, cv2.WND_PROP_VISIBLE) < 1:
                    break
                elif key == ord('f'):
                    # Toggle fullscreen
                    cv2.setWindowProperty(self.cv_window_name, cv2.WND_PROP_FULLSCREEN, 
                                        cv2.WINDOW_FULLSCREEN)
                
                time.sleep(0.01)  # Small delay to prevent CPU spinning
                
        except KeyboardInterrupt:
            print("Interrupted by user")
        finally:
            self.stop()
        
        return 0
    
    def _on_frame_ready(self, frame, detections, game_state, fps):
        """Handle new frame and inference results."""
        # Draw detections on frame for OpenCV display
        engine = CardDetectionEngine()  # Temporary instance for drawing
        frame_with_detections = engine.draw_detections(frame, detections)
        
        # Show in OpenCV window
        cv2.imshow(self.cv_window_name, frame_with_detections)
        
        # Update PyQt overlay
        if self.overlay:
            self.overlay.update_data(detections, game_state, fps)
    
    def stop(self):
        """Stop the gaming system."""
        if self.processor:
            self.processor.stop()
        if self.overlay:
            self.overlay.close()
        cv2.destroyAllWindows()

def main():
    """Start the unified gaming system."""
    print("Unified Gaming System")
    print("=" * 40)
    
    try:
        system = UnifiedGamingSystem()
        return system.start()
    except Exception as e:
        print(f"Gaming system failed: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())