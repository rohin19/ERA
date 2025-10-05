import sys
import time
from pathlib import Path
import numpy as np
import cv2

import threading
import queue
from newGameState import GameState

try:
    import mss
    from runtime.overlay.qt_capture_view import load_capture_config
    from runtime.infer.onnx_engine import CardDetectionEngine
    DEPENDENCIES_OK = True
except ImportError as e:
    print(f"❌ Dependencies missing: {e}")
    print("📝 Install: pip install mss numpy opencv-python onnxruntime")
    DEPENDENCIES_OK = False

class OpenCVCaptureEngine:
    def initialize(self):
        try:
            self.inference_engine = CardDetectionEngine(self.model_path)
            print("✅ Inference engine initialized")
            return True
        except Exception as e:
            print(f"❌ Failed to initialize inference engine: {e}")
            return False
    """Handles screen capture and inference using OpenCV for display"""
    def __init__(self, capture_config, model_path):
        self.capture_config = capture_config
        self.model_path = model_path
        self.running = False
        self.inference_engine = None
        self.game_state = GameState()  # Use newGameState.GameState
        self.fps = 0.0
        self.frame_count = 0
        self.last_fps_time = time.time()

    def run(self):
        if not self.initialize():
            return
        self.running = True
        print("🚀 OpenCV Capture engine started")
        with mss.mss() as sct:
            monitor_config = {
                "top": int(self.capture_config.top),
                "left": int(self.capture_config.left),
                "width": int(self.capture_config.width),
                "height": int(self.capture_config.height)
            }
            window_name = "Clash Royale OpenCV Assistant"
            cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

            # Threading setup
            frame_queue = queue.Queue(maxsize=1)
            detections_queue = queue.Queue(maxsize=1)
            stop_event = threading.Event()

            def inference_worker():
                while not stop_event.is_set():
                    try:
                        frame = frame_queue.get(timeout=0.1)
                        detections = self.inference_engine.predict(frame, confidence_threshold=0.5)
                        if detections_queue.full():
                            detections_queue.get_nowait()
                        detections_queue.put(detections)
                    except queue.Empty:
                        continue

            thread = threading.Thread(target=inference_worker, daemon=True)
            thread.start()

            latest_detections = []
            while self.running:
                try:
                    screenshot = sct.grab(monitor_config)
                    frame = np.array(screenshot)
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

                    # Send frame to inference thread
                    if not frame_queue.full():
                        frame_queue.put_nowait(frame.copy())

                    # Get latest detections if available
                    try:
                        latest_detections = detections_queue.get_nowait()
                    except queue.Empty:
                        pass

                    # Draw detections and update game state
                    frame_h, frame_w = frame.shape[:2]
                    MODEL_INPUT_SIZE = 640
                    scale_x = frame_w / MODEL_INPUT_SIZE
                    scale_y = frame_h / MODEL_INPUT_SIZE
                    detection_dicts = []
                    if latest_detections:
                        print("Detections:")
                        for det in latest_detections:
                            bbox = det['bbox']
                            x1 = int(bbox[0] * scale_x)
                            y1 = int(bbox[1] * scale_y)
                            x2 = int(bbox[2] * scale_x)
                            y2 = int(bbox[3] * scale_y)
                            label = f"{det['class_name']} ({det['confidence']:.0%})"
                            print(f"  {label} at [{x1}, {y1}, {x2}, {y2}]")
                            cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
                            cv2.putText(frame, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
                            detection_dicts.append({
                                'card': det['class_name'],
                                'bbox': [x1, y1, x2, y2],
                                'confidence': det['confidence'],
                                'frame': None
                            })
                    # Update game state with detection info
                    self.game_state.update(detection_dicts)
                    state = self.game_state.get_state()
                    print(f"Opponent Elixir: {state['elixir_opponent']:.2f} | Last Played: {state['last_played']} | Deck: {state['deck']}")

                    # Draw FPS counter in top-left
                    fps_text = f"FPS: {self.fps:.1f}"
                    cv2.putText(frame, fps_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,255), 2)

                    # Listen for spacebar to start elixir counting
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        self.stop()
                        break
                    if key == ord(' '):
                        if not self.game_state.elixir_counting_enabled:
                            self.game_state.start_elixir_counting()
                            print('Elixir counting started!')
                    # Show overlay if elixir counting not started
                    if not self.game_state.elixir_counting_enabled:
                        cv2.putText(frame, 'Press SPACE to start elixir counting', (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)

                    cv2.imshow(window_name, frame)

                    self._update_fps()
                except Exception as e:
                    print(f"❌ Capture loop error: {e}")
                    time.sleep(0.1)
            stop_event.set()
            thread.join()
            cv2.destroyAllWindows()

    def _update_fps(self):
        self.frame_count += 1
        current_time = time.time()
        if current_time - self.last_fps_time >= 1.0:
            self.fps = self.frame_count / (current_time - self.last_fps_time)
            print(f"FPS: {self.fps:.1f}")
            self.frame_count = 0
            self.last_fps_time = current_time

    def stop(self):
        self.running = False

def main():
    if not DEPENDENCIES_OK:
        return 1
    print("🎮 Clash Royale OpenCV Assistant")
    print("=" * 50)
    capture_config = load_capture_config()
    if not capture_config:
        print("No capture area configured. Run screen capture tool first.")
        return 1
    model_path = Path("model/weights/best.onnx")
    if not model_path.exists():
        print(f"Model not found: {model_path}")
        return 1
    engine = OpenCVCaptureEngine(capture_config, str(model_path))
    engine.run()
    return 0

if __name__ == "__main__":
    # DEBUG: Print GameState class and its attributes
    print('DEBUG: GameState imported from:', GameState.__module__)
    print('DEBUG: GameState attributes:', dir(GameState))

    sys.exit(main())
