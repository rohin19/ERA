import sys
import time
from pathlib import Path
import numpy as np
import cv2

try:
    import mss
    from runtime.overlay.qt_capture_view import load_capture_config
    from runtime.infer.onnx_engine import CardDetectionEngine
    from runtime.gamestate.state import GameState
    DEPENDENCIES_OK = True
except ImportError as e:
    print(f"❌ Dependencies missing: {e}")
    print("📝 Install: pip install mss numpy opencv-python onnxruntime")
    DEPENDENCIES_OK = False

class OpenCVCaptureEngine:
    """Handles screen capture and inference using OpenCV for display"""
    def __init__(self, capture_config, model_path):
        self.capture_config = capture_config
        self.model_path = model_path
        self.running = False
        self.inference_engine = None
        self.game_state = GameState()
        self.fps = 0.0
        self.frame_count = 0
        self.last_fps_time = time.time()

    def initialize(self):
        try:
            self.inference_engine = CardDetectionEngine(self.model_path)
            print("✅ Inference engine initialized")
            return True
        except Exception as e:
            print(f"❌ Failed to initialize inference engine: {e}")
            return False

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
            while self.running:
                try:
                    screenshot = sct.grab(monitor_config)
                    frame = np.array(screenshot)
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
                    # Optionally, run inference and draw boxes (uncomment if desired)
                    detections = self.inference_engine.predict(frame, confidence_threshold=0.5)
                    # if detections:
                    #     print("Detections:")
                    #     frame_h, frame_w = frame.shape[:2]
                    #     MODEL_INPUT_SIZE = 640  # Change if your model uses a different input size
                    #     scale_x = frame_w / MODEL_INPUT_SIZE
                    #     scale_y = frame_h / MODEL_INPUT_SIZE
                    #     for det in detections:
                    #         bbox = det['bbox']
                    #         # Assume bbox is in model input size coordinates
                    #         x1 = int(bbox[0] * scale_x)
                    #         y1 = int(bbox[1] * scale_y)
                    #         x2 = int(bbox[2] * scale_x)
                    #         y2 = int(bbox[3] * scale_y)
                    #         label = f"{det['class_name']} ({det['confidence']:.0%})"
                    #         print(f"  {label} at [{x1}, {y1}, {x2}, {y2}]")
                    #         cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
                    #         cv2.putText(frame, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)

                    if detections:
                        print("Detections:")
                        for det in detections:
                            x1 = int(bbox[0])
                            y1 = int(bbox[1])
                            x2 = int(bbox[2])
                            y2 = int(bbox[3])
                            bbox = det['bbox']
                            label = f"{det['class_name']} ({det['confidence']:.0%})"

                            print(f"  {label} at [{x1}, {y1}, {x2}, {y2}]")

                    cv2.imshow(window_name, frame)
                    if cv2.waitKey(30) & 0xFF == ord('q'):
                        self.stop()
                        break
                    self._update_fps()
                except Exception as e:
                    print(f"❌ Capture loop error: {e}")
                    time.sleep(0.1)
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
    sys.exit(main())
