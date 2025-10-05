import sys
import time
from pathlib import Path
import numpy as np
import cv2
import os

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

# Place this near the top of the file, after imports

# --- Card image mapping ---
ASSETS_DIR = os.path.join(os.path.dirname(__file__), 'assets')
# BACKGROUND_PATH = os.path.join(ASSETS_DIR, 'Clash-Royale-background.jpg')
BACKGROUND_PATH = os.path.join(ASSETS_DIR, 'overlay.png')

HEADER_PATH = os.path.join(ASSETS_DIR, 'edgeheader.png')
CARD_IMG_SIZE = (70, 90)

def get_placeholder_card():
    img = np.full((CARD_IMG_SIZE[1], CARD_IMG_SIZE[0], 3), 180, dtype=np.uint8)
    cv2.rectangle(img, (0,0), (CARD_IMG_SIZE[0]-1,CARD_IMG_SIZE[1]-1), (120,120,120), 2)
    cv2.putText(img, '?', (CARD_IMG_SIZE[0]//2-15, CARD_IMG_SIZE[1]//2+15), cv2.FONT_HERSHEY_SIMPLEX, 2, (100,100,100), 3)
    return img

def load_card_images():
    mapping = {}
    for fname in os.listdir(ASSETS_DIR):
        if fname.endswith('.png'):
            key = fname.lower().replace('.png','').replace(' ', '').replace('_','')
            img = cv2.imread(os.path.join(ASSETS_DIR, fname))
            if img is not None:
                mapping[key] = cv2.resize(img, CARD_IMG_SIZE)
    return mapping

CARD_IMAGES = load_card_images()
CARD_IMG = get_placeholder_card()

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
        self.background_img = None
        if os.path.exists(BACKGROUND_PATH):
            bg = cv2.imread(BACKGROUND_PATH)
            if bg is not None:
                self.background_img = bg
            else:
                print(f"Warning: Could not load background image at {BACKGROUND_PATH}")
        else:
            print(f"Warning: Background image not found at {BACKGROUND_PATH}")
            self.header_img = None
            if os.path.exists(HEADER_PATH):
                header = cv2.imread(HEADER_PATH)
                if header is not None:
                    self.header_img = header
                else:
                    print(f"Warning: Could not load header image at {HEADER_PATH}")
            else:
                print(f"Warning: Header image not found at {HEADER_PATH}")
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

            while self.running:
                try:
                    screenshot = sct.grab(monitor_config)
                    frame = np.array(screenshot)
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

                    # Run inference directly (single-threaded)
                    detections = self.inference_engine.predict(frame, confidence_threshold=0.5)

                    # Draw detections and update game state
                    frame_h, frame_w = frame.shape[:2]
                    MODEL_INPUT_SIZE = 640
                    scale_x = frame_w / MODEL_INPUT_SIZE
                    scale_y = frame_h / MODEL_INPUT_SIZE
                    detection_dicts = []
                    if detections:
                        # print("Detections:")
                        for det in detections:
                            bbox = det['bbox']
                            x1 = int(bbox[0] * scale_x)
                            y1 = int(bbox[1] * scale_y)
                            x2 = int(bbox[2] * scale_x)
                            y2 = int(bbox[3] * scale_y)
                            label = f"{det['class_name']} ({det['confidence']:.0%})"
                            # print(f"  {label} at [{x1}, {y1}, {x2}, {y2}]")
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
                    # print(f"Opponent Elixir: {state['elixir_opponent']:.2f} | Last Played: {state['last_played']} | Deck: {state['deck']}")

                    # Draw FPS counter in top-left
                    fps_text = f"FPS: {self.fps:.1f}"
                    cv2.putText(frame, fps_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,255), 2)

                    # Listen for spacebar to start elixir counting
                    key = cv2.waitKey(45) & 0xFF
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

                    # --- Sidebar overlay ---
                    sidebar_width = 400
                    h, w = frame.shape[:2]
                    overlay = np.zeros((h, sidebar_width, 3), dtype=np.uint8)

                    # Elixir bar (centered at top)
                    elixir = int(round(state['elixir_opponent']))
                    max_elixir = 10
                    bar_w, bar_h = 220, 24
                    bar_x = (sidebar_width - bar_w) // 2 + 20
                    bar_y = 40
                    seg_w = int(bar_w / max_elixir) - 4
                    # Draw elixir icon (purple drop)
                    cv2.ellipse(overlay, (bar_x-25, bar_y+bar_h//2), (16,20), 0, 0, 360, (180, 60, 220), -1)
                    # Draw elixir bar segments
                    for i in range(max_elixir):
                        color = (200, 80, 255) if i < elixir else (60, 40, 80)
                        x = bar_x + i * (seg_w + 4)
                        cv2.rectangle(overlay, (x, bar_y), (x+seg_w, bar_y+bar_h), color, -1)
                    # Elixir text
                    cv2.putText(overlay, f"{elixir} / 10", (bar_x+bar_w+10, bar_y+bar_h-2), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,255,255), 2)
                    cv2.putText(overlay, "Opponent Elixir", (bar_x-10, bar_y-15), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)

                    # Card slots as 2 rows of 4 columns
                    cv2.putText(overlay, "Opponent Deck", (sidebar_width//2-90, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,255,255), 2)
                    card_w, card_h = 70, 90
                    gap_x, gap_y = 18, 18
                    start_x = (sidebar_width - (4*card_w + 3*gap_x)) // 2
                    start_y = 140

                    for i in range(8):
                        row = i // 4
                        col = i % 4
                        x = start_x + col * (card_w + gap_x)
                        y = start_y + row * (card_h + gap_y)
                        card = state['deck'][i]
                        # Normalize card name for lookup
                        card_key = None
                        if card:
                            card_key = card.lower().replace(' ', '').replace('_','')
                        card_img = CARD_IMAGES.get(card_key, CARD_IMG)
                        overlay[y:y+card_h, x:x+card_w] = card_img.copy()
                        # Draw border: white for current hand (0-3), yellow for slot 4 (next up), gray for others
                        if i < 4:
                            cv2.rectangle(overlay, (x, y), (x+card_w, y+card_h), (255,255,255), 3)
                        elif i == 4:
                            cv2.rectangle(overlay, (x, y), (x+card_w, y+card_h), (0,255,255), 3)
                        else:
                            cv2.rectangle(overlay, (x, y), (x+card_w, y+card_h), (120,120,120), 2)
                        # Card name (if available)
                        if card:
                            cv2.putText(overlay, card[:12], (x+4, y+card_h-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 2)



                    # Compose output dimensions
                    border_width = 400
                    sidebar_width = 400
                    h, w = frame.shape[:2]
                    total_w = border_width + w + sidebar_width

                    # Prepare background
                    if self.background_img is not None:
                        bg_resized = cv2.resize(self.background_img, (total_w, h))
                    else:
                        bg_resized = np.zeros((h, total_w, 3), dtype=np.uint8)

                    # Overlay the frame in the center, with yellow bars on each side
                    yellow_bar_width = 25
                    yellow_color = (0, 160, 218)  # BGR for OpenCV
                    # Draw left yellow bar
                    bg_resized[:, border_width:border_width+yellow_bar_width] = yellow_color
                    # Draw right yellow bar
                    bg_resized[:, border_width+w-yellow_bar_width:border_width+w] = yellow_color
                    # Overlay the frame (between yellow bars)
                    bg_resized[:, border_width+yellow_bar_width:border_width+w-yellow_bar_width] = frame[:, yellow_bar_width:w-yellow_bar_width]

                    # Overlay the sidebar content (only non-black parts)
                    sidebar_mask = np.any(overlay != 0, axis=2)
                    for c in range(3):
                        bg_resized[:, border_width+w:, c][sidebar_mask] = overlay[:,:,c][sidebar_mask]

                    final = bg_resized

                    # --- Draw edgeheader.png at the top-left, max width 400px, height scaled proportionally ---
                    header_img = None
                    HEADER_PATH = os.path.join(ASSETS_DIR, 'edgeheader.png')
                    if os.path.exists(HEADER_PATH):
                        header_img = cv2.imread(HEADER_PATH)
                    if header_img is not None:
                        max_header_width = 400
                        h_ratio = max_header_width / header_img.shape[1]
                        header_h = int(header_img.shape[0] * h_ratio)
                        header_resized = cv2.resize(header_img, (max_header_width, header_h))
                        # Overlay header on the top-left of the final image, fill only header_h rows
                        final[:header_h, :max_header_width] = header_resized

                    cv2.imshow(window_name, final)

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
    # DEBUG: Print GameState class and its attributes
    print('DEBUG: GameState imported from:', GameState.__module__)
    print('DEBUG: GameState attributes:', dir(GameState))

    sys.exit(main())
