"""
Edge Royale Analytics - Using LOCAL YOLO Model (best.pt)
NO API needed - Direct local inference!
"""
import sys
import time
import cv2
import json
from pathlib import Path
import numpy as np

project_root = Path(__file__).parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from runtime.gamestate.gamestate import GameState
import mss

# Import Ultralytics YOLO
try:
    from ultralytics import YOLO
    print("Ultralytics YOLO loaded")
except ImportError:
    print("Ultralytics not found. Install: pip install ultralytics")
    sys.exit(1)


class ClashRoyaleAssistant:
    """Simple assistant using LOCAL YOLO model."""
    
    # EASY CONFIDENCE CONTROLS - ADJUST THESE!
    DETECTION_CONFIDENCE = 0.4  # Lower = more detections, but more false positives
    DISPLAY_CONFIDENCE = 0.4    # Show all detections above this on screen
    GAMESTATE_CONFIDENCE = 0.6   # Only feed high-confidence to GameState
    
    def __init__(self):
        # Load LOCAL YOLO model - FIXED PATH
        model_path = Path("model/runs/weights/best.pt")
        
        if not model_path.exists():
            # Try alternative paths
            alt_paths = [
                Path("model/runs3/weights/best.pt"),
                Path("model/weights/best.pt"),
                Path("model/runs4/weights/best.pt"),
            ]
            print(f"Model not found at: {model_path}")
            print(f"   Trying alternatives...")
            for alt_path in alt_paths:
                if alt_path.exists():
                    model_path = alt_path
                    print(f"   Found: {alt_path}")
                    break
            else:
                raise FileNotFoundError(f"Model not found at {model_path} or alternatives")
        
        print(f"📦 Loading LOCAL YOLO model: {model_path}")
        self.model = YOLO(str(model_path))
        print(f"YOLO model loaded (LOCAL inference - NO API!)")
        print(f"Detection confidence: {self.DETECTION_CONFIDENCE}")
        print(f"Display confidence: {self.DISPLAY_CONFIDENCE}")
        print(f"GameState confidence: {self.GAMESTATE_CONFIDENCE}")
        
        self.capture_region = self._load_capture_config()
        self.gamestate = GameState()
        self.sct = mss.mss()
        
        self.card_images = self._load_card_assets()
        
        # Card colors (BGR)
        self.card_colors = {
            "Baby Dragon": (50, 140, 255),
            "Bomber": (100, 100, 255),
            "Dart Goblin": (100, 255, 100),
            "Giant": (255, 150, 150),
            "Hog Rider": (100, 200, 255),
            "Knight": (255, 100, 200),
            "Mini Pekka": (100, 255, 255),
            "Valkyrie": (200, 150, 255),
        }
        
        self.window_title = "Edge Royale Analytics Overlay"
        self.fps = 0.0
        self.frame_count = 0
        self.last_fps_time = time.time()
        
        self.frame_counter = 0
        self.state_update_interval = 5
        self.current_state = {}
        
        print(f"Loaded {len(self.card_images)} card assets")
        print(f"Capture: {self.capture_region['width']}x{self.capture_region['height']}")
    
    def _load_capture_config(self):
        """Load capture configuration."""
        config_file = Path("capture_config.json")
        if not config_file.exists():
            raise FileNotFoundError("No capture area configured")
        
        with open(config_file) as f:
            config = json.load(f)
        
        if "monitor" in config:
            monitor = config["monitor"]
            return {
                "left": monitor["left"],
                "top": monitor["top"],
                "width": monitor["width"],
                "height": monitor["height"]
            }
        
        raise ValueError("Invalid capture config")
    
    def _load_card_assets(self):
        """Load card images."""
        assets_dir = Path("assets")
        card_images = {}
        
        card_files = {
            "Baby Dragon": "babydragon.png",
            "Bomber": "bomber.png",
            "Dart Goblin": "dartgoblin.png",
            "Giant": "giant.png",
            "Hog Rider": "hogrider.png",
            "Knight": "knight.png",
            "Mini Pekka": "minipekka.png",
            "Valkyrie": "valkyrie.png"
        }
        
        for card_name, filename in card_files.items():
            image_path = assets_dir / filename
            if image_path.exists():
                img = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
                if img is not None:
                    img = cv2.resize(img, (48, 48))
                    if len(img.shape) == 3 and img.shape[2] == 4:
                        alpha = img[:, :, 3:4] / 255.0
                        rgb = img[:, :, :3]
                        white = np.ones_like(rgb) * 255
                        img = (rgb * alpha + white * (1 - alpha)).astype(np.uint8)
                    card_images[card_name] = img
        
        return card_images
    
    def run(self):
        """Main loop - LOCAL YOLO with DIAGNOSTICS."""
        print("\nControls:")
        print("  Q = Quit")
        print("  R = Reset GameState")
        print("  S = Screenshot")
        print("  D = Toggle debug mode")
        print("  + = Increase detection confidence (+0.05)")
        print("  - = Decrease detection confidence (-0.05)")
        print("\nUsing LOCAL YOLO model - FAST & NO API!\n")
        
        cv2.namedWindow(self.window_title, cv2.WINDOW_NORMAL)
        initial_w = int(self.capture_region["width"] * 0.8)
        initial_h = int(self.capture_region["height"] * 0.8)
        cv2.resizeWindow(self.window_title, initial_w, initial_h)
        
        debug_mode = False
        frame_saved = False
        
        try:
            while True:
                # Capture
                screenshot = self.sct.grab(self.capture_region)
                frame = np.array(screenshot)
                frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
                
                h, w = frame.shape[:2]
                
                # SAVE FIRST FRAME FOR DEBUGGING
                if not frame_saved:
                    cv2.imwrite("debug_test_frame.jpg", frame)
                    print(f" Saved test frame: debug_test_frame.jpg ({w}x{h})")
                    print(f"   Check this image - does it show the game?")
                    frame_saved = True
                
                # YOLO Inference with ADJUSTABLE threshold
                results = self.model.predict(frame, conf=self.DETECTION_CONFIDENCE, verbose=debug_mode)
                
                # Extract predictions
                predictions = []
                if len(results) > 0:
                    result = results[0]
                    boxes = result.boxes
                    
                    # DEBUG: Print what we got
                    if debug_mode or self.frame_counter % 60 == 0:
                        print(f"\n🔍 Frame {self.frame_counter}:")
                        print(f"   Raw detections: {len(boxes)} (conf >= {self.DETECTION_CONFIDENCE})")
                    
                    for box in boxes:
                        # Get coordinates (xyxy format)
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        conf = float(box.conf[0])
                        cls_id = int(box.cls[0])
                        
                        # Get class name
                        class_name = result.names[cls_id]
                        
                        # DEBUG: Print all detections
                        if debug_mode:
                            print(f"   - {class_name}: {conf:.2%} at ({x1:.0f},{y1:.0f})")
                        
                        # Convert to center format (like V1)
                        center_x = (x1 + x2) / 2
                        center_y = (y1 + y2) / 2
                        width = x2 - x1
                        height = y2 - y1
                        
                        predictions.append({
                            'x': center_x,
                            'y': center_y,
                            'width': width,
                            'height': height,
                            'confidence': conf,
                            'class': class_name
                        })
                
                # Convert to GameState format (high confidence only)
                detections = self._convert_predictions(predictions)
                
                # GameState update
                if self.frame_counter % self.state_update_interval == 0:
                    self.gamestate.ingest_detections(detections, time.time())
                    self.current_state = self.gamestate.get_state()
                
                self.frame_counter += 1
                
                # Drawing - show detections above DISPLAY_CONFIDENCE
                display_predictions = [p for p in predictions if p['confidence'] >= self.DISPLAY_CONFIDENCE]
                display_frame = self._draw_overlay_v1(frame, display_predictions, self.current_state, w, h)
                
                # Add detection count and confidence info
                total_dets = len(predictions)
                shown_dets = len(display_predictions)
                high_conf_dets = len([p for p in predictions if p['confidence'] >= self.GAMESTATE_CONFIDENCE])
                
                info_text = f"Detections: {shown_dets}/{total_dets} shown | {high_conf_dets} high-conf | Conf: {self.DETECTION_CONFIDENCE:.2f}"
                cv2.putText(display_frame, info_text, (10, h - 20),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                
                # Show
                cv2.imshow(self.window_title, display_frame)
                
                # Update FPS
                self._update_fps()
                
                # Print stats every 60 frames
                if self.frame_counter % 60 == 0:
                    print(f"FPS: {self.fps:.1f} | Total: {total_dets} | Shown: {shown_dets} | High: {high_conf_dets}")
                
                # Handle input
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q") or cv2.getWindowProperty(self.window_title, cv2.WND_PROP_VISIBLE) < 1:
                    break
                elif key == ord("r"):
                    self.gamestate.reset()
                    print("Reset")
                elif key == ord("s"):
                    self._save_screenshot(display_frame)
                elif key == ord("d"):
                    debug_mode = not debug_mode
                    print(f"Debug mode: {'ON' if debug_mode else 'OFF'}")
                elif key == ord("+") or key == ord("="):
                    self.DETECTION_CONFIDENCE = min(1.0, self.DETECTION_CONFIDENCE + 0.05)
                    print(f"Detection confidence: {self.DETECTION_CONFIDENCE:.2f}")
                elif key == ord("-") or key == ord("_"):
                    self.DETECTION_CONFIDENCE = max(0.01, self.DETECTION_CONFIDENCE - 0.05)
                    print(f"Detection confidence: {self.DETECTION_CONFIDENCE:.2f}")
        
        except KeyboardInterrupt:
            print("\nStopped")
        finally:
            cv2.destroyAllWindows()
    
    def _convert_predictions(self, predictions):
        """Convert YOLO predictions to GameState format."""
        detections = []
        for pred in predictions:
            if pred.get('confidence', 0) >= self.GAMESTATE_CONFIDENCE:
                center_x, center_y = pred['x'], pred['y']
                w, h = pred['width'], pred['height']
                x1 = center_x - w/2
                y1 = center_y - h/2
                x2 = center_x + w/2
                y2 = center_y + h/2
                
                detections.append({
                    'card': pred['class'],
                    'bbox': [x1, y1, x2, y2],
                    'confidence': pred['confidence']
                })
        return detections
    
    def _draw_overlay_v1(self, frame, predictions, state, frame_w, frame_h):
        """V1 overlay - EXACT COPY from working version."""
        
        # Draw detections V1 EXACT
        self._draw_detections_v1(frame, predictions)
        
        # Bottom panel
        if state:
            self._draw_bottom_panel(frame, state, frame_w, frame_h)
        
        # FPS
        self._draw_fps(frame, frame_w, frame_h)
        
        return frame
    
    def _draw_detections_v1(self, frame, predictions):
        """V1 detection drawing - EXACT COPY from V1 era_launcher.py"""
        for prediction in predictions:
            # V1 bounding box extraction
            x = int(prediction['x'] - prediction['width'] / 2)
            y = int(prediction['y'] - prediction['height'] / 2)
            w = int(prediction['width'])
            h = int(prediction['height'])
            
            confidence = prediction['confidence']
            card_name = prediction['class']
            
            # V1 color coding
            if confidence >= self.GAMESTATE_CONFIDENCE:
                color = (128, 0, 128)  # Purple - high confidence
                thickness = 3
            else:
                color = (100, 100, 100)  # Gray - low confidence
                thickness = 1
            
            # Draw box
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, thickness)
            
            # Draw label (V1 style)
            label = f"{card_name}: {confidence:.2f}"
            font_scale = 0.4
            text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)[0]
            
            # Label background
            cv2.rectangle(frame, (x, y - text_size[1] - 8), (x + text_size[0] + 6, y), (0, 0, 0), -1)
            cv2.rectangle(frame, (x, y - text_size[1] - 8), (x + text_size[0] + 6, y), color, 1)
            
            # Label text
            cv2.putText(frame, label, (x + 3, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 
                       font_scale, (255, 255, 255), 1)
    
    def _draw_bottom_panel(self, frame, state, w, h):
        """Bottom panel with proper hand logic and AI predictions."""
        panel_h = 200
        panel_y = h - panel_h
        
        cv2.rectangle(frame, (0, panel_y), (w, h), (25, 30, 35), -1)
        
        # Elixir bar
        elixir = state.get("elixir_opponent", 0)
        bar_x, bar_y = 15, panel_y + 10
        bar_w, bar_h = 250, 20
        
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (50, 50, 50), -1)
        fill_w = int(min(elixir, 10.0) / 10.0 * bar_w)
        if fill_w > 0:
            cv2.rectangle(frame, (bar_x, bar_y), (bar_x + fill_w, bar_y + bar_h), (200, 100, 255), -1)
        
        cv2.putText(frame, f"Elixir: {elixir:.1f}/10", (bar_x + 5, bar_y + 15),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # === CURRENT HAND ===
        deck = state.get("deck", [])
        current_hand = deck[:4] if len(deck) >= 4 else []
        
        # Count discovered cards
        cards_discovered = len([c for c in deck if c is not None])
        
        hand_x, hand_y = 15, panel_y + 45
        
        # Show current hand once we have 4+ cards (first cycle complete)
        if cards_discovered >= 4:
            cv2.putText(frame, "CURRENT HAND (Available):", (hand_x, hand_y - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 255, 100), 1)
            
            for i in range(4):
                card = current_hand[i] if i < len(current_hand) and current_hand[i] is not None else None
                x = hand_x + (i * 55)
                
                if card and card in self.card_images:
                    try:
                        frame[hand_y:hand_y+48, x:x+48] = self.card_images[card]
                        cv2.rectangle(frame, (x-1, hand_y-1), (x+49, hand_y+49),
                                     self.card_colors.get(card, (255, 255, 255)), 2)
                        cv2.putText(frame, card[:6], (x, hand_y + 60),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.3, (200, 200, 200), 1)
                    except Exception as e:
                        pass
                else:
                    cv2.rectangle(frame, (x, hand_y), (x+48, hand_y+48), (60, 60, 60), -1)
                    cv2.rectangle(frame, (x, hand_y), (x+48, hand_y+48), (120, 120, 120), 1)
                    cv2.putText(frame, "?", (x + 18, hand_y + 30),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (100, 100, 100), 2)
        else:
            # Early game - show discovery progress
            cv2.putText(frame, f"DISCOVERING HAND... ({cards_discovered}/4)", (hand_x, hand_y - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 200, 100), 1)
            
            # Show progress bar
            progress_w = 220
            progress_h = 15
            progress_y = hand_y + 10
            
            cv2.rectangle(frame, (hand_x, progress_y), (hand_x + progress_w, progress_y + progress_h), 
                         (50, 50, 50), -1)
            
            if cards_discovered > 0:
                fill_w = int((cards_discovered / 4.0) * progress_w)
                cv2.rectangle(frame, (hand_x, progress_y), (hand_x + fill_w, progress_y + progress_h), 
                             (100, 255, 255), -1)
            
            cv2.putText(frame, "Need 4 cards to identify hand", (hand_x, progress_y + 35),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1)
        
        # === NEXT CARD (show if we have 5+ cards) ===
        if cards_discovered >= 5:
            next_card = deck[4] if len(deck) > 4 and deck[4] is not None else None
            
            next_x = hand_x + 230
            next_y = hand_y
            
            cv2.putText(frame, "NEXT:", (next_x - 50, next_y + 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 100), 1)
            
            if next_card and next_card in self.card_images:
                try:
                    frame[next_y:next_y+48, next_x:next_x+48] = self.card_images[next_card]
                    cv2.rectangle(frame, (next_x-1, next_y-1), (next_x+49, next_y+49),
                                 (255, 255, 100), 3)
                except Exception as e:
                    pass
            else:
                cv2.rectangle(frame, (next_x, next_y), (next_x+48, next_y+48), (60, 60, 60), -1)
                cv2.rectangle(frame, (next_x, next_y), (next_x+48, next_y+48), (120, 120, 120), 1)
                cv2.putText(frame, "?", (next_x + 18, next_y + 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (100, 100, 100), 2)
        
        # === AI PREDICTIONS (show if we have 4+ cards) ===
        predictions = state.get("next_prediction", [])
        prediction_conf = state.get("prediction_confidence", 0.0)
        
        pred_x, pred_y = hand_x, hand_y + 75
        
        if cards_discovered >= 4 and predictions and len(predictions) > 0:
            conf_color = (100, 255, 100) if prediction_conf >= 0.7 else (255, 200, 100) if prediction_conf >= 0.4 else (255, 100, 100)
            cv2.putText(frame, f"AI PREDICTIONS ({prediction_conf:.0%}):", (pred_x, pred_y - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, conf_color, 1)
            
            for i in range(min(3, len(predictions))):
                card = predictions[i]
                x = pred_x + (i * 55)
                
                if card and card in self.card_images:
                    try:
                        if i == 0:
                            cv2.rectangle(frame, (x-3, pred_y-3), (x+51, pred_y+51), conf_color, 2)
                        
                        frame[pred_y:pred_y+48, x:x+48] = self.card_images[card]
                        cv2.rectangle(frame, (x-1, pred_y-1), (x+49, pred_y+49),
                                     self.card_colors.get(card, (255, 255, 255)), 2)
                        
                        rank_text = f"#{i+1}"
                        cv2.putText(frame, rank_text, (x + 2, pred_y + 12),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 3)
                        cv2.putText(frame, rank_text, (x + 2, pred_y + 12),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, conf_color, 1)
                    except Exception as e:
                        pass
        elif cards_discovered < 4:
            cv2.putText(frame, "AI PREDICTIONS: (need 4 cards)", (pred_x, pred_y + 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (120, 120, 120), 1)
        else:
            cv2.putText(frame, "AI PREDICTIONS: (learning...)", (pred_x, pred_y + 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (120, 120, 120), 1)
        
        # === FULL DECK (always show discovered cards) ===
        deck_x, deck_y = 350, panel_y + 10
        cv2.putText(frame, f"DISCOVERED DECK ({cards_discovered}/8):", (deck_x, deck_y - 5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)
        
        for i in range(8):
            card = deck[i] if i < len(deck) and deck[i] is not None else None
            row, col = i // 4, i % 4
            x, y = deck_x + (col * 55), deck_y + (row * 55)
            
            if card and card in self.card_images:
                try:
                    frame[y:y+48, x:x+48] = self.card_images[card]
                    
                    # Visual indicators based on deck knowledge
                    if cards_discovered >= 4:
                        if i < 4:
                            border_color = (100, 255, 100)  # Green for current hand
                            thickness = 2
                        elif i == 4:
                            border_color = (255, 255, 100)  # Yellow for next
                            thickness = 3
                        else:
                            border_color = (100, 255, 255)  # Cyan for queue
                            thickness = 1
                    else:
                        # Early game - all discovered cards shown in cyan (queue)
                        border_color = (100, 255, 255)
                        thickness = 1
                    
                    cv2.rectangle(frame, (x-1, y-1), (x+49, y+49), border_color, thickness)
                except Exception as e:
                    pass
            else:
                cv2.rectangle(frame, (x, y), (x+48, y+48), (60, 60, 60), -1)
                cv2.rectangle(frame, (x, y), (x+48, y+48), (120, 120, 120), 1)
        
        # === MATCH INFO ===
        info_x = deck_x + 230
        info_y = panel_y + 15
        
        match_time = state.get("match_time", 0)
        play_count = len(state.get("play_history", []))
        last_played = state.get("last_played", None)
        
        cv2.putText(frame, f"Match: {match_time:.0f}s", (info_x, info_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
        cv2.putText(frame, f"Plays: {play_count}", (info_x, info_y + 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
        
        if last_played:
            last_text = last_played[:8] if len(last_played) > 8 else last_played
        else:
            last_text = "None"
        cv2.putText(frame, f"Last: {last_text}", (info_x, info_y + 40),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
        
        # === PATTERN INFO ===
        if prediction_conf > 0 and cards_discovered >= 4:
            pattern_y = info_y + 65
            cv2.putText(frame, "AI Status:", (info_x, pattern_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 255), 1)
            
            if prediction_conf >= 0.7:
                status = "High Confidence"
                status_color = (100, 255, 100)
            elif prediction_conf >= 0.4:
                status = "Learning..."
                status_color = (255, 200, 100)
            else:
                status = "Gathering Data"
                status_color = (255, 100, 100)
            
            cv2.putText(frame, status, (info_x, pattern_y + 15),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.35, status_color, 1)
    
    def _draw_fps(self, frame, w, h):
        """Draw FPS."""
        fps_color = (60, 200, 80) if self.fps >= 20 else (255, 180, 70) if self.fps >= 10 else (230, 70, 70)
        cv2.rectangle(frame, (10, 10), (110, 40), (0, 0, 0), -1)
        cv2.putText(frame, f"FPS: {self.fps:.1f}", (15, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, fps_color, 2)
    
    def _update_fps(self):
        """Update FPS."""
        self.frame_count += 1
        current_time = time.time()
        if current_time - self.last_fps_time >= 1.0:
            self.fps = self.frame_count / (current_time - self.last_fps_time)
            self.frame_count = 0
            self.last_fps_time = current_time
    
    def _save_screenshot(self, frame):
        """Save screenshot."""
        timestamp = int(time.time())
        filename = f"era_screenshot_{timestamp}.png"
        cv2.imwrite(filename, frame)
        print(f"📸 Screenshot saved: {filename}")


def main():
    """Entry point."""
    if not Path("capture_config.json").exists():
        print("❌ No capture area configured")
        return 1
    
    if not Path("assets").exists():
        print("❌ Assets folder not found")
        return 1
    
    try:
        assistant = ClashRoyaleAssistant()
        assistant.run()
        return 0
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())