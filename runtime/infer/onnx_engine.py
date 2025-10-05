"""Real-time inference engine that integrates with screen capture system."""
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional
import cv2
import time

try:
    import onnxruntime as ort
except ImportError:
    ort = None

# Import screen capture functions
try:
    import sys
    from pathlib import Path
    # Ensure we can import from project root
    project_root = Path(__file__).parent.parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    
    from runtime.capture.screen import capture_single_frame, get_current_capture_config, is_qt_capture_available
    CAPTURE_AVAILABLE = True
except ImportError as e:
    print(f"Screen capture integration not available: {e}")
    capture_single_frame = None
    get_current_capture_config = None
    is_qt_capture_available = None
    CAPTURE_AVAILABLE = False

class CardDetectionEngine:
    """Real-time card detection using pre-exported ONNX model."""
    
    def __init__(self, model_path: str = "model/weights/best.onnx"):
        """Initialize inference engine.
        
        Args:
            model_path: Path to pre-exported ONNX model file
        """
        self.model_path = Path(model_path)
        
        if not self.model_path.exists():
            raise FileNotFoundError(
                f"ONNX model not found: {model_path}\n"
                f"Export it first with: python scripts/export_model.py"
            )
        
        if ort is None:
            raise ImportError("Install with: pip install onnxruntime opencv-python")
        
        # Initialize ONNX Runtime
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        self.session = ort.InferenceSession(str(self.model_path), providers=providers)
        
        # Get model info
        self.input_name = self.session.get_inputs()[0].name
        self.output_names = [output.name for output in self.session.get_outputs()]
        
        input_shape = self.session.get_inputs()[0].shape
        self.input_size = input_shape[2]  # Should be 640
        
        # Class names from training
        self.class_names = [
            "Baby Dragon", "Bomber", "Dart Goblin", "Giant",
            "Hog Rider", "Knight", "Mini Pekka", "Valkyrie"
        ]
        
        print(f"Inference engine loaded: {model_path}")
        print(f"Input shape: {input_shape}")
        print(f"Provider: {self.session.get_providers()[0]}")
        print(f"Classes: {len(self.class_names)}")
    
    def preprocess_frame(self, frame: np.ndarray) -> np.ndarray:
        """Convert screen capture frame to model input format.
        
        Args:
            frame: BGR frame from screen capture (H, W, 3)
            
        Returns:
            Model input tensor (1, 3, 640, 640)
        """
        # Convert BGR to RGB (screen capture gives BGR, model expects RGB)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Resize to model input size (640x640)
        resized = cv2.resize(rgb_frame, (self.input_size, self.input_size))
        
        # Normalize to [0, 1]
        normalized = resized.astype(np.float32) / 255.0
        
        # Change from HWC to CHW format
        transposed = np.transpose(normalized, (2, 0, 1))
        
        # Add batch dimension
        batched = np.expand_dims(transposed, axis=0)
        
        return batched
    
    def postprocess_detections(self, model_output: np.ndarray, 
                              confidence_threshold: float = 0.5) -> List[Dict[str, Any]]:
        """Convert model output to card detections.
        
        Args:
            model_output: Raw YOLO output (1, 84, 8400)
            confidence_threshold: Minimum confidence to keep
            
        Returns:
            List of detected cards with positions
        """
        predictions = model_output[0].T  # (8400, 84)
        detections = []
        
        for prediction in predictions:
            # Extract bbox coordinates and class scores
            x_center, y_center, width, height = prediction[:4]
            class_scores = prediction[4:4 + len(self.class_names)]
            
            # Find most confident class
            class_id = np.argmax(class_scores)
            confidence = class_scores[class_id]
            
            if confidence >= confidence_threshold:
                # Convert to pixel coordinates (relative to 640x640)
                x1 = int((x_center - width / 2) * self.input_size)
                y1 = int((y_center - height / 2) * self.input_size)
                x2 = int((x_center + width / 2) * self.input_size)
                y2 = int((y_center + height / 2) * self.input_size)
                
                detection = {
                    'class_id': int(class_id),
                    'class_name': self.class_names[class_id],
                    'confidence': float(confidence),
                    'bbox': [x1, y1, x2, y2],
                    'center': [int(x_center * self.input_size), int(y_center * self.input_size)]
                }
                detections.append(detection)
        
        return detections
    
    def predict(self, frame: np.ndarray, confidence_threshold: float = 0.5) -> List[Dict[str, Any]]:
        """Run inference on a screen capture frame.
        
        Args:
            frame: BGR frame from screen capture
            confidence_threshold: Minimum confidence for detections
            
        Returns:
            List of card detections
        """
        # Preprocess the frame
        input_tensor = self.preprocess_frame(frame)
        
        # Run inference
        outputs = self.session.run(self.output_names, {self.input_name: input_tensor})
        
        # Postprocess to detections
        detections = self.postprocess_detections(outputs[0], confidence_threshold)
        
        return detections
    
    def draw_detections(self, frame: np.ndarray, detections: List[Dict[str, Any]], 
                       scale_to_original: bool = True) -> np.ndarray:
        """Draw detection boxes on the frame for visualization.
        
        Args:
            frame: Original BGR frame from screen capture
            detections: List of detections from predict()
            scale_to_original: Whether to scale boxes to original frame size
            
        Returns:
            Frame with detection boxes drawn
        """
        if not detections:
            return frame
        
        result_frame = frame.copy()
        
        # Calculate scale factors
        if scale_to_original:
            scale_x = frame.shape[1] / self.input_size
            scale_y = frame.shape[0] / self.input_size
        else:
            scale_x = scale_y = 1.0
        
        for detection in detections:
            # Scale bounding box to original frame size
            x1, y1, x2, y2 = detection['bbox']
            x1 = int(x1 * scale_x)
            y1 = int(y1 * scale_y)
            x2 = int(x2 * scale_x)
            y2 = int(y2 * scale_y)
            
            # Draw bounding box
            cv2.rectangle(result_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Draw label with background
            label = f"{detection['class_name']}: {detection['confidence']:.2f}"
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
            cv2.rectangle(result_frame, (x1, y1 - label_size[1] - 10), 
                         (x1 + label_size[0], y1), (0, 255, 0), -1)
            cv2.putText(result_frame, label, (x1, y1 - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
        
        return result_frame
    
    # Not sure if right placement <--

    def start_gaming_overlay(self):
        """Start the beautiful gaming overlay interface.
        
        This launches the gaming overlay that shows detections in real-time
        over your configured capture area.
        """
        print("🎮 Starting Gaming Overlay...")
        print("This will show a beautiful overlay over your game area")
        
        try:
            from runtime.overlay.gaming_overlay import OverlayManager
            manager = OverlayManager()
            return manager.start()
            
        except ImportError as e:
            print(f"❌ Gaming overlay not available: {e}")
            print("Make sure PyQt5 is installed: pip install PyQt5")
            return False
        except Exception as e:
            print(f"❌ Gaming overlay failed: {e}")
            return False


def test_integration():
    """Test inference engine integration with screen capture system."""
    print("Testing inference engine integration...")
    print("=" * 50)
    
    # Check if capture system is available
    if not CAPTURE_AVAILABLE:
        print("ERROR: Screen capture system not available")
        print("Make sure runtime/capture/screen.py is accessible")
        return False
    
    print("Screen capture system: Available")
    print(f"Qt interface available: {is_qt_capture_available()}")
    
    # Check current capture configuration
    config = get_current_capture_config()
    if config:
        print(f"Current capture area: {config['width']}x{config['height']} at ({config['left']}, {config['top']})")
    else:
        print("No saved capture configuration (will use default)")
    
    try:
        # Initialize inference engine
        print("\nInitializing inference engine...")
        engine = CardDetectionEngine()
        
        # Capture a frame using the integrated system
        print("Capturing frame...")
        frame = capture_single_frame()
        
        if frame is None:
            print("ERROR: Failed to capture frame")
            return False
        
        print(f"Captured frame: {frame.shape}")
        
        # Run inference
        print("Running inference...")
        start_time = time.time()
        detections = engine.predict(frame, confidence_threshold=0.3)
        inference_time = (time.time() - start_time) * 1000
        
        print(f"Inference completed in {inference_time:.1f}ms")
        print(f"Found {len(detections)} detections:")
        
        # Show results
        for i, det in enumerate(detections):
            print(f"  {i+1}. {det['class_name']}: {det['confidence']:.2f} at {det['bbox']}")
        
        # Draw detections and show result
        result_frame = engine.draw_detections(frame, detections)
        
        cv2.imshow("Inference Test - Press any key to close", result_frame)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
        print("\nIntegration test completed successfully!")
        return True
        
    except Exception as e:
        print(f"ERROR: Test failed: {e}")
        return False


def test_live_inference():
    """Test live inference with screen capture (requires model)."""
    print("Testing live inference...")
    print("Press 'q' to quit, 'c' to change confidence threshold")
    
    if not CAPTURE_AVAILABLE:
        print("ERROR: Screen capture not available")
        return False
    
    try:
        engine = CardDetectionEngine()
        confidence = 0.5
        
        print(f"\nLive inference started (confidence: {confidence})")
        print("Controls: 'q' = quit, 'c' = change confidence, 's' = screenshot")
        
        while True:
            # Capture frame
            frame = capture_single_frame()
            if frame is None:
                print("Capture failed")
                break
            
            # Run inference
            start_time = time.time()
            detections = engine.predict(frame, confidence_threshold=confidence)
            inference_time = (time.time() - start_time) * 1000
            
            # Draw detections
            result_frame = engine.draw_detections(frame, detections)
            
            # Add info overlay
            info_text = f"Detections: {len(detections)} | Time: {inference_time:.1f}ms | Conf: {confidence:.1f}"
            cv2.putText(result_frame, info_text, (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            cv2.imshow("Live Inference - q:quit c:confidence s:save", result_frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('c'):
                # Cycle through confidence levels
                confidence = 0.3 if confidence >= 0.7 else confidence + 0.2
                print(f"Confidence threshold: {confidence}")
            elif key == ord('s'):
                # Save screenshot
                timestamp = int(time.time())
                filename = f"inference_result_{timestamp}.png"
                cv2.imwrite(filename, result_frame)
                print(f"Saved: {filename}")
        
        cv2.destroyAllWindows()
        return True
        
    except Exception as e:
        print(f"Live inference failed: {e}")
        return False


# Updated for overlay support

if __name__ == "__main__":
    print("ONNX Inference Engine Test Suite")
    print("=" * 40)
    
    print("Available tests:")
    print("1. Integration test (basic)")
    print("2. Live inference (OpenCV window)")
    print("3. 🎮 Gaming overlay (beautiful PyQt overlay)")
    print()
    
    choice = input("Choose test (1-3, default=3): ").strip() or "3"
    
    if choice == "1":
        success = test_integration()
    elif choice == "2":
        success = test_live_inference()
    elif choice == "3":
        # Gaming overlay mode
        try:
            engine = CardDetectionEngine()
            success = engine.start_gaming_overlay()
        except Exception as e:
            print(f"Gaming overlay failed: {e}")
            success = False
    else:
        print("Invalid choice")
        success = False
    
    if not success:
        print("Test failed!")