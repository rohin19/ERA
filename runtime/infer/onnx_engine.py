"""Real-time inference engine that processes screen capture frames."""
import numpy as np
from pathlib import Path
from typing import List, Dict, Any
import cv2

try:
    import onnxruntime as ort
except ImportError:
    ort = None

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
        """Convert screen capture frame to model input.
        
        Takes the BGR frame from your screen.py and converts it to
        the exact format the AI model expects.
        
        Args:
            frame: BGR frame from MSS screen capture (H, W, 3)
            
        Returns:
            Model input tensor (1, 3, 640, 640)
        """
        # Convert BGR to RGB (screen.py gives BGR, model expects RGB)
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
        
        This is the main function called from your screen capture loop.
        
        Args:
            frame: BGR frame from screen.py
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
                       original_size: tuple = None) -> np.ndarray:
        """Draw detection boxes on the frame for visualization.
        
        Args:
            frame: Original BGR frame
            detections: List of detections from predict()
            original_size: (width, height) of original frame before resize
            
        Returns:
            Frame with detection boxes drawn
        """
        if not detections:
            return frame
        
        # Calculate scale factors if frame was resized
        if original_size:
            scale_x = original_size[0] / self.input_size
            scale_y = original_size[1] / self.input_size
        else:
            scale_x = frame.shape[1] / self.input_size
            scale_y = frame.shape[0] / self.input_size
        
        result_frame = frame.copy()
        
        for detection in detections:
            # Scale bounding box back to original frame size
            x1, y1, x2, y2 = detection['bbox']
            x1 = int(x1 * scale_x)
            y1 = int(y1 * scale_y)
            x2 = int(x2 * scale_x)
            y2 = int(y2 * scale_y)
            
            # Draw bounding box
            cv2.rectangle(result_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Draw label
            label = f"{detection['class_name']}: {detection['confidence']:.2f}"
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
            cv2.rectangle(result_frame, (x1, y1 - label_size[1] - 10), 
                         (x1 + label_size[0], y1), (0, 255, 0), -1)
            cv2.putText(result_frame, label, (x1, y1 - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
        
        return result_frame


def test_with_screen_capture():
    """Test inference engine with screen capture from your screen.py."""
    print("Testing inference engine with screen capture...")
    
    try:
        # Initialize inference engine
        engine = CardDetectionEngine()
        
        # Try to capture screen like your screen.py does
        import mss
        with mss.mss() as sct:
            # Use same default area as screen.py
            monitor = {
                "top": 100,
                "left": 100,
                "width": 800,
                "height": 600
            }
            
            # Capture frame
            frame = np.array(sct.grab(monitor))
            frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
            
            print(f"Captured frame: {frame.shape}")
            
            # Run inference
            detections = engine.predict(frame, confidence_threshold=0.3)
            
            # Draw detections
            result_frame = engine.draw_detections(frame, detections)
            
            print(f"Found {len(detections)} detections:")
            for i, det in enumerate(detections):
                print(f"  {i+1}. {det['class_name']}: {det['confidence']:.2f}")
            
            # Show result
            cv2.imshow("Inference Test", result_frame)
            cv2.waitKey(3000)  # Show for 3 seconds
            cv2.destroyAllWindows()
            
            return True
            
    except Exception as e:
        print(f"Test failed: {e}")
        return False


if __name__ == "__main__":
    test_with_screen_capture()