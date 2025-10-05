import sys
from pathlib import Path
from ultralytics import YOLO

def find_latest_model():
    """Find the most recent trained model."""
    model_dir = Path("model")
    best_files = list(model_dir.glob("runs*/weights/best.pt"))
    
    if not best_files:
        print("ERROR: No trained models found!")
        print("Train a model first with: python scripts/train_yolo.py")
        return None
    
    # Get the newest one
    latest_model = max(best_files, key=lambda x: x.stat().st_mtime)
    print(f"Found latest model: {latest_model}")
    return latest_model

def export_model(model_path):
    """Convert PyTorch model to ONNX format."""
    print(f"Converting model: {model_path}")
    
    # Load the trained model
    model = YOLO(str(model_path))
    
    # Convert to ONNX (like making a reference card from a textbook)
    onnx_path = model.export(
        format='onnx',
        opset=17,          # Version of ONNX to use
        simplify=True,     # Make it cleaner
        dynamic=False,     # Fixed size for speed
        imgsz=640          # Input image size
    )
    
    # Move to a standard location
    output_dir = Path("model/weights")
    output_dir.mkdir(exist_ok=True)
    
    import shutil
    final_path = output_dir / "best.onnx"
    shutil.copy(onnx_path, final_path)
    
    print(f"Model copied to standard location: {final_path}")
    return final_path

def test_exported_model(onnx_path):
    """Make sure our converted model works."""
    try:
        import onnxruntime as ort
        session = ort.InferenceSession(str(onnx_path))
        
        # Check what the model expects
        input_info = session.get_inputs()[0]
        output_info = session.get_outputs()[0]
        
        print("Model conversion successful!")
        print(f"Input name: {input_info.name}")
        print(f"Input shape: {input_info.shape}")
        print(f"Output name: {output_info.name}")
        print(f"Output shape: {output_info.shape}")
        return True
        
    except ImportError:
        print("ERROR: Need to install onnxruntime")
        print("Run: pip install onnxruntime")
        return False
    except Exception as e:
        print(f"ERROR: Model test failed: {e}")
        return False

def main():
    # Step 1: Find our trained model
    model_path = find_latest_model()
    if not model_path:
        sys.exit(1)
    
    try:
        # Step 2: Convert it to ONNX
        onnx_path = export_model(model_path)
        
        # Step 3: Test that it works
        if test_exported_model(onnx_path):
            print("SUCCESS: Model ready for real-time inference!")
            print(f"Use this file: {onnx_path}")
        else:
            print("WARNING: Model exported but couldn't test it")
            
    except Exception as e:
        print(f"ERROR: Export failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()