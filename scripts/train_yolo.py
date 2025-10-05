#!/usr/bin/env python3
"""Fixed YOLO training script that ensures correct working directory.
Resolves path resolution issues with Ultralytics by running from project root.
Usage: python scripts/train_yolo.py [epochs] [batch_size]
"""
import os
import sys
from pathlib import Path

def main():
    # Ensure we're in the project root (where cr_data.yaml can find data/)
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    os.chdir(project_root)
    
    print(f"Working directory: {os.getcwd()}")
    
    # Get command line args with defaults
    epochs = int(sys.argv[1]) if len(sys.argv) > 1 else 50
    batch_size = int(sys.argv[2]) if len(sys.argv) > 2 else 16
    
    print(f"Training: epochs={epochs}, batch={batch_size}")
    
    try:
        from ultralytics import YOLO
        
        # Load model and train
        model = YOLO('yolov8n.pt')  # or 'yolo11n.pt' for v11
        
        model.train(
            data='model/cr_data.yaml',
            epochs=epochs,
            imgsz=640,
            batch=batch_size,
            patience=10,
            project='model',
            name='runs',
            pretrained=True,
            workers=4,
        )
        
        print("✅ Training completed successfully!")
        print("📂 Check: model/runs/*/weights/best.pt")
        
    except ImportError:
        print("❌ ultralytics not installed. Run: pip install ultralytics")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Training failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()