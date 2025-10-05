"""ERA Launcher - Clean integration with advanced GameState."""
import sys
from pathlib import Path

# Ensure project root is in path
project_root = Path(__file__).parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

def check_dependencies():
    """Check if all required packages are installed."""
    missing = []
    
    try:
        import cv2
    except ImportError:
        missing.append("opencv-python")
    
    try:
        import mss
    except ImportError:
        missing.append("mss")
    
    try:
        import onnxruntime
    except ImportError:
        missing.append("onnxruntime")
    
    try:
        import numpy
    except ImportError:
        missing.append("numpy")
    
    if missing:
        print("❌ Missing dependencies:")
        for pkg in missing:
            print(f"   pip install {pkg}")
        return False
    
    return True

def check_setup():
    """Check if system is ready."""
    issues = []
    
    # Check model
    model_path = Path("model/weights/best.onnx")
    if not model_path.exists():
        issues.append(f"Model not found: {model_path}")
        issues.append("  Fix: Export your model to ONNX format")
    
    # Check capture config
    capture_config = Path("capture_config.json")
    if not capture_config.exists():
        issues.append("No capture area configured")
        issues.append("  Fix: Run option 1 to set up capture area")
    else:
        print(f"✅ Capture config found: {capture_config}")
    
    # Check GameState
    gamestate_path = Path("runtime/gamestate/gamestate.py")
    if gamestate_path.exists():
        print("✅ Advanced GameState with AI predictions available")
    else:
        issues.append("GameState file missing")
    
    return issues

def main():
    """Launch the ERA system."""
    print("🎮 Edge Royale Analytics (ERA) Launcher")
    print("=" * 45)
    
    # Check dependencies first
    if not check_dependencies():
        print("\nInstall missing packages and run again.")
        return 1
    
    # Check setup
    issues = check_setup()
    if issues:
        print("❌ Setup Issues Found:")
        for issue in issues:
            print(f"   {issue}")
        print("\nFix these issues and run again.")
        return 1
    
    print("✅ System ready!")
    print("\nChoose mode:")
    print("1. 📍 Setup capture area (select game region)")
    print("2. 🧪 Test inference (check model works)")
    print("3. 🎮 Simple Assistant (Clean & Fast) ⭐ RECOMMENDED")
    print("4. 📤 Export model (PyTorch → ONNX)")
    
    try:
        choice = input("\nEnter choice (1-4, default=3): ").strip() or "3"
        
        if choice == "1":
            print("📍 Starting capture area setup...")
            from runtime.capture.screen import main as capture_main
            capture_main()
            
        elif choice == "2":
            print("🧪 Starting inference test...")
            from runtime.infer.onnx_engine import test_live_inference
            test_live_inference()
            
        elif choice == "3":
            print("🎮 Starting Simple Assistant...")
            print("✨ Features: Advanced GameState, AI predictions, smooth rendering")
            from clash_royale_simple import main as simple_main
            return simple_main()
            
        elif choice == "4":
            print("📤 Starting model export...")
            try:
                from scripts.export_model import main as export_main
                export_main()
            except ImportError:
                print("❌ Export script not found")
                return 1
            
        else:
            print("Invalid choice")
            return 1
            
    except KeyboardInterrupt:
        print("\nStopped by user")
        return 0
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())