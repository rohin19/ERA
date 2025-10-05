"""ERA Launcher - Simple integration of all components."""
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
        import PyQt5
    except ImportError:
        missing.append("PyQt5")
    
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
        issues.append("  Fix: Run 'python scripts/export_model.py'")
    
    # Check capture config
    try:
        from runtime.overlay.qt_capture_view import load_capture_config
        config = load_capture_config()
        if not config:
            issues.append("No capture area configured")
            issues.append("  Fix: Run 'python runtime/capture/screen.py' and press 'R' to select area")
        else:
            print(f"✅ Capture area: {config.width}x{config.height} at ({config.left}, {config.top})")
    except ImportError:
        issues.append("Capture system not available")
    
    return issues

def main():
    """Launch the complete ERA system."""
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
    print("1. Setup capture area (select game region)")
    print("2. Test inference (OpenCV window)")
    print("3. 🎮 Gaming Overlay (unified system)")
    print("4. 🏆 Clash Royale Assistant (PRODUCTION)")
    print("5. Export model (PyTorch → ONNX)")
    
    try:
        choice = input("\nEnter choice (1-5, default=4): ").strip() or "4"
        
        if choice == "1":
            print("Starting capture area setup...")
            from runtime.capture.screen import main as capture_main
            capture_main()
            
        elif choice == "2":
            print("Starting inference test...")
            from runtime.infer.onnx_engine import test_live_inference
            test_live_inference()

        elif choice == "3":
            print("🚀 Starting Unified Gaming System...")
            print("📊 Complete pipeline: Gameplay → Capture → Inference → Game State → Display")
            from runtime.overlay.unified_gaming_overlay import main as unified_main
            return unified_main()
            
        elif choice == "4":
            print("🏆 Starting Clash Royale Assistant (Production)...")
            print("📊 Features: Opponent tracking, elixir estimation, deck analysis")
            from clash_royale_assistant import main as assistant_main
            return assistant_main()
            
        elif choice == "5":
            print("Starting model export...")
            from scripts.export_model import main as export_main
            export_main()
            
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