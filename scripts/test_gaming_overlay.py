"""Test script specifically for the gaming overlay."""
import sys
from pathlib import Path

# Ensure project root is in path
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

def main():
    """Test the gaming overlay system."""
    print("🎮 Gaming Overlay Test")
    print("=" * 30)
    
    # Check if model exists
    model_path = Path("model/weights/best.onnx")
    if not model_path.exists():
        print(f"❌ Model not found: {model_path}")
        print("📝 Export your model first with: python scripts/export_model.py")
        return 1
    
    # Check if capture configuration exists
    try:
        from runtime.overlay.qt_capture_view import load_capture_config
        config = load_capture_config()
        if config is None:
            print("❌ No capture area configured.")
            print("📝 Run the capture tool first: python runtime/capture/screen.py")
            print("📝 Use 'R' to select your game area, then close the window")
            return 1
        else:
            print(f"✅ Using capture area: {config.width}x{config.height} at ({config.left}, {config.top})")
    except ImportError:
        print("❌ Capture system not available")
        return 1
    
    # Start the gaming overlay
    try:
        from runtime.overlay.gaming_overlay import OverlayManager
        
        print("\n🚀 Starting gaming overlay...")
        print("📋 The overlay will appear over your configured capture area")
        print("📋 Controls:")
        print("   ESC = Close overlay")
        print("   F11 = Toggle fullscreen")
        print("   +/- buttons = Adjust confidence threshold")
        print("\n⏳ Starting in 3 seconds...")
        
        import time
        time.sleep(3)
        
        manager = OverlayManager()
        return manager.start()
        
    except KeyboardInterrupt:
        print("\n🛑 Stopped by user")
        return 0
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())