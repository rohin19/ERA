import mss
import numpy as np
import cv2
import time
import json
import os

# Global variables for rectangle selection
drawing = False
rect_start = None
rect_end = None
selection_mode = False

def mouse_callback(event, x, y, flags, param):
    global drawing, rect_start, rect_end
    
    if not selection_mode:
        return
    
    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        rect_start = (x, y)
        rect_end = (x, y)
    
    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            rect_end = (x, y)
    
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        rect_end = (x, y)

def load_capture_config():
    """Load saved capture area configuration"""
    config_file = "capture_config.json"
    if os.path.exists(config_file):
        try:
            with open(config_file, 'r') as f:
                config = json.load(f)
                return config.get("monitor", None)
        except:
            pass
    return None

def save_capture_config(monitor):
    """Save capture area configuration"""
    config = {"monitor": monitor}
    with open("capture_config.json", 'w') as f:
        json.dump(config, f, indent=2)
    print(f"Saved capture area: {monitor}")

def selection_overlay_mode(sct, screen_width, screen_height):
    """Mode for selecting capture area with transparent overlay"""
    global selection_mode, rect_start, rect_end, drawing
    
    # Destroy all existing windows and wait for them to actually disappear
    cv2.destroyAllWindows()
    cv2.waitKey(1)  # Process window destruction
    time.sleep(0.5)  # Wait for windows to actually close on screen
    
    selection_mode = True
    rect_start = None
    rect_end = None
    
    # Take a screenshot of the full screen to use as the background BEFORE creating any windows
    full_monitor = sct.monitors[1]
    screenshot = np.array(sct.grab(full_monitor))
    screenshot = cv2.cvtColor(screenshot, cv2.COLOR_BGRA2BGR)
    
    # Create windowed overlay that covers entire screen
    cv2.namedWindow("Selection Overlay", cv2.WINDOW_NORMAL)
    cv2.setWindowProperty("Selection Overlay", cv2.WND_PROP_TOPMOST, 1)
    cv2.moveWindow("Selection Overlay", 0, 0)
    cv2.resizeWindow("Selection Overlay", screen_width, screen_height)
    
    # Force window to stay at full screen size (disable resizing)
    cv2.setWindowProperty("Selection Overlay", cv2.WND_PROP_ASPECT_RATIO, cv2.WINDOW_FREERATIO)
    cv2.setMouseCallback("Selection Overlay", mouse_callback)
    
    # Give the window time to resize properly
    cv2.waitKey(100)
    
    print("\nSelection Mode Active:")
    print("- Click and drag to select capture area")
    print("- Press 's' to save selection")
    print("- Press 'c' to cancel")
    print("- Current selection will be shown in green")

    while True:
        # Use the screenshot as the background
        overlay = screenshot.copy()

        # Add instructions
        cv2.putText(overlay, "SELECTION MODE", (50, 50), 
                   cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 255), 3)
        cv2.putText(overlay, "Click and drag to select area", (50, 100), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(overlay, "Press 's' to save, 'c' to cancel", (50, 140), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        # Draw selection rectangle
        if rect_start and rect_end:
            x1, y1 = rect_start
            x2, y2 = rect_end

            # Ensure coordinates are in correct order
            left = min(x1, x2)
            top = min(y1, y2)
            right = max(x1, x2)
            bottom = max(y1, y2)

            # Draw rectangle
            cv2.rectangle(overlay, (left, top), (right, bottom), (0, 255, 0), 3)

            # Add coordinates text
            width = right - left
            height = bottom - top
            cv2.putText(overlay, f"Area: {width}x{height} at ({left},{top})", 
                       (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        cv2.imshow("Selection Overlay", overlay)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('s') or key == ord('S'):  # Save selection
            if rect_start and rect_end:
                x1, y1 = rect_start
                x2, y2 = rect_end

                left = min(x1, x2)
                top = min(y1, y2)
                width = abs(x2 - x1)
                height = abs(y2 - y1)

                if width > 10 and height > 10:  # Minimum size check
                    monitor = {
                        "top": top,
                        "left": left,
                        "width": width,
                        "height": height
                    }
                    save_capture_config(monitor)
                    cv2.destroyWindow("Selection Overlay")
                    selection_mode = False
                    return monitor
                else:
                    print("Selection too small! Please select a larger area.")
            else:
                print("No selection made! Please draw a rectangle first.")

        elif key == ord('c') or key == ord('C') or key == 27:  # Cancel
            cv2.destroyWindow("Selection Overlay")
            selection_mode = False
            return None

def main():
    global selection_mode
    
    # Initialize MSS for screen capture
    sct = mss.mss()
    # Debug: Print all monitor information
    print("=== Monitor Debug Information ===")
    for i, monitor in enumerate(sct.monitors):
        print(f"Monitor {i}: {monitor}")
    
    # Get the primary monitor's full dimensions
    primary_monitor = sct.monitors[1]
    screen_width = primary_monitor["width"]
    screen_height = primary_monitor["height"]
    
    # print(f"\n=== Display Analysis ===")
    print(f"Detected logical resolution: {screen_width}x{screen_height}")

    
    print(f"Detected screen resolution: {screen_width}x{screen_height}")
    print("\nScreen Capture with Selection Tool")
    print("=" * 40)
    print("Controls:")
    print("- Press 'r' to enter selection mode")
    print("- Press 'q' to quit")
    print("- Press 's' to take screenshot of current capture area")
    
    # Try to load saved capture configuration
    monitor = load_capture_config()
    if monitor:
        print(f"Loaded saved capture area: {monitor['width']}x{monitor['height']} at ({monitor['left']}, {monitor['top']})")
    else:
        # Default capture area
        monitor = {
            "top": 100,
            "left": 100,
            "width": 800,
            "height": 600
        }
        print(f"Using default capture area: {monitor['width']}x{monitor['height']} at ({monitor['left']}, {monitor['top']})")
    
    # Track FPS
    fps_counter = 0
    start_time = time.time()
    
    try:
        while True:
            if not selection_mode:
                # Normal capture mode
                frame = np.array(sct.grab(monitor))
                frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

                
                # Add FPS and info
                fps_counter += 1
                elapsed_time = time.time() - start_time
                if elapsed_time > 0:
                    fps = fps_counter / elapsed_time
                    cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30), 
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
                cv2.putText(frame, f"Area: {monitor['width']}x{monitor['height']}", (10, 70), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.putText(frame, "Press 'r' for selection mode", (10, 110), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
                
                cv2.imshow("Screen Capture", frame)
            
            key = cv2.waitKey(30) & 0xFF
            if key == ord('q') or key == ord('Q') or key == 27:  # Quit
                break
            elif key == ord('r') or key == ord('R'):  # Selection mode
                cv2.destroyAllWindows()
                
                new_monitor = selection_overlay_mode(sct, screen_width, screen_height)
                if new_monitor:
                    monitor = new_monitor
                    # Reset FPS counter
                    fps_counter = 0
                    start_time = time.time()
            elif key == ord('s') or key == ord('S'):  # Save screenshot
                if not selection_mode:
                    frame = np.array(sct.grab(monitor))
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
                    timestamp = int(time.time())
                    filename = f"screenshot_{timestamp}.png"
                    cv2.imwrite(filename, frame)
                    print(f"Screenshot saved as {filename}")
                    
    except KeyboardInterrupt:
        print("\nCapture interrupted by user")
    
    finally:
        cv2.destroyAllWindows()
        if fps_counter > 0 and time.time() - start_time > 0:
            final_fps = fps_counter / (time.time() - start_time)
            print(f"Final FPS: {final_fps:.1f}")
        print("Screen capture ended")

if __name__ == "__main__":
    main()