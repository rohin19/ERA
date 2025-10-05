"""Main entry point for ERA (Elixir and Rotation Assistant).
Handles game state tracking with test mode and model mode (commented out).
"""
import msvcrt
import time
import os
from runtime.gamestate.gamestate import GameState

# Model-related imports (commented out until needed)
# import cv2
# import numpy as np
# from runtime.config import ONNX_MODEL_PATH
# from runtime.infer.onnx_engine import CardDetectionEngine
# from runtime.capture.screen import run_qt_capture
# from PyQt5.QtWidgets import QApplication
# import sys

def clear_screen():
    """Clear the terminal screen"""
    os.system('cls' if os.name == 'nt' else 'clear')

def print_initial_screen():
    """Print the initial welcome screen"""
    clear_screen()
    print("\nERA (Edge Royale Analytics)")
    print("===================================")
    print("\nPress Enter to start tracking when your game begins")
    print("Press Q to quit")

def print_game_state(state):
    """Print current game state information"""
    clear_screen()
    print("\nERA (Edge Royale Analytics)")
    print("===================================")
    snapshot = state.snapshot()
    
    print(f"\nGame Active: {'Yes' if snapshot['game_active'] else 'No'}")
    print(f"Current Elixir: {snapshot['elixir']:.1f}")
    
    # Show all cards seen in the opponent's total deck
    deck_list = [play['card'] for play in snapshot['plays']]
    print("\nOpponent's Total Deck:")
    print(', '.join(deck_list[-8:]) if deck_list else "No cards seen yet")
    
    # Show opponent's current hand (cards 5-8 in rotation after seeing all 8)
    print("\nOpponent's Current Hand:")
    print(', '.join(snapshot['available_cards']) if snapshot['available_cards'] else "Need to see more cards")
    
    # Show the next card (card that will come after current hand)
    print("\nOpponent's Next Card:")
    print(snapshot['next_card'] if snapshot['next_card'] else "Need to see more cards")
    
    print("\nControls:")
    print("  'R' - Reset game state")
    print("  'T' - Test card play (simulates detecting a card)")
    print("  'Q' - Quit program")

def main():
    # Initialize components
    state = GameState()
    
    # Test cards (these should match your model's classes)
    test_cards = [
        "Baby Dragon", "Bomber", "Dart Goblin", "Giant",
        "Hog Rider", "Knight", "Mini Pekka", "Valkyrie"
    ]
    current_test_card = 0

    # === Model-based version (commented out) ===
    """
    # Initialize model components
    engine = CardDetectionEngine(ONNX_MODEL_PATH)
    
    def process_frame(frame):
        if frame is None:
            return frame

        # Run inference
        detections = engine.predict(frame)
        state.ingest_detections(detections)
        
        # Update display
        game_state = state.snapshot()
        
        # Draw overlay info on frame
        info_text = []
        info_text.append(f"Game Active: {'Yes' if game_state['game_active'] else 'No'}")
        if game_state['game_active']:
            info_text.append(f"Elixir: {game_state['elixir']:.1f}")
            info_text.append(f"Available Cards: {', '.join(game_state['available_cards'])}")
            if game_state['next_card']:
                info_text.append(f"Next Card: {game_state['next_card']}")
            info_text.append(f"Cards Seen: {game_state['total_cards_seen']}")
        
        # Add text to frame
        y = 30
        for text in info_text:
            cv2.putText(frame, text, (10, y), 
                      cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            y += 30
        
        # Draw detection boxes
        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), 
                        (0, 255, 0), 2)
            cv2.putText(frame, f"{det['class']} {det['conf']:.2f}", 
                      (int(x1), int(y1)-10), cv2.FONT_HERSHEY_SIMPLEX, 
                      0.5, (0, 255, 0), 2)
        
        return frame

    def handle_key(key):
        if key == 's' and not state.game_started:
            state.start_game()
            print("\nGame tracking started!")
        elif key == 'r':
            state.stop_game()
            print("\nGame tracking reset. Press 'S' to start new game.")
        elif key == 'q':
            print("\nQuitting...")
            return True
        return False

    # Run the Qt capture loop
    run_qt_capture(
        process_frame=process_frame,
        handle_key=handle_key,
        title="ERA - Elixir and Rotation Assistant",
        target_fps=30
    )
    """
    
    # === Test version (active) ===
    print_initial_screen()
    running = True

    try:
        # Wait for Enter to start or Q to quit
        while running and not state.game_started:
            if msvcrt.kbhit():
                key = msvcrt.getch().decode('utf-8').lower()
                if key == '\r':  # Enter key
                    state.start_game()
                    clear_screen()
                    print("\nGame tracking started!")
                elif key == 'q':
                    print("\nQuitting...")
                    running = False
            time.sleep(0.1)

        # Main game loop - only runs after Enter is pressed
        last_update = time.time()
        while running and state.game_started:
            # Handle keyboard input
            if msvcrt.kbhit():
                key = msvcrt.getch().decode('utf-8').lower()
                if key == 'r':
                    state.stop_game()
                    print("\nGame tracking reset. Press Enter to start new game.")
                    break  # Go back to waiting for Enter
                elif key == 't':
                    # Simulate a card being played
                    card = test_cards[current_test_card]
                    state.record_play(card)
                    print(f"\nSimulated play of {card}")
                    current_test_card = (current_test_card + 1) % len(test_cards)
                elif key == 'q':
                    print("\nQuitting...")
                    running = False

            # Update display every second
            if time.time() - last_update >= 0.5:
                print_game_state(state)
                last_update = time.time()

            # Small sleep to prevent CPU overuse
            time.sleep(0.05)

    except KeyboardInterrupt:
        print("\nExiting...")

if __name__ == "__main__":
    main()
