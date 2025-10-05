# clash_state.py

class CardDetection:
    def __init__(self, card_name, bbox, confidence, frame=None):
        self.card_name = card_name
        self.bbox = bbox  # [x1, y1, x2, y2]
        self.confidence = confidence
        self.frame = frame

class GameState:
    def __init__(self):
        self.visible_cards = []  # List of CardDetection (current frame)
        self.prev_visible_cards = []  # List of CardDetection (previous frame)
        self.elixir_opponent = 5.0
        self.last_played = None
        self.play_history = []  # List of card names
        self.next_prediction = []  # List of card names
        self.deck = [None] * 8  # Track opponent's deck: [0-3] current hand, [4-7] queue
        import time
        self.last_elixir_update = time.time()
        self.match_start_time = time.time()  # Track when match started for double elixir
        self.elixir_counting_enabled = False

    def start_elixir_counting(self):
        import time
        self.elixir_counting_enabled = True
        self.last_elixir_update = time.time()
        self.match_start_time = time.time()

    def update(self, detections):
        """
        Update game state with new detections.
        detections: list of dicts with keys 'card', 'bbox', 'confidence', 'frame' (optional)
        Also updates elixir incrementally based on time.
        Implements deck discovery and hand cycling logic.
        """
        self.update_elixir()

        # Card elixir costs (example, add more as needed)
        card_elixir_cost = {
            "Pekka": 7,
            "Giant": 5,
            "Goblin": 2,
            "Wizard": 5,
            "Knight": 3,
            "Bomber": 3,
            "Hog Rider": 4,
            "Dart Goblin": 5,
            "Mini Pekka": 4,
            "Baby Dragon": 4,
            "Valkyrie": 4,
            # ... add all relevant cards
        }

        # Track play order for deck discovery
        played_this_frame = []
        for d in detections:
            card_name = d["card"]
            if card_name not in played_this_frame:
                played_this_frame.append(card_name)

        # Process each played card
        for card_name in played_this_frame:
            # If card is already in hand (positions 0-3) and elixir is high enough, play and cycle
            cost = card_elixir_cost.get(card_name, None)
            if card_name in self.deck[:4] and cost is not None and self.elixir_opponent >= cost:
                self.elixir_opponent -= cost
                self.elixir_opponent = max(self.elixir_opponent, 0)
                self.last_played = card_name
                self.play_history.append(card_name)
                
                # Find played card position in hand (0-3)
                played_pos = self.deck.index(card_name)
                
                # Move played card to back of deck (position 7)
                played_card = self.deck[played_pos]
                
                # Shift cards: move next card from queue (position 4) to hand
                if self.deck[4] is not None:
                    self.deck[played_pos] = self.deck[4]
                    # Shift queue cards forward
                    for i in range(4, 7):
                        self.deck[i] = self.deck[i + 1]
                    # Put played card at back of queue
                    self.deck[7] = played_card
                else:
                    # If no card in queue, just move played card to back
                    self.deck[played_pos] = None
                    self.deck[7] = played_card
            
            # If card is new (not in deck), add it and deduct elixir
            elif card_name not in self.deck:
                # Deduct elixir for new card played
                if self.elixir_opponent >= cost:
                    self.elixir_opponent -= cost
                    self.elixir_opponent = max(self.elixir_opponent, 0)
                    self.last_played = card_name
                    self.play_history.append(card_name)
                
                # Shift entire deck forward by 1, new card goes to position 7
                for i in range(7):
                    self.deck[i] = self.deck[i + 1]
                self.deck[7] = card_name

        # Update visible cards
        self.visible_cards = [CardDetection(
            card_name=d["card"],
            bbox=d["bbox"],
            confidence=d["confidence"],
            frame=d.get("frame")
        ) for d in detections]

    def update_elixir(self):
        if not getattr(self, 'elixir_counting_enabled', False):
            return
        import time
        now = time.time()
        elapsed = now - self.last_elixir_update
        # Check if we're in double elixir time (after 2 minutes = 120 seconds)
        match_elapsed = now - self.match_start_time
        is_double_elixir = match_elapsed >= 120
        # Double elixir time: increment every 0.14 seconds instead of 0.28
        increment_interval = 0.14 if is_double_elixir else 0.28
        increments = int(elapsed / increment_interval)
        if increments > 0:
            self.elixir_opponent += 0.1 * increments
            self.elixir_opponent = min(self.elixir_opponent, 10)  # Max elixir is 10
            self.last_elixir_update += increment_interval * increments

    def get_state(self):
        """
        Return current game state as a dict.
        """
        return {
            "visible_cards": [d.card_name for d in self.visible_cards],
            "elixir_opponent": self.elixir_opponent,
            "last_played": self.last_played,
            "next_prediction": self.next_prediction,
            "deck": self.deck,
            "current_hand": self.deck[:4],  # Current hand is positions 0-3
            "next_in_queue": self.deck[4] if len(self.deck) > 4 else None
        }

    def reset(self):
        """
        Reset game state for a new match.
        """
        import time
        self.visible_cards = []
        self.elixir_opponent = 0
        self.last_played = None
        self.play_history = []
        self.next_prediction = []
        self.deck = [None] * 8
        self.match_start_time = time.time()  # Reset match timer
