"""GameState for tracking opponent's deck, cards, and elixir in Clash Royale."""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
import time
from collections import deque

# Constants for elixir mechanics
ELIXIR_GENERATION_RATE = 2.8  # seconds per elixir
MAX_ELIXIR = 10.0
START_ELIXIR = 5.0

@dataclass
class Card:
    name: str
    elixir_cost: float

# Dictionary of available cards
AVAILABLE_CARDS = {
    'Baby Dragon': Card('Baby Dragon', 4.0),
    'Bomber': Card('Bomber', 2.0),
    'Dart Goblin': Card('Dart Goblin', 3.0),
    'Giant': Card('Giant', 5.0),
    'Hog Rider': Card('Hog Rider', 4.0),
    'Knight': Card('Knight', 3.0),
    'Mini Pekka': Card('Mini Pekka', 4.0),
    'Valkyrie': Card('Valkyrie', 4.0),
}

@dataclass
class PlayEvent:
    t: float
    card: str

@dataclass
class GameState:
    plays: List[PlayEvent] = field(default_factory=list)
    last_detections: List[Dict[str, Any]] = field(default_factory=list)
    elixir: float = START_ELIXIR
    deck: deque = field(default_factory=lambda: deque(maxlen=8))
    game_started: bool = False
    last_elixir_update: float = field(default_factory=time.time)

    def start_game(self):
        """Initialize or reset the game state"""
        self.game_started = True
        self.elixir = START_ELIXIR
        self.last_elixir_update = time.time()
        self.deck.clear()
        self.plays.clear()

    def stop_game(self):
        """Stop the game state tracking"""
        self.game_started = False

    def update_elixir(self):
        """Update elixir based on time passed"""
        if not self.game_started:
            return

        current_time = time.time()
        elapsed = current_time - self.last_elixir_update
        
        # Calculate elixir gained since last update
        elixir_gained = (elapsed / ELIXIR_GENERATION_RATE)
        self.elixir = min(MAX_ELIXIR, self.elixir + elixir_gained)
        self.last_elixir_update = current_time

    def ingest_detections(self, detections: List[Dict[str, Any]]):
        """Store and process new card detections"""
        self.last_detections = detections
        # Future: Add logic to filter/validate detections

    def record_play(self, card: str):
        """Record a card play and update game state"""
        if not self.game_started or card not in AVAILABLE_CARDS:
            return

        # Update elixir before recording play
        self.update_elixir()
        current_time = time.time()
        
        # Record the play
        self.plays.append(PlayEvent(current_time, card))
        if len(self.plays) > 100:
            self.plays = self.plays[-100:]
        
        # Add to deck rotation
        self.deck.append(card)
        
        # Subtract elixir cost
        card_cost = AVAILABLE_CARDS[card].elixir_cost
        self.elixir = max(0.0, self.elixir - card_cost)
        self.last_elixir_update = current_time

    def get_available_cards(self) -> List[str]:
        """Get the 4 cards currently available to opponent"""
        deck_list = list(self.deck)
        if len(deck_list) < 4:
            return deck_list
        return deck_list[-4:]

    def get_next_card(self) -> Optional[str]:
        """Get the next card in rotation (5th card)"""
        deck_list = list(self.deck)
        if len(deck_list) < 5:
            return None
        return deck_list[-5]

    def snapshot(self) -> Dict[str, Any]:
        """Get current game state"""
        self.update_elixir()
        return {
            'version': 1,
            'game_active': self.game_started,
            'elixir': round(self.elixir, 2),
            'last_detections': self.last_detections,
            'plays': [{'t': p.t, 'card': p.card} for p in self.plays],
            'available_cards': self.get_available_cards(),
            'next_card': self.get_next_card(),
            'total_cards_seen': len(self.deck)
        }
