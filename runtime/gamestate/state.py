 """GameState skeleton for card tracking and elixir estimation."""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Dict, Any
import time

@dataclass
class PlayEvent:
    t: float
    card: str

@dataclass
class GameState:
    plays: List[PlayEvent] = field(default_factory=list)
    last_detections: List[Dict[str, Any]] = field(default_factory=list)
    elixir: float = 5.0  # starting midpoint guess

    def ingest_detections(self, detections: List[Dict[str, Any]]):
        # For now just store detections. Later: derive plays & update elixir.
        self.last_detections = detections

    def record_play(self, card: str):
        self.plays.append(PlayEvent(time.time(), card))
        if len(self.plays) > 100:
            self.plays = self.plays[-100:]

    def snapshot(self) -> Dict[str, Any]:
        return {
            'version': 1,
            'elixir': round(self.elixir, 2),
            'last_detections': self.last_detections,
            'plays': [ {'t': p.t, 'card': p.card} for p in self.plays ],
        }
