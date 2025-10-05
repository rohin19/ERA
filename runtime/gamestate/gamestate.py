"""gamestate.py
Core opponent state tracking logic. Only this file is modified to avoid merge conflicts.

External model integration contract:
Call GameState.ingest_detections(predictions, frame_ts=None) each frame where
  pre        self._last_play_time[card_name] = now
        self._last_absence_times.pop(card_name, None)
        
        # Record elixir level at play time for pattern analysis
        if card_name not in self._elixir_at_play:
            self._elixir_at_play[card_name] = []
        self._elixir_at_play[card_name].append(self.elixir_opponent)
        
        # Track play sequences (keep last 10 plays)
        if len(self.play_history) >= 2:
            recent_sequence = self.play_history[-2:]
            if len(recent_sequence) == 2:
                self._play_sequences.append(recent_sequence)
                # Keep only recent sequences
                if len(self._play_sequences) > 50:
                    self._play_sequences = self._play_sequences[-50:]
        
        self._integrate_into_deck(card)
        # Update predictions after each play
        self._update_predictions()tions: list[dict] with keys (case-insensitive accepted for flexibility):
    - card OR class_name: str (normalized card name Title Case)
    - bbox OR bbox_xyxy: [x1, y1, x2, y2] in pixel coordinates
    - confidence: float (0..1)
    - (optional) frame: int | None (frame index)

Primary responsibilities:
 1. Maintain a rolling set of currently visible cards & last-seen times.
 2. Infer "play events" when a card newly appears (debounced so persistence
    across consecutive frames does not repeatedly count as a play).
 3. Simulate opponent elixir regeneration & spending when a card play is inferred.
 4. Reconstruct / maintain deck ordering & current hand (first 4 indices) in a
    best-effort way as cards are discovered.

Important assumptions / simplifications (MVP):
 - A card play is inferred when a detection for a card appears after it was not
   visible for at least PLAY_COOLDOWN_SEC seconds (prevents multi-frame duplicates).
 - If card unknown in deck we append it (deck discovery phase).
 - Hand cycling: when a card in the first 4 (hand) is played, it is moved to the
   back after queue shift emulating Clash Royale cycle (simplified when unknowns exist).
 - We do not attempt to handle mirrored / duplicate card variants.

Extend / integrate: The model provider can simply format outputs to the contract
above; no other code change needed.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import List, Dict, Optional, Any
import time
import math

############################
# Configuration Constants  #
############################

PLAY_COOLDOWN_SEC = 1.0          # Minimum time a card must be absent before a new detection counts as a play
VISIBILITY_TTL_SEC = 2.0         # Remove a card from visible set if not re-seen for this long
MIN_CONFIDENCE = 0.30            # Ignore detections below this confidence
ELIXIR_INCREMENT_INTERVAL = 0.28 # Normal elixir tick (adds 0.1)
ELIXIR_DOUBLE_TIME = 120.0       # Seconds after start when double elixir begins
MAX_ELIXIR = 10.0
FRAME_TIME_FALLBACK = 1/15       # Used if frame timestamp not provided to approximate timing

# Basic elixir cost map (extend as needed)
CARD_ELIXIR_COST: Dict[str, int] = {
    "Giant": 5, "Knight": 3, "Bomber": 3, 
    "Hog Rider": 4, "Dart Goblin": 3, "Mini Pekka": 4,
    "Baby Dragon": 4, "Valkyrie": 4
}

# Card synergy and counter patterns for prediction
CARD_SYNERGIES = {
    "Giant": ["Dart Goblin", "Baby Dragon", "Valkyrie"],
    "Hog Rider": ["Knight", "Bomber"],
    "Knight": ["Hog Rider", "Giant"],
    "Valkyrie": ["Giant", "Baby Dragon"],
    "Baby Dragon": ["Giant", "Valkyrie"],
    "Bomber": ["Hog Rider", "Knight"],
    "Dart Goblin": ["Giant", "Mini Pekka"],
    "Mini Pekka": ["Dart Goblin"]
}


@dataclass
class CardDetection:
    card_name: str
    bbox: List[float]  # [x1, y1, x2, y2]
    confidence: float
    frame: Optional[int] = None
    first_seen: float = 0.0
    last_seen: float = 0.0


class GameState:
    def __init__(self):
        # Visibility tracking
        self.visible_cards: Dict[str, CardDetection] = {}
        # Deck & play tracking
        self.deck: List[Optional[str]] = [None] * 8   # indices 0-3 hand, 4-7 queue
        self.play_history: List[str] = []
        self.last_played: Optional[str] = None
        # Elixir
        self.elixir_opponent: float = MAX_ELIXIR
        self.last_elixir_update: float = time.time()
        self.match_start_time: float = self.last_elixir_update
        # Predictions & helpers
        self.next_prediction: List[str] = []  # AI-powered card predictions
        self.prediction_confidence: float = 0.0  # Confidence in next prediction
        self._last_absence_times: Dict[str, float] = {}  # last time we stopped seeing a card
        self._last_play_time: Dict[str, float] = {}      # last time a card was considered played
        # Pattern tracking for prediction engine
        self._play_sequences: List[List[str]] = []  # Track card play sequences
        self._elixir_at_play: Dict[str, List[float]] = {}  # Elixir levels when cards were played

    # ------------------------ Public API ------------------------ #
    def ingest_detections(self, detections: List[Dict[str, Any]], frame_ts: Optional[float] = None):
        """Main entry point called each frame with raw model detections.

        detections: list of dicts with keys (loose schema):
           card|class_name : str
           bbox|bbox_xyxy  : list[float] length 4 (x1,y1,x2,y2)
           confidence      : float
           frame           : (optional) int
        frame_ts: optional explicit timestamp (seconds). If None, uses time.time().
        """
        now = frame_ts if frame_ts is not None else time.time()
        self._update_elixir(now)

        # Normalize & filter
        norm = self._normalize_detections(detections)
        # Deduplicate per card keep highest confidence
        reduced = self._dedupe_keep_best(norm)
        # Update visibility map
        self._update_visibility(reduced, now)
        # Determine new play events
        plays = self._infer_plays(reduced, now)
        # Apply deck & elixir effects of plays
        for card in plays:
            self._apply_play(card, now)
        # Housekeeping remove stale visibility
        self._prune_visibility(now)

    def get_state(self) -> Dict[str, Any]:
        """Return current state summary (serializable)."""
        visible_export = [
            {
                "card": cd.card_name,
                "bbox": cd.bbox,
                "confidence": round(cd.confidence, 4),
                "last_seen": cd.last_seen,
            } for cd in self.visible_cards.values()
        ]
        return {
            "visible_cards": visible_export,
            "elixir_opponent": round(self.elixir_opponent, 2),
            "last_played": self.last_played,
            "play_history": self.play_history[-20:],  # recent window
            "deck": self.deck,
            "current_hand": self.deck[:4],
            "queue": self.deck[4:],
            "next_prediction": self.next_prediction,
            "prediction_confidence": round(self.prediction_confidence, 3),
            "match_time": round(time.time() - self.match_start_time, 2)
        }

    def reset(self):
        self.__init__()

    # --------------------- Internal Helpers --------------------- #
    def _normalize_detections(self, raw: List[Dict[str, Any]]) -> List[CardDetection]:
        out: List[CardDetection] = []
        for d in raw:
            if not isinstance(d, dict):
                continue
            name = d.get("card") or d.get("class_name") or d.get("name")
            if not name:
                continue
            bbox = d.get("bbox") or d.get("bbox_xyxy")
            if not bbox or len(bbox) != 4:
                continue
            conf = float(d.get("confidence", 0.0))
            if conf < MIN_CONFIDENCE:
                continue
            name_norm = self._normalize_card_name(name)
            out.append(CardDetection(
                card_name=name_norm,
                bbox=[float(x) for x in bbox],
                confidence=conf,
                frame=d.get("frame"),
                first_seen=0.0,
                last_seen=0.0,
            ))
        return out

    def _dedupe_keep_best(self, dets: List[CardDetection]) -> List[CardDetection]:
        best: Dict[str, CardDetection] = {}
        for d in dets:
            prev = best.get(d.card_name)
            if prev is None or d.confidence > prev.confidence:
                best[d.card_name] = d
        return list(best.values())

    def _update_visibility(self, dets: List[CardDetection], now: float):
        for d in dets:
            existing = self.visible_cards.get(d.card_name)
            if existing:
                existing.bbox = d.bbox
                existing.confidence = d.confidence
                existing.last_seen = now
            else:
                d.first_seen = now
                d.last_seen = now
                self.visible_cards[d.card_name] = d

    def _infer_plays(self, dets: List[CardDetection], now: float) -> List[str]:
        """
        Infer card plays from detections.
        
        KEY INSIGHT: When we DETECT a card on screen, it means the opponent JUST PLAYED IT.
        The card appears on the battlefield, which triggers our detection.
        
        We use cooldowns to prevent counting the same play multiple times across frames.
        """
        plays: List[str] = []
        current_names = {d.card_name for d in dets}
        
        # Mark absence times for cards that disappeared
        for name, cd in list(self.visible_cards.items()):
            if name not in current_names and (now - cd.last_seen) > 0.0:
                self._last_absence_times[name] = now
        
        # A play is detected when:
        # 1. We see a card on screen (detection)
        # 2. AND enough time has passed since the last play (cooldown)
        # 3. OR it's the first time we've ever seen this card
        
        for d in dets:
            last_play_t = self._last_play_time.get(d.card_name, 0.0)
            absence_t = self._last_absence_times.get(d.card_name, None)
            
            # First time seeing this card ever - definitely a play (deck discovery)
            if last_play_t == 0.0 and d.card_name not in self.play_history:
                plays.append(d.card_name)
                continue
            
            # Card reappeared after being absent - check cooldown
            if absence_t is not None:
                time_absent = now - absence_t
                time_since_last_play = now - last_play_t
                
                # Only count as new play if cooldown has passed
                if time_since_last_play >= PLAY_COOLDOWN_SEC:
                    plays.append(d.card_name)
                    continue
        
        return plays

    def _apply_play(self, card_name: str, now: float):
            cost = CARD_ELIXIR_COST.get(card_name)
            if cost is not None and self.elixir_opponent >= cost:
                self.elixir_opponent = max(0.0, self.elixir_opponent - cost)
            self.last_played = card_name
            self.play_history.append(card_name)
            self._last_play_time[card_name] = now
            self._last_absence_times.pop(card_name, None)
            
            # Record elixir level at play time for pattern analysis
            if card_name not in self._elixir_at_play:
                self._elixir_at_play[card_name] = []
            self._elixir_at_play[card_name].append(self.elixir_opponent)
            
            # Track play sequences (keep last 10 plays)
            if len(self.play_history) >= 2:
                recent_sequence = self.play_history[-2:]
                if len(recent_sequence) == 2:
                    self._play_sequences.append(recent_sequence)
                    # Keep only recent sequences
                    if len(self._play_sequences) > 50:
                        self._play_sequences = self._play_sequences[-50:]
            
            self._integrate_into_deck(card_name)
            
            # UPDATE PREDICTIONS AFTER EACH PLAY
            self._update_predictions()

    def _integrate_into_deck(self, card: str):
        """
        Proper Clash Royale cycle logic with early-game handling:
        - When a card is played (detected), it goes to the BACK of the queue (position 7)
        - EARLY GAME (< 4 cards): New discoveries go to positions 4-7, hand stays unknown
        - AFTER 4 CARDS: We can start building the hand from discovered cards
        - Hand is positions 0-3, Queue is positions 4-7
        """
        
        # Count how many unique cards we've discovered
        cards_discovered = len([c for c in self.deck if c is not None])
        
        # CASE 1: Card is already in the hand (positions 0-3) - This is a PLAY
        if card in self.deck[:4]:
            hand_index = self.deck.index(card)
            played_card = self.deck[hand_index]
            
            if self.deck[4] is not None:
                # Pull next card from queue into the hand slot
                self.deck[hand_index] = self.deck[4]
                
                # Shift queue forward
                for i in range(4, 7):
                    self.deck[i] = self.deck[i + 1]
                
                # Played card goes to back
                self.deck[7] = played_card
            else:
                # Queue not fully known yet - leave gap and put played card at end
                self.deck[hand_index] = None
                for i in range(4, 7):
                    self.deck[i] = self.deck[i + 1]
                self.deck[7] = played_card
            
            self._compact_deck()
            return
        
        # CASE 2: Card is in queue (positions 4-7) - This is DISCOVERY + IMMEDIATE PLAY
        if card in self.deck[4:]:
            queue_index = self.deck.index(card)
            
            # Remove it from queue and put at back (it was just played)
            played_card = self.deck[queue_index]
            
            # Shift everything after it forward
            for i in range(queue_index, 7):
                self.deck[i] = self.deck[i + 1]
            
            # Put played card at back
            self.deck[7] = played_card
            
            self._compact_deck()
            return
        
        # CASE 3: Brand new card discovery - IT WAS JUST PLAYED!
        # Put new discoveries in queue (positions 4-7)
        # Once we have 4+ cards, _compact_deck will move them to hand
        
        placed = False
        
        # Try to place in queue (positions 4-7)
        for i in range(4, 8):
            if self.deck[i] is None:
                self.deck[i] = card
                placed = True
                break
        
        # If queue is full, shift and place at position 7
        if not placed:
            for i in range(4, 7):
                self.deck[i] = self.deck[i + 1]
            self.deck[7] = card
        
        self._compact_deck()
    
    def _compact_deck(self):
        """
        Keep order but fill hand from queue when appropriate.
        IMPORTANT: Only start filling hand once we have at least 4 cards discovered.
        """
        hand = self.deck[:4]
        queue = self.deck[4:]
        
        # Count total discovered cards
        total_cards = len([c for c in self.deck if c is not None])
        
        # Only fill hand if we have at least 4 cards discovered
        # This means the first 4 plays have cycled through
        if total_cards >= 4:
            # Now we can build the hand from queue
            for i in range(4):
                if hand[i] is None:
                    # Find first non-None in queue
                    for j, qv in enumerate(queue):
                        if qv is not None:
                            hand[i] = qv
                            queue[j] = None
                            break
        
        self.deck = hand + queue

    def _prune_visibility(self, now: float):
        for name in list(self.visible_cards.keys()):
            cd = self.visible_cards[name]
            if (now - cd.last_seen) > VISIBILITY_TTL_SEC:
                self.visible_cards.pop(name, None)
                self._last_absence_times[name] = now

    def _update_elixir(self, now: float):
        elapsed = now - self.last_elixir_update
        if elapsed <= 0:
            return
        match_elapsed = now - self.match_start_time
        double_time = match_elapsed >= ELIXIR_DOUBLE_TIME
        interval = ELIXIR_INCREMENT_INTERVAL / (2 if double_time else 1)
        ticks = int(elapsed / interval)
        if ticks > 0:
            self.elixir_opponent = min(MAX_ELIXIR, self.elixir_opponent + 0.1 * ticks)
            self.last_elixir_update += ticks * interval

    def _normalize_card_name(self, raw: str) -> str:
        # Basic normalization (Title Case & strip)
        return raw.strip().title()

    def _update_predictions(self):
        """Advanced AI prediction engine based on multiple factors."""
        if len(self.play_history) < 2:
            return
        
        # Initialize prediction scores
        card_scores: Dict[str, float] = {}
        
        # Factor 1: Cycle prediction (highest weight)
        cycle_predictions = self._predict_from_cycle()
        for card, score in cycle_predictions.items():
            card_scores[card] = card_scores.get(card, 0) + score * 0.4
        
        # Factor 2: Elixir-based prediction
        elixir_predictions = self._predict_from_elixir()
        for card, score in elixir_predictions.items():
            card_scores[card] = card_scores.get(card, 0) + score * 0.25
        
        # Factor 3: Sequence pattern prediction
        sequence_predictions = self._predict_from_sequences()
        for card, score in sequence_predictions.items():
            card_scores[card] = card_scores.get(card, 0) + score * 0.2
        
        # Factor 4: Synergy prediction
        synergy_predictions = self._predict_from_synergies()
        for card, score in synergy_predictions.items():
            card_scores[card] = card_scores.get(card, 0) + score * 0.15
        
        # Sort by score and update predictions
        if card_scores:
            sorted_predictions = sorted(card_scores.items(), key=lambda x: x[1], reverse=True)
            self.next_prediction = [card for card, _ in sorted_predictions[:3]]  # Top 3 predictions
            self.prediction_confidence = min(1.0, sorted_predictions[0][1]) if sorted_predictions else 0.0
        else:
            self.next_prediction = []
            self.prediction_confidence = 0.0

    def _predict_from_cycle(self) -> Dict[str, float]:
        """Predict based on card cycle position."""
        scores: Dict[str, float] = {}
        
        # If we know the current hand, predict next card in queue
        current_hand = self.deck[:4]
        queue = self.deck[4:]
        
        # Cards in queue are likely to be played next
        for i, card in enumerate(queue):
            if card is not None:
                # Higher score for cards closer to the front of queue
                position_score = (4 - i) / 4.0
                scores[card] = position_score
        
        return scores

    def _predict_from_elixir(self) -> Dict[str, float]:
        """Predict based on current elixir and card costs."""
        scores: Dict[str, float] = {}
        
        for card in self.deck:
            if card is None:
                continue
                
            cost = CARD_ELIXIR_COST.get(card, 4)
            
            # Higher score if opponent has enough elixir
            if self.elixir_opponent >= cost:
                elixir_efficiency = min(1.0, self.elixir_opponent / cost)
                scores[card] = elixir_efficiency * 0.8
            else:
                # Lower score if not enough elixir
                scores[card] = 0.2
        
        return scores

    def _predict_from_sequences(self) -> Dict[str, float]:
        """Predict based on historical play sequences."""
        scores: Dict[str, float] = {}
        
        if len(self.play_history) < 1:
            return scores
            
        last_played = self.play_history[-1]
        
        # Count how often each card follows the last played card
        follow_counts: Dict[str, int] = {}
        total_follows = 0
        
        for sequence in self._play_sequences:
            if len(sequence) >= 2 and sequence[0] == last_played:
                follow_card = sequence[1]
                follow_counts[follow_card] = follow_counts.get(follow_card, 0) + 1
                total_follows += 1
        
        # Convert counts to probabilities
        if total_follows > 0:
            for card, count in follow_counts.items():
                probability = count / total_follows
                scores[card] = probability
        
        return scores

    def _predict_from_synergies(self) -> Dict[str, float]:
        """Predict based on card synergies and combinations."""
        scores: Dict[str, float] = {}
        
        if len(self.play_history) < 1:
            return scores
            
        last_played = self.play_history[-1]
        
        # Look for cards that synergize with recently played cards
        synergistic_cards = CARD_SYNERGIES.get(last_played, [])
        
        for card in synergistic_cards:
            if card in self.deck:
                # Base synergy score
                synergy_score = 0.6
                
                # Boost if enough elixir for combo
                cost = CARD_ELIXIR_COST.get(card, 4)
                if self.elixir_opponent >= cost:
                    synergy_score *= 1.5
                
                scores[card] = min(1.0, synergy_score)
        
        return scores


# -------------------------- Example (Manual Test) -------------------------- #
if __name__ == "__main__":  # simple manual sanity test
    gs = GameState()
    # Simulate first frame
    gs.ingest_detections([
        {"card": "Giant", "bbox": [10,10,50,90], "confidence": 0.9},
        {"card": "Goblin", "bbox": [60,15,90,85], "confidence": 0.82},
    ])
    print("After frame 1", gs.get_state())
    time.sleep(0.5)
    # Same cards again (should NOT double-count plays due to cooldown)
    gs.ingest_detections([
        {"card": "Giant", "bbox": [12,12,52,92], "confidence": 0.88},
    ])
    print("After frame 2", gs.get_state())
    # Simulate disappearance & reappearance
    time.sleep(1.2)
    gs.ingest_detections([
        {"card": "Giant", "bbox": [14,14,54,94], "confidence": 0.9},
    ])
    print("After frame 3 (giant replay)", gs.get_state())