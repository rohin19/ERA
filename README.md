# Edge Royale Analytics (ERA)

Real-time Clash Royale opponent card tracking and prediction system using computer vision and machine learning.     

[Demo Video](https://www.youtube.com/watch?v=d3FYPzs-ZBQ)

## Overview
ERA uses a custom-trained YOLOv8 model to detect opponent card plays in real-time, track their deck composition, predict their next moves, and monitor elixir levels. The system runs entirely locally with no external API dependencies.

## Core Features

### Card Detection
- Real-time card detection using YOLOv8 computer vision model
- Configurable confidence thresholds for detection, display, and game state tracking
- Support for 8 card classes: Baby Dragon, Bomber, Dart Goblin, Giant, Hog Rider, Knight, Mini Pekka, Valkyrie
- Automatic play detection with cooldown-based duplicate prevention

### Deck Tracking
- Automatic deck composition discovery (8-card deck)
- Queue-based cycle tracking matching Clash Royale's 4-card hand + 4-card queue system
- Visual display of current hand (positions 0-3) and upcoming cards (positions 4-7)
- Progressive discovery mode showing deck learning progress

### Prediction Engine
The AI prediction system uses weighted analysis across four factors:

1. **Cycle Prediction (40% weight)**: Cards in queue positions 4-7 are scored based on proximity to entering the hand
2. **Elixir Analysis (25% weight)**: Filters predictions based on current elixir availability
3. **Sequence Patterns (20% weight)**: Learns common card play sequences from match history
4. **Card Synergies (15% weight)**: Predefined synergy bonuses for commonly paired cards

Predictions look 2-4 plays ahead, showing cards likely to be played soon rather than just immediate options.

### Elixir Tracking
- Automatic elixir level tracking starting at 5.0
- Dynamic elixir generation rate (1 per 2.8 seconds standard, 2 per 2.8 seconds during double elixir)
- Automatic elixir deduction when cards are played
- Manual match start control via spacebar to prevent prematch tracking

## Technical Architecture

### Project Structure
```
ERA/
├── clash_royale_simple.py      # Main application entry point
├── runtime/
│   └── gamestate/
│       └── gamestate.py        # Core state tracking and prediction logic
├── model/
│   ├── runs/weights/best.pt    # Trained YOLO model
│   └── cr_data.yaml            # Dataset configuration
├── data/
│   ├── dataset_yolov11/        # Training dataset
│   │   ├── train/
│   │   │   ├── images/
│   │   │   └── labels/
│   │   └── valid/
│   │       ├── images/
│   │       └── labels/
│   └── classes.txt             # Card class definitions
├── assets/                     # Card image assets for overlay
└── capture_config.json         # Screen capture region configuration
```

### GameState Logic

**Deck Positions**:
- Positions 0-3: Current hand (cards available to play now)
- Positions 4-7: Queue (cards waiting to cycle into hand)

**Card Play Flow**:
1. Card detected on screen → Play event triggered
2. Played card moves to position 7 (back of queue)
3. Card at position 4 moves into empty hand slot
4. All queue cards shift forward one position

**Discovery Phase**:
- First 4 cards detected populate queue (positions 4-7)
- Hand remains unknown until 4+ cards discovered
- After 4 cards, system begins populating hand from queue
- Full deck tracking active after 8 unique cards detected

### Detection Pipeline

1. **Screen Capture**: MSS library captures configured screen region
2. **YOLO Inference**: Local YOLOv8 model processes frame
3. **Confidence Filtering**: Three-tier filtering system
   - DETECTION_CONFIDENCE (0.4): Minimum for YOLO inference
   - DISPLAY_CONFIDENCE (0.4): Minimum for on-screen display
   - GAMESTATE_CONFIDENCE (0.6): Minimum for state tracking
4. **Play Inference**: Cooldown-based duplicate detection (3 seconds)
5. **State Update**: GameState processes plays and updates predictions

## Installation

### Requirements
- Python 3.8+
- CUDA-capable GPU (recommended for real-time performance)
- Windows OS (MSS screen capture)

### Dependencies
```bash
pip install -r requirements.txt
```

### Setup

1. Clone repository
2. Install dependencies
3. Configure capture region:
   ```bash
   python era_launcher.py
   # Select option 1 to configure capture area
   ```
4. Ensure model exists at `model/runs/weights/best.pt`
5. Verify card assets in `assets/` directory

## Usage

### Basic Operation
```bash
python clash_royale_simple.py
```

### Controls
- **SPACEBAR**: Start match tracking (begins elixir monitoring)
- **R**: Reset game state (prepare for new match)
- **Q**: Quit application
- **S**: Save screenshot
- **D**: Toggle debug mode (verbose detection logging)
- **+/-**: Adjust detection confidence threshold

### Workflow
1. Launch application
2. Position overlay window to monitor game
3. Enter Clash Royale match
4. Press SPACEBAR when match starts
5. System automatically tracks cards, deck, and elixir
6. View predictions and deck state in real-time
7. Press R between matches to reset

## Display Interface

### Top Section
- FPS counter with performance color coding
- Detection statistics (total/shown/high-confidence)
- Bounding boxes with confidence scores on detected cards

### Bottom Panel
- **Elixir Bar**: Current opponent elixir level (0-10)
- **Current Hand**: 4 cards available to play (green border)
- **Next Card**: Card at position 4 ready to enter hand (yellow border)
- **AI Predictions**: Top 3 predicted upcoming plays with confidence
- **Full Deck**: Complete 8-card deck with color-coded positions
- **Match Info**: Time elapsed, total plays, last card played
- **AI Status**: Prediction confidence level

### Color Coding
- **Green**: Current hand cards (positions 0-3)
- **Yellow**: Next card entering hand (position 4)
- **Cyan**: Queue cards (positions 5-7)
- **Purple**: High-confidence detections (≥0.6)
- **Gray**: Low-confidence detections (<0.6)

## Model Training

### Dataset Format
YOLO format with normalized bounding boxes:
```
class_id x_center y_center width height
```

All coordinates normalized to 0-1 range.

### Training Process
```bash
python scripts/train_yolo.py [epochs] [batch_size]
```

Default configuration:
- Base model: YOLOv8n
- Image size: 640x640
- Batch size: 16
- Epochs: 50
- Patience: 10

### Dataset Validation
```bash
python scripts/check_dataset.py
```

Validates:
- Image and label file presence
- Class ID ranges
- YAML configuration alignment

## Performance

### Typical Metrics
- FPS: 20-30 (with GPU)
- Detection latency: <50ms
- Prediction update: Every 5 frames
- Memory usage: ~2GB (model + application)

### Optimization Tips
- Use CUDA-enabled GPU
- Reduce batch size if memory limited
- Adjust state_update_interval for slower systems
- Lower DETECTION_CONFIDENCE for more detections (higher false positives)
- Raise GAMESTATE_CONFIDENCE for cleaner tracking (fewer updates)

## Known Limitations

1. **Starting Hand Unknown**: Cannot determine initial hand order until cards are played
2. **Double Elixir Detection**: Manual mode switching not implemented
3. **Mirror Card**: Not currently supported in 8-card tracking system
4. **Occlusion Handling**: Cards may not detect if heavily obscured
5. **Multi-Monitor**: Capture region must be manually configured per setup

## Future Enhancements

- Automatic double/triple elixir detection
- Extended card library support
- Multi-opponent tracking for 2v2 modes
- Historical match analytics
- Pattern learning across multiple matches
- Automatic capture region detection

## Development

### Adding New Cards

1. Add class to `data/classes.txt`
2. Collect and label training images
3. Update `cr_data.yaml` with new class
4. Retrain model
5. Add card asset to `assets/` directory
6. Update `card_colors` and `card_files` in `clash_royale_simple.py`
7. Add synergies to `CARD_SYNERGIES` in `gamestate.py`

### Testing

```bash
# Test inference
python era_launcher.py
# Select option 2

# Validate dataset
python scripts/check_dataset.py

# Train with validation
bash model/train.sh
bash model/validate.sh
```

## Contributors

Jayden Troung | [kendymann](https://github.com/kendymann)   
Rohin Aulakh | [rohin19](https://github.com/rohin19)   
Hugo Najafi | [HugoNajafi](https://github.com/HugoNajafi)    
Michael Bazett | [Bazinator](https://github.com/Bazinator)   
