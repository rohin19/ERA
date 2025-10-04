"""Entry point: capture -> infer -> gamestate -> overlay (skeleton).
Currently stubbed so different team members can fill parts independently.
"""
from runtime.config import ONNX_MODEL_PATH
from runtime.infer.onnx_engine import OnnxEngine
from runtime.gamestate.state import GameState


def main():
    # Initialize components
    engine = OnnxEngine(ONNX_MODEL_PATH)
    state = GameState()

    # Pseudocode loop placeholder
    # while True:
    #     frame = capture_frame()
    #     detections = engine.predict(frame)
    #     state.ingest_detections(detections)
    #     overlay.render(state.snapshot())
    print("Skeleton runtime main executed (no loop yet).")


if __name__ == "__main__":
    main()
