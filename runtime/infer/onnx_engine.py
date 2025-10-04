"""ONNX runtime inference engine skeleton."""
from __future__ import annotations
from pathlib import Path
from typing import List, Dict, Any
import numpy as np

try:
    import onnxruntime as ort  # type: ignore
except ImportError:  # pragma: no cover - optional at this stage
    ort = None  # placeholder

from runtime import config


class OnnxEngine:
    def __init__(self, model_path: Path):
        self.model_path = Path(model_path)
        if ort and self.model_path.exists():
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
            self.session = ort.InferenceSession(str(self.model_path), providers=providers)
            self.input_name = self.session.get_inputs()[0].name
            self.output_name = self.session.get_outputs()[0].name
        else:
            self.session = None
            self.input_name = 'input'
            self.output_name = 'output'

    def preprocess(self, frame: np.ndarray) -> np.ndarray:
        # Resize & normalize stub (return zeros with expected shape)
        sz = config.INPUT_SIZE
        return np.zeros((1, 3, sz, sz), dtype=np.float32)

    def postprocess(self, raw_output: np.ndarray) -> List[Dict[str, Any]]:
        # Stub: return empty detection list
        return []

    def predict(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        inp = self.preprocess(frame)
        if self.session:
            out = self.session.run([self.output_name], {self.input_name: inp})[0]
        else:
            out = np.zeros((1, 84, 8400), dtype=np.float32)  # Placeholder shape
        return self.postprocess(out)
