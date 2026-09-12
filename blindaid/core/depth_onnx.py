"""ONNX-based depth estimation using MiDaS-Small.

Drop-in replacement for depth.py (PyTorch DPT-Hybrid-MiDaS ~470MB)
with MiDaS-Small ONNX (~24MB).

Phase 1 of the research paper implementation.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# Default model path
DEFAULT_MODEL_PATH = (
    Path(__file__).resolve().parent.parent.parent / "resources" / "models" / "midas_small.onnx"
)

# MiDaS-Small preprocessing constants (ImageNet normalization)
MIDAS_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
MIDAS_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
MIDAS_INPUT_SIZE = 256


class DepthAnalyzerONNX:
    """Monocular depth estimation using MiDaS-Small ONNX.

    Same interface as DepthAnalyzer (compute_depth) so it can
    be swapped in without changing guardian_mode.py.

    Key differences from PyTorch version:
    - onnxruntime instead of torch + transformers
    - MiDaS-Small (24MB) instead of DPT-Hybrid (470MB)
    - CPU-only, no GPU required
    """

    def __init__(self, model_path: Optional[str] = None):
        self._model_path = Path(model_path) if model_path else DEFAULT_MODEL_PATH
        self._session = None
        self._input_name = None

    def _ensure_loaded(self) -> None:
        """Lazy-load the ONNX model on first use."""
        if self._session is not None:
            return

        import onnxruntime as ort

        if not self._model_path.exists():
            raise FileNotFoundError(
                f"MiDaS ONNX model not found at {self._model_path}.\n"
                f"Run: python scripts/download_onnx_models.py"
            )

        logger.info("Loading MiDaS-Small ONNX from %s", self._model_path)

        self._session = ort.InferenceSession(
            str(self._model_path),
            providers=["CPUExecutionProvider"],
        )
        self._input_name = self._session.get_inputs()[0].name

        logger.info("MiDaS-Small ONNX loaded successfully")

    def _preprocess(self, frame: np.ndarray) -> np.ndarray:
        """Preprocess BGR frame for MiDaS-Small input.

        Returns (1, 3, 256, 256) float32 tensor.
        """
        # BGR -> RGB
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Resize to model input size
        resized = cv2.resize(
            rgb, (MIDAS_INPUT_SIZE, MIDAS_INPUT_SIZE), interpolation=cv2.INTER_AREA
        )

        # Normalize: [0,255] -> [0,1] -> ImageNet normalization
        normalized = resized.astype(np.float32) / 255.0
        normalized = (normalized - MIDAS_MEAN) / MIDAS_STD

        # HWC -> CHW, add batch dimension
        tensor = normalized.transpose(2, 0, 1)  # (3, 256, 256)
        tensor = np.expand_dims(tensor, axis=0)  # (1, 3, 256, 256)

        return tensor

    def compute_depth(self, frame: np.ndarray) -> np.ndarray:
        """Estimate relative depth from a single frame.

        Matches the interface of DepthAnalyzer.compute_depth().

        Args:
            frame: BGR image from OpenCV (H, W, 3), uint8

        Returns:
            Depth map (H, W), float32, normalized to [0, 1].
            Higher values = closer to camera.
        """
        self._ensure_loaded()

        h, w = frame.shape[:2]

        # Preprocess
        input_tensor = self._preprocess(frame)

        # Run inference
        output = self._session.run(None, {self._input_name: input_tensor})
        depth = output[0]  # Shape varies: (1, 256, 256) or (256, 256)

        # Remove batch dimension if present
        if depth.ndim == 3:
            depth = depth[0]

        # Resize back to original frame size
        depth_resized = cv2.resize(depth, (w, h), interpolation=cv2.INTER_CUBIC)

        # Normalize to [0, 1] — same logic as the PyTorch version
        d_min = depth_resized.min()
        d_max = depth_resized.max()
        if d_max - d_min > 1e-6:
            depth_normalized = (depth_resized - d_min) / (d_max - d_min)
        else:
            depth_normalized = np.zeros_like(depth_resized)

        return depth_normalized.astype(np.float32)


__all__ = ["DepthAnalyzerONNX"]
