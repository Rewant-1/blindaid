"""Depth estimation helpers using monocular models."""
from __future__ import annotations

import logging
from typing import List, Tuple, Any

import cv2
import numpy as np

from blindaid.modes.scene.scene_mode import Detection

logger = logging.getLogger(__name__)


class DepthAnalyzer:
    """Provides depth estimation for a frame and detections."""

    def __init__(self, device: str | None = None):
        self.device = device or "cpu"
        self.processor = None
        self.model = None
        self._torch: Any | None = None

    def _ensure_loaded(self):
        if self.processor is not None and self.model is not None:
            return
        try:
            import torch  # pylint: disable=import-error
            from transformers import (  # pylint: disable=import-error
                DPTForDepthEstimation,
                DPTImageProcessor,
            )
        except ImportError as exc:
            raise RuntimeError(
                "Depth analysis requires the optional 'advanced' dependencies."
                " Install them via 'pip install -r requirements.txt' (full extras) or"
                " 'pip install .[advanced]' before pressing 'D'."
            ) from exc

        if self.device == "cpu" and torch.cuda.is_available():
            self.device = "cuda"

        logger.info("Loading depth estimation model (%s)", self.device)
        processor = DPTImageProcessor.from_pretrained("Intel/dpt-hybrid-midas")
        model = DPTForDepthEstimation.from_pretrained("Intel/dpt-hybrid-midas")
        model = model.to(self.device)
        model.eval()

        self.processor = processor
        self.model = model
        self._torch = torch
        logger.info("Depth estimation model ready")

    # ------------------------------------------------------------------
    def compute_depth(self, frame: np.ndarray) -> np.ndarray:
        self._ensure_loaded()
        assert self.processor is not None and self.model is not None  # For type checkers
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        inputs = self.processor(images=rgb, return_tensors="pt").to(self.device)
        assert self._torch is not None
        with self._torch.no_grad():
            outputs = self.model(**inputs)
            pred = outputs.predicted_depth
        depth = pred.squeeze().cpu().numpy()
        depth = cv2.resize(depth, (frame.shape[1], frame.shape[0]))
        # Normalize for visualization
        depth = depth - depth.min()
        if depth.max() > 0:
            depth = depth / depth.max()
        return depth

    # ------------------------------------------------------------------
    def describe_detections(self, depth_map: np.ndarray, detections: List[Detection]) -> Tuple[str, List[str]]:
        if not detections:
            return "No detections available for depth analysis.", []

        h, w = depth_map.shape
        messages = []
        debug_lines = []
        for det in detections:
            x1, y1, x2, y2 = det.box
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w - 1, x2), min(h - 1, y2)
            if x2 <= x1 or y2 <= y1:
                continue
            region = depth_map[y1:y2, x1:x2]
            if region.size == 0:
                continue
            median_depth = float(np.median(region))
            distance_label, approx_meters = self._categorize_depth(median_depth)
            messages.append(f"{det.label} appears {distance_label}")
            debug_lines.append(
                f"{det.label}: depth={median_depth:.2f} -> {distance_label} (~{approx_meters:.1f}m)"
            )
        if not messages:
            return "Unable to estimate depth for the detected objects.", debug_lines
        summary = " ".join(messages)
        return summary, debug_lines

    def _categorize_depth(self, depth_value: float) -> Tuple[str, float]:
        # depth_value is normalized 0 (near) to 1 (far)
        inverted = 1.0 - depth_value
        approx_meters = max(0.3, inverted * 4.0)
        if depth_value < 0.3:
            return "very close", approx_meters
        if depth_value < 0.6:
            return "at medium distance", approx_meters
        return "far", approx_meters
