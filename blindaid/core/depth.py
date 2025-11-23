"""Lightweight wrapper around the MiDaS depth model."""
from __future__ import annotations

import logging
from typing import Any

import cv2
import numpy as np

logger = logging.getLogger(__name__)


class DepthAnalyzer:
    def __init__(self, device: str | None = None):
        self.device = device or "cpu"
        self.processor = None
        self.model = None
        self._torch: Any | None = None
        self._env_ready = False

    def _prepare_env(self) -> None:
        if self._env_ready:
            return
        import os
        import warnings

        os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
        warnings.filterwarnings("ignore", category=FutureWarning)
        self._env_ready = True

    def _ensure_loaded(self) -> None:
        if self.processor is not None and self.model is not None:
            return
        self._prepare_env()
        try:
            import torch
            from transformers import DPTForDepthEstimation, DPTImageProcessor
        except ImportError as exc:
            raise RuntimeError(
                "Depth estimation requires the optional transformers dependencies."
            ) from exc

        if self.device == "cpu" and torch.cuda.is_available():
            self.device = "cuda"

        logger.info("Loading depth model on %s", self.device)
        processor = DPTImageProcessor.from_pretrained("Intel/dpt-hybrid-midas")
        model = DPTForDepthEstimation.from_pretrained("Intel/dpt-hybrid-midas").to(self.device)
        model.eval()

        self.processor = processor
        self.model = model
        self._torch = torch

    def compute_depth(self, frame: np.ndarray) -> np.ndarray:
        self._ensure_loaded()
        assert self.processor is not None and self.model is not None and self._torch is not None

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        inputs = self.processor(images=rgb, return_tensors="pt").to(self.device)
        with self._torch.no_grad():
            outputs = self.model(**inputs)
            pred = outputs.predicted_depth
        depth = pred.squeeze().cpu().numpy()
        depth = cv2.resize(depth, (frame.shape[1], frame.shape[0]))
        depth = depth - depth.min()
        if depth.max() > 0:
            depth = depth / depth.max()
        return depth
