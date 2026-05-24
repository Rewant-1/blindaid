"""ONNX-based OCR using RapidOCR.

Drop-in replacement for PaddleOCR. Uses rapidocr-onnxruntime which runs
PaddleOCR models via pure onnxruntime — no paddlepaddle dependency.

Phase 3 of the research paper implementation.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class OCRResult:
    """Single OCR text detection result."""
    text: str
    confidence: float
    bbox: list  # [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]


class OCREngineONNX:
    """OCR engine using RapidOCR (ONNX-based PaddleOCR models).

    Advantages over PaddleOCR:
    - No paddlepaddle dependency (large, complex install)
    - Pure onnxruntime inference
    - Lighter memory footprint
    - Same model accuracy (uses same underlying PP-OCR models)

    Usage:
        ocr = OCREngineONNX()
        results = ocr.read_text(frame)
        for r in results:
            print(f"'{r.text}' (conf: {r.confidence:.2f})")
    """

    def __init__(self, language: str = "en", conf_threshold: float = 0.5):
        self.language = language
        self.conf_threshold = conf_threshold
        self._engine = None

    def _ensure_loaded(self) -> None:
        """Lazy-load RapidOCR on first use."""
        if self._engine is not None:
            return

        logger.info("Loading RapidOCR ONNX engine...")

        from rapidocr_onnxruntime import RapidOCR

        self._engine = RapidOCR()

        logger.info("RapidOCR ONNX loaded successfully")

    def read_text(self, frame: np.ndarray) -> list[OCRResult]:
        """Run OCR on a frame.

        Args:
            frame: BGR image from OpenCV (H, W, 3), uint8

        Returns:
            List of OCRResult with text, confidence, and bounding box.
        """
        self._ensure_loaded()

        result, _ = self._engine(frame)

        if result is None:
            return []

        ocr_results = []
        for item in result:
            # RapidOCR returns: [bbox, text, confidence]
            bbox = item[0]
            text = str(item[1])
            confidence = float(item[2])

            if confidence >= self.conf_threshold and text.strip():
                ocr_results.append(OCRResult(
                    text=text.strip(),
                    confidence=float(confidence),
                    bbox=bbox,
                ))

        return ocr_results

    def read_text_simple(self, frame: np.ndarray) -> Optional[str]:
        """Read text and return as single concatenated string.

        Convenience method matching the pattern used in reading_mode.
        Returns None if no text detected.
        """
        results = self.read_text(frame)
        if not results:
            return None

        # Sort by vertical position (top to bottom), then left to right
        results.sort(key=lambda r: (r.bbox[0][1], r.bbox[0][0]))

        full_text = " ".join(r.text for r in results)
        return full_text if full_text.strip() else None


__all__ = ["OCREngineONNX", "OCRResult"]
