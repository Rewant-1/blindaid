"""OCR mode using RapidOCR (ONNX-based, no PaddleOCR dependency).

Research branch: Uses ONNX OCR + Adaptive Frame Processing.
"""
from __future__ import annotations

import logging
import time
from typing import List, Tuple

import cv2
import numpy as np

from blindaid.core import config
from blindaid.core.adaptive_processor import AdaptiveFrameProcessor
from blindaid.core.ocr_onnx import OCREngineONNX

logger = logging.getLogger(__name__)


class ReadingMode:
    def __init__(self, audio_enabled: bool = True, language: str = "en"):
        self.audio_enabled = audio_enabled
        self.language = language
        self.ocr = None
        self._ocr_failed = False
        self.afp = AdaptiveFrameProcessor()

        self.frame_count = 0
        self.cooldown = config.OCR_COOLDOWN_SECONDS
        self.confidence_threshold = config.OCR_CONFIDENCE_THRESHOLD
        self.last_spoken = 0.0
        self.last_text = ""
        self.stable_text_count = 0
        self._cached_display = None
        self._cached_info = []

    def _ensure_ocr(self):
        if self.ocr is not None or self._ocr_failed:
            return self.ocr

        try:
            logger.info("Loading RapidOCR ONNX (%s)", self.language)
            self.ocr = OCREngineONNX(language=self.language)
        except Exception as exc:
            self._ocr_failed = True
            logger.error("Failed to initialise RapidOCR: %s", exc)
        return self.ocr

    def _run_ocr(self, frame: np.ndarray):
        engine = self._ensure_ocr()
        if engine is None:
            return []
        return engine.read_text(frame)

    def process_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, List[str], List[str]]:
        display = frame.copy()
        info_lines: List[str] = ["Mode: Reading (ONNX)"]
        speech: List[str] = []

        self.frame_count += 1

        # AFP decides whether to run OCR on this frame
        should_run = self.afp.should_process("reading", frame)

        if should_run:
            results = self._run_ocr(frame)

            if results:
                texts = [r.text for r in results]
                info_text = " ".join(texts)

                if info_text:
                    info_lines.append(info_text)

                    now = time.time()
                    high_conf = [r.text for r in results if r.confidence >= self.confidence_threshold]

                    if info_text == self.last_text:
                        self.stable_text_count += 1
                    else:
                        self.stable_text_count = 0
                        self.last_text = info_text

                    if (
                        self.audio_enabled
                        and high_conf
                        and self.stable_text_count >= 2
                        and (now - self.last_spoken) > self.cooldown
                    ):
                        speech_text = " ".join(high_conf)
                        speech.append(speech_text)
                        self.last_spoken = now

                    # Draw bounding boxes
                    for r in results:
                        if r.bbox and len(r.bbox) >= 4:
                            pts = np.array(r.bbox, dtype=np.int32)
                            cv2.polylines(display, [pts], True, (0, 255, 0), 2)
                            cv2.putText(display, f"{r.text} ({r.confidence:.0%})",
                                        (int(pts[0][0]), int(pts[0][1]) - 5),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
            else:
                info_lines.append("No text detected")

            # AFP stats
            metrics = self.afp.get_metrics("reading")
            skip_pct = metrics.get("cpu_savings_pct", 0)
            h = frame.shape[0]
            cv2.putText(display, f"AFP: {skip_pct:.0f}% saved",
                        (10, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)

            self._cached_display = display
            self._cached_info = info_lines
        else:
            # Return cached during skip
            if self._cached_display is not None:
                return self._cached_display, self._cached_info, []
            info_lines.append("Warming up...")

        return display, info_lines, speech

    def on_enter(self):
        self.frame_count = 0
        self.last_text = ""
        self.stable_text_count = 0
        logger.info("Reading Mode Active (ONNX + AFP)")

    def on_exit(self):
        if self.afp:
            metrics = self.afp.get_metrics("reading")
            logger.info("Reading AFP metrics: %s", metrics)
