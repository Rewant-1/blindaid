"""Reading mode (OCR) for BlindAid."""
from __future__ import annotations

import time
import logging
from typing import List, Tuple

import cv2
import numpy as np
from paddleocr import PaddleOCR

from blindaid.core import config

logger = logging.getLogger(__name__)


class ReadingMode:
    """OCR reading mode suitable for shared camera controller."""

    def __init__(self, audio_enabled: bool = True, language: str = "en"):
        self.audio_enabled = audio_enabled
        self.language = language
        logger.info("Initializing PaddleOCR for reading mode")
        self.ocr = PaddleOCR(
            lang=language,
            use_angle_cls=True,
            text_det_limit_side_len=640,
        )
        logger.info("PaddleOCR ready")

        self.frame_count = 0
        self.skip = max(0, config.OCR_FRAME_SKIP)
        self.cooldown = config.OCR_COOLDOWN_SECONDS
        self.confidence_threshold = config.OCR_CONFIDENCE_THRESHOLD
        self.last_spoken = 0.0
        self.last_text = ""
        self.last_text_data: List[Tuple[str, float, np.ndarray]] = []
        self.info_lines: List[str] = []

    # ------------------------------------------------------------------
    def _parse_result(self, result, frame_shape) -> List[Tuple[str, float, np.ndarray]]:
        parsed: List[Tuple[str, float, np.ndarray]] = []
        if not result:
            return parsed

        first = result[0]
        if not first:
            return parsed

        # Support both OCRResult object and legacy list outputs
        if hasattr(first, "get"):
            texts = first.get("rec_texts", [])
            scores = first.get("rec_scores", [])
            polys = first.get("rec_polys", first.get("dt_polys", []))
            for idx, text in enumerate(texts):
                if not text:
                    continue
                score = scores[idx] if idx < len(scores) else 1.0
                poly = polys[idx] if idx < len(polys) else None
                if poly is None:
                    continue
                box = np.array(poly, dtype=np.int32)
                parsed.append((str(text), float(score), box))
        else:
            for line in first:
                if not isinstance(line, (list, tuple)) or len(line) < 2:
                    continue
                box, text_info = line[0], line[1]
                try:
                    box_array = np.array(box, dtype=np.int32)
                except Exception:  # noqa: BLE001 - skip malformed boxes
                    continue
                if isinstance(text_info, (list, tuple)):
                    text = text_info[0]
                    score = float(text_info[1]) if len(text_info) > 1 else 1.0
                else:
                    text = str(text_info)
                    score = 1.0
                parsed.append((text, score, box_array))
        return parsed

    # ------------------------------------------------------------------
    def process_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, List[str], List[str]]:
        """Process frame, returning annotated frame, info lines, speech messages."""
        display = frame.copy()
        info_lines: List[str] = []
        speech: List[str] = []

        # Draw previous boxes by default
        for text, score, box in self.last_text_data:
            color = (0, 255, 0) if score >= self.confidence_threshold else (0, 255, 255)
            cv2.polylines(display, [box], True, color, 2)
            x, y = box[0]
            cv2.putText(
                display,
                f"{text} ({score:.2f})",
                (int(x), int(y) - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 0, 0),
                2,
            )

        self.frame_count += 1
        should_run = (self.frame_count % (self.skip + 1)) == 0
        if should_run:
            result = self.ocr.ocr(display)
            parsed = self._parse_result(result, display.shape)
            if parsed:
                self.last_text_data = parsed
            else:
                self.last_text_data = []

        if self.last_text_data:
            texts = [text for text, score, _ in self.last_text_data if text]
            info_text = " ".join(texts)
            if info_text:
                info_lines.append(info_text)
                now = time.time()
                high_conf_texts = [text for text, score, _ in self.last_text_data if score >= self.confidence_threshold]
                if (
                    self.audio_enabled
                    and high_conf_texts
                    and info_text != self.last_text
                    and (now - self.last_spoken) > self.cooldown
                ):
                    speech_text = " ".join(high_conf_texts)
                    speech.append(speech_text)
                    self.last_text = info_text
                    self.last_spoken = now
        else:
            info_lines.append("No text detected - show printed text to the camera")

        return display, info_lines, speech

    # ------------------------------------------------------------------
    def on_enter(self):
        self.frame_count = 0
        self.last_text = ""
        self.last_text_data = []
        self.info_lines = []

    def on_exit(self):
        return
