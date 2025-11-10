"""Caption generation (image-to-text) utilities."""
from __future__ import annotations

import logging
from typing import Optional

import cv2
import torch
from transformers import BlipForConditionalGeneration, BlipProcessor

logger = logging.getLogger(__name__)


class CaptionGenerator:
    """Lazy BLIP caption generator."""

    def __init__(self, device: Optional[str] = None):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.processor: Optional[BlipProcessor] = None
        self.model: Optional[BlipForConditionalGeneration] = None

    # ------------------------------------------------------------------
    def _ensure_loaded(self):
        if self.processor is not None and self.model is not None:
            return
        logger.info("Loading BLIP captioning model (%s)", self.device)
        self.processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
        self.model = BlipForConditionalGeneration.from_pretrained(
            "Salesforce/blip-image-captioning-base"
        )
        self.model = self.model.to(self.device)
        self.model.eval()
        logger.info("BLIP model ready")

    # ------------------------------------------------------------------
    def generate_caption(self, frame) -> str:
        self._ensure_loaded()
        assert self.processor is not None and self.model is not None  # For type checkers
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        inputs = self.processor(images=image, return_tensors="pt").to(self.device)
        with torch.no_grad():
            output = self.model.generate(**inputs, max_length=60)
        caption = self.processor.decode(output[0], skip_special_tokens=True)
        return caption.strip()
