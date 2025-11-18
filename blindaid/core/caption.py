"""Caption generation (image-to-text) utilities."""
from __future__ import annotations

import logging
from typing import Optional, Any

import cv2

logger = logging.getLogger(__name__)


class CaptionGenerator:
    """Lazy BLIP caption generator that only loads heavy deps when requested."""

    def __init__(self, device: Optional[str] = None):
        self.device = device or "cpu"
        self.processor = None
        self.model = None
        self._torch: Optional[Any] = None

    # ------------------------------------------------------------------
    def _ensure_loaded(self):
        if self.processor is not None and self.model is not None:
            return
        try:
            import torch  # pylint: disable=import-error
            from transformers import (  # pylint: disable=import-error
                BlipForConditionalGeneration,
                BlipProcessor,
            )
        except ImportError as exc:  # noqa: F401
            raise RuntimeError(
                "Captioning requires the optional 'advanced' dependencies."
                " Install them via 'pip install -r requirements.txt' (full extras) or"
                " 'pip install .[advanced]' before pressing 'C'."
            ) from exc

        if self.device == "cpu" and torch.cuda.is_available():
            self.device = "cuda"

        logger.info("Loading BLIP captioning model (%s)", self.device)
        processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
        model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
        model = model.to(self.device)
        model.eval()

        self.processor = processor
        self.model = model
        self._torch = torch  # cache for no_grad context
        logger.info("BLIP model ready")

    # ------------------------------------------------------------------
    def generate_caption(self, frame) -> str:
        self._ensure_loaded()
        assert self.processor is not None and self.model is not None  # For type checkers
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        inputs = self.processor(images=image, return_tensors="pt").to(self.device)
        with self._torch.no_grad():  # type: ignore[attr-defined]
            output = self.model.generate(**inputs, max_length=60)
        caption = self.processor.decode(output[0], skip_special_tokens=True)
        return caption.strip()
