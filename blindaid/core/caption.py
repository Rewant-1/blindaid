"""Caption and VQA helpers built on BLIP models."""
from __future__ import annotations

import logging
from typing import Any, Optional

import cv2

logger = logging.getLogger(__name__)


class VisualAssistant:
    def __init__(self, device: Optional[str] = None):
        self.device = device or "cpu"
        self.processor = None
        self.model = None
        self.vqa_processor = None
        self.vqa_model = None
        self._torch: Optional[Any] = None
        self._env_configured = False

    def _configure_transformers(self) -> None:
        if self._env_configured:
            return
        import os
        import warnings

        os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
        warnings.filterwarnings("ignore", category=FutureWarning)
        self._env_configured = True

    def _ensure_torch(self) -> None:
        if self._torch is not None:
            return
        try:
            import torch

            self._torch = torch
        except ImportError as exc:
            raise RuntimeError("Captioning/VQA requires the optional transformers stack.") from exc

    def _select_device(self) -> None:
        assert self._torch is not None
        if self.device == "cpu" and self._torch.cuda.is_available():
            self.device = "cuda"

    def _ensure_caption_model(self) -> None:
        if self.processor is not None and self.model is not None:
            return
        self._ensure_torch()
        self._configure_transformers()
        self._select_device()

        logger.info("Loading BLIP caption model on %s", self.device)
        from transformers import BlipForConditionalGeneration, BlipProcessor

        self.processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
        self.model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to(self.device)
        self.model.eval()

    def _ensure_vqa_model(self) -> None:
        if self.vqa_processor is not None and self.vqa_model is not None:
            return
        self._ensure_torch()
        self._configure_transformers()
        self._select_device()

        logger.info("Loading BLIP VQA model on %s", self.device)
        from transformers import BlipForQuestionAnswering, BlipProcessor

        self.vqa_processor = BlipProcessor.from_pretrained("Salesforce/blip-vqa-base")
        self.vqa_model = BlipForQuestionAnswering.from_pretrained("Salesforce/blip-vqa-base").to(self.device)
        self.vqa_model.eval()

    def generate_caption(self, frame) -> str:
        self._ensure_caption_model()
        assert self.processor is not None and self.model is not None and self._torch is not None

        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        inputs = self.processor(images=image, return_tensors="pt").to(self.device)
        with self._torch.no_grad():
            output = self.model.generate(**inputs, max_length=60)
        return self.processor.decode(output[0], skip_special_tokens=True).strip()

    def answer_question(self, frame, question: str) -> str:
        self._ensure_vqa_model()
        assert self.vqa_processor is not None and self.vqa_model is not None and self._torch is not None

        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        inputs = self.vqa_processor(images=image, text=question, return_tensors="pt").to(self.device)
        with self._torch.no_grad():
            output = self.vqa_model.generate(**inputs)
        return self.vqa_processor.decode(output[0], skip_special_tokens=True).strip()
