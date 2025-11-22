"""Caption generation (image-to-text) utilities."""
from __future__ import annotations

import logging
from typing import Optional, Any

import cv2

logger = logging.getLogger(__name__)


class VisualAssistant:
    """Lazy BLIP caption generator and VQA assistant."""

    def __init__(self, device: Optional[str] = None):
        self.device = device or "cpu"
        self.processor = None
        self.model = None
        self.vqa_processor = None
        self.vqa_model = None
        self._torch: Optional[Any] = None

    # ------------------------------------------------------------------
    def _ensure_caption_model(self):
        if self.processor is not None and self.model is not None:
            return
        self._check_deps()
        self._setup_device()

        logger.info("Loading BLIP captioning model (%s)", self.device)
        from transformers import BlipForConditionalGeneration, BlipProcessor
        self.processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
        self.model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to(self.device)
        self.model.eval()
        logger.info("BLIP caption model ready")

    def _ensure_vqa_model(self):
        if self.vqa_processor is not None and self.vqa_model is not None:
            return
        self._check_deps()
        self._setup_device()

        logger.info("Loading BLIP VQA model (%s)", self.device)
        from transformers import BlipForQuestionAnswering, BlipProcessor
        self.vqa_processor = BlipProcessor.from_pretrained("Salesforce/blip-vqa-base")
        self.vqa_model = BlipForQuestionAnswering.from_pretrained("Salesforce/blip-vqa-base").to(self.device)
        self.vqa_model.eval()
        logger.info("BLIP VQA model ready")

    def _check_deps(self):
        if self._torch is not None:
            return
        try:
            import torch
            self._torch = torch
        except ImportError as exc:
            raise RuntimeError(
                "Captioning/VQA requires 'advanced' dependencies."
            ) from exc

    def _setup_device(self):
        if self.device == "cpu" and self._torch.cuda.is_available():
            self.device = "cuda"

    # ------------------------------------------------------------------
    def generate_caption(self, frame) -> str:
        self._ensure_caption_model()
        assert self.processor is not None and self.model is not None
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        inputs = self.processor(images=image, return_tensors="pt").to(self.device)
        with self._torch.no_grad():  # type: ignore[attr-defined]
            output = self.model.generate(**inputs, max_length=60)
        caption = self.processor.decode(output[0], skip_special_tokens=True)
        return caption.strip()

    def answer_question(self, frame, question: str) -> str:
        """Answer a question about the image using VQA."""
        self._ensure_vqa_model()
        assert self.vqa_processor is not None and self.vqa_model is not None
        
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        inputs = self.vqa_processor(images=image, text=question, return_tensors="pt").to(self.device)
        
        with self._torch.no_grad():
            output = self.vqa_model.generate(**inputs)
            
        answer = self.vqa_processor.decode(output[0], skip_special_tokens=True)
        return answer.strip()
