"""ONNX-based object detection using YOLOv8-Nano.

Pure onnxruntime inference — no ultralytics or torch at runtime.
Ultralytics is only needed once to export the ONNX model.

Phase 2 of the research paper implementation.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)

DEFAULT_MODEL_PATH = (
    Path(__file__).resolve().parent.parent.parent / "resources" / "models" / "yolov8n.onnx"
)

# COCO class names (80 classes)
COCO_CLASSES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck",
    "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench",
    "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra",
    "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove",
    "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
    "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
    "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
    "hair drier", "toothbrush",
]


@dataclass
class Detection:
    """Single object detection result."""
    class_id: int
    class_name: str
    confidence: float
    x1: int  # bbox top-left x
    y1: int  # bbox top-left y
    x2: int  # bbox bottom-right x
    y2: int  # bbox bottom-right y


class ObjectDetectorONNX:
    """Object detection using YOLOv8-Nano ONNX.

    Runs pure onnxruntime inference — no torch, no ultralytics at runtime.

    Usage:
        detector = ObjectDetectorONNX()
        detections = detector.detect(frame)
        for d in detections:
            print(f"{d.class_name} ({d.confidence:.2f}) at [{d.x1},{d.y1},{d.x2},{d.y2}]")
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.45,
        input_size: int = 640,
    ):
        self._model_path = Path(model_path) if model_path else DEFAULT_MODEL_PATH
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.input_size = input_size
        self._session = None
        self._input_name = None

    def _ensure_loaded(self) -> None:
        """Lazy-load the ONNX model."""
        if self._session is not None:
            return

        import onnxruntime as ort

        if not self._model_path.exists():
            raise FileNotFoundError(
                f"YOLOv8n ONNX model not found at {self._model_path}.\n"
                f"Run: python scripts/download_onnx_models.py"
            )

        logger.info("Loading YOLOv8n ONNX from %s", self._model_path)

        self._session = ort.InferenceSession(
            str(self._model_path),
            providers=["CPUExecutionProvider"],
        )
        self._input_name = self._session.get_inputs()[0].name

        logger.info("YOLOv8n ONNX loaded successfully")

    def _preprocess(self, frame: np.ndarray) -> tuple[np.ndarray, float, int, int]:
        """Letterbox resize + normalize for YOLOv8 input.

        Returns:
            tensor: (1, 3, 640, 640) float32
            ratio: scale ratio applied
            pad_w: horizontal padding
            pad_h: vertical padding
        """
        h, w = frame.shape[:2]
        target = self.input_size

        # Compute scale to fit within target while maintaining aspect ratio
        ratio = min(target / h, target / w)
        new_h, new_w = int(h * ratio), int(w * ratio)

        # Resize
        resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        # Pad to square (center pad)
        pad_h = (target - new_h) // 2
        pad_w = (target - new_w) // 2
        padded = np.full((target, target, 3), 114, dtype=np.uint8)
        padded[pad_h:pad_h + new_h, pad_w:pad_w + new_w] = resized

        # BGR -> RGB, normalize, HWC -> CHW, add batch
        rgb = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB)
        tensor = rgb.astype(np.float32) / 255.0
        tensor = tensor.transpose(2, 0, 1)  # (3, 640, 640)
        tensor = np.expand_dims(tensor, 0)  # (1, 3, 640, 640)

        return tensor, ratio, pad_w, pad_h

    def _postprocess(
        self, output: np.ndarray, ratio: float, pad_w: int, pad_h: int, orig_h: int, orig_w: int
    ) -> list[Detection]:
        """Process YOLOv8 ONNX output into detections.

        YOLOv8 ONNX output shape: (1, 84, 8400)
        - 84 = 4 (bbox: cx, cy, w, h) + 80 (class scores)
        - 8400 = number of anchor boxes
        """
        # output shape: (1, 84, 8400) -> transpose to (8400, 84)
        predictions = output[0].T  # (8400, 84)

        # Split into boxes and class scores
        boxes = predictions[:, :4]  # (8400, 4) - cx, cy, w, h
        scores = predictions[:, 4:]  # (8400, 80) - class scores

        # Get best class per box
        class_ids = np.argmax(scores, axis=1)
        confidences = scores[np.arange(len(scores)), class_ids]

        # Filter by confidence
        mask = confidences > self.conf_threshold
        boxes = boxes[mask]
        class_ids = class_ids[mask]
        confidences = confidences[mask]

        if len(boxes) == 0:
            return []

        # Convert cx,cy,w,h -> x1,y1,x2,y2
        x1 = boxes[:, 0] - boxes[:, 2] / 2
        y1 = boxes[:, 1] - boxes[:, 3] / 2
        x2 = boxes[:, 0] + boxes[:, 2] / 2
        y2 = boxes[:, 1] + boxes[:, 3] / 2

        # Remove padding and rescale to original image coordinates
        x1 = (x1 - pad_w) / ratio
        y1 = (y1 - pad_h) / ratio
        x2 = (x2 - pad_w) / ratio
        y2 = (y2 - pad_h) / ratio

        # Clip to image bounds
        x1 = np.clip(x1, 0, orig_w).astype(int)
        y1 = np.clip(y1, 0, orig_h).astype(int)
        x2 = np.clip(x2, 0, orig_w).astype(int)
        y2 = np.clip(y2, 0, orig_h).astype(int)

        # NMS (Non-Maximum Suppression)
        indices = cv2.dnn.NMSBoxes(
            bboxes=[(int(x), int(y), int(w - x), int(h - y))
                    for x, y, w, h in zip(x1, y1, x2, y2)],
            scores=confidences.tolist(),
            score_threshold=self.conf_threshold,
            nms_threshold=self.iou_threshold,
        )

        detections = []
        if len(indices) > 0:
            # cv2.dnn.NMSBoxes returns different formats depending on version
            if isinstance(indices, np.ndarray):
                indices = indices.flatten()
            else:
                indices = [i for i in indices]

            for i in indices:
                detections.append(Detection(
                    class_id=int(class_ids[i]),
                    class_name=COCO_CLASSES[int(class_ids[i])] if int(class_ids[i]) < len(COCO_CLASSES) else "unknown",
                    confidence=float(confidences[i]),
                    x1=int(x1[i]),
                    y1=int(y1[i]),
                    x2=int(x2[i]),
                    y2=int(y2[i]),
                ))

        return detections

    def detect(self, frame: np.ndarray) -> list[Detection]:
        """Run object detection on a frame.

        Args:
            frame: BGR image from OpenCV (H, W, 3), uint8

        Returns:
            List of Detection objects with class, confidence, and bbox.
        """
        self._ensure_loaded()

        orig_h, orig_w = frame.shape[:2]

        # Preprocess
        tensor, ratio, pad_w, pad_h = self._preprocess(frame)

        # Run inference
        output = self._session.run(None, {self._input_name: tensor})

        # Postprocess
        detections = self._postprocess(output[0], ratio, pad_w, pad_h, orig_h, orig_w)

        return detections


__all__ = ["ObjectDetectorONNX", "Detection", "COCO_CLASSES"]
