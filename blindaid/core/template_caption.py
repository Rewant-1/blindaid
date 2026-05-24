"""Template-based scene captioning using YOLO detections.

Replaces BLIP (400MB, ~1600ms) with YOLO-Nano (12MB, ~33ms) + templates.
Generates natural language scene descriptions from object detections.

Phase 5 of the research paper implementation.
"""
from __future__ import annotations

import logging
from typing import Optional

import numpy as np

from blindaid.core.detector_onnx import ObjectDetectorONNX, Detection

logger = logging.getLogger(__name__)


# Spatial regions for template generation
REGION_NAMES = {
    "left": "on the left",
    "center": "in the center",
    "right": "on the right",
}

DISTANCE_NAMES = {
    "near": "nearby",
    "mid": "",
    "far": "in the distance",
}

# Objects particularly important for assistive navigation
OBSTACLE_CLASSES = {
    "person", "bicycle", "car", "motorcycle", "bus", "truck", "train",
    "bench", "chair", "couch", "bed", "dining table", "toilet",
    "dog", "cat", "horse", "cow", "elephant", "bear",
    "suitcase", "backpack",
}

HAZARD_CLASSES = {
    "car", "motorcycle", "bus", "truck", "train", "bicycle",
}


class TemplateCaption:
    """Generate scene descriptions from YOLO detections + templates.

    Pipeline:
    1. YOLO-Nano detects objects → [(class, bbox, confidence)]
    2. Map each bbox to spatial region (left/center/right, near/far)
    3. Template NLG generates natural language description

    Example:
        Input: YOLO detects [chair @ center-near, table @ left-far]
        Output: "There is a chair in the center, nearby. A table on the left."

    Advantages over BLIP:
    - ~50x faster (33ms vs 1600ms)
    - ~30x smaller (12MB vs 400MB)
    - Deterministic and explainable
    - Trade-off: less descriptive (no colors, textures, actions)
    """

    def __init__(self, detector: Optional[ObjectDetectorONNX] = None, conf_threshold: float = 0.3):
        self._detector = detector or ObjectDetectorONNX(conf_threshold=conf_threshold)

    def _classify_region(self, detection: Detection, frame_w: int) -> str:
        """Map bounding box to left/center/right region."""
        # Use center of bounding box
        cx = (detection.x1 + detection.x2) / 2

        third = frame_w / 3
        if cx < third:
            return "left"
        elif cx < 2 * third:
            return "center"
        else:
            return "right"

    def _classify_distance(self, detection: Detection, frame_h: int) -> str:
        """Estimate relative distance from bounding box size.

        Larger bounding box = closer to camera.
        """
        box_height = detection.y2 - detection.y1
        height_ratio = box_height / frame_h

        if height_ratio > 0.4:
            return "near"
        elif height_ratio > 0.15:
            return "mid"
        else:
            return "far"

    def _is_hazard(self, detection: Detection) -> bool:
        """Check if detection is a potential hazard for navigation."""
        return detection.class_name in HAZARD_CLASSES

    def _is_obstacle(self, detection: Detection) -> bool:
        """Check if detection is an obstacle (blocks path)."""
        return detection.class_name in OBSTACLE_CLASSES

    def generate_caption(self, frame: np.ndarray) -> str:
        """Generate a template-based scene description.

        Args:
            frame: BGR image from OpenCV (H, W, 3), uint8

        Returns:
            Natural language scene description.
        """
        detections = self._detector.detect(frame)

        if not detections:
            return "The area appears clear."

        h, w = frame.shape[:2]

        # Annotate each detection with spatial info
        annotated = []
        for det in detections:
            region = self._classify_region(det, w)
            distance = self._classify_distance(det, h)
            is_hazard = self._is_hazard(det)
            annotated.append((det, region, distance, is_hazard))

        # Sort: hazards first, then by distance (near first), then by region
        distance_order = {"near": 0, "mid": 1, "far": 2}
        annotated.sort(key=lambda x: (not x[3], distance_order[x[2]], x[1]))

        # Generate sentences
        sentences = []

        # Hazard warnings first
        hazards = [(d, r, dist, _) for d, r, dist, h in annotated if h]
        if hazards:
            for det, region, distance, _ in hazards:
                dist_str = DISTANCE_NAMES[distance]
                region_str = REGION_NAMES[region]
                if dist_str:
                    sentences.append(f"Warning: {det.class_name} {region_str}, {dist_str}")
                else:
                    sentences.append(f"Warning: {det.class_name} {region_str}")

        # Then regular objects (deduplicate by class+region)
        seen = set()
        for det, region, distance, is_hazard in annotated:
            if is_hazard:
                continue  # Already handled

            key = (det.class_name, region)
            if key in seen:
                continue
            seen.add(key)

            dist_str = DISTANCE_NAMES[distance]
            region_str = REGION_NAMES[region]

            if dist_str:
                sentences.append(f"A {det.class_name} {region_str}, {dist_str}")
            else:
                sentences.append(f"A {det.class_name} {region_str}")

        # Combine into paragraph
        if not sentences:
            return "The area appears clear."

        # Capitalize first word
        result = ". ".join(sentences) + "."
        return result

    def generate_navigation_summary(self, frame: np.ndarray) -> str:
        """Generate a brief navigation-focused summary.

        Shorter than full caption — just obstacles and hazards.
        Suitable for real-time audio during walking.
        """
        detections = self._detector.detect(frame)

        if not detections:
            return "Path clear."

        h, w = frame.shape[:2]

        # Only report obstacles and hazards
        relevant = []
        for det in detections:
            if self._is_obstacle(det) or self._is_hazard(det):
                region = self._classify_region(det, w)
                distance = self._classify_distance(det, h)
                relevant.append((det, region, distance))

        if not relevant:
            return "Path clear."

        # Sort by distance (nearest first)
        distance_order = {"near": 0, "mid": 1, "far": 2}
        relevant.sort(key=lambda x: distance_order[x[2]])

        parts = []
        for det, region, distance in relevant[:3]:  # Max 3 for brevity
            parts.append(f"{det.class_name} {REGION_NAMES[region]}")

        return "; ".join(parts) + "."


__all__ = ["TemplateCaption"]
