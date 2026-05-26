"""Depth + YOLO Fusion — combines object detection with depth estimation.

Instead of: "person on the left"
We get:     "person ahead, approximately 2 meters"

Uses MiDaS relative depth mapped to approximate distance categories.
MiDaS gives inverse relative depth (higher value = closer to camera).

Phase 8 of the research paper implementation.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np

from blindaid.core.detector_onnx import Detection

logger = logging.getLogger(__name__)


@dataclass
class FusedDetection:
    """Detection enriched with depth information."""
    class_name: str
    confidence: float
    x1: int
    y1: int
    x2: int
    y2: int

    # Depth info
    raw_depth: float        # Median MiDaS value in bbox (0-1, higher=closer)
    distance_category: str  # "very_close", "close", "nearby", "moderate", "far"
    distance_label: str     # Human-readable: "less than 1 meter"

    # Spatial info
    region: str             # "left", "center", "right"
    vertical: str           # "ground", "mid", "upper"

    @property
    def is_collision_risk(self) -> bool:
        """Is this object dangerously close?"""
        return self.distance_category in ("very_close", "close")

    @property
    def bbox_area(self) -> int:
        return (self.x2 - self.x1) * (self.y2 - self.y1)


# Distance mapping from MiDaS relative depth values
# These thresholds were determined empirically — MiDaS outputs are relative,
# not metric, so exact meters are approximate.
DISTANCE_THRESHOLDS = [
    (0.80, "very_close", "less than 1 meter"),
    (0.60, "close",      "about 1-2 meters"),
    (0.40, "nearby",     "about 2-4 meters"),
    (0.25, "moderate",   "about 4-6 meters"),
    (0.00, "far",        "more than 6 meters"),
]

# Short labels for speech output
DISTANCE_SPEECH = {
    "very_close": "very close",
    "close":      "about 2 meters",
    "nearby":     "about 3 meters",
    "moderate":   "about 5 meters",
    "far":        "far away",
}


def classify_distance(depth_value: float) -> tuple[str, str]:
    """Map MiDaS depth value to distance category and human label.

    Args:
        depth_value: Normalized MiDaS depth (0-1, higher = closer)

    Returns:
        (category, label) tuple
    """
    for threshold, category, label in DISTANCE_THRESHOLDS:
        if depth_value >= threshold:
            return category, label
    return "far", "more than 6 meters"


def classify_region(cx: float, frame_w: int) -> str:
    """Map horizontal position to region."""
    third = frame_w / 3
    if cx < third:
        return "left"
    elif cx < 2 * third:
        return "center"
    return "right"


def classify_vertical(cy: float, frame_h: int) -> str:
    """Map vertical position to height zone."""
    third = frame_h / 3
    if cy > 2 * third:
        return "ground"
    elif cy > third:
        return "mid"
    return "upper"


def fuse_detections(
    detections: list[Detection],
    depth_map: np.ndarray,
    frame_shape: tuple[int, int],
) -> list[FusedDetection]:
    """Combine YOLO detections with MiDaS depth map.

    For each detected object:
    1. Extract the depth values within its bounding box
    2. Compute median depth (robust to outliers)
    3. Map to distance category
    4. Add spatial region info

    Args:
        detections: YOLO detection results
        depth_map: MiDaS depth map (H, W), values 0-1, higher = closer
        frame_shape: (height, width) of original frame

    Returns:
        List of FusedDetection with distance info
    """
    frame_h, frame_w = frame_shape
    depth_h, depth_w = depth_map.shape[:2]

    fused = []

    for det in detections:
        # Scale bbox to depth map coordinates if sizes differ
        scale_x = depth_w / frame_w
        scale_y = depth_h / frame_h

        dx1 = max(0, int(det.x1 * scale_x))
        dy1 = max(0, int(det.y1 * scale_y))
        dx2 = min(depth_w, int(det.x2 * scale_x))
        dy2 = min(depth_h, int(det.y2 * scale_y))

        # Extract depth within bounding box
        if dx2 > dx1 and dy2 > dy1:
            bbox_depth = depth_map[dy1:dy2, dx1:dx2]
            # Use median for robustness (ignores background pixels)
            raw_depth = float(np.median(bbox_depth))
        else:
            raw_depth = 0.0

        # Classify
        dist_category, dist_label = classify_distance(raw_depth)

        cx = (det.x1 + det.x2) / 2
        cy = (det.y1 + det.y2) / 2
        region = classify_region(cx, frame_w)
        vertical = classify_vertical(cy, frame_h)

        fused.append(FusedDetection(
            class_name=det.class_name,
            confidence=det.confidence,
            x1=det.x1, y1=det.y1, x2=det.x2, y2=det.y2,
            raw_depth=raw_depth,
            distance_category=dist_category,
            distance_label=dist_label,
            region=region,
            vertical=vertical,
        ))

    # Sort by distance (closest first — highest depth value)
    fused.sort(key=lambda f: f.raw_depth, reverse=True)

    return fused


def generate_fused_speech(fused_detections: list[FusedDetection], max_items: int = 3) -> str:
    """Generate concise speech from fused detections.

    Examples:
        "Person ahead, about 2 meters."
        "Car on the left, very close. Chair in the center, about 3 meters."
    """
    if not fused_detections:
        return "Path clear."

    # Prioritize: collision risks first, then by distance
    collision_risks = [f for f in fused_detections if f.is_collision_risk]
    others = [f for f in fused_detections if not f.is_collision_risk]

    parts = []

    # Collision warnings first
    for det in collision_risks[:2]:
        dist_speech = DISTANCE_SPEECH.get(det.distance_category, "")
        if det.region == "center":
            parts.append(f"Warning: {det.class_name} ahead, {dist_speech}")
        else:
            parts.append(f"Warning: {det.class_name} on the {det.region}, {dist_speech}")

    # Then other objects (up to max_items total)
    remaining = max_items - len(parts)
    for det in others[:remaining]:
        dist_speech = DISTANCE_SPEECH.get(det.distance_category, "")
        if det.region == "center":
            parts.append(f"{det.class_name} ahead, {dist_speech}")
        else:
            parts.append(f"{det.class_name} on the {det.region}, {dist_speech}")

    return ". ".join(parts) + "." if parts else "Path clear."


__all__ = [
    "FusedDetection", "fuse_detections", "generate_fused_speech",
    "classify_distance", "DISTANCE_THRESHOLDS", "DISTANCE_SPEECH",
]
