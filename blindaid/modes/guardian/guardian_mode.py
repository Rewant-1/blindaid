"""Navigation mode — layered perception stack.

Camera → YOLO + MiDaS → Depth Fusion → Scene State Engine → Speech

Research branch: ONNX models + AFP + Depth Fusion + Scene State.
"""
from __future__ import annotations

import logging
import time

import cv2
import numpy as np

from blindaid.core.adaptive_processor import AdaptiveFrameProcessor
from blindaid.core.depth_onnx import DepthAnalyzerONNX
from blindaid.core.detector_onnx import ObjectDetectorONNX
from blindaid.core.depth_fusion import fuse_detections, FusedDetection, DISTANCE_SPEECH
from blindaid.core.scene_state import SceneStateEngine

logger = logging.getLogger(__name__)


class GuardianMode:
    def __init__(self, audio_enabled: bool = True):
        self.audio_enabled = audio_enabled
        self.depth_analyzer = None
        self.object_detector = None
        self.afp = AdaptiveFrameProcessor()
        self.scene_state = SceneStateEngine()

        self.frame_counter = 0

        # Cache for display during skipped frames
        self._last_display_frame = None
        self._last_info_lines = ["Mode: Smart Navigation"]
        self._last_speech = []
        self._last_depth_map = None

    def _ensure_models(self):
        if self.depth_analyzer is None:
            self.depth_analyzer = DepthAnalyzerONNX()
        if self.object_detector is None:
            self.object_detector = ObjectDetectorONNX()

    def process_frame(self, frame: np.ndarray):
        self.frame_counter += 1
        self._ensure_models()

        # AFP decides whether to process this frame
        should_process = self.afp.should_process(
            "guardian", frame, self._last_depth_map
        )

        if not should_process:
            # Return cached display during skip
            if self._last_display_frame is not None:
                return self._last_display_frame, self._last_info_lines, []
            return frame.copy(), ["Mode: Smart Navigation (warming up)"], []

        # === Full processing pipeline ===
        info_lines = ["Mode: Smart Navigation"]
        speech_messages = []
        display_frame = frame.copy()

        try:
            h_frame, w_frame = frame.shape[:2]

            # --- Layer 1: Perception ---
            depth_map = self.depth_analyzer.compute_depth(frame)
            self._last_depth_map = depth_map
            self.afp.update_depth(depth_map)

            detections = self.object_detector.detect(frame)

            # --- Layer 2: Depth Fusion ---
            fused = fuse_detections(detections, depth_map, (h_frame, w_frame), enable_fallback=True)

            # --- Layer 3: Scene State Engine ---
            events = self.scene_state.update(fused)

            # Generate speech from events (not raw detections)
            for event in events:
                speech_messages.append(event.text)
                info_lines.append(event.text)

            # If no events but we have objects, show info without speaking
            if not events and fused:
                closest = fused[0]
                dist = DISTANCE_SPEECH.get(closest.distance_category, "")
                info_lines.append(f"Tracking: {closest.class_name} {closest.region}, {dist}")

            if not fused:
                info_lines.append("Path clear")

            # --- Display: depth overlay + detection boxes with distance ---
            colored_depth = cv2.applyColorMap(
                (depth_map * 255).astype(np.uint8), cv2.COLORMAP_MAGMA
            )
            display_frame = cv2.addWeighted(frame, 0.7, colored_depth, 0.3, 0)

            # Draw fused detections with distance labels
            for det in fused:
                # Color by risk level
                if det.is_collision_risk:
                    color = (0, 0, 255)  # Red
                elif det.distance_category in ("nearby",):
                    color = (0, 165, 255)  # Orange
                else:
                    color = (0, 255, 0)  # Green

                cv2.rectangle(display_frame, (det.x1, det.y1), (det.x2, det.y2), color, 2)

                # Label with distance
                dist_short = DISTANCE_SPEECH.get(det.distance_category, "")
                label = f"{det.class_name} {dist_short}"
                cv2.putText(display_frame, label, (det.x1, det.y1 - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)

            # Stats bar
            afp_metrics = self.afp.get_metrics("guardian")
            scene_metrics = self.scene_state.get_metrics()
            skip_pct = afp_metrics.get("cpu_savings_pct", 0)
            tracks = scene_metrics.get("active_tracks", 0)

            stats_text = f"AFP:{skip_pct:.0f}%skip | Tracks:{tracks}"
            cv2.putText(display_frame, stats_text, (10, h_frame - 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)

        except Exception as e:
            logger.error("Guardian processing error: %s", e)
            info_lines.append(f"Error: {e}")

        # Cache for skipped frames
        self._last_display_frame = display_frame
        self._last_info_lines = info_lines
        self._last_speech = speech_messages

        return display_frame, info_lines, speech_messages

    def on_enter(self):
        logger.info("Smart Nav Active (ONNX + AFP + Scene State)")

    def on_exit(self):
        if self.afp:
            metrics = self.afp.get_metrics("guardian")
            logger.info("Guardian AFP metrics: %s", metrics)
        if self.scene_state:
            metrics = self.scene_state.get_metrics()
            logger.info("Guardian Scene State metrics: %s", metrics)