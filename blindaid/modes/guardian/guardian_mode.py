"""Navigation mode - warns about obstacles using depth + YOLO detection.

Research branch: Uses ONNX models + Adaptive Frame Processing.
"""
from __future__ import annotations

import logging
import time

import cv2
import numpy as np

from blindaid.core.adaptive_processor import AdaptiveFrameProcessor
from blindaid.core.depth_onnx import DepthAnalyzerONNX
from blindaid.core.detector_onnx import ObjectDetectorONNX
from blindaid.core.template_caption import TemplateCaption

logger = logging.getLogger(__name__)


class GuardianMode:
    def __init__(self, audio_enabled: bool = True):
        self.audio_enabled = audio_enabled
        self.depth_analyzer = None
        self.object_detector = None
        self.template_caption = None
        self.afp = AdaptiveFrameProcessor()

        self.frame_counter = 0
        self.last_warning_time = 0.0
        self.warning_cooldown = 2.5

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
        if self.template_caption is None:
            self.template_caption = TemplateCaption(detector=self.object_detector)

    def get_distance_label(self, depth_val):
        if depth_val > 0.8:
            return "Very Close (< 0.5m)"
        if depth_val > 0.6:
            return "Close (1m)"
        if depth_val > 0.4:
            return "Nearby (2m)"
        return "Safe"

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

        # === Full processing ===
        info_lines = ["Mode: Smart Navigation"]
        speech_messages = []
        display_frame = frame.copy()

        try:
            # Depth estimation
            depth_map = self.depth_analyzer.compute_depth(frame)
            self._last_depth_map = depth_map

            # Update AFP with new depth for next decision
            self.afp.update_depth(depth_map)

            # Object detection + template caption
            detections = self.object_detector.detect(frame)
            if detections:
                caption = self.template_caption.generate_navigation_summary(frame)
                if caption and caption != "Path clear.":
                    info_lines.append(f"Scene: {caption}")

            h, w = depth_map.shape

            # Screen ko 3 parts mein divide
            left_part = depth_map[:, : w // 3]
            center_part = depth_map[:, w // 3 : 2 * w // 3]
            right_part = depth_map[:, 2 * w // 3 :]

            l_val = np.mean(left_part)
            c_val = np.mean(center_part)
            r_val = np.mean(right_part)

            # Obstacle warning logic
            msg = ""
            if c_val > 0.7:
                msg = "Stop! Obstacle Ahead."
            elif l_val > 0.75:
                msg = "Obstacle on Left."
            elif r_val > 0.75:
                msg = "Obstacle on Right."

            now = time.time()
            if msg and (now - self.last_warning_time > self.warning_cooldown):
                speech_messages.append(msg)
                self.last_warning_time = now
                info_lines.append(f"WARNING: {msg}")

            # Draw YOLO detections on frame
            for det in detections:
                color = (0, 0, 255) if det.class_name in {"car", "truck", "bus", "motorcycle"} else (0, 255, 0)
                cv2.rectangle(display_frame, (det.x1, det.y1), (det.x2, det.y2), color, 2)
                label = f"{det.class_name} {det.confidence:.0%}"
                cv2.putText(display_frame, label, (det.x1, det.y1 - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

            # Depth heatmap overlay
            colored_depth = cv2.applyColorMap(
                (depth_map * 255).astype(np.uint8), cv2.COLORMAP_MAGMA
            )
            display_frame = cv2.addWeighted(frame, 0.7, colored_depth, 0.3, 0)

            # Redraw detections on top of overlay
            for det in detections:
                color = (0, 0, 255) if det.class_name in {"car", "truck", "bus", "motorcycle"} else (0, 255, 0)
                cv2.rectangle(display_frame, (det.x1, det.y1), (det.x2, det.y2), color, 2)
                label = f"{det.class_name} {det.confidence:.0%}"
                cv2.putText(display_frame, label, (det.x1, det.y1 - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

            # Stats bar
            metrics = self.afp.get_metrics("guardian")
            skip_pct = metrics.get("cpu_savings_pct", 0)
            cv2.putText(
                display_frame,
                f"L:{l_val:.2f} C:{c_val:.2f} R:{r_val:.2f} AFP:{skip_pct:.0f}%skip",
                (10, h - 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                2,
            )

        except Exception as e:
            logger.error("Guardian processing error: %s", e)
            info_lines.append(f"Error: {e}")

        # Cache for skipped frames
        self._last_display_frame = display_frame
        self._last_info_lines = info_lines
        self._last_speech = speech_messages

        return display_frame, info_lines, speech_messages

    def on_enter(self):
        logger.info("Smart Nav Active (ONNX + AFP)")

    def on_exit(self):
        # Log AFP metrics
        if self.afp:
            metrics = self.afp.get_metrics("guardian")
            logger.info("Guardian AFP metrics: %s", metrics)