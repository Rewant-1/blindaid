"""Scene understanding mode combining object detection and face recognition."""
from __future__ import annotations

import time
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple, Dict

import cv2
import numpy as np
import face_recognition
from ultralytics import YOLO

from blindaid.core import config

logger = logging.getLogger(__name__)


@dataclass
class Detection:
    """Represents a detected entity in the frame."""

    label: str
    box: Tuple[int, int, int, int]
    confidence: float
    kind: str  # "object" or "face"
    position: str


class SceneMode:
    """Combined scene understanding mode (objects + faces)."""

    def __init__(self, audio_enabled: bool = True):
        self.audio_enabled = audio_enabled

        # Models
        self.object_model = YOLO(str(config.OBJECT_DETECTION_MODEL))
        logger.info("Object detection model loaded for scene mode")
        self.face_detector = YOLO(str(config.FACE_RECOGNITION_MODEL))
        logger.info("Face detection model loaded for scene mode")

        # Known faces
        (
            self.known_face_encodings,
            self.known_face_names,
        ) = self._load_known_faces(config.KNOWN_FACES_DIR)

        # State
        self.object_confidence = config.OBJECT_DETECTION_CONFIDENCE
        self.last_object_speech_time = 0.0
        self.object_speech_cooldown = getattr(config, "SCENE_OBJECT_COOLDOWN_SECONDS", 4.0)
        self.last_face_speech: Dict[str, float] = {}
        self.face_debounce = config.FACE_DEBOUNCE_SECONDS
        self.frame_counter = 0
        self.process_every = max(1, config.SCENE_PROCESS_EVERY)
        self.last_detections: List[Detection] = []
        self.last_object_summary: Optional[str] = None
        self.last_face_summary: Optional[str] = None

    # ------------------------------------------------------------------
    # Face utilities
    # ------------------------------------------------------------------
    def _load_known_faces(self, directory: Path):
        encodings = []
        names = []
        path = Path(directory)
        if not path.is_dir():
            path = config.KNOWN_FACES_DIR

        if not path.is_dir():
            logger.warning("Known faces directory %s not found", path)
            return encodings, names

        logger.info("Loading known faces from %s", path)
        for person_dir in path.iterdir():
            if not person_dir.is_dir():
                continue
            name = person_dir.name
            for image_path in person_dir.glob("*.*"):
                try:
                    image = face_recognition.load_image_file(str(image_path))
                    encs = face_recognition.face_encodings(image)
                    if encs:
                        encodings.append(encs[0])
                        names.append(name)
                except Exception as exc:  # noqa: BLE001 - log and continue
                    logger.warning("Failed to load face image %s: %s", image_path, exc)
        logger.info("Loaded %d known face embeddings", len(encodings))
        return encodings, names

    def _recognize_face(self, encoding: np.ndarray) -> Tuple[str, float]:
        if not self.known_face_encodings:
            return "Unknown", 0.0
        dists = face_recognition.face_distance(self.known_face_encodings, encoding)
        best_idx = np.argmin(dists)
        best_distance = dists[best_idx]
        confidence = float(max(0.0, 1.0 - best_distance))
        if best_distance <= config.FACE_THRESHOLD:
            return self.known_face_names[best_idx], confidence
        return "Unknown", confidence

    # ------------------------------------------------------------------
    @staticmethod
    def _object_position(x_center: float, frame_width: int) -> str:
        if x_center < frame_width / 3:
            return "left"
        if x_center < (2 * frame_width) / 3:
            return "center"
        return "right"

    def _summarize_objects(self, detections: List[Detection]) -> Optional[str]:
        if not detections:
            return None
        labels = {}
        for det in detections:
            labels.setdefault(det.label, set()).add(det.position)
        pieces = []
        for label, positions in labels.items():
            pos_text = ", ".join(sorted(positions))
            pieces.append(f"{label} on the {pos_text}")
        if pieces:
            return "I see " + ", ".join(pieces)
        return None

    def _build_face_messages(self, detections: List[Detection], now: float) -> List[str]:
        messages = []
        for det in detections:
            if det.label == "Unknown":
                continue
            last_time = self.last_face_speech.get(det.label, 0.0)
            if now - last_time > self.face_debounce:
                messages.append(f"{det.label} is on the {det.position} side")
                self.last_face_speech[det.label] = now
        return messages

    # ------------------------------------------------------------------
    def process_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, List[Detection], List[str], List[str]]:
        """Process a frame and return annotated frame, detections, info lines, speech."""
        display_frame = frame.copy()
        h, w = display_frame.shape[:2]
        now = time.time()
        speech: List[str] = []
        info_lines: List[str] = []
        detections: List[Detection] = []

        self.frame_counter += 1
        should_process = (self.frame_counter % self.process_every) == 0

        if should_process:
            # Object detection -------------------------------------------------
            object_results = self.object_model(display_frame, conf=self.object_confidence, verbose=False)
            object_dets: List[Detection] = []
            if object_results:
                boxes = object_results[0].boxes
                for box in boxes:
                    cls_id = int(box.cls[0])
                    label = self.object_model.names.get(cls_id, f"cls_{cls_id}")
                    conf = float(box.conf[0])
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    x1, y1 = max(x1, 0), max(y1, 0)
                    x2, y2 = min(x2, w - 1), min(y2, h - 1)
                    position = self._object_position((x1 + x2) / 2, w)
                    object_dets.append(
                        Detection(
                            label=label,
                            box=(x1, y1, x2, y2),
                            confidence=conf,
                            kind="object",
                            position=position,
                        )
                    )
                    cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 165, 255), 2)
                    cv2.putText(
                        display_frame,
                        f"{label} {conf*100:.0f}%",
                        (x1, max(20, y1 - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 165, 255),
                        2,
                    )

            # Face detection ---------------------------------------------------
            rgb_small = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
            face_results = self.face_detector(rgb_small, verbose=False)
            face_dets: List[Detection] = []
            face_locations = []
            if face_results:
                boxes = face_results[0].boxes.xyxy.cpu().numpy().astype(int)
                for (x1, y1, x2, y2) in boxes:
                    x1, y1 = max(x1, 0), max(y1, 0)
                    x2, y2 = min(x2, w - 1), min(y2, h - 1)
                    face_locations.append((y1, x2, y2, x1))

                encodings = face_recognition.face_encodings(rgb_small, face_locations)
                for (top, right, bottom, left), enc in zip(face_locations, encodings):
                    name, conf = self._recognize_face(enc)
                    position = self._object_position((left + right) / 2, w)
                    face_dets.append(
                        Detection(
                            label=name,
                            box=(left, top, right, bottom),
                            confidence=conf,
                            kind="face",
                            position=position,
                        )
                    )
                    color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
                    cv2.rectangle(display_frame, (left, top), (right, bottom), color, 2)
                    label_text = name if name != "Unknown" else "Unknown person"
                    if name != "Unknown":
                        label_text += f" {conf*100:.0f}%"
                    cv2.rectangle(display_frame, (left, bottom - 30), (right, bottom), color, cv2.FILLED)
                    cv2.putText(
                        display_frame,
                        f"{label_text} ({position})",
                        (left + 5, bottom - 8),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (255, 255, 255),
                        1,
                    )

            detections = object_dets + face_dets
            self.last_detections = detections
        else:
            detections = self.last_detections

        # Summaries & speech --------------------------------------------------
        object_dets = [d for d in detections if d.kind == "object"]
        face_dets = [d for d in detections if d.kind == "face"]

        if object_dets:
            summary = self._summarize_objects(object_dets)
            if summary:
                self.last_object_summary = summary
                if self.audio_enabled and now - self.last_object_speech_time > self.object_speech_cooldown:
                    speech.append(summary)
                    self.last_object_speech_time = now

        if face_dets:
            face_messages = self._build_face_messages(face_dets, now)
            if self.audio_enabled and face_messages:
                speech.extend(face_messages)
            if any(det.label != "Unknown" for det in face_dets):
                labeled = [f"{det.label} ({det.position})" for det in face_dets]
                self.last_face_summary = ", ".join(labeled)
        
        if self.last_object_summary:
            info_lines.append(self.last_object_summary)
        if self.last_face_summary:
            info_lines.append(self.last_face_summary)

        return display_frame, detections, info_lines, speech

    # ------------------------------------------------------------------
    def on_enter(self):  # noqa: D401 - simple hook
        """Reset counters on entering mode."""
        self.frame_counter = 0

    def on_exit(self):  # noqa: D401 - simple hook
        """Placeholder for future cleanup."""
        return

    def get_last_detections(self) -> List[Detection]:
        return self.last_detections
