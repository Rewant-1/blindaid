"""Face recognition mode."""
from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import List, Optional, Set, Tuple

import cv2
import face_recognition
import numpy as np
from ultralytics import YOLO

from blindaid.core import config

logger = logging.getLogger(__name__)


class PeopleMode:
    def __init__(self, audio_enabled: bool = True):
        self.audio_enabled = audio_enabled
        self.start_time = 0.0
        self.duration = 5.0
        self.finished = False

        self.face_detector: Optional[YOLO] = None
        self.known_face_encodings: List[np.ndarray] = []
        self.known_face_names: List[str] = []
        self._loaded = False

        self.detected_people: Set[str] = set()

    def _ensure_loaded(self) -> None:
        if self._loaded:
            return
        import warnings

        warnings.filterwarnings("ignore", category=UserWarning)
        warnings.filterwarnings("ignore", category=FutureWarning)
        try:
            self.face_detector = YOLO(str(config.FACE_RECOGNITION_MODEL), verbose=False)
            self._load_known_faces(config.KNOWN_FACES_DIR)
            self._loaded = True
            logger.info("Face datasets loaded (%d known)", len(self.known_face_encodings))
        except Exception as exc:  # noqa: BLE001
            logger.error("Failed to load face models: %s", exc)

    def _load_known_faces(self, directory: Path) -> None:
        path = Path(directory)
        if not path.is_dir():
            logger.warning("Known faces directory %s not found", path)
            return

        for person_dir in path.iterdir():
            if not person_dir.is_dir():
                continue
            name = person_dir.name
            for image_path in person_dir.glob("*.*"):
                try:
                    image = face_recognition.load_image_file(str(image_path))
                    encodings = face_recognition.face_encodings(image)
                    if encodings:
                        self.known_face_encodings.append(encodings[0])
                        self.known_face_names.append(name)
                except Exception:  # noqa: BLE001
                    continue

    def _recognize_face(self, encoding: np.ndarray) -> Tuple[str, float]:
        if not self.known_face_encodings:
            return "Unknown", 0.0
        distances = face_recognition.face_distance(self.known_face_encodings, encoding)
        best_idx = int(np.argmin(distances))
        best_distance = float(distances[best_idx])
        confidence = max(0.0, 1.0 - best_distance)
        if best_distance <= config.FACE_THRESHOLD:
            return self.known_face_names[best_idx], confidence
        return "Unknown", confidence

    def on_enter(self) -> None:
        self._ensure_loaded()
        self.start_time = time.monotonic()
        self.finished = False
        self.detected_people = set()
        logger.info("People mode started")

    def on_exit(self) -> None:
        logger.info("People mode stopped")

    def is_finished(self) -> bool:
        return self.finished

    def _summarise(self) -> Tuple[List[str], List[str]]:
        info_lines = ["Scan complete"]
        speech: List[str] = []
        if not self.detected_people:
            info_lines.append("No one found")
            speech.append("No one found.")
            return info_lines, speech

        known_names = [name for name in self.detected_people if name != "Unknown"]
        if known_names:
            summary = "I see " + " and ".join(sorted(known_names))
            info_lines.append(summary)
            speech.append(summary)
        else:
            info_lines.append("No one recognized")
            speech.append("No one recognized.")
        return info_lines, speech

    def process_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, List[str], List[str]]:
        display_frame = frame.copy()
        if self.finished:
            return display_frame, ["Scan complete"], []

        elapsed = time.monotonic() - self.start_time
        if elapsed > self.duration:
            self.finished = True
            return display_frame, *self._summarise()

        info_lines = ["Scanning for people..."]
        speech_messages: List[str] = []

        if self.face_detector is None:
            return display_frame, info_lines, speech_messages

        h, w = frame.shape[:2]
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_detector(rgb_frame, verbose=False)
        if not results:
            return display_frame, info_lines, speech_messages

        boxes = results[0].boxes.xyxy.cpu().numpy().astype(int)
        face_locations = []
        for (x1, y1, x2, y2) in boxes:
            x1, y1 = max(x1, 0), max(y1, 0)
            x2, y2 = min(x2, w - 1), min(y2, h - 1)
            face_locations.append((y1, x2, y2, x1))

        encodings = face_recognition.face_encodings(rgb_frame, face_locations)
        for (top, right, bottom, left), encoding in zip(face_locations, encodings):
            name, _ = self._recognize_face(encoding)
            self.detected_people.add(name)

            color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
            cv2.rectangle(display_frame, (left, top), (right, bottom), color, 2)
            cv2.putText(display_frame, name, (left, max(0, top - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        return display_frame, info_lines, speech_messages
