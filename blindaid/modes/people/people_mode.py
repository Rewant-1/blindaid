"""People Mode: One-shot face recognition."""
from __future__ import annotations

import time
import logging
from pathlib import Path
from typing import List, Tuple, Optional, Dict

import cv2
import numpy as np
import face_recognition
from ultralytics import YOLO

from blindaid.core import config

logger = logging.getLogger(__name__)

class PeopleMode:
    """
    People Mode (One-Shot).
    Scans for 3 seconds to list people nearby.
    """

    def __init__(self, audio_enabled: bool = True):
        self.audio_enabled = audio_enabled
        self.start_time = 0.0
        self.duration = 5.0 # Give it 5 seconds to be sure
        self.finished = False
        
        # Models
        self.face_detector: Optional[YOLO] = None
        self.known_face_encodings: List[np.ndarray] = []
        self.known_face_names: List[str] = []
        self._loaded = False
        
        # Results
        self.detected_people = set()

    def _ensure_loaded(self):
        if self._loaded:
            return
        import os
        import warnings
        # Suppress YOLO and face_recognition verbose output
        os.environ['YOLO_VERBOSE'] = 'False'
        warnings.filterwarnings('ignore', category=UserWarning)
        warnings.filterwarnings('ignore', category=FutureWarning)
        try:
            logger.info("Loading face detection models...")
            self.face_detector = YOLO(str(config.FACE_RECOGNITION_MODEL), verbose=False)
            self._load_known_faces(config.KNOWN_FACES_DIR)
            self._loaded = True
        except Exception as e:
            logger.error(f"Failed to load face models: {e}")

    def _load_known_faces(self, directory: Path):
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
                    encs = face_recognition.face_encodings(image)
                    if encs:
                        self.known_face_encodings.append(encs[0])
                        self.known_face_names.append(name)
                except Exception:
                    pass
        logger.info("Loaded %d known face embeddings", len(self.known_face_encodings))

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

    def on_enter(self):
        self._ensure_loaded()
        self.start_time = time.time()
        self.finished = False
        self.detected_people = set()
        logger.info("People Mode Started")

    def on_exit(self):
        pass

    def is_finished(self) -> bool:
        return self.finished

    def process_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, List[str], List[str]]:
        display_frame = frame.copy()
        info_lines = ["Scanning for people..."]
        speech_messages = []
        
        if self.finished:
            return display_frame, ["Scan Complete"], []

        now = time.time()
        if now - self.start_time > self.duration:
            self.finished = True
            # Summarize results - only speak about known people
            if self.detected_people:
                names = list(self.detected_people)
                known_names = [n for n in names if n != "Unknown"]
                
                if known_names:
                    summary = "I see " + " and ".join(known_names)
                    speech_messages.append(summary)
                    info_lines.append(summary)
                else:
                    speech_messages.append("No one recognized.")
                    info_lines.append("No one recognized.")
            else:
                speech_messages.append("No one found.")
                info_lines.append("No one found.")
            
            return display_frame, info_lines, speech_messages

        # Detection Logic
        h, w = frame.shape[:2]
        rgb_small = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        if self.face_detector:
            results = self.face_detector(rgb_small, verbose=False)
            if results:
                boxes = results[0].boxes.xyxy.cpu().numpy().astype(int)
                face_locations = []
                for (x1, y1, x2, y2) in boxes:
                    x1, y1 = max(x1, 0), max(y1, 0)
                    x2, y2 = min(x2, w - 1), min(y2, h - 1)
                    face_locations.append((y1, x2, y2, x1)) # face_recognition uses top, right, bottom, left

                encodings = face_recognition.face_encodings(rgb_small, face_locations)
                for (top, right, bottom, left), enc in zip(face_locations, encodings):
                    name, conf = self._recognize_face(enc)
                    self.detected_people.add(name)
                    
                    color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
                    cv2.rectangle(display_frame, (left, top), (right, bottom), color, 2)
                    cv2.putText(display_frame, name, (left, top-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        return display_frame, info_lines, speech_messages
