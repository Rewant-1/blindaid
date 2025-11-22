"""
Configuration management for BlindAid system.
"""
import os
from pathlib import Path

# Project root directory
PROJECT_ROOT = Path(__file__).parent.parent.parent
RESOURCES_DIR = PROJECT_ROOT / "resources"
MODELS_DIR = RESOURCES_DIR / "models"
KNOWN_FACES_DIR = RESOURCES_DIR / "known_faces"

# Camera settings
DEFAULT_CAMERA_INDEX = 0

# Object Detection settings
OBJECT_DETECTION_MODEL = MODELS_DIR / "object_blind_aide.onnx"
OBJECT_DETECTION_CONFIDENCE = 0.6
OBJECT_DETECTION_FRAME_SKIP = 3
SCENE_PROCESS_EVERY = 1

# OCR settings
OCR_LANGUAGE = 'en'
OCR_CONFIDENCE_THRESHOLD = 0.9
OCR_COOLDOWN_SECONDS = 5
OCR_FRAME_SKIP = 4

# Face Recognition settings
FACE_RECOGNITION_MODEL = MODELS_DIR / "yolov9t-face-lindevs.onnx"
FACE_THRESHOLD = 0.5
FACE_DETECTION_MODEL = "hog"  # "hog" or "cnn"
FACE_FRAME_SCALE = 0.25
FACE_PROCESS_EVERY_N_FRAMES = 2
FACE_DEBOUNCE_SECONDS = 15.0
FACE_OVERLAY_TIMEOUT = 0.6

# Scene mode defaults
SCENE_OBJECT_COOLDOWN_SECONDS = 4.0
SCENE_HINT_TEXT = "1:Scene 2:Read P:People V:Ask Space:Cap Q:Quit"

# Audio/TTS settings
AUDIO_ENABLED = True
TTS_RATE = 150
TTS_VOLUME = 0.9
# If True, force use of online TTS (gTTS + pygame) instead of offline pyttsx3.
# Useful when pyttsx3 cannot access a Windows voice driver or is unstable.
TTS_FORCE_ONLINE = False

# Display settings
DISPLAY_FPS = True
BOUNDING_BOX_COLOR_KNOWN = (0, 255, 0)  # Green
BOUNDING_BOX_COLOR_UNKNOWN = (0, 0, 255)  # Red
BOUNDING_BOX_THICKNESS = 2

# Performance
FRAME_WIDTH = 320
FRAME_HEIGHT = 240
