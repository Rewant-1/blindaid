"""Config values."""
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
OCR_CONFIDENCE_THRESHOLD = 0.95
OCR_COOLDOWN_SECONDS = 5
OCR_FRAME_SKIP = 4

# Face Recognition settings
FACE_RECOGNITION_MODEL = MODELS_DIR / "yolov9t-face-lindevs.pt"
FACE_THRESHOLD = 0.5
FACE_DETECTION_MODEL = "hog"  # "hog" or "cnn"
FACE_FRAME_SCALE = 0.25
FACE_PROCESS_EVERY_N_FRAMES = 2
FACE_DEBOUNCE_SECONDS = 15.0
FACE_OVERLAY_TIMEOUT = 0.6

# Scene mode defaults
SCENE_OBJECT_COOLDOWN_SECONDS = 4.0
SCENE_HINT_TEXT = "0:Sit 1:Walk 2:Read 3:Ppl 4:Ask 5:Cap Q:Quit"

# Audio/TTS settings
AUDIO_ENABLED = True
TTS_RATE = 150
TTS_VOLUME = 0.9
# Always use online TTS (gTTS + pygame) for reliability
TTS_FORCE_ONLINE = True

# Display settings
DISPLAY_FPS = True
BOUNDING_BOX_COLOR_KNOWN = (0, 255, 0)  # Green
BOUNDING_BOX_COLOR_UNKNOWN = (0, 0, 255)  # Red
BOUNDING_BOX_THICKNESS = 2

# Performance
FRAME_WIDTH = 320
FRAME_HEIGHT = 240
