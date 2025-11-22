"""
Raspberry Pi optimized configuration for BlindAid system.
Copy this to blindaid/core/config.py on your Raspberry Pi for better performance.
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

# Object Detection settings - OPTIMIZED FOR PI
OBJECT_DETECTION_MODEL = MODELS_DIR / "object_blind_aide.onnx"
OBJECT_DETECTION_CONFIDENCE = 0.5  # Lowered for faster processing
OBJECT_DETECTION_FRAME_SKIP = 5    # Process every 5th frame
SCENE_PROCESS_EVERY = 3

# OCR settings - OPTIMIZED FOR PI
OCR_LANGUAGE = 'en'
OCR_CONFIDENCE_THRESHOLD = 0.85  # Slightly lower for better detection
OCR_COOLDOWN_SECONDS = 5
OCR_FRAME_SKIP = 5  # Process every 5th frame for better performance

# Face Recognition settings - OPTIMIZED FOR PI
FACE_RECOGNITION_MODEL = MODELS_DIR / "yolov9t-face-lindevs.onnx"
FACE_THRESHOLD = 0.5
FACE_DETECTION_MODEL = "hog"  # HOG is faster than CNN on CPU
FACE_FRAME_SCALE = 0.15  # Aggressive downscaling for Pi (was 0.25)
FACE_PROCESS_EVERY_N_FRAMES = 5  # Process every 5 frames
FACE_DEBOUNCE_SECONDS = 15.0
FACE_OVERLAY_TIMEOUT = 0.6

# Scene mode defaults
SCENE_OBJECT_COOLDOWN_SECONDS = 4.0
SCENE_HINT_TEXT = "1: Scene  2: Reading  Q: Quit"  # Removed heavy features

# Audio/TTS settings
AUDIO_ENABLED = True
TTS_RATE = 150
TTS_VOLUME = 0.9

# Display settings
DISPLAY_FPS = True
BOUNDING_BOX_COLOR_KNOWN = (0, 255, 0)  # Green
BOUNDING_BOX_COLOR_UNKNOWN = (0, 0, 255)  # Red
BOUNDING_BOX_THICKNESS = 2

# Performance - OPTIMIZED FOR PI 5
FRAME_WIDTH = 320   # Reduced from 640 for better FPS
FRAME_HEIGHT = 240  # Reduced from 480 for better FPS

# Note: Caption and Depth features disabled by default on Pi
# They require PyTorch which is very heavy on Raspberry Pi
# If you have 8GB RAM Pi 5 and want to try them:
# 1. Install torch, transformers in requirements-pi.txt
# 2. Expect slower performance (may need to disable other modes)
