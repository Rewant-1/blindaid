"""
Configuration defaults for Face Recognition System
=================================================
This module contains default configuration values used by the face recognizer.
You can modify these defaults or override them via command-line arguments.
"""

# Directory paths
DEFAULT_KNOWN_FACES_DIR = "known_faces"

# Camera settings
DEFAULT_CAMERA_INDEX = 0

# Recognition parameters
DEFAULT_FACE_THRESHOLD = 0.5  # Lower = stricter matching (range 0.0-1.0)
DEFAULT_DETECTION_MODEL = "hog"  # Options: "hog" (faster) or "cnn" (more accurate)

# Performance settings
DEFAULT_FRAME_SCALE = 0.25  # Scale factor for processing (smaller = faster)
DEFAULT_PROCESS_EVERY_N_FRAMES = 2  # Process every Nth frame

# Audio settings
DEFAULT_DEBOUNCE_SECONDS = 15.0  # Seconds between repeated announcements
DEFAULT_TTS_RATE = 150  # Words per minute for text-to-speech
DEFAULT_TTS_VOLUME = 0.9  # Volume level (0.0 to 1.0)

# Display settings
FPS_DISPLAY_INTERVAL = 30  # Calculate FPS every N frames
BOUNDING_BOX_COLOR_KNOWN = (0, 255, 0)  # Green for recognized faces
BOUNDING_BOX_COLOR_UNKNOWN = (0, 0, 255)  # Red for unknown faces
BOUNDING_BOX_THICKNESS = 2
LABEL_FONT = 0  # cv2.FONT_HERSHEY_SIMPLEX
LABEL_FONT_SCALE = 0.6
LABEL_FONT_THICKNESS = 1
LABEL_TEXT_COLOR = (255, 255, 255)  # White

# Position detection thresholds
POSITION_LEFT_THRESHOLD = 0.33  # Left third of frame
POSITION_RIGHT_THRESHOLD = 0.67  # Right third of frame

# Logging configuration
LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'

# Audio queue settings
AUDIO_QUEUE_MAX_SIZE = 10  # Maximum pending audio messages

# Supported image formats
SUPPORTED_IMAGE_FORMATS = ('.jpg', '.jpeg', '.png', '.bmp', '.gif')
