"""
BlindAid - Main Application Entry Point
========================================
A consolidated assistive technology system with multiple modes:
- Integrated Controller: Unified scene understanding and reading with hotkeys
- Object Detection: Detect and locate objects in the environment
- OCR: Read text from images/documents
- Face Recognition: Recognize people and their positions

Usage:
    python -m blindaid                 # Integrated controller
    python -m blindaid --mode object-detection
    python -m blindaid --mode ocr
    python -m blindaid --mode face
"""
import sys
import argparse
import logging
import signal
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from blindaid.core import config
from blindaid.controller import ModeController
from blindaid.modes.object_detection.detector import ObjectDetectionService
from blindaid.modes.ocr.reader import OCRService
from blindaid.modes.face_recognition.recognizer import FaceRecognitionService

# Logger
logger = logging.getLogger(__name__)


def setup_logging(debug=False):
    """Configure logging."""
    level = logging.DEBUG if debug else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler()]
    )


def signal_handler(signum, frame):
    """Handle shutdown signals gracefully."""
    logger.info(f"Received signal {signum}, shutting down...")
    sys.exit(0)


def run_object_detection(args):
    """Run object detection mode."""
    logger.info("Starting Object Detection Mode")
    service = ObjectDetectionService(
        camera_index=args.camera,
        model_path=args.model,
        confidence=args.confidence,
        audio_enabled=args.audio
    )
    service.run()


def run_ocr(args):
    """Run OCR mode."""
    logger.info("Starting OCR Mode")
    service = OCRService(
        camera_index=args.camera,
        language=args.language,
        confidence=args.confidence,
        audio_enabled=args.audio
    )
    service.run()


def run_face_recognition(args):
    """Run face recognition mode."""
    logger.info("Starting Face Recognition Mode")
    service = FaceRecognitionService(
        camera_index=args.camera,
        known_faces_dir=args.known_faces,
        threshold=args.threshold,
        audio_enabled=args.audio
    )
    service.run()


def run_integrated_controller(args):
    """Run the unified keyboard-controlled controller."""
    logger.info("Starting Integrated Controller Mode")
    controller = ModeController(
        camera_index=args.camera,
        audio_enabled=args.audio,
    )
    controller.run()


def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="BlindAid - Assistive Technology System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Run integrated controller (default)
    python -m blindaid

  # Run object detection mode
  python -m blindaid --mode object-detection
  
  # Run OCR mode with higher confidence
  python -m blindaid --mode ocr --confidence 0.95
  
  # Run face recognition mode with custom directory
  python -m blindaid --mode face --known-faces ./my_faces
  
  # Disable audio feedback
  python -m blindaid --mode object-detection --no-audio
  
  # Use different camera
  python -m blindaid --mode face --camera 1
        """
    )
    
    # Required arguments
    parser.add_argument('--mode', type=str, default='integrated',
                       choices=['integrated', 'object-detection', 'ocr', 'face'],
                       help='Mode to run: integrated (default), object-detection, ocr, or face')
    
    # Common arguments
    parser.add_argument('--camera', type=int, default=config.DEFAULT_CAMERA_INDEX,
                       help=f'Camera device index (default: {config.DEFAULT_CAMERA_INDEX})')
    parser.add_argument('--no-audio', dest='audio', action='store_false',
                       help='Disable audio feedback')
    parser.add_argument('--debug', action='store_true',
                       help='Enable debug logging')
    
    # Object detection arguments
    parser.add_argument('--model', type=str,
                       help='Path to custom YOLO model (object detection mode)')
    parser.add_argument('--confidence', type=float, default=0.6,
                       help='Confidence threshold (default: 0.6)')
    
    # OCR arguments
    parser.add_argument('--language', type=str, default='en',
                       help='OCR language (default: en)')
    
    # Face recognition arguments
    parser.add_argument('--known-faces', type=str,
                       help='Path to known faces directory (face recognition mode)')
    parser.add_argument('--threshold', type=float, default=0.5,
                       help='Face recognition threshold (default: 0.5)')
    
    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_arguments()
    setup_logging(args.debug)
    
    # Register signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    if hasattr(signal, 'SIGTERM'):
        signal.signal(signal.SIGTERM, signal_handler)
    
    logger.info("="*70)
    logger.info(f"BlindAid System - {args.mode.upper()} Mode")
    logger.info("="*70)
    
    # Route to appropriate mode
    try:
        if args.mode == 'integrated':
            run_integrated_controller(args)
        elif args.mode == 'object-detection':
            run_object_detection(args)
        elif args.mode == 'ocr':
            run_ocr(args)
        elif args.mode == 'face':
            run_face_recognition(args)
        else:
            logger.error(f"Unknown mode: {args.mode}")
            return 1
    except Exception as e:
        logger.exception(f"Fatal error: {e}")
        return 1
    
    logger.info("BlindAid System Stopped")
    return 0


if __name__ == "__main__":
    sys.exit(main())
