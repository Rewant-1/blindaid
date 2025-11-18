"""
BlindAid - Main Application Entry Point
========================================
Unified controller that handles both scene exploration and reading assistance.

Usage:
    python -m blindaid                 # Start in scene mode (default)
    python -m blindaid --start-mode reading
"""
import sys
import argparse
import logging
import signal
from pathlib import Path
from typing import Sequence, Optional

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from blindaid.core import config

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
    from blindaid.modes.object_detection.detector import ObjectDetectionService

    logger.info("Starting Object Detection Mode")
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
    
    return parser.parse_args(argv)


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
