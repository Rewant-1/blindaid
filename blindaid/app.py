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
    level = logging.DEBUG if debug else logging.ERROR  # ERROR by default for faster startup
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler()]
    )


def signal_handler(signum, frame):
    """Handle shutdown signals gracefully."""
    logger.info(f"Received signal {signum}, shutting down...")
    sys.exit(0)


def parse_arguments(argv: Optional[Sequence[str]] = None):
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="BlindAid - Assistive Technology System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Start in scene mode (default)
    python -m blindaid

    # Start in reading mode
    python -m blindaid --start-mode reading

    # Use a different camera with audio disabled
    python -m blindaid --camera 1 --no-audio
        """
    )

    parser.add_argument(
        "--start-mode",
        type=str,
        default="sitting",
        choices=["sitting", "guardian", "reading", "people"],
        help="Mode to start in (default: sitting)",
    )
    parser.add_argument(
        "--camera",
        type=int,
        default=config.DEFAULT_CAMERA_INDEX,
        help=f"Camera device index (default: {config.DEFAULT_CAMERA_INDEX})",
    )
    parser.add_argument(
        "--no-audio",
        dest="audio",
        action="store_false",
        help="Disable audio feedback",
    )
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")

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
    logger.info("BlindAid System - Integrated Controller")
    logger.info("="*70)
    try:
        # Import controller lazily to avoid heavy deps (cv2) at module import time.
        from blindaid.controller import ModeController

        controller = ModeController(
            camera_index=args.camera,
            audio_enabled=args.audio,
            initial_mode=args.start_mode,
        )
        controller.run()
    except Exception as e:  # noqa: BLE001
        logger.exception("Fatal error: %s", e)
        return 1
    
    logger.info("BlindAid System Stopped")
    return 0


if __name__ == "__main__":
    sys.exit(main())
