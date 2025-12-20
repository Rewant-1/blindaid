"""CLI entry point."""
import argparse
import logging
import signal
import sys
from typing import Sequence, Optional

from blindaid.core import config

logger = logging.getLogger(__name__)


def setup_logging(debug: bool = False) -> None:
    level = logging.DEBUG if debug else logging.INFO
    logging.basicConfig(level=level, format="%(asctime)s %(levelname)s %(name)s: %(message)s")


def signal_handler(signum, _frame):
    logger.info("Received signal %s, shutting down", signum)
    sys.exit(0)


def parse_arguments(argv: Optional[Sequence[str]] = None):
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
    args = parse_arguments()
    setup_logging(args.debug)
    signal.signal(signal.SIGINT, signal_handler)
    if hasattr(signal, "SIGTERM"):
        signal.signal(signal.SIGTERM, signal_handler)
    logger.info("Starting BlindAid (mode=%s, camera=%s)", args.start_mode, args.camera)
    try:
        from blindaid.controller import ModeController

        controller = ModeController(
            camera_index=args.camera,
            audio_enabled=args.audio,
            initial_mode=args.start_mode,
        )
        controller.run()
    except Exception:  # noqa: BLE001
        logger.exception("Fatal error")
        return 1
    logger.info("BlindAid stopped")
    return 0


if __name__ == "__main__":
    sys.exit(main())
