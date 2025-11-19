"""
Audio utilities for text-to-speech and audio feedback.
"""
import logging
from queue import Queue, Empty
from threading import Thread, Event

logger = logging.getLogger(__name__)

# Global shutdown event
shutdown_event = Event()


class AudioPlayer:
    """Thread-safe audio player using pyttsx3 for offline TTS."""

    def __init__(self, rate: int = 150, volume: float = 0.9):
        self.rate = rate
        self.volume = volume
        self.queue: Queue[str | None] = Queue(maxsize=10)
        self.worker_thread = Thread(target=self._worker, daemon=True)
        self.worker_thread.start()
        logger.info("Audio player initialized with pyttsx3")

    def _worker(self):
        """Worker thread that processes audio queue."""
        try:
            import pyttsx3

            engine = pyttsx3.init()
            engine.setProperty("rate", self.rate)
            engine.setProperty("volume", self.volume)

            while not shutdown_event.is_set():
                try:
                    message = self.queue.get(timeout=0.5)
                    if message is None:  # Poison pill
                        break
                    logger.debug("Playing audio: %s", message)
                    engine.say(message)
                    engine.runAndWait()
                    self.queue.task_done()
                except Empty:
                    continue
                except Exception as exc:  # noqa: BLE001
                    if not shutdown_event.is_set():
                        logger.exception("Audio playback error: %s", exc)
        except ImportError:
            logger.error("pyttsx3 not available - audio disabled")

    def speak(self, message: str):
        """Add message to audio queue (non-blocking)."""
        try:
            self.queue.put_nowait(message)
        except Exception:  # noqa: BLE001
            logger.warning("Audio queue full, skipping message")

    def shutdown(self):
        """Gracefully shutdown audio player."""
        self.queue.put(None)
        self.worker_thread.join(timeout=2.0)
