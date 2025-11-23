"""
Audio utilities for text-to-speech and audio feedback.
"""
import logging
import os
import tempfile
import time
from queue import Queue, Empty
from threading import Thread, Event

logger = logging.getLogger(__name__)

# Global shutdown event
shutdown_event = Event()


class AudioPlayer:
    """Thread-safe audio player that uses Google TTS (gTTS) + pygame for playback.

    We intentionally removed pyttsx3 to avoid SAPI driver problems on Windows and
    force use of online gTTS as requested.
    """

    def __init__(self, rate: int = 150, volume: float = 0.9, use_online: bool = False):
        self.rate = rate
        self.volume = volume
        self.use_online = use_online
        self._pytt_engine = None
        self._pygame_initialized = False
        self.queue: Queue[str | None] = Queue(maxsize=10)
        self.worker_thread = Thread(target=self._worker, daemon=True)
        self.worker_thread.start()
        logger.info("Audio player initialized (Online: %s)", self.use_online)

    def _worker(self):
        """Worker thread that processes audio queue."""
        # We always use gTTS/pygame in the worker thread
        self.use_online = True

        while not shutdown_event.is_set():
            try:
                message = self.queue.get(timeout=0.5)
                if message is None:  # Poison pill
                    break
                
                logger.debug("Playing audio: %s", message)
                
                if self.use_online:
                    # Always use the online TTS path
                    self._ensure_pygame()
                    self._speak_gtts(message)
                else:
                    # Should never happen (we force gTTS), but fall back just in case
                    self._ensure_pygame()
                    self._speak_gtts(message)

                self.queue.task_done()
            except Empty:
                continue
            except Exception as exc:  # noqa: BLE001
                if not shutdown_event.is_set():
                    logger.exception("Audio playback error: %s", exc)

    def _speak_gtts(self, text: str):
        """Speak using Google Text-to-Speech and pygame."""
        try:
            from gtts import gTTS
            import pygame
            
            # Initialize pygame mixer if not already
            if not pygame.mixer.get_init():
                # Use small buffer on low power machines
                pygame.mixer.init(frequency=22050, size=-16, channels=2, buffer=4096)
            self._pygame_initialized = True

            tts = gTTS(text=text, lang='en')
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as fp:
                temp_filename = fp.name
                
            tts.save(temp_filename)
            
            pygame.mixer.music.load(temp_filename)
            pygame.mixer.music.play()
            while pygame.mixer.music.get_busy():
                if shutdown_event.is_set():
                    pygame.mixer.music.stop()
                    break
                time.sleep(0.1)
                
            try:
                pygame.mixer.music.unload()
            except Exception:
                # Older pygame lacks unload; stop and load a blank
                try:
                    pygame.mixer.music.stop()
                except Exception:
                    pass
            os.remove(temp_filename)
            
        except ImportError:
            logger.error("gTTS or pygame not installed. Cannot play audio.")
        except Exception as exc:
            logger.error("gTTS playback failed: %s", exc)

    def _ensure_pygame(self):
        """Ensure pygame mixer is initialized (idempotent)."""
        if self._pygame_initialized:
            return
        try:
            import pygame

            if not pygame.mixer.get_init():
                pygame.mixer.init(frequency=22050, size=-16, channels=2, buffer=4096)
            self._pygame_initialized = True
        except Exception as exc:  # noqa: BLE001
            logger.exception("Failed to initialize pygame mixer: %s", exc)
            raise

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
        # Cleanup mixer if we used it
        if self._pygame_initialized:
            try:
                import pygame

                try:
                    pygame.mixer.quit()
                except Exception:
                    pass
            except Exception:
                pass
