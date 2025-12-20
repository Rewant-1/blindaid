"""TTS using gTTS + pygame. pyttsx3 was crashing so switched to this."""
from __future__ import annotations

import logging
import os
import tempfile
import time
from pathlib import Path
from queue import Empty, Full, Queue
from threading import Event, Thread

os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")

logger = logging.getLogger(__name__)

class AudioPlayer:
    """Plays TTS in background thread so video doesnt freeze."""

    def __init__(self, rate: int = 150, volume: float = 0.9, use_online: bool = False):
        self.rate = rate
        self.volume = volume
        self.use_online = use_online
        self._pygame_initialized = False
        self.queue: Queue[str | None] = Queue(maxsize=10)
        self._stop = Event()
        self.worker_thread = Thread(target=self._worker, daemon=True)
        self.worker_thread.start()
        logger.info("Audio player ready (online=%s)", self.use_online)

    def _worker(self):
        self.use_online = True

        while not self._stop.is_set():
            try:
                message = self.queue.get(timeout=0.5)
                if message is None:  # Poison pill
                    self.queue.task_done()
                    break
                
                logger.debug("Playing audio: %s", message)
                
                self._ensure_pygame()
                self._speak_gtts(message)

                self.queue.task_done()
            except Empty:
                continue
            except Exception as exc:  # noqa: BLE001
                if not self._stop.is_set():
                    logger.exception("Audio playback error: %s", exc)

    def _speak_gtts(self, text: str):
        temp_path: Path | None = None
        try:
            from gtts import gTTS
            import pygame

            if not pygame.mixer.get_init():
                pygame.mixer.init(frequency=22050, size=-16, channels=2, buffer=4096)
            self._pygame_initialized = True

            tts = gTTS(text=text, lang="en")
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as fp:
                temp_path = Path(fp.name)

            tts.save(temp_path.as_posix())
            pygame.mixer.music.load(temp_path.as_posix())
            pygame.mixer.music.play()
            while pygame.mixer.music.get_busy():
                if self._stop.is_set():
                    pygame.mixer.music.stop()
                    break
                time.sleep(0.1)

            try:
                pygame.mixer.music.unload()
            except Exception:
                try:
                    pygame.mixer.music.stop()
                except Exception:
                    pass
        except ImportError:
            logger.error("gTTS or pygame not installed. Cannot play audio.")
        except Exception as exc:
            logger.error("gTTS playback failed: %s", exc)
        finally:
            if temp_path is not None:
                temp_path.unlink(missing_ok=True)
        return

    def _ensure_pygame(self):
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
        try:
            self.queue.put_nowait(message)
        except Full:
            logger.warning("Audio queue full, skipping message")

    def shutdown(self):
        self._stop.set()
        try:
            self.queue.put_nowait(None)
        except Full:
            logger.debug("Audio queue full during shutdown; waiting for worker")
            self.queue.put(None)
        self.worker_thread.join(timeout=2.0)
        if self._pygame_initialized:
            try:
                import pygame

                try:
                    pygame.mixer.quit()
                except Exception:
                    pass
            except Exception:
                pass
