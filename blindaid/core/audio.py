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
    """Thread-safe audio player supporting pyttsx3 (offline) and gTTS (online)."""

    def __init__(self, rate: int = 150, volume: float = 0.9, use_online: bool = False):
        self.rate = rate
        self.volume = volume
        self.use_online = use_online
        self.queue: Queue[str | None] = Queue(maxsize=10)
        self.worker_thread = Thread(target=self._worker, daemon=True)
        self.worker_thread.start()
        logger.info("Audio player initialized (Online: %s)", self.use_online)

    def _worker(self):
        """Worker thread that processes audio queue."""
        # Try initializing pyttsx3 first if offline
        engine = None
        if not self.use_online:
            try:
                import pyttsx3
                engine = pyttsx3.init()
                engine.setProperty("rate", self.rate)
                engine.setProperty("volume", self.volume)
            except ImportError:
                logger.warning("pyttsx3 not found, falling back to gTTS if available")
                self.use_online = True
            except Exception as exc:
                logger.warning("pyttsx3 init failed: %s. Falling back to gTTS.", exc)
                self.use_online = True

        while not shutdown_event.is_set():
            try:
                message = self.queue.get(timeout=0.5)
                if message is None:  # Poison pill
                    break
                
                logger.debug("Playing audio: %s", message)
                
                if self.use_online:
                    self._speak_gtts(message)
                else:
                    try:
                        # Re-init engine properties just in case
                        if engine:
                            engine.say(message)
                            engine.runAndWait()
                    except Exception as exc:
                        logger.error("pyttsx3 error: %s. Switching to gTTS.", exc)
                        self.use_online = True
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
                pygame.mixer.init()

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
                
            pygame.mixer.music.unload()
            os.remove(temp_filename)
            
        except ImportError:
            logger.error("gTTS or pygame not installed. Cannot play audio.")
        except Exception as exc:
            logger.error("gTTS playback failed: %s", exc)

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
