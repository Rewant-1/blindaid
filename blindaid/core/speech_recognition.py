"""
Speech recognition utility for voice commands.
"""
import logging
from typing import Optional

logger = logging.getLogger(__name__)


class SpeechListener:
    """Handles microphone input and speech-to-text conversion."""

    def __init__(self):
        self.recognizer = None
        self.microphone = None
        self._available = False
        self._ensure_loaded()

    def _ensure_loaded(self):
        if self._available:
            return
        try:
            import speech_recognition as sr
            self.recognizer = sr.Recognizer()
            self.microphone = sr.Microphone()
            self._available = True
            logger.info("Speech recognition initialized")
        except ImportError as exc:
            logger.warning("SpeechRecognition or PyAudio not installed: %s", exc)
        except Exception as exc:
            logger.error("Failed to initialize microphone: %s", exc)

    def listen_for_command(self, timeout: int = 5) -> Optional[str]:
        """
        Listens for a voice command and returns the transcribed text.
        Returns None if no speech is detected or if there's an error.
        """
        self._ensure_loaded()
        if not self._available:
            return None

        import speech_recognition as sr

        try:
            with self.microphone as source:
                logger.info("Adjusting for ambient noise...")
                self.recognizer.adjust_for_ambient_noise(source, duration=0.5)
                logger.info("Listening for command...")
                try:
                    audio = self.recognizer.listen(source, timeout=timeout, phrase_time_limit=5)
                except sr.WaitTimeoutError:
                    logger.info("Listening timed out (no speech detected)")
                    return None

            logger.info("Recognizing speech...")
            text = self.recognizer.recognize_google(audio)
            logger.info("Heard: %s", text)
            return text

        except sr.UnknownValueError:
            logger.info("Speech was unintelligible")
            return None
        except sr.RequestError as exc:
            logger.error("Could not request results from speech service: %s", exc)
            return None
        except Exception as exc:
            logger.exception("Error during speech recognition: %s", exc)
            return None
