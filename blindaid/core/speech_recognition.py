"""Voice input using Google Speech API."""
import logging
from typing import Optional

logger = logging.getLogger(__name__)


class SpeechListener:
    def __init__(self):
        self.recognizer = None
        self.microphone = None
        self._available = False
        self._ensure_loaded()

    def _ensure_loaded(self) -> None:
        if self._available:
            return
        try:
            import speech_recognition as sr

            self.recognizer = sr.Recognizer()
            self.microphone = sr.Microphone()
            self._available = True
            logger.debug("Speech recognition stack initialized")
        except ImportError as exc:
            logger.warning("SpeechRecognition or PyAudio missing: %s", exc)
        except Exception as exc:
            logger.error("Microphone initialisation failed: %s", exc)

    def listen_for_command(self, timeout: int = 5) -> Optional[str]:
        self._ensure_loaded()
        if not self._available:
            return None

        import speech_recognition as sr

        try:
            with self.microphone as source:
                self.recognizer.adjust_for_ambient_noise(source, duration=0.5)
                try:
                    audio = self.recognizer.listen(source, timeout=timeout, phrase_time_limit=5)
                except sr.WaitTimeoutError:
                    logger.debug("Speech timeout reached")
                    return None

            text = self.recognizer.recognize_google(audio)
            logger.info("Heard: %s", text)
            return text
        except sr.UnknownValueError:
            logger.debug("Speech unintelligible")
            return None
        except sr.RequestError as exc:
            logger.error("Speech service request failed: %s", exc)
            return None
        except Exception as exc:
            logger.exception("Speech recognition error: %s", exc)
            return None
