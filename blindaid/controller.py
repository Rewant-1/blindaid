"""Unified keyboard-controlled orchestrator for BlindAid modes."""
from __future__ import annotations

import logging
import time
import threading
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence

try:
    import cv2
except ModuleNotFoundError as exc:
    raise RuntimeError(
        "OpenCV is required to run the ModeController."
    ) from exc

from blindaid.core import config
from blindaid.core.audio import AudioPlayer
from blindaid.core.caption import VisualAssistant
from blindaid.core.depth import DepthAnalyzer
from blindaid.core.speech_recognition import SpeechListener
from blindaid.modes.guardian.guardian_mode import GuardianMode
from blindaid.modes.ocr.reading_mode import ReadingMode
from blindaid.modes.people.people_mode import PeopleMode


logger = logging.getLogger(__name__)


@dataclass
class OverlayMessage:
    """UI helper representing a transient on-screen message."""
    text: str
    expires_at: float


class ModeController:
    """Main event loop that multiplexes between modes with hotkeys."""

    WINDOW_NAME = "BlindAid"

    def __init__(
        self,
        camera_index: Optional[int] = None,
        audio_enabled: Optional[bool] = None,
        initial_mode: str | None = None,
    ):
        self.camera_index = camera_index if camera_index is not None else config.DEFAULT_CAMERA_INDEX
        default_audio = config.AUDIO_ENABLED
        self.audio_enabled = default_audio if audio_enabled is None else (audio_enabled and default_audio)

        self.audio_player: Optional[AudioPlayer] = None
        if self.audio_enabled:
            try:
                # Always use online TTS (gTTS) as configured
                self.audio_player = AudioPlayer(
                    rate=config.TTS_RATE, volume=config.TTS_VOLUME, use_online=getattr(config, 'TTS_FORCE_ONLINE', True)
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning("Audio player initialization failed: %s", exc)
                self.audio_player = None

        self._mode_factories: Dict[str, Callable[[], object]] = {
            "sitting": lambda: None,  # Default idle mode - no processing
            "guardian": lambda: GuardianMode(audio_enabled=self.audio_enabled),
            "reading": lambda: ReadingMode(
                audio_enabled=self.audio_enabled,
                language=config.OCR_LANGUAGE,
            ),
            "people": lambda: PeopleMode(audio_enabled=self.audio_enabled),
        }
        self._mode_instances: Dict[str, object] = {}
        self.mode_labels: Dict[str, str] = {
            "sitting": "Sitting (Idle)",
            "guardian": "Walking",
            "reading": "Reading",
            "people": "People",
        }
        
        # Default to sitting mode if not specified or unknown
        requested_mode = (initial_mode or "sitting").lower()
        if requested_mode not in self._mode_factories:
            logger.warning("Unknown initial mode '%s', defaulting to sitting", requested_mode)
            requested_mode = "sitting"
            
        self.current_mode_key = requested_mode
        self.previous_mode_key = requested_mode

        self.visual_assistant: Optional[VisualAssistant] = None
        self.speech_listener: Optional[SpeechListener] = None
        self.depth_analyzer: Optional[DepthAnalyzer] = None

        self.overlays: List[OverlayMessage] = []
        self.fps_counter = 0
        self.fps_last_time = time.time()
        self.fps_value = 0.0
        
        # Background model preloader
        self._preload_thread: Optional[threading.Thread] = None
        self._preload_running = False

    # ------------------------------------------------------------------
    def _start_background_preload(self) -> None:
        """Start background thread to preload models gradually."""
        if self._preload_running:
            return
        
        def preload_worker():
            """Gradually preload models in background without blocking main thread."""
            try:
                logger.info("Background preload started")
                time.sleep(2)  # Let the app start first
                
                # Preload in order of likely usage
                preload_order = ["guardian", "reading", "people"]
                
                for mode_key in preload_order:
                    if not self._preload_running:
                        break
                    try:
                        logger.info(f"Preloading {mode_key} mode...")
                        mode = self._get_mode(mode_key)
                        # Call _ensure_loaded if available to trigger model loading
                        if hasattr(mode, '_ensure_loaded'):
                            mode._ensure_loaded()
                        elif hasattr(mode, '_ensure_ocr'):
                            mode._ensure_ocr()
                        time.sleep(1)  # Pause between loads to keep UI responsive
                    except Exception as e:
                        logger.debug(f"Preload failed for {mode_key}: {e}")
                
                # Preload visual assistant for caption/VQA
                if self._preload_running:
                    try:
                        logger.info("Preloading visual assistant...")
                        if self.visual_assistant is None:
                            self.visual_assistant = VisualAssistant(device=getattr(config, 'CAPTION_DEVICE', 'cpu'))
                        # Trigger lazy load of caption model (more commonly used)
                        self.visual_assistant._ensure_caption_model()
                    except Exception as e:
                        logger.debug(f"Preload failed for visual assistant: {e}")
                
                logger.info("Background preload complete")
            except Exception as e:
                logger.error(f"Background preload error: {e}")
        
        self._preload_running = True
        self._preload_thread = threading.Thread(target=preload_worker, daemon=True)
        self._preload_thread.start()
    
    def _get_mode(self, key: str) -> object:
        factory = self._mode_factories.get(key)
        if factory is None:
            raise KeyError(key)
        if key not in self._mode_instances:
            logger.info("Lazy-loading %s mode", key)
            self._mode_instances[key] = factory()
        return self._mode_instances[key]

    def _switch_mode(self, target_key: str) -> None:
        if target_key == self.current_mode_key:
            return
        if target_key not in self._mode_factories:
            logger.warning("Unknown mode key requested: %s", target_key)
            return

        logger.info("Switching mode: %s -> %s", self.current_mode_key, target_key)
        current = self._get_mode(self.current_mode_key)
        if hasattr(current, "on_exit"):
            try:
                current.on_exit()
            except Exception as exc:  # noqa: BLE001
                logger.debug("Mode on_exit failed: %s", exc)

        self.current_mode_key = target_key

        new_mode = self._get_mode(self.current_mode_key)
        if hasattr(new_mode, "on_enter"):
            try:
                new_mode.on_enter()
            except Exception as exc:  # noqa: BLE001
                logger.debug("Mode on_enter failed: %s", exc)

        self._add_overlay(f"Switched to {self.mode_labels[self.current_mode_key]} mode", duration=2.5)

    def _add_overlay(self, text: str, duration: float = 4.0) -> None:
        expiry = time.time() + duration
        self.overlays.append(OverlayMessage(text=text, expires_at=expiry))

    def _active_overlays(self) -> List[str]:
        now = time.time()
        active: List[OverlayMessage] = []
        messages: List[str] = []
        for overlay in self.overlays:
            if overlay.expires_at > now:
                active.append(overlay)
                messages.append(overlay.text)
        self.overlays = active
        return messages

    def _speak_messages(self, messages: Sequence[str]) -> None:
        if not messages:
            return
        for message in messages:
            if not message:
                continue
            if self.audio_player is not None:
                self.audio_player.speak(message)
            else:
                logger.info("[Speech] %s", message)

    def _update_fps(self) -> None:
        self.fps_counter += 1
        if self.fps_counter >= 20:
            now = time.time()
            elapsed = now - self.fps_last_time
            if elapsed > 0:
                self.fps_value = self.fps_counter / elapsed
            self.fps_counter = 0
            self.fps_last_time = now

    def _ensure_visual_assistant(self) -> VisualAssistant:
        if self.visual_assistant is None:
            self.visual_assistant = VisualAssistant()
        return self.visual_assistant

    def _ensure_speech_listener(self) -> SpeechListener:
        if self.speech_listener is None:
            self.speech_listener = SpeechListener()
        return self.speech_listener

    def _handle_caption_request(self, frame) -> None:
        try:
            self._add_overlay("Analyzing scene...", duration=2.0)
            # Force a redraw to show "Analyzing..."
            cv2.waitKey(1)
            
            assistant = self._ensure_visual_assistant()
            caption = assistant.generate_caption(frame)
            if caption:
                self._add_overlay(f"Caption: {caption}", duration=6.0)
                self._speak_messages([caption])
                logger.info("Caption: %s", caption)
            else:
                self._add_overlay("Caption: (no description)", duration=3.0)
        except Exception as exc:  # noqa: BLE001
            logger.exception("Caption generation failed: %s", exc)
            self._add_overlay("Caption error - see logs", duration=3.0)

    def _handle_vqa_request(self, frame) -> None:
        try:
            listener = self._ensure_speech_listener()
            
            self._add_overlay("Listening... (Speak now)", duration=5.0)
            self._speak_messages(["Listening..."])
            cv2.waitKey(1)  # Update UI
            
            # Pause briefly to let TTS start/finish "Listening"
            time.sleep(0.5)
            
            question = listener.listen_for_command(timeout=5)
            
            if not question:
                self._add_overlay("No question heard.", duration=2.0)
                self._speak_messages(["I didn't hear a question."])
                return

            self._add_overlay(f"Q: {question}", duration=4.0)
            self._speak_messages([f"You asked: {question}"])
            cv2.waitKey(1)

            self._add_overlay("Thinking...", duration=2.0)
            assistant = self._ensure_visual_assistant()
            answer = assistant.answer_question(frame, question)
            
            if answer:
                self._add_overlay(f"A: {answer}", duration=6.0)
                self._speak_messages([answer])
                logger.info("VQA: Q='%s' A='%s'", question, answer)
            else:
                self._add_overlay("Could not answer.", duration=2.0)
                self._speak_messages(["I couldn't find an answer."])

        except Exception as exc:
            logger.exception("VQA failed: %s", exc)
            self._add_overlay("Error processing question", duration=3.0)
            self._speak_messages(["Sorry, I encountered an error."])

    def _draw_overlay_text(self, frame, info_lines: Sequence[str], extra_lines: Sequence[str]) -> None:
        h, w = frame.shape[:2]
        header_y = 20
        mode_label = self.mode_labels.get(self.current_mode_key, self.current_mode_key)
        cv2.putText(
            frame,
            f"M:{mode_label}",
            (5, header_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (0, 255, 255),
            1,
        )

        fps_text = f"FPS:{self.fps_value:.0f}" if self.fps_value else "FPS:--"
        cv2.putText(
            frame,
            fps_text,
            (w - 60, header_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (0, 255, 0),
            1,
        )

        lines_to_draw: List[str] = list(info_lines)
        for overlay in extra_lines:
            lines_to_draw.append(overlay)

        # Draw the static hint text at the very bottom
        bottom_y = h - 5
        cv2.putText(
            frame,
            config.SCENE_HINT_TEXT,
            (5, bottom_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.35,
            (200, 200, 200),
            1,
        )

        # Draw dynamic messages above the hint text
        bottom_y -= 18
        for line in reversed(lines_to_draw[-6:]):  # limit clutter
            cv2.putText(
                frame,
                line,
                (5, bottom_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                (255, 255, 255),
                1,
            )
            bottom_y -= 16

    # ------------------------------------------------------------------
    def run(self) -> None:
        logger.info("Starting BlindAid controller (camera %s)", self.camera_index)
        cv2.namedWindow(self.WINDOW_NAME, cv2.WINDOW_NORMAL)

        capture = cv2.VideoCapture(self.camera_index)
        if not capture.isOpened():
            logger.error("Unable to open camera index %s", self.camera_index)
            return

        if config.FRAME_WIDTH:
            capture.set(cv2.CAP_PROP_FRAME_WIDTH, config.FRAME_WIDTH)
        if config.FRAME_HEIGHT:
            capture.set(cv2.CAP_PROP_FRAME_HEIGHT, config.FRAME_HEIGHT)
        
        # Start background model preloading for smooth mode switching
        self._start_background_preload()

        initial_mode = self._get_mode(self.current_mode_key)
        if hasattr(initial_mode, "on_enter"):
            try:
                initial_mode.on_enter()
            except Exception as exc:  # noqa: BLE001
                logger.debug("Initial on_enter failed: %s", exc)

        self._add_overlay("System Ready. Sitting Mode.", duration=3.0)
        self._speak_messages(["BlindAid Online. Ready."])

        try:
            while True:
                ret, frame = capture.read()
                if not ret:
                    logger.warning("Camera frame grab failed, stopping controller")
                    break

                current_mode = self._get_mode(self.current_mode_key)
                
                # Check if PeopleMode finished
                if self.current_mode_key == "people" and hasattr(current_mode, "is_finished") and current_mode.is_finished():
                     self._switch_mode(self.previous_mode_key)
                     # Continue to process frame in the restored mode
                     current_mode = self._get_mode(self.current_mode_key)

                info_lines: Sequence[str]
                speech_messages: List[str]

                # Polymorphic call - all modes should implement process_frame
                if current_mode is None or not hasattr(current_mode, "process_frame"):
                    # Sitting mode - no processing, just show camera feed
                    display_frame = frame.copy()
                    info_lines = ["Sitting Mode - Press 1-5 for features"]
                    speech_messages = []
                else:
                    display_frame, info_lines, speech_messages = current_mode.process_frame(frame)

                self._speak_messages(speech_messages)

                self._update_fps()
                overlay_texts = self._active_overlays()
                self._draw_overlay_text(display_frame, info_lines, overlay_texts)

                cv2.imshow(self.WINDOW_NAME, display_frame)

                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    logger.info("Quit requested by user")
                    break
                
                # Key Bindings (0-5 for modes)
                if key == ord("0"):
                    self._switch_mode("sitting")  # Default idle mode
                elif key == ord("1"):
                    self._switch_mode("guardian")  # Walking mode
                elif key == ord("2"):
                    self._switch_mode("reading")
                elif key == ord("3"):
                    if self.current_mode_key != "people":
                        self.previous_mode_key = self.current_mode_key
                        self._switch_mode("people")
                elif key == ord("4"):
                    self._handle_vqa_request(frame)
                elif key == ord("5"):
                    self._handle_caption_request(frame)
                elif key in (ord("t"), ord("T")):
                    # Quick TTS test to validate audio at runtime
                    self._add_overlay("TTS Test", duration=2.0)
                    self._speak_messages(["Audio check one two three."])

            logger.info("Controller loop exited")
        finally:
            # Stop background preloader
            self._preload_running = False
            if self._preload_thread and self._preload_thread.is_alive():
                self._preload_thread.join(timeout=1.0)
            
            capture.release()
            cv2.destroyAllWindows()
            for mode in self._mode_instances.values():
                if hasattr(mode, "on_exit"):
                    try:
                        mode.on_exit()
                    except Exception:  # noqa: BLE001
                        pass
            if self.audio_player is not None:
                self.audio_player.shutdown()


__all__ = ["ModeController"]
