"""Unified keyboard-controlled orchestrator for BlindAid modes."""
from __future__ import annotations

import logging
import time
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
from blindaid.core.caption import CaptionGenerator
from blindaid.core.depth import DepthAnalyzer
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
                self.audio_player = AudioPlayer(rate=config.TTS_RATE, volume=config.TTS_VOLUME)
            except Exception as exc:  # noqa: BLE001
                logger.warning("Audio player initialization failed: %s", exc)
                self.audio_player = None

        self._mode_factories: Dict[str, Callable[[], object]] = {
            "guardian": lambda: GuardianMode(audio_enabled=self.audio_enabled),
            "reading": lambda: ReadingMode(
                audio_enabled=self.audio_enabled,
                language=config.OCR_LANGUAGE,
            ),
            "people": lambda: PeopleMode(audio_enabled=self.audio_enabled),
        }
        self._mode_instances: Dict[str, object] = {}
        self.mode_labels: Dict[str, str] = {
            "guardian": "Guardian (Safety)",
            "reading": "Reader (Text)",
            "people": "People Scanner",
        }
        
        # Default to guardian if not specified or unknown
        requested_mode = (initial_mode or "guardian").lower()
        if requested_mode not in self._mode_factories:
            logger.warning("Unknown initial mode '%s', defaulting to guardian", requested_mode)
            requested_mode = "guardian"
            
        self.current_mode_key = requested_mode
        self.previous_mode_key = requested_mode

        self.caption_generator: Optional[CaptionGenerator] = None
        self.depth_analyzer: Optional[DepthAnalyzer] = None

        self.overlays: List[OverlayMessage] = []
        self.fps_counter = 0
        self.fps_last_time = time.time()
        self.fps_value = 0.0

    # ------------------------------------------------------------------
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

    def _ensure_caption_generator(self) -> CaptionGenerator:
        if self.caption_generator is None:
            self.caption_generator = CaptionGenerator()
        return self.caption_generator

    def _handle_caption_request(self, frame) -> None:
        try:
            self._add_overlay("Analyzing scene...", duration=2.0)
            # Force a redraw to show "Analyzing..."
            cv2.waitKey(1)
            
            captioner = self._ensure_caption_generator()
            caption = captioner.generate_caption(frame)
            if caption:
                self._add_overlay(f"Caption: {caption}", duration=6.0)
                self._speak_messages([caption])
                logger.info("Caption: %s", caption)
            else:
                self._add_overlay("Caption: (no description)", duration=3.0)
        except Exception as exc:  # noqa: BLE001
            logger.exception("Caption generation failed: %s", exc)
            self._add_overlay("Caption error - see logs", duration=3.0)

    def _draw_overlay_text(self, frame, info_lines: Sequence[str], extra_lines: Sequence[str]) -> None:
        h, w = frame.shape[:2]
        header_y = 30
        mode_label = self.mode_labels.get(self.current_mode_key, self.current_mode_key)
        cv2.putText(
            frame,
            f"Mode: {mode_label}",
            (10, header_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 255),
            2,
        )

        fps_text = f"FPS: {self.fps_value:.1f}" if self.fps_value else "FPS: --"
        cv2.putText(
            frame,
            fps_text,
            (w - 140, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2,
        )

        lines_to_draw: List[str] = list(info_lines)
        for overlay in extra_lines:
            lines_to_draw.append(overlay)

        bottom_y = h - 20
        for line in reversed(lines_to_draw[-6:]):  # limit clutter
            cv2.putText(
                frame,
                line,
                (10, bottom_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2,
            )
            bottom_y -= 22

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

        initial_mode = self._get_mode(self.current_mode_key)
        if hasattr(initial_mode, "on_enter"):
            try:
                initial_mode.on_enter()
            except Exception as exc:  # noqa: BLE001
                logger.debug("Initial on_enter failed: %s", exc)

        self._add_overlay("System Ready. Guardian Mode Active.", duration=3.0)
        self._speak_messages(["System Ready"])

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
                if hasattr(current_mode, "process_frame"):
                    display_frame, info_lines, speech_messages = current_mode.process_frame(frame)
                else:
                    display_frame = frame.copy()
                    info_lines = []
                    speech_messages = []

                self._speak_messages(speech_messages)

                self._update_fps()
                overlay_texts = self._active_overlays()
                self._draw_overlay_text(display_frame, info_lines, overlay_texts)

                cv2.imshow(self.WINDOW_NAME, display_frame)

                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    logger.info("Quit requested by user")
                    break
                
                # Key Bindings
                if key == ord("1"):
                    self._switch_mode("guardian")
                elif key == ord("2"):
                    self._switch_mode("reading")
                elif key == 32: # Space
                    self._handle_caption_request(frame)
                elif key in (ord("p"), ord("P")):
                    if self.current_mode_key != "people":
                        self.previous_mode_key = self.current_mode_key
                        self._switch_mode("people")

            logger.info("Controller loop exited")
        finally:
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
