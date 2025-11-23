"""Shared interface for optional standalone modes."""
from abc import ABC, abstractmethod
import logging


class BaseMode(ABC):
    def __init__(self, camera_index: int = 0, audio_enabled: bool = True):
        self.camera_index = camera_index
        self.audio_enabled = audio_enabled
        self.logger = logging.getLogger(self.__class__.__name__)

    @abstractmethod
    def run(self) -> None:  # pragma: no cover - interface only
        raise NotImplementedError

    def cleanup(self) -> None:
        return
