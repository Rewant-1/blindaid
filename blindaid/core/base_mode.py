"""
Base class for all mode implementations.
"""
from abc import ABC, abstractmethod
import logging

logger = logging.getLogger(__name__)


class BaseMode(ABC):
    """Abstract base class for all assistive modes."""
    
    def __init__(self, camera_index=0, audio_enabled=True):
        self.camera_index = camera_index
        self.audio_enabled = audio_enabled
        self.logger = logging.getLogger(self.__class__.__name__)
    
    @abstractmethod
    def run(self):
        """Run the mode. Must be implemented by subclasses."""
        pass
    
    def cleanup(self):
        """Optional cleanup method."""
        pass
