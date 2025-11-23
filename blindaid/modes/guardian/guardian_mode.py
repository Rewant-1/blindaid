"""Guardian Mode: The Smart Default Safety Mode."""
from __future__ import annotations

import time
import logging
import cv2
import numpy as np
from typing import Tuple, List, Optional
from blindaid.core.depth import DepthAnalyzer
from blindaid.core import config

logger = logging.getLogger(__name__)

class GuardianMode:
    """
    Walking Mode (Renamed from Guardian).
    - Always active (no idle state).
    - Monitors Depth Map continuously.
    - Warns if obstacle < 1m detected.
    """

    def __init__(self, audio_enabled: bool = True):
        self.audio_enabled = audio_enabled
        self.depth_analyzer: Optional[DepthAnalyzer] = None
        
        self.frame_counter = 0
        
        # Safety
        self.last_warning_time = 0.0
        self.warning_cooldown = 2.0
        
        # FPS Control (always active, process every 30 frames)
        self.process_interval = 30

    def _ensure_depth_analyzer(self) -> DepthAnalyzer:
        if self.depth_analyzer is None:
            self.depth_analyzer = DepthAnalyzer()
        return self.depth_analyzer

    def process_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, List[str], List[str]]:
        """
        Process frame - always active walking mode.
        Returns: display_frame, info_lines, speech_messages
        """
        self.frame_counter += 1
        now = time.time()
        info_lines = []
        speech_messages = []
        display_frame = frame.copy()

        # Process depth every N frames
        should_process = (self.frame_counter % self.process_interval) == 0

        if should_process:
            try:
                analyzer = self._ensure_depth_analyzer()
                depth_map = analyzer.compute_depth(frame)
                
                # 3. Safety Check (Center & Bottom regions)
                h, w = depth_map.shape
                # ROI: Center-Bottom (where obstacles like walls/furniture are)
                # x: w/4 to 3w/4 (middle half)
                # y: h/2 to h (bottom half)
                
                roi = depth_map[int(h/2):h, int(w/4):int(3*w/4)]
                
                if roi.size > 0:
                    # Check if significant portion is close
                    # depth > 0.75 means < 1m
                    close_pixels = np.sum(roi > 0.75)
                    roi_pixels = roi.size
                    
                    if (close_pixels / roi_pixels) > 0.1: # If > 10% of ROI is close
                        # Danger!
                        if now - self.last_warning_time > self.warning_cooldown:
                            msg = "Stop"
                            speech_messages.append(msg)
                            self.last_warning_time = now
                            
                        cv2.rectangle(display_frame, (int(w/4), int(h/2)), (int(3*w/4), h), (0, 0, 255), 3)
                        cv2.putText(display_frame, "STOP", (int(w/2)-40, int(h/2)+40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

            except Exception as e:
                logger.error(f"Depth processing failed: {e}")
                info_lines.append("Depth Error")

        info_lines.append("Walking Mode")
        
        return display_frame, info_lines, speech_messages

    def on_enter(self):
        self.frame_counter = 0
        logger.info("Entering Walking Mode (Guardian)")

    def on_exit(self):
        logger.info("Exiting Walking Mode (Guardian)")
