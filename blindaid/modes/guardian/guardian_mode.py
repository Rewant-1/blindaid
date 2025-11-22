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
    Guardian Mode (The Smart Default).
    - Monitors Depth Map.
    - Idle State (Sitting): Low power, silence.
    - Active State (Walking): Real-time safety, warns if obstacle < 1m.
    """

    STATE_IDLE = "Sitting (Idle)"
    STATE_ACTIVE = "Walking (Active)"

    def __init__(self, audio_enabled: bool = True):
        self.audio_enabled = audio_enabled
        self.depth_analyzer: Optional[DepthAnalyzer] = None
        
        # State
        self.current_state = self.STATE_IDLE
        self.frame_counter = 0
        
        # Motion Detection
        self.prev_gray_frame = None
        self.motion_threshold = 1000000  # Adjusted for sum of pixels. 
        # If 640x480, max diff is 255 * 307200. 
        # Let's use a simpler metric: percentage of changed pixels.
        self.last_motion_time = time.time()
        self.idle_timeout = 3.0  # Seconds of no motion to switch to IDLE
        
        # Safety
        self.last_warning_time = 0.0
        self.warning_cooldown = 2.0
        
        # FPS Control
        self.idle_process_interval = 10  # Process every 10th frame
        self.active_process_interval = 30   # Process every 15th frame (lower frequency to reduce GPU load)

    def _ensure_depth_analyzer(self) -> DepthAnalyzer:
        if self.depth_analyzer is None:
            self.depth_analyzer = DepthAnalyzer()
        return self.depth_analyzer

    def _detect_motion(self, frame: np.ndarray) -> bool:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (21, 21), 0)
        
        if self.prev_gray_frame is None:
            self.prev_gray_frame = gray
            return False
            
        frame_delta = cv2.absdiff(self.prev_gray_frame, gray)
        thresh = cv2.threshold(frame_delta, 25, 255, cv2.THRESH_BINARY)[1]
        
        # Count non-zero pixels
        changed_pixels = cv2.countNonZero(thresh)
        total_pixels = frame.shape[0] * frame.shape[1]
        change_ratio = changed_pixels / total_pixels
        
        self.prev_gray_frame = gray
        
        # If more than 1% of the screen changed, it's motion
        return change_ratio > 0.01

    def process_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, List[str], List[str]]:
        """
        Process frame based on state.
        Returns: display_frame, info_lines, speech_messages
        """
        self.frame_counter += 1
        now = time.time()
        info_lines = []
        speech_messages = []
        display_frame = frame.copy()

        # 1. Motion Detection (Always run to switch states)
        is_moving = self._detect_motion(frame)
        
        if is_moving:
            self.last_motion_time = now
            if self.current_state == self.STATE_IDLE:
                self.current_state = self.STATE_ACTIVE
        else:
            if self.current_state == self.STATE_ACTIVE and (now - self.last_motion_time > self.idle_timeout):
                self.current_state = self.STATE_IDLE

        # 2. Determine if we should process depth
        interval = self.active_process_interval if self.current_state == self.STATE_ACTIVE else self.idle_process_interval
        should_process = (self.frame_counter % interval) == 0

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

        info_lines.append(f"State: {self.current_state}")
        
        return display_frame, info_lines, speech_messages

    def on_enter(self):
        self.frame_counter = 0
        self.current_state = self.STATE_IDLE
        logger.info("Entering Guardian Mode")

    def on_exit(self):
        logger.info("Exiting Guardian Mode")
