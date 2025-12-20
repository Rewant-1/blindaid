"""Navigation mode - warns about obstacles using depth."""
from __future__ import annotations
import time
import logging
import cv2
import numpy as np
from blindaid.core.depth import DepthAnalyzer

logger = logging.getLogger(__name__)

class GuardianMode:
    def __init__(self, audio_enabled: bool = True):
        self.audio_enabled = audio_enabled
        self.depth_analyzer = None
        self.frame_counter = 0
        self.last_warning_time = 0.0
        self.warning_cooldown = 2.5  # Thoda gap rakhenge taaki irritate na kare

    def _ensure_depth_analyzer(self):
        if self.depth_analyzer is None:
            self.depth_analyzer = DepthAnalyzer()
        return self.depth_analyzer

    def get_distance_label(self, depth_val):
        # Depth map 0 (far) to 1 (close) hota hai usually.
        # Yeh rough estimation hai:
        if depth_val > 0.8: return "Very Close (< 0.5m)"
        if depth_val > 0.6: return "Close (1m)"
        if depth_val > 0.4: return "Nearby (2m)"
        return "Safe"

    def process_frame(self, frame: np.ndarray):
        self.frame_counter += 1
        info_lines = ["Mode: Smart Navigation"]
        speech_messages = []
        display_frame = frame.copy()

        # Har 15th frame pe check karega (Lag kam karne ke liye)
        if self.frame_counter % 15 == 0:
            try:
                analyzer = self._ensure_depth_analyzer()
                depth_map = analyzer.compute_depth(frame)
                
                h, w = depth_map.shape
                # Screen ko 3 parts mein divide kar rahe hain
                left_part = depth_map[:, :w//3]
                center_part = depth_map[:, w//3:2*w//3]
                right_part = depth_map[:, 2*w//3:]

                # Har part ka average depth nikalo
                l_val = np.mean(left_part)
                c_val = np.mean(center_part)
                r_val = np.mean(right_part)

                # Logic: Kahan sabse zyada khatra hai?
                msg = ""
                if c_val > 0.7:  # Center mein obstacle
                    msg = "Stop! Obstacle Ahead."
                elif l_val > 0.75:
                    msg = "Obstacle on Left."
                elif r_val > 0.75:
                    msg = "Obstacle on Right."
                
                # Agar koi warning hai aur cooldown khatam ho gaya hai
                now = time.time()
                if msg and (now - self.last_warning_time > self.warning_cooldown):
                    speech_messages.append(msg)
                    self.last_warning_time = now
                    info_lines.append(f"WARNING: {msg}")

                # Visualization for Professor (Cool Factor)
                # Screen pe depth ka heatmap overlay kar
                colored_depth = cv2.applyColorMap((depth_map * 255).astype(np.uint8), cv2.COLORMAP_MAGMA)
                
                # Overlay: Original frame pe thoda transparent depth dikhao
                display_frame = cv2.addWeighted(frame, 0.7, colored_depth, 0.3, 0)
                
                # Text Stats
                cv2.putText(display_frame, f"L:{l_val:.2f} C:{c_val:.2f} R:{r_val:.2f}", (10, h-50), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            except Exception as e:
                logger.error(f"Depth error: {e}")

        return display_frame, info_lines, speech_messages

    def on_enter(self):
        logger.info("Smart Nav Active")
    
    def on_exit(self):
        pass