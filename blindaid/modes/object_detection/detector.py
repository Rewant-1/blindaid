"""
Object Detection Service using YOLO.
"""
import cv2
import time
from ultralytics import YOLO
from blindaid.core.base_mode import BaseMode
from blindaid.core.audio import AudioPlayer
from blindaid.core import config


class ObjectDetectionService(BaseMode):
    """Object detection and location identification service."""
    
    def __init__(self, camera_index=0, model_path=None, confidence=0.6, audio_enabled=True):
        super().__init__(camera_index, audio_enabled)
        
        self.model_path = model_path or config.OBJECT_DETECTION_MODEL
        self.confidence = confidence
        self.audio_player = AudioPlayer() if audio_enabled else None
        self.frame_count = 0
        self.fps = 0
        self.fps_start_time = time.time()
        
        # Load YOLO model
        self.logger.info(f"Loading YOLO model from: {self.model_path}")
        self.model = YOLO(str(self.model_path))
        self.logger.info("Object detection model loaded successfully")
    
    def _calculate_fps(self):
        """Calculate FPS."""
        self.frame_count += 1
        if self.frame_count % 30 == 0:
            elapsed = time.time() - self.fps_start_time
            self.fps = 30 / elapsed if elapsed > 0 else 0
            self.fps_start_time = time.time()
    
    def _get_object_position(self, x_center, frame_width):
        """Determine position of object (left/center/right)."""
        if x_center < frame_width / 3:
            return "left"
        elif x_center < 2 * frame_width / 3:
            return "center"
        else:
            return "right"
    
    def _announce_detections(self, results, frame_width):
        """Announce detected objects and their positions."""
        if not self.audio_enabled or not results:
            return
        
        detected_objects = []
        for r in results:
            boxes = r.boxes
            for box in boxes:
                cls = int(box.cls[0])
                conf = float(box.conf[0])
                if conf >= self.confidence:
                    # Get object name
                    obj_name = self.model.names[cls]
                    # Get position
                    x1, y1, x2, y2 = box.xyxy[0]
                    x_center = (x1 + x2) / 2
                    position = self._get_object_position(x_center, frame_width)
                    detected_objects.append(f"{obj_name} on the {position}")
        
        if detected_objects:
            message = ", ".join(detected_objects[:3])  # Limit to 3 objects
            self.logger.info(f"Announcing: {message}")
            self.audio_player.speak(message)
    
    def run(self):
        """Run object detection mode."""
        cap = cv2.VideoCapture(self.camera_index)
        
        if not cap.isOpened():
            self.logger.error(f"Failed to open camera {self.camera_index}")
            return
        
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.logger.info(f"Camera opened: {frame_width}x{frame_height}")
        self.logger.info(f"Object Detection Mode - Press 'q' to quit, 's' to speak detections")
        
        last_announcement = 0
        announcement_cooldown = 3  # seconds
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    self.logger.warning("Failed to read frame")
                    break
                
                self._calculate_fps()
                
                # Run detection
                results = self.model(frame, conf=self.confidence, verbose=False)
                
                # Draw results
                annotated_frame = results[0].plot()
                
                # Display FPS
                fps_text = f"FPS: {self.fps:.1f}" if self.fps > 0 else "FPS: --"
                cv2.putText(annotated_frame, fps_text, (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                # Display mode
                cv2.putText(annotated_frame, "Object Detection Mode", (10, 60),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
                
                cv2.imshow('BlindAid - Object Detection', annotated_frame)
                
                # Auto-announce every few seconds
                current_time = time.time()
                if current_time - last_announcement > announcement_cooldown:
                    self._announce_detections(results, frame_width)
                    last_announcement = current_time
                
                # Handle key presses
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    self.logger.info("User quit")
                    break
                elif key == ord('s'):
                    self._announce_detections(results, frame_width)
        
        except KeyboardInterrupt:
            self.logger.info("Interrupted by user")
        except Exception as e:
            self.logger.exception(f"Error in object detection: {e}")
        finally:
            self.cleanup()
            cap.release()
            cv2.destroyAllWindows()
    
    def cleanup(self):
        """Cleanup resources."""
        if self.audio_player:
            self.audio_player.shutdown()
