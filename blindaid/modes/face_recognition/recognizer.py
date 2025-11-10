"""
Face Recognition Service with position detection and audio feedback.
"""
import cv2
import time
import os
import numpy as np
import face_recognition
from ultralytics import YOLO
from blindaid.core.base_mode import BaseMode
from blindaid.core.audio import AudioPlayer
from blindaid.core import config


class FaceRecognitionService(BaseMode):
    """Face recognition with position detection service."""
    
    def __init__(self, camera_index=0, known_faces_dir=None, threshold=0.5, audio_enabled=True):
        super().__init__(camera_index, audio_enabled)
        
        self.known_faces_dir = known_faces_dir or config.KNOWN_FACES_DIR
        self.threshold = threshold
        self.audio_player = AudioPlayer() if audio_enabled else None
        
        # Load YOLO face detection model
        self.logger.info(f"Loading YOLO face detection model...")
        self.yolo_model = YOLO(str(config.FACE_RECOGNITION_MODEL))
        
        # Face recognition data
        self.known_face_encodings = []
        self.known_face_names = []
        self.last_spoken = {}
        self.last_detections = []
        
        # Performance settings
        self.scale = config.FACE_FRAME_SCALE
        self.process_every = config.FACE_PROCESS_EVERY_N_FRAMES
        self.overlay_timeout = config.FACE_OVERLAY_TIMEOUT
        self.debounce = config.FACE_DEBOUNCE_SECONDS
        
        self.frame_count = 0
        self.fps_start_time = time.time()
        self.fps = 0
        
        # Load known faces
        self._load_known_faces()
    
    def _load_known_faces(self):
        """Load and encode known faces from directory."""
        if not os.path.isdir(self.known_faces_dir):
            self.logger.warning(f"Known faces directory '{self.known_faces_dir}' not found")
            return
        
        self.logger.info(f"Loading known faces from: {self.known_faces_dir}")
        loaded_count = 0
        
        for person_name in os.listdir(self.known_faces_dir):
            person_dir = os.path.join(self.known_faces_dir, person_name)
            if os.path.isdir(person_dir):
                for image_name in os.listdir(person_dir):
                    image_path = os.path.join(person_dir, image_name)
                    try:
                        image = face_recognition.load_image_file(image_path)
                        encodings = face_recognition.face_encodings(image)
                        if encodings:
                            self.known_face_encodings.append(encodings[0])
                            self.known_face_names.append(person_name)
                            loaded_count += 1
                    except Exception as e:
                        self.logger.error(f"Failed to load {image_path}: {e}")
        
        unique_people = len(set(self.known_face_names))
        self.logger.info(f"Loaded {loaded_count} face encodings for {unique_people} people")
    
    def _calculate_fps(self):
        """Calculate FPS."""
        self.frame_count += 1
        if self.frame_count % 30 == 0:
            elapsed = time.time() - self.fps_start_time
            self.fps = 30 / elapsed if elapsed > 0 else 0
            self.fps_start_time = time.time()
    
    def _get_position(self, left, frame_width):
        """Determine position of face (left/middle/right)."""
        if left < frame_width / 3:
            return "left"
        elif left < 2 * frame_width / 3:
            return "middle"
        else:
            return "right"
    
    def _recognize_face(self, face_encoding):
        """Recognize a face and return (name, confidence)."""
        if not self.known_face_encodings:
            return "Unknown", 0.0
        
        face_distances = face_recognition.face_distance(self.known_face_encodings, face_encoding)
        best_match_index = np.argmin(face_distances)
        best_distance = face_distances[best_match_index]
        confidence = 1.0 - best_distance
        
        if best_distance <= self.threshold:
            name = self.known_face_names[best_match_index]
            return name, confidence
        else:
            return "Unknown", confidence
    
    def _should_announce(self, name):
        """Check if enough time has passed since last announcement."""
        current_time = time.time()
        if name not in self.last_spoken:
            return True
        return (current_time - self.last_spoken[name]) > self.debounce
    
    def _draw_detections(self, frame, detections):
        """Draw bounding boxes and labels for detections."""
        now = time.time()
        for det in detections:
            if now - det.get('timestamp', 0) > self.overlay_timeout:
                continue
            
            left, top, right, bottom = det['left'], det['top'], det['right'], det['bottom']
            name = det.get('name', 'Unknown')
            confidence = det.get('confidence', 0.0)
            position = det.get('position', '')
            
            color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
            cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
            
            label = f"{name} ({position})"
            if name != "Unknown":
                label += f" {confidence*100:.0f}%"
            
            cv2.rectangle(frame, (left, bottom - 35), (right, bottom), color, cv2.FILLED)
            cv2.putText(frame, label, (left + 6, bottom - 6),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    
    def run(self):
        """Run face recognition mode."""
        cap = cv2.VideoCapture(self.camera_index)
        
        if not cap.isOpened():
            self.logger.error(f"Failed to open camera {self.camera_index}")
            return
        
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.logger.info(f"Camera opened: {frame_width}x{frame_height}")
        self.logger.info("Face Recognition Mode - Press 'q' to quit")
        
        process_frame_number = 0
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    self.logger.warning("Failed to read frame")
                    break
                
                self._calculate_fps()
                
                process_frame_number += 1
                if process_frame_number % self.process_every != 0:
                    self._draw_detections(frame, self.last_detections)
                    
                    fps_text = f"FPS: {self.fps:.1f}" if self.fps > 0 else "FPS: --"
                    cv2.putText(frame, fps_text, (10, 30),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    cv2.putText(frame, "Face Recognition Mode", (10, 60),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
                    
                    cv2.imshow('BlindAid - Face Recognition', frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                    continue
                
                # Downscale for processing
                small_frame = cv2.resize(frame, (0, 0), fx=self.scale, fy=self.scale)
                rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
                
                # Use YOLO for face detection
                yolo_results = self.yolo_model(rgb_small_frame, verbose=False)
                yolo_boxes = yolo_results[0].boxes.xyxy.cpu().numpy().astype(int)
                face_locations = [(y1, x2, y2, x1) for (x1, y1, x2, y2) in yolo_boxes]
                
                # Get face encodings
                face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
                
                detections = []
                for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
                    # Scale back coordinates
                    top = int(top / self.scale)
                    right = int(right / self.scale)
                    bottom = int(bottom / self.scale)
                    left = int(left / self.scale)
                    
                    # Recognize face
                    name, confidence = self._recognize_face(face_encoding)
                    position = self._get_position(left, frame_width)
                    
                    detections.append({
                        'left': left, 'top': top, 'right': right, 'bottom': bottom,
                        'name': name, 'confidence': confidence, 'position': position,
                        'timestamp': time.time()
                    })
                    
                    # Announce if recognized
                    if name != "Unknown" and self._should_announce(name):
                        message = f"{name} is on the {position} side."
                        self.logger.info(f"Announcing: {message}")
                        if self.audio_player:
                            self.audio_player.speak(message)
                        self.last_spoken[name] = time.time()
                
                if detections:
                    self.last_detections = detections
                
                # Draw detections
                self._draw_detections(frame, detections if detections else self.last_detections)
                
                # Display FPS and mode
                fps_text = f"FPS: {self.fps:.1f}" if self.fps > 0 else "FPS: --"
                cv2.putText(frame, fps_text, (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(frame, "Face Recognition Mode", (10, 60),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
                
                cv2.imshow('BlindAid - Face Recognition', frame)
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    self.logger.info("User quit")
                    break
        
        except KeyboardInterrupt:
            self.logger.info("Interrupted by user")
        except Exception as e:
            self.logger.exception(f"Error in face recognition: {e}")
        finally:
            self.cleanup()
            cap.release()
            cv2.destroyAllWindows()
    
    def cleanup(self):
        """Cleanup resources."""
        if self.audio_player:
            self.audio_player.shutdown()
