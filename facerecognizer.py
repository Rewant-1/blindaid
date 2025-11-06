"""
Face Recognition System with Position Detection and Audio Feedback
===================================================================
A real-time face recognition system that identifies people from a webcam feed,
detects their position (left/middle/right), and provides audio announcements.

Features:
- Real-time face detection and recognition
- Position detection (left, middle, right)
- Offline text-to-speech audio feedback
- Configurable via command-line arguments
- Performance optimizations (frame scaling, frame skipping)
- Robust error handling and logging
"""

import face_recognition
import cv2
import numpy as np
import os
import logging
import time
import argparse
import signal
import sys
from queue import Queue
from threading import Thread, Event
import pyttsx3
from ultralytics import YOLO


# Module-level logger
logger = logging.getLogger(__name__)

# Global shutdown event
shutdown_event = Event()


class AudioPlayer:
    """Thread-safe audio player using pyttsx3 for offline TTS."""
    
    def __init__(self):
        self.queue = Queue(maxsize=10)
        self.worker_thread = Thread(target=self._worker, daemon=True)
        self.worker_thread.start()
        logger.info("Audio player initialized with pyttsx3")
    
    def _worker(self):
        """Worker thread that processes audio queue."""
        from queue import Empty
        engine = pyttsx3.init()
        # Configure voice properties
        engine.setProperty('rate', 150)  # Speed of speech
        engine.setProperty('volume', 0.9)  # Volume (0.0 to 1.0)
        
        while not shutdown_event.is_set():
            try:
                message = self.queue.get(timeout=0.5)
                if message is None:  # Poison pill
                    break
                logger.debug(f"Playing audio: {message}")
                engine.say(message)
                engine.runAndWait()
                self.queue.task_done()
            except Empty:
                # Timeout is expected, just continue
                continue
            except Exception as e:
                if not shutdown_event.is_set():
                    logger.exception(f"Audio playback error: {e}")
    
    def speak(self, message):
        """Add message to audio queue (non-blocking)."""
        try:
            self.queue.put_nowait(message)
        except:
            logger.warning("Audio queue full, skipping message")
    
    def shutdown(self):
        """Gracefully shutdown audio player."""
        self.queue.put(None)
        self.worker_thread.join(timeout=2.0)



class FaceRecognizer:
    """Main face recognition system."""
    
    def __init__(self, args):
        self.args = args
        self.known_face_encodings = []
        self.known_face_names = []
        self.last_spoken = {}
        # Keep last detections to persist overlays between processed frames
        self.last_detections = []  # list of dicts: {left, top, right, bottom, name, confidence, position, timestamp}
        # How long (seconds) to keep drawing the last detection on skipped frames
        self.overlay_timeout = getattr(args, 'overlay_timeout', 0.6)
        self.audio_player = AudioPlayer()
        self.yolo_model = YOLO('yolov8n.pt')
        self.frame_count = 0
        self.fps_start_time = time.time()
        self.fps = 0
        
        # Load known faces
        self._load_known_faces()
    
    def _load_known_faces(self):
        """Load and encode known faces from directory."""
        if not os.path.isdir(self.args.known_faces):
            logger.warning(f"Known faces directory '{self.args.known_faces}' not found. Continuing with empty database.")
            return
        
        logger.info(f"Loading known faces from: {self.args.known_faces}")
        loaded_count = 0
        error_count = 0
        
        for person_name in os.listdir(self.args.known_faces):
            person_dir = os.path.join(self.args.known_faces, person_name)
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
                        else:
                            logger.warning(f"No face found in image: {image_path}")
                    except Exception as e:
                        error_count += 1
                        logger.error(f"Failed to load image {image_path}: {e}")
        
        unique_people = len(set(self.known_face_names))
        logger.info(f"Loaded {loaded_count} face encodings for {unique_people} people ({error_count} errors)")
    
    def _calculate_fps(self):
        """Calculate and log FPS periodically."""
        self.frame_count += 1
        if self.frame_count % 30 == 0:
            elapsed = time.time() - self.fps_start_time
            self.fps = 30 / elapsed if elapsed > 0 else 0
            logger.debug(f"FPS: {self.fps:.1f}")
            self.fps_start_time = time.time()

    def _draw_detections(self, frame, detections):
        """Draw bounding boxes and labels for the provided detections on the frame.

        Detections should be a list of dicts with keys: left, top, right, bottom, name, confidence, position, timestamp
        This method filters out stale detections using self.overlay_timeout.
        """
        now = time.time()
        for det in detections:
            # skip stale
            if now - det.get('timestamp', 0) > self.overlay_timeout:
                continue

            left = det['left']
            top = det['top']
            right = det['right']
            bottom = det['bottom']
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
    
    def _get_position(self, left, frame_width):
        """Determine position of face in frame (left/middle/right)."""
        if left < frame_width / 3:
            return "left"
        elif left < 2 * frame_width / 3:
            return "middle"
        else:
            return "right"
    
    def _recognize_face(self, face_encoding):
        """
        Recognize a face and return (name, confidence).
        
        Returns:
            tuple: (name, confidence) where confidence is 0.0-1.0
        """
        if not self.known_face_encodings:
            return "Unknown", 0.0
        
        # Calculate distances to all known faces
        face_distances = face_recognition.face_distance(self.known_face_encodings, face_encoding)
        best_match_index = np.argmin(face_distances)
        best_distance = face_distances[best_match_index]
        
        # Convert distance to confidence (inverse relationship)
        confidence = 1.0 - best_distance
        
        # Check if best match is above threshold
        if best_distance <= self.args.threshold:
            name = self.known_face_names[best_match_index]
            logger.debug(f"Recognized {name} with confidence {confidence:.2f}")
            return name, confidence
        else:
            logger.debug(f"No match found (best distance: {best_distance:.2f}, threshold: {self.args.threshold})")
            return "Unknown", confidence
    
    def _should_announce(self, name):
        """Check if enough time has passed since last announcement."""
        current_time = time.time()
        if name not in self.last_spoken:
            return True
        return (current_time - self.last_spoken[name]) > self.args.debounce
    
    def run(self):
        """Main recognition loop."""
        # Open video capture
        video_capture = cv2.VideoCapture(self.args.camera)
        
        if not video_capture.isOpened():
            logger.error(f"Failed to open camera {self.args.camera}")
            return
        
        # Get camera properties
        frame_width = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        logger.info(f"Camera opened: {frame_width}x{frame_height}")
        logger.info(f"Configuration: scale={self.args.scale}, process_every={self.args.process_every}, "
                   f"threshold={self.args.threshold}, model={self.args.model}")
        
        process_frame_number = 0
        
        try:
            while not shutdown_event.is_set():
                ret, frame = video_capture.read()
                if not ret:
                    logger.warning("Failed to read frame from camera")
                    break
                
                # Calculate FPS
                self._calculate_fps()
                
                # Process only every Nth frame for performance
                process_frame_number += 1
                if process_frame_number % self.args.process_every != 0:
                    # Draw last known detections (persist overlays) while skipping processing
                    try:
                        self._draw_detections(frame, self.last_detections)
                    except Exception:
                        # Drawing should not crash the loop; log and continue
                        logger.exception("Error drawing persisted detections")

                    cv2.imshow('Face Recognition', frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                    continue
                
                # Downscale frame for faster processing
                small_frame = cv2.resize(frame, (0, 0), fx=self.args.scale, fy=self.args.scale)
                rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
                
                # --- Our New Code ---
# 1. Use YOLO to find 'person' objects (fast)
                yolo_results = self.yolo_model(rgb_small_frame, classes=[0], verbose=False) # 0 is 'person' class

# 2. Convert YOLO boxes to face_recognition's (top, right, bottom, left) format
                yolo_boxes = yolo_results[0].boxes.xyxy.cpu().numpy().astype(int)
# (yolo format is x1, y1, x2, y2)
                face_locations = [(y1, x2, y2, x1) for (x1, y1, x2, y2) in yolo_boxes]

# 3. Now, get encodings ONLY for the faces YOLO found
                face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
# --- End New Code ---
                
                # Process each face and collect detections so we can persist overlays
                detections = []
                for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
                    # Scale back coordinates to original frame size
                    top = int(top / self.args.scale)
                    right = int(right / self.args.scale)
                    bottom = int(bottom / self.args.scale)
                    left = int(left / self.args.scale)

                    # Recognize face
                    name, confidence = self._recognize_face(face_encoding)

                    # Determine position
                    position = self._get_position(left, frame_width)

                    # Prepare detection dict with timestamp so it can be persisted
                    detections.append({
                        'left': left,
                        'top': top,
                        'right': right,
                        'bottom': bottom,
                        'name': name,
                        'confidence': confidence,
                        'position': position,
                        'timestamp': time.time()
                    })

                    # Announce if recognized and not recently spoken
                    if name != "Unknown" and self._should_announce(name):
                        message = f"{name} is on the {position} side."
                        logger.info(f"Announcing: {message}")
                        self.audio_player.speak(message)
                        self.last_spoken[name] = time.time()

                # If we have any detections, update last_detections so overlays persist on skipped frames.
                # If there are no detections, keep previous last_detections until they expire instead of clearing immediately
                if detections:
                    self.last_detections = detections

                # Draw detections (either current or persisted ones)
                try:
                    # draw current detections first (will filter by timestamp internally)
                    if detections:
                        self._draw_detections(frame, detections)
                    else:
                        # draw persisted detections
                        self._draw_detections(frame, self.last_detections)
                except Exception:
                    logger.exception("Error drawing detections")
                
                # Display FPS on frame
                fps_text = f"FPS: {self.fps:.1f}" if self.fps > 0 else "FPS: --"
                cv2.putText(frame, fps_text, (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                # Show the frame
                cv2.imshow('Face Recognition', frame)
                
                # Quit on 'q' key
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    logger.info("User pressed 'q' to quit")
                    break
        
        except KeyboardInterrupt:
            logger.info("Interrupted by user")
        except Exception as e:
            logger.exception(f"Error in main loop: {e}")
        finally:
            # Cleanup
            logger.info("Shutting down...")
            video_capture.release()
            cv2.destroyAllWindows()
            self.audio_player.shutdown()


def signal_handler(signum, frame):
    """Handle shutdown signals gracefully."""
    logger.info(f"Received signal {signum}, shutting down...")
    shutdown_event.set()


def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Real-time Face Recognition with Position Detection",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run with defaults
  python facerecognizer.py
  
  # Use different camera and custom threshold
  python facerecognizer.py --camera 1 --threshold 0.4
  
  # High performance mode (lower quality but faster)
  python facerecognizer.py --scale 0.2 --process-every 3 --model hog
  
  # High accuracy mode (slower but more accurate)
  python facerecognizer.py --scale 0.5 --process-every 1 --threshold 0.4 --model cnn
        """
    )
    
    parser.add_argument('--known-faces', type=str, default='known_faces',
                       help='Path to known faces directory (default: known_faces)')
    parser.add_argument('--camera', type=int, default=0,
                       help='Camera device index (default: 0)')
    parser.add_argument('--debounce', type=float, default=15.0,
                       help='Seconds between repeated announcements (default: 15.0)')
    parser.add_argument('--scale', type=float, default=0.25,
                       help='Frame scale factor for processing (default: 0.25, smaller=faster)')
    parser.add_argument('--process-every', type=int, default=2,
                       help='Process every Nth frame (default: 2, higher=faster)')
    parser.add_argument('--threshold', type=float, default=0.5,
                       help='Face match threshold 0.0-1.0 (default: 0.5, lower=stricter)')
    parser.add_argument('--model', type=str, default='hog', choices=['hog', 'cnn'],
                       help='Face detection model: hog (faster) or cnn (more accurate, needs GPU)')
    parser.add_argument('--overlay-timeout', type=float, default=0.6,
                       help='Seconds to keep last detection overlay on skipped frames (default: 0.6)')
    parser.add_argument('--debug', action='store_true',
                       help='Enable debug logging')
    
    return parser.parse_args()


def setup_logging(debug=False):
    """Configure logging."""
    level = logging.DEBUG if debug else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler()]
    )


def main():
    """Main entry point."""
    args = parse_arguments()
    setup_logging(args.debug)
    
    # Register signal handlers for graceful shutdown
    signal.signal(signal.SIGINT, signal_handler)
    if hasattr(signal, 'SIGTERM'):
        signal.signal(signal.SIGTERM, signal_handler)
    
    logger.info("="*60)
    logger.info("Face Recognition System Starting")
    logger.info("="*60)
    
    # Create and run recognizer
    recognizer = FaceRecognizer(args)
    recognizer.run()
    
    logger.info("Face Recognition System Stopped")


if __name__ == "__main__":
    main()
