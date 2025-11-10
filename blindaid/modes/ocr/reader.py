"""
OCR Service using PaddleOCR for real-time text reading.
"""
import cv2
import time
import numpy as np
from paddleocr import PaddleOCR
from blindaid.core.base_mode import BaseMode
from blindaid.core.audio import GTTSAudioPlayer
from blindaid.core import config


class OCRService(BaseMode):
    """Optical Character Recognition service for reading text."""
    
    def __init__(self, camera_index=0, language='en', confidence=0.9, audio_enabled=True):
        super().__init__(camera_index, audio_enabled)
        
        self.language = language
        self.confidence_threshold = confidence
        self.audio_player = GTTSAudioPlayer() if audio_enabled else None
        
        # Initialize OCR
        self.logger.info("Initializing PaddleOCR...")
        self.ocr = PaddleOCR(lang=language, use_textline_orientation=True, 
                            text_det_limit_side_len=640, show_log=False)
        self.logger.info("OCR initialized successfully")
        
        self.last_text = ""
        self.last_spoken = 0
        self.cooldown = config.OCR_COOLDOWN_SECONDS
        self.frame_skip = config.OCR_FRAME_SKIP
        self.frame_count = 0
        
        # Store last detected text data
        self.last_text_data = []
        self.last_text_list = []
        self.last_high_confidence_text_list = []
    
    def _extract_text_from_result(self, result):
        """Extract text from OCR result."""
        text_list = []
        high_confidence_text_list = []
        current_text_data = []
        
        if not result or not result[0]:
            return text_list, high_confidence_text_list, current_text_data
        
        ocr_result = result[0]
        
        # Handle OCRResult object format
        if hasattr(ocr_result, '__getitem__') or hasattr(ocr_result, 'get'):
            try:
                if hasattr(ocr_result, 'get'):
                    rec_texts = ocr_result.get('rec_texts', [])
                    rec_scores = ocr_result.get('rec_scores', [])
                    rec_polys = ocr_result.get('rec_polys', ocr_result.get('dt_polys', []))
                else:
                    rec_texts = ocr_result['rec_texts']
                    rec_scores = ocr_result['rec_scores']
                    rec_polys = ocr_result.get('rec_polys', ocr_result.get('dt_polys', []))
                
                if not isinstance(rec_texts, list):
                    rec_texts = [rec_texts] if rec_texts else []
                if not isinstance(rec_scores, list):
                    rec_scores = [rec_scores] if rec_scores else [1.0] * len(rec_texts)
                if not isinstance(rec_polys, list):
                    rec_polys = [rec_polys] if rec_polys else []
                
                for i, text in enumerate(rec_texts):
                    if text and str(text).strip():
                        text = str(text).strip()
                        score = rec_scores[i] if i < len(rec_scores) else 1.0
                        text_list.append(text)
                        
                        if score > self.confidence_threshold:
                            high_confidence_text_list.append(text)
                        
                        box = rec_polys[i] if i < len(rec_polys) else None
                        if box is not None:
                            try:
                                box_array = np.array(box, dtype=np.float32)
                                if box_array.ndim == 2 and box_array.shape[1] == 2:
                                    box_coords = box_array.astype(int)
                                else:
                                    box_coords = box_array.reshape(-1, 2).astype(int)
                                
                                if box_coords.shape[0] >= 4:
                                    current_text_data.append((text, score, box_coords))
                            except:
                                pass
            except:
                pass
        
        # Handle list format
        elif isinstance(ocr_result, list):
            for line in ocr_result:
                try:
                    if not isinstance(line, (list, tuple)) or len(line) < 2:
                        continue
                    
                    box = line[0]
                    text_info = line[1]
                    
                    if isinstance(box, (list, tuple)) and len(box) >= 4:
                        try:
                            box_array = np.array(box, dtype=np.float32)
                            if box_array.size >= 4:
                                box_coords = box_array.reshape(-1, 2).astype(int)
                            else:
                                continue
                        except (ValueError, TypeError):
                            continue
                    else:
                        continue
                    
                    if isinstance(text_info, tuple) and len(text_info) >= 2:
                        text, score = text_info[0], text_info[1]
                    elif isinstance(text_info, tuple) and len(text_info) == 1:
                        text, score = text_info[0], 1.0
                    elif isinstance(text_info, str):
                        text, score = text_info, 1.0
                    else:
                        text, score = str(text_info), 1.0
                    
                    text_list.append(text)
                    
                    if score > self.confidence_threshold:
                        high_confidence_text_list.append(text)
                    
                    if box_coords.shape[0] >= 4:
                        current_text_data.append((text, score, box_coords))
                except:
                    continue
        
        return text_list, high_confidence_text_list, current_text_data
    
    def _draw_text_boxes(self, frame, text_data):
        """Draw bounding boxes and text on frame."""
        for text, score, box_coords in text_data:
            try:
                box_color = (0, 255, 0) if score > self.confidence_threshold else (0, 255, 255)
                cv2.polylines(frame, [box_coords], True, box_color, 2)
                top_left = box_coords[0]
                cv2.putText(frame, f"{text} ({score:.2f})",
                           (int(top_left[0]), int(top_left[1]) - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
            except:
                pass
    
    def run(self):
        """Run OCR mode."""
        cap = cv2.VideoCapture(self.camera_index)
        
        if not cap.isOpened():
            self.logger.error(f"Failed to open camera {self.camera_index}")
            return
        
        self.logger.info("OCR Mode - Press 'q' to quit, 's' to speak detected text")
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    self.logger.warning("Failed to read frame")
                    break
                
                # Resize for faster processing
                frame = cv2.resize(frame, (640, 480))
                
                # Run OCR on every Nth frame
                result = None
                if self.frame_count % (self.frame_skip + 1) == 0:
                    result = self.ocr.ocr(frame)
                self.frame_count += 1
                
                text_list, high_confidence_text_list, current_text_data = [], [], []
                
                if result:
                    text_list, high_confidence_text_list, current_text_data = self._extract_text_from_result(result)
                    
                    if current_text_data:
                        self.last_text_data = current_text_data.copy()
                        self.last_text_list = text_list.copy()
                        self.last_high_confidence_text_list = high_confidence_text_list.copy()
                    
                    # Draw text boxes
                    self._draw_text_boxes(frame, current_text_data)
                    
                    # Auto-speak high confidence text
                    high_confidence_text = " ".join(high_confidence_text_list).strip()
                    if (self.audio_enabled and high_confidence_text and 
                        high_confidence_text != self.last_text and 
                        (time.time() - self.last_spoken > self.cooldown)):
                        self.logger.info(f"Detected: {high_confidence_text}")
                        self.audio_player.speak(high_confidence_text)
                        self.last_text = high_confidence_text
                        self.last_spoken = time.time()
                else:
                    # Draw last detected boxes
                    if self.last_text_data:
                        self._draw_text_boxes(frame, self.last_text_data)
                
                # Status overlay
                if text_list:
                    status_text = f"Text detected: {len(text_list)} lines"
                    status_color = (0, 255, 0)
                else:
                    status_text = "No text detected - Show text to camera"
                    status_color = (0, 255, 255)
                
                cv2.putText(frame, status_text, (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
                cv2.putText(frame, "OCR Mode - Press 'q' to quit, 's' to speak", 
                           (10, frame.shape[0] - 20),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                
                cv2.imshow("BlindAid - OCR", frame)
                
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    self.logger.info("User quit")
                    break
                elif key == ord('s'):
                    if self.last_high_confidence_text_list:
                        text = " ".join(self.last_high_confidence_text_list)
                        self.audio_player.speak(text)
        
        except KeyboardInterrupt:
            self.logger.info("Interrupted by user")
        except Exception as e:
            self.logger.exception(f"Error in OCR: {e}")
        finally:
            self.cleanup()
            cap.release()
            cv2.destroyAllWindows()
    
    def cleanup(self):
        """Cleanup resources."""
        if self.audio_player:
            self.audio_player.shutdown()
