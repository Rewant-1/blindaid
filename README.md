# BlindAid - Assistive Technology System

A consolidated multi-modal assistive technology system designed to help visually impaired individuals navigate their environment through object detection, text reading (OCR), and face recognition.

## Features

### 🕹️ Integrated Controller (New)

- Single window experience with hotkeys for scene understanding, reading, captions, and depth cues
- Scene mode merges YOLO object detection with known-face recognition and speech cooldowns
- On-demand BLIP captioning (`C`) for rich descriptions
- MiDaS depth summaries (`D`) to approximate how far objects are

### 🎯 Object Detection Mode

- Real-time detection of objects using YOLO
- Position identification (left/center/right)
- Audio announcements of detected objects and their locations
- Customizable confidence thresholds

### 📖 OCR Mode

- Real-time text recognition using PaddleOCR
- Text-to-speech for reading detected text
- High-confidence filtering for accurate readings
- Visual bounding boxes around detected text

### 👤 Face Recognition Mode

- Real-time face detection and recognition
- Position detection (left/middle/right)
- Audio announcements with person identification
- Support for custom face databases
- Optimized performance with YOLO-based face detection

## Quick Start

### Installation

1. **Install Python dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Set up resources:**
   - Copy YOLO object detection model to: `resources/models/object_blind_aide.pt`
   - Copy YOLO face detection model to: `resources/models/yolov9t-face-lindevs.pt`
   - Copy known faces to: `resources/known_faces/<person_name>/*.jpg`

### Running the System

#### Integrated Controller (Recommended)

```bash
python -m blindaid
```

##### Hotkeys

- `1`: Scene understanding (objects + faces)
- `2`: Reading mode (OCR)
- `C`: Generate a descriptive caption for the current frame
- `D`: Run depth analysis on the latest scene detections
- `Q`: Quit the application

#### Object Detection Mode

```bash
python -m blindaid --mode object-detection
```

##### Controls (Object Detection)

- Press `q` to quit
- Press `s` to manually trigger audio announcement

#### OCR Mode

```bash
python -m blindaid --mode ocr
```

##### Controls (OCR)

- Press `q` to quit
- Press `s` to speak detected text
- Show text to camera for automatic reading

#### Face Recognition Mode

```bash
python -m blindaid --mode face
```

##### Controls (Face)

- Press `q` to quit
- Automatic announcements when faces are recognized

## Advanced Usage

### Custom Camera
```bash
python -m blindaid --mode object-detection --camera 1
```

### Adjust Confidence Threshold
```bash
python -m blindaid --mode object-detection --confidence 0.7
```

### Disable Audio
```bash
python -m blindaid --mode ocr --no-audio
```

### Custom Known Faces Directory
```bash
python -m blindaid --mode face --known-faces ./my_faces
```

### Debug Mode
```bash
python -m blindaid --mode face --debug
```

## Project Structure

```
sec-project/
├── blindaid/                    # Main application package
│   ├── __init__.py
│   ├── __main__.py             # Package entry point
│   ├── app.py                  # CLI application
│   ├── core/                   # Shared utilities
│   │   ├── config.py           # Configuration
│   │   ├── audio.py            # Audio/TTS utilities
│   │   └── base_mode.py        # Base class for modes
│   └── modes/                  # Mode implementations
│       ├── object_detection/
│       │   └── detector.py     # Object detection service
│       ├── ocr/
│       │   └── reader.py       # OCR service
│       └── face_recognition/
│           └── recognizer.py   # Face recognition service
├── resources/                   # Models and data
│   ├── models/                 # YOLO models
│   └── known_faces/            # Face database
└── requirements.txt            # Dependencies

Legacy directories (reference only):
├── Blind_aide_specs/           # Original object detection code
└── sec-1/                      # Original face recognition code
```

## Demo Instructions for Professor

### Preparation
1. Ensure camera is connected and working
2. Install all dependencies: `pip install -r requirements.txt`
3. Copy models to `resources/models/` directory
4. Add at least 2-3 known faces to `resources/known_faces/`

### Demonstration Sequence

#### 1. Object Detection Demo (2-3 minutes)
```bash
python -m blindaid --mode object-detection
```
- Show common objects (pen, book, bag, bottle)
- Move objects to different positions (left/center/right)
- Demonstrate audio announcements
- Press 's' to manually trigger announcements

#### 2. OCR Demo (2-3 minutes)
```bash
python -m blindaid --mode ocr
```
- Show printed text (book cover, document, sign)
- Demonstrate automatic text reading
- Show confidence-based filtering with low-quality text
- Press 's' for manual reading

#### 3. Face Recognition Demo (2-3 minutes)
```bash
python -m blindaid --mode face
```
- Show registered faces and their recognition
- Demonstrate position detection (left/middle/right)
- Show "Unknown" detection for unregistered faces
- Audio announcements with names and positions

#### 4. Integrated Scene Demo (Optional bonus)
```bash
python -m blindaid
```
- Toggle between scene (`1`) and reading (`2`) modes in real time
- Trigger caption (`C`) for descriptions and depth (`D`) for distance hints
- Highlight combined speech feedback for people and objects

### Tips for Successful Demo
- Good lighting is essential
- Keep objects/text/faces at reasonable distance (1-3 feet)
- Ensure camera is stable
- Test each mode before the demo
- Have backup objects/text samples ready

## Troubleshooting

### Camera not opening
- Check camera index with `--camera 0` or `--camera 1`
- Ensure no other application is using the camera

### Models not found
- Verify model paths in `resources/models/`
- Check file names match configuration in `core/config.py`

### Audio not working
- Check system audio settings
- Try `--no-audio` flag to run without audio
- For OCR, ensure internet connection (uses gTTS)

### Poor detection accuracy
- Improve lighting conditions
- Adjust `--confidence` threshold
- Move closer to camera
- Ensure camera is focused

## Configuration

All default configurations are in `blindaid/core/config.py`. You can modify:
- Model paths
- Confidence thresholds
- Camera settings
- Audio settings
- Performance parameters

## Requirements

### Desktop/Laptop
- Python 3.8+
- Webcam
- Windows/Linux/MacOS
- 4GB+ RAM recommended
- Internet connection (for gTTS audio)

### Raspberry Pi
- Raspberry Pi 5 (4GB or 8GB)
- Raspberry Pi Camera Module v2/v3 or USB webcam
- See **[RASPBERRY_PI.md](RASPBERRY_PI.md)** for complete Pi setup guide

## Future Enhancements

- [ ] Multi-modal pipeline (combine detection + OCR)
- [ ] Depth sensing for distance estimation
- [ ] Obstacle detection and warnings
- [ ] Gesture recognition
- [ ] Mobile app integration
- [ ] Cloud synchronization for face database

## License

Educational project for assistive technology demonstration.

## Credits

Developed using:
- YOLOv8/v9 for object and face detection
- PaddleOCR for text recognition
- face_recognition library
- OpenCV for computer vision
- pyttsx3/gTTS for text-to-speech
