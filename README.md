# BlindAid - Assistive Technology System

A consolidated multi-modal assistive technology system designed to help visually impaired individuals navigate their environment through object detection, text reading (OCR), and face recognition.

## Features

### 🕹️ Integrated Controller

- One entry point (`python -m blindaid`) powers every experience
- Hotkeys let you toggle modes (`1` Scene, `2` Reading) without restarting
- Optional caption (`C`) and depth (`D`) overlays spin up only when requested
- Lazy dependency loading keeps startup fast even on modest hardware

### 👀 Scene Mode (Objects + Faces)

- YOLO-based object detection fused with the known-faces directory
- Announces spatial hints (left/center/right) with configurable cooldowns
- Designed for navigation: salient object summaries + optional depth cues

### 📖 Reading Mode (OCR)

- PaddleOCR pipeline tuned for handheld documents and signage
- TTS playback for detected text with manual repeat whenever you press `S`
- Smart filtering to avoid noisy partial detections

### 🧠 Optional AI Enhancements

- BLIP captioning for natural descriptions (`C`)
- MiDaS depth estimation for approximate distances (`D`)
- Both features live behind the `advanced`/`full` extras so lighter installs stay slim

## Quick Start

### Installation

1. **Create a virtual environment & install dependencies:**

   ```bash
   python -m venv .venv
   # Windows
   .venv\\Scripts\\activate
   # Linux / macOS
   source .venv/bin/activate
   pip install -r requirements.txt
   ```
   
   The requirements file now installs the editable `blindaid` package with every runtime extra.
   See `docs/INSTALL.md` for uv/pipx alternatives and developer tooling instructions.

2. **Set up resources:**
   - Copy YOLO object detection model to: `resources/models/object_blind_aide.pt`
   - Copy YOLO face detection model to: `resources/models/yolov9t-face-lindevs.pt`
   - Copy known faces to: `resources/known_faces/<person_name>/*.jpg`

### Running the System

#### Integrated Controller (Recommended)

```bash
python -m blindaid
```

Want to land directly in reading mode? Start the controller with a different initial state:

```bash
python -m blindaid --start-mode reading
```

##### Hotkeys

- `1`: Scene understanding (objects + faces)
- `2`: Reading mode (OCR)
- `C`: Generate a descriptive caption for the current frame *(requires optional advanced extras)*
- `D`: Run depth analysis on the latest scene detections *(requires optional advanced extras)*
- `S`: Manually replay the last spoken message (mode-specific)
- `Q`: Quit the application

##### Helpful flags

```bash
python -m blindaid --camera 1        # Use a different camera index
python -m blindaid --no-audio        # Mute TTS output
python -m blindaid --debug           # Verbose logging
```

Fine-grained thresholds (confidence, cooldowns, etc.) now live in `blindaid/core/config.py` for clarity.

## Advanced Usage

- **Start modes**: `--start-mode scene|reading` controls the initial screen without removing hotkey toggles.
- **Camera + audio**: Mix `--camera`, `--no-audio`, and `--debug` flags as shown above to fit your setup.
- **Confidence + thresholds**: Adjust `OBJECT_CONFIDENCE`, `TEXT_CONFIDENCE`, and cooldown timers inside `blindaid/core/config.py`.
- **Known faces**: Drop JPEGs into `resources/known_faces/<person_name>/` (no CLI flag needed anymore).
- **Optional extras**: Install `pip install -e .[advanced]` or `.[full]` to enable caption/depth helpers.

## Project Structure

```text
sec-project/
├── blindaid/                    # Main application package
│   ├── __init__.py
│   ├── __main__.py             # Package entry point
│   ├── app.py                  # CLI application
│   ├── core/                   # Shared utilities
│   │   ├── config.py           # Configuration
│   │   ├── audio.py            # Audio/TTS utilities
│   │   ├── caption.py          # Optional BLIP captions
│   │   └── depth.py            # Optional MiDaS depth analysis
│   └── modes/                  # Mode implementations
│       ├── scene/
│       │   └── scene_mode.py   # Objects + faces combined for ModeController
│       └── ocr/
│           └── reading_mode.py # OCR pipeline used by ModeController
├── resources/                   # Models and data
│   ├── models/                 # YOLO models
│   └── known_faces/            # Face database
├── pyproject.toml              # Packaging + dependency metadata
├── requirements.txt            # Editable install (full runtime extras)
├── requirements-dev.txt        # Developer tooling (ruff, pytest, mypy)
├── Makefile                    # Helper targets (setup, lint, test)
├── docs/                       # Install, structure, and Raspberry Pi guides
├── .venv/                      # Local virtual environment (gitignored)

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

1. **Launch the controller**

   ```bash
   python -m blindaid
   ```

   - Explain that this single window now handles everything.

2. **Scene walk-through (hotkey `1`)**
   - Show common objects (pen, book, bag, bottle) and move them left/center/right.
   - Point out audio announcements combining object labels and relative positions.
   - Let the system auto-switch between objects and known faces as people enter the frame.

3. **Reading showcase (press `2` or start in reading mode)**

   ```bash
   python -m blindaid --start-mode reading
   ```

   - Hold a sign/book in front of the camera to trigger OCR.
   - Demonstrate `S` for manual replay and how noisy text gets filtered.

4. **Face recognition (still in Scene mode)**
   - Introduce registered faces to highlight greetings and left/middle/right cues.
   - Show an unregistered face to emphasize "Unknown" handling.

5. **Optional AI extras**
   - Press `C` for a BLIP caption and `D` for depth summaries (if advanced extras installed).
   - Describe how these features are opt-in so the controller stays lightweight.

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

- Python 3.9+
- Webcam
- Windows/Linux/MacOS
- 4GB+ RAM recommended
- Internet connection (for gTTS audio)

### Raspberry Pi

- Raspberry Pi 5 (4GB or 8GB)
- Raspberry Pi Camera Module v2/v3 or USB webcam
- See **[docs/RASPBERRY_PI.md](docs/RASPBERRY_PI.md)** for complete Pi setup guide

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
