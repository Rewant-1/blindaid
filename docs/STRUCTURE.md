# BlindAid Project Structure

```
sec-project/
├── blindaid/                      # Main unified package
│   ├── __init__.py               # Package initialization
│   ├── __main__.py               # Entry point (python -m blindaid)
│   ├── app.py                    # CLI application with mode routing
│   ├── controller.py             # Integrated keyboard-controlled orchestrator
│   │
│   ├── core/                     # Shared core utilities
│   │   ├── __init__.py
│   │   ├── config.py             # Centralized configuration
│   │   ├── audio.py              # Text-to-speech utilities
│   │   ├── base_mode.py          # Base class for modes
│   │   ├── caption.py            # BLIP image captioning
│   │   └── depth.py              # MiDaS depth estimation
│   │
│   └── modes/                    # Mode implementations
│       ├── __init__.py
│       ├── scene/                # Integrated scene understanding
│       │   ├── __init__.py
│       │   └── scene_mode.py     # Objects + faces combined
│       ├── ocr/                  # Text reading
│       │   ├── __init__.py
│       │   ├── reader.py         # Legacy OCR service
│       │   └── reading_mode.py   # Controller-ready OCR
│       ├── object_detection/     # Object detection (legacy)
│       │   ├── __init__.py
│       │   └── detector.py
│       └── face_recognition/     # Face recognition (legacy)
│           ├── __init__.py
│           └── recognizer.py
│
├── resources/                    # Models and data (gitignored)
│   ├── models/                   # YOLO weights
│   │   ├── .gitkeep
│   │   ├── object_blind_aide.pt  # Object detection model
│   │   └── yolov9t-face-lindevs.pt # Face detection model
│   └── known_faces/              # Face embeddings database
│       ├── .gitkeep
│       └── <person_name>/        # One folder per person
│           ├── photo1.jpg
│           ├── photo2.jpg
│           └── photo3.jpg
│
├── .venv/                        # Preferred virtual environment (gitignored)
│
├── _legacy/                      # Archived original implementations
│   ├── README.md                 # Legacy archive documentation
│   ├── Blind_aide_specs/         # Original object detection code
│   ├── sec-1/                    # Original face recognition code
│   └── tasks.md                  # Old task tracking
│
├── .gitignore                    # Git ignore patterns
├── pyproject.toml                # Packaging + dependency metadata
├── requirements.txt              # Editable install (full runtime extras)
├── requirements-dev.txt          # Developer tooling extras
├── README.md                     # Main documentation
├── docs/                         # INSTALL / RASPBERRY_PI / STRUCTURE guides
├── setup.ps1                     # Windows setup helper (optional)
└── setup.sh                      # Linux/Mac setup helper (optional)
```

## Key Features

### Unified Structure

- **Single package**: `blindaid/` contains all functionality
- **Single packaging file**: `pyproject.toml` sources dependencies + extras
- **Editable install**: `requirements.txt` pins the local editable build with all extras
- **Single venv**: `.venv/` at project root (ignored by git)
- **Single resources**: All models and data in `resources/`

### Entry Points

1. **Integrated Controller** (Recommended):

   ```bash
   python -m blindaid
   ```

   - Hotkeys: `1` (Scene), `2` (Reading), `C` (Caption), `D` (Depth), `Q` (Quit)

2. **Legacy Single Modes**:

   ```bash
   python -m blindaid --mode object-detection
   python -m blindaid --mode ocr
   python -m blindaid --mode face
   ```


### Clean Organization

- **No scattered requirements**: All dependencies in root `requirements.txt`
- **No duplicate code**: Legacy implementations archived in `_legacy/`
- **No nested venvs**: Single venv at project root
- **Proper gitignore**: Models, venvs, and cache files excluded

### Migration Complete

All original functionality from scattered directories has been:

- ✅ Consolidated into `blindaid/` package
- ✅ Unified with pyproject-driven dependency management
- ✅ Enhanced with integrated controller
- ✅ Extended with AI features (captioning, depth)
- ✅ Properly structured and documented
- ✅ Legacy code archived for reference
