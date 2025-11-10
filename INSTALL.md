# BlindAid - Quick Setup Guide

## Installation

1. **Create and activate virtual environment:**
   ```bash
   python -m venv venv
   
   # On Windows
   venv\Scripts\activate
   
   # On Linux/Mac
   source venv/bin/activate
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up resources:**
   - Download YOLO models:
     - Object detection: `object_blind_aide.pt` → `resources/models/`
     - Face detection: `yolov9t-face-lindevs.pt` → `resources/models/`
   
   - Add known faces:
     - Create folders: `resources/known_faces/<person_name>/`
     - Add 2-3 photos per person (JPG/PNG)

## Running

**Integrated controller (recommended):**
```bash
python -m blindaid
```

**Legacy single modes:**
```bash
python -m blindaid --mode object-detection
python -m blindaid --mode ocr
python -m blindaid --mode face
```

## Quick Test

```bash
# Check installation
python -c "import cv2, ultralytics, face_recognition, paddleocr, torch, transformers; print('All imports OK')"

# Run integrated mode
python -m blindaid
```

See `README.md` for full documentation.
