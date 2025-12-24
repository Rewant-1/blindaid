![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white)
![OpenCV](https://img.shields.io/badge/opencv-%23white.svg?style=for-the-badge&logo=opencv&logoColor=white)

# BlindAid

Real-time assistive vision system for visually impaired users.

Built this as a practical CV project. Uses depth for navigation, OCR for reading text, face recognition to identify people - all through a single webcam.

## Features

| Mode | What it does |
|------|-------------|
| Guardian | Warns about obstacles using depth estimation |
| Reading | Reads text aloud (PaddleOCR + gTTS) |
| People | Identifies known faces |
| Ask | Answer questions about scene (BLIP VQA) |
| Caption | Describe what camera sees |

## Why this architecture

1. **Lazy loading** - Models are 400MB+. Loading all at startup = 30+ seconds wait. Each mode loads on first use instead.

2. **Frame skipping** - OCR/depth at 30fps is pointless. Running every 4th frame saves 75% compute with same result.

3. **Threaded audio** - TTS blocks main thread. Without background worker, video freezes while speaking.

4. **Speech cooldowns** - 2.5s gap between warnings so it doesn't spam "obstacle obstacle obstacle".

## Limitations

- Depth is relative not metric (can't say "2 meters away")
- Face recognition needs 3-5 photos per person in `resources/known_faces/{name}/`
- gTTS needs internet, no offline fallback because pytt3x is often hanging on windows by speaking only once and then holding the audio port without continuing speaking.

## Setup

```bash
# Install dependencies
pip install -r requirements.txt

# Or install as editable package
pip install -e .

# Run the system
python -m blindaid
```

## Controls

- `0` - Sitting (idle)
- `1` - Navigation
- `2` - Reading
- `3` - People
- `4` - Ask question
- `5` - Caption
- `q` - Quit

## Stack

- YOLOv8/v9 + face_recognition for detection
- PaddleOCR for text
- BLIP for scene understanding
- MiDaS for depth
- gTTS + pygame for audio

## What I'd improve

**Model size problem:** BLIP is 400MB which is too heavy. Better options:
- Use MobileVLM or TinyLLaVA (~100MB) for captioning
- Or run BLIP on a server and call via API
- For production, would use ONNX quantized models

**Single model approach:** Currently using 4 separate models (YOLO, BLIP, MiDaS, PaddleOCR). A single multimodal model like LLaVA-1.5-7B could handle detection + captioning + VQA together, but needs GPU.

**Offline TTS:** pyttsx3 works offline but has threading bugs on Windows. Could use Coqui TTS or Piper for offline + better quality.

**Depth estimation:** MiDaS gives relative depth. For actual distance, would need a stereo camera or LiDAR sensor.
