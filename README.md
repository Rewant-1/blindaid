# Face Recognition System

Real-time face recognition system with position detection and audio announcements. Built with Python, OpenCV, and face_recognition library.

## Features

✨ **Real-time face detection and recognition** from webcam feed  
📍 **Position detection** (left, middle, right) for each recognized person  
🔊 **Offline audio announcements** using text-to-speech  
⚡ **Performance optimizations** with frame scaling and skipping  
🎛️ **Configurable parameters** via command-line arguments  
🛡️ **Robust error handling** with comprehensive logging  
📊 **FPS counter** and confidence metrics  

## Requirements

- Python 3.7 or higher
- Webcam/camera device
- Windows, macOS, or Linux

## Installation

### 1. Install Visual Studio Build Tools (Windows Only)

`dlib` requires C++ build tools on Windows. Download and install:
- [Visual Studio Build Tools](https://visualstudio.microsoft.com/downloads/)
- During installation, select "Desktop development with C++"

Alternatively, install CMake:
```powershell
# Using Chocolatey
choco install cmake

# Or download from https://cmake.org/download/
```

### 2. Create Virtual Environment (Recommended)

```powershell
# Create virtual environment
python -m venv .venv

# Activate on Windows
.venv\Scripts\activate

# Activate on macOS/Linux
source .venv/bin/activate
```

### 3. Install Dependencies

```powershell
pip install -r requirements.txt
```

**Note:** Installing `dlib` and `face_recognition` may take several minutes as dlib compiles from source.

#### Troubleshooting Installation

**If dlib installation fails on Windows:**
```powershell
# Option 1: Install pre-built wheel
pip install https://github.com/jloh02/dlib/releases/download/v19.22/dlib-19.22.0-cp311-cp311-win_amd64.whl

# Option 2: Use conda (if you have Anaconda/Miniconda)
conda install -c conda-forge dlib
```

**If face_recognition installation fails:**
```powershell
pip install cmake
pip install dlib
pip install face_recognition
```

## Setup

### 1. Organize Known Faces

Create a directory structure like this:
```
known_faces/
├── person1/
│   ├── photo1.jpg
│   ├── photo2.jpg
│   └── photo3.jpg
├── person2/
│   ├── photo1.jpg
│   └── photo2.jpg
└── person3/
    └── photo1.jpg
```

**Tips for best results:**
- Use clear, front-facing photos
- Good lighting conditions
- One face per photo
- Multiple photos per person improves accuracy
- Supported formats: JPG, PNG, BMP

### 2. Test Your Camera

```powershell
python -c "import cv2; cap = cv2.VideoCapture(0); print('Camera OK' if cap.isOpened() else 'Camera FAILED'); cap.release()"
```

## Usage

### Basic Usage

```powershell
# Run with default settings
python facerecognizer.py
```

Press `q` to quit the application.

### Command-Line Options

```powershell
python facerecognizer.py --help
```

#### Available Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--known-faces` | str | `known_faces` | Path to directory containing known faces |
| `--camera` | int | `0` | Camera device index (0 for default camera) |
| `--debounce` | float | `15.0` | Seconds between repeated announcements for same person |
| `--scale` | float | `0.25` | Frame scale factor for processing (smaller = faster) |
| `--process-every` | int | `2` | Process every Nth frame (higher = faster, lower quality) |
| `--threshold` | float | `0.5` | Face match threshold 0.0-1.0 (lower = stricter matching) |
| `--model` | str | `hog` | Detection model: `hog` (faster) or `cnn` (accurate, needs GPU) |
| `--debug` | flag | `False` | Enable debug logging |

### Usage Examples

**High performance mode (recommended for most laptops):**
```powershell
python facerecognizer.py --scale 0.2 --process-every 3 --model hog
```

**High accuracy mode (requires good CPU/GPU):**
```powershell
python facerecognizer.py --scale 0.5 --process-every 1 --threshold 0.4
```

**Use CNN model (requires GPU for real-time):**
```powershell
python facerecognizer.py --model cnn --process-every 1
```

**Multiple cameras (use camera 1 instead of default):**
```powershell
python facerecognizer.py --camera 1
```

**Custom known faces directory:**
```powershell
python facerecognizer.py --known-faces "C:\MyFaces" --threshold 0.4
```

**Debug mode with verbose logging:**
```powershell
python facerecognizer.py --debug
```

## Configuration Tuning

### Performance vs Accuracy Trade-offs

| Setting | Fast/Low CPU | Balanced | Accurate/High CPU |
|---------|--------------|----------|-------------------|
| `--scale` | 0.2 | 0.25 | 0.5 |
| `--process-every` | 3 | 2 | 1 |
| `--model` | hog | hog | cnn |

### Threshold Tuning

- **0.3-0.4**: Very strict, fewer false positives, may miss some matches
- **0.5**: Default, balanced
- **0.6-0.7**: Lenient, more matches but higher false positive rate

Test with your specific dataset and adjust accordingly.

## How It Works

1. **Face Detection**: Detects faces in video frames using HOG or CNN model
2. **Face Encoding**: Generates 128-dimensional face encodings
3. **Face Recognition**: Compares encodings with known faces database
4. **Position Detection**: Determines if person is on left/middle/right of frame
5. **Audio Announcement**: Speaks person's name and position (with debouncing)

### Performance Optimizations

- **Frame Downscaling**: Processes smaller frames (default 0.25x) for faster detection
- **Frame Skipping**: Processes every Nth frame (default every 2nd frame)
- **HOG Model**: Uses histogram of gradients (faster than CNN)
- **Threaded Audio**: Audio playback in background thread doesn't block video

## Troubleshooting

### Camera Issues

**Camera not opening:**
```powershell
# Try different camera index
python facerecognizer.py --camera 1

# Check available cameras
python -c "import cv2; [print(f'Camera {i}: {cv2.VideoCapture(i).isOpened()}') for i in range(5)]"
```

### Recognition Issues

**Low accuracy / not recognizing faces:**
- Add more photos per person (3-5 recommended)
- Use better quality photos (good lighting, clear faces)
- Lower threshold: `--threshold 0.4`
- Increase processing: `--scale 0.5 --process-every 1`

**False positives (wrong person recognized):**
- Increase threshold: `--threshold 0.6`
- Use stricter model settings
- Ensure known faces photos are distinct

### Performance Issues

**Low FPS / laggy video:**
- Decrease scale: `--scale 0.2`
- Increase frame skip: `--process-every 3`
- Use HOG model: `--model hog`
- Close other applications

**High CPU usage:**
- Default settings are optimized for balance
- Try: `--scale 0.2 --process-every 3`

### Audio Issues

**No audio / TTS not working:**
```powershell
# Test pyttsx3
python -c "import pyttsx3; engine = pyttsx3.init(); engine.say('Test'); engine.runAndWait()"
```

**On Windows**, pyttsx3 uses SAPI5. If issues occur:
```powershell
pip install pywin32
```

**On Linux**, install espeak:
```bash
sudo apt-get install espeak
```

**On macOS**, uses built-in NSSpeechSynthesizer (should work out of box).

### Installation Issues

**dlib won't install:**
See "Troubleshooting Installation" section above.

**Import errors:**
```powershell
# Ensure virtual environment is activated
.venv\Scripts\activate

# Reinstall dependencies
pip install --upgrade -r requirements.txt
```

## Development

### Project Structure

```
sec-project/
├── facerecognizer.py      # Main application
├── requirements.txt        # Python dependencies
├── README.md              # This file
├── known_faces/           # Known faces database
│   ├── person1/
│   └── person2/
└── .venv/                 # Virtual environment (created)
```

### Logging

View detailed logs:
```powershell
python facerecognizer.py --debug
```

Logs include:
- Loaded faces count
- Camera resolution
- FPS counter
- Recognition events
- Configuration settings

### Extending the System

The code is structured with classes for easy extension:
- `AudioPlayer`: Handles TTS audio playback
- `FaceRecognizer`: Main recognition logic

You can modify:
- Audio messages in `run()` method
- Position detection logic in `_get_position()`
- Recognition threshold in arguments
- Add new features (e.g., save snapshots, database logging)

## Performance Benchmarks

Typical performance on different systems:

| System | Settings | FPS | Notes |
|--------|----------|-----|-------|
| Gaming Laptop (RTX 3060) | Default | 25-30 | Smooth |
| Modern Laptop (i7) | Default | 15-20 | Good |
| Older Laptop (i5) | Fast mode | 10-15 | Usable |
| Raspberry Pi 4 | Fast + HOG | 5-8 | Limited |

## Security & Privacy

⚠️ **Important Notes:**
- Face data is stored locally only
- No network connectivity required (offline TTS)
- No data is sent to external services
- Known faces images remain in your `known_faces/` directory

## License

This project uses the following open-source libraries:
- `face_recognition` - MIT License
- `dlib` - Boost Software License
- `OpenCV` - Apache 2.0 License
- `pyttsx3` - MPL 2.0 License

## Credits

Built with:
- [face_recognition](https://github.com/ageitgey/face_recognition) by Adam Geitgey
- [dlib](http://dlib.net/) by Davis King
- [OpenCV](https://opencv.org/)
- [pyttsx3](https://github.com/nateshmbhat/pyttsx3)

## Support

For issues or questions:
1. Check troubleshooting section above
2. Verify all dependencies are installed correctly
3. Test with `--debug` flag for detailed logs
4. Ensure camera and known faces are properly configured

---

**Made with ❤️ for real-time face recognition**
