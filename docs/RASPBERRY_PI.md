# Running BlindAid on Raspberry Pi 5

This guide covers installation and optimization for Raspberry Pi 5 (4GB or 8GB recommended).

## Hardware Requirements

### Minimum Setup
- **Raspberry Pi 5** (4GB RAM minimum, 8GB recommended)
- **MicroSD Card**: 64GB+ (Class 10 or better)
- **Camera**: 
  - Raspberry Pi Camera Module v2/v3 (recommended for best integration)
  - USB webcam (alternative)
- **Power Supply**: Official Raspberry Pi 5 USB-C power adapter (27W)
- **Cooling**: Active cooling fan or heatsink (important for sustained AI workloads)

### Optional but Recommended
- **Speaker/Headphones**: For audio feedback
- **Display**: HDMI monitor for setup and debugging
- **Storage**: External SSD via USB 3.0 for better performance

## Operating System Setup

### 1. Install Raspberry Pi OS (64-bit)

Use **Raspberry Pi OS (64-bit)** - required for better performance with AI models.

```bash
# Use Raspberry Pi Imager to write:
# Raspberry Pi OS (64-bit) - Debian Bookworm
```

**Important**: Use 64-bit version for optimal NumPy, OpenCV, and PyTorch performance.

### 2. Initial System Configuration

```bash
# Update system
sudo apt update
sudo apt upgrade -y

# Install system dependencies
sudo apt install -y \
    python3-dev \
    python3-pip \
    python3-venv \
    cmake \
    build-essential \
    pkg-config \
    libjpeg-dev \
    libtiff5-dev \
    libpng-dev \
    libavcodec-dev \
    libavformat-dev \
    libswscale-dev \
    libv4l-dev \
    libxvidcore-dev \
    libx264-dev \
    libfontconfig1-dev \
    libcairo2-dev \
    libgdk-pixbuf2.0-dev \
    libpango1.0-dev \
    libgtk2.0-dev \
    libgtk-3-dev \
    libatlas-base-dev \
    gfortran \
    libhdf5-dev \
    libhdf5-serial-dev \
    libhdf5-103 \
    libqt5gui5 \
    libqt5webkit5 \
    libqt5test5 \
    python3-pyqt5 \
    libegl1-mesa-dev \
    libgles2-mesa-dev \
    libgbm-dev

# Audio dependencies for TTS
sudo apt install -y \
    espeak \
    ffmpeg \
    libespeak1 \
    portaudio19-dev \
    python3-pyaudio

# Install git if not present
sudo apt install -y git
```

### 3. Enable Camera

```bash
# For Raspberry Pi Camera Module
sudo raspi-config
# Navigate to: Interface Options → Camera → Enable

# Reboot
sudo reboot
```

## BlindAid Installation

### 1. Clone Repository

```bash
cd ~
git clone https://github.com/Rewant-1/blindaid.git
cd blindaid
```

### 2. Create Virtual Environment

```bash
# Create venv
python3 -m venv venv

# Activate
source venv/bin/activate
```

### 3. Install Dependencies (Optimized for Pi)

Create a Raspberry Pi specific requirements file:

```bash
# Install lightweight versions first
pip install --upgrade pip setuptools wheel

# Core dependencies (Pi-optimized versions)
pip install numpy
pip install opencv-python-headless  # Headless for better performance
pip install Pillow

# YOLO - use CPU-optimized version
pip install ultralytics

# Face recognition - this takes time on Pi
sudo apt install -y libopenblas-dev liblapack-dev
pip install dlib --no-cache-dir  # Compile from source (15-30 min)
pip install face-recognition

# OCR - PaddleOCR alternative for Pi
pip install paddlepaddle  # CPU version
pip install paddleocr

# Skip heavy AI models initially
# pip install torch torchvision transformers  # Optional, very slow

# Audio
pip install pyttsx3
pip install gTTS
pip install pygame

# Other utilities
pip install matplotlib pandas
```

**Note**: Full installation may take 1-2 hours on Raspberry Pi 5.

### 4. Lightweight Alternative Installation

For faster setup, create `requirements-pi.txt`:

```txt
# Lightweight Raspberry Pi requirements
numpy>=1.24.0
opencv-python-headless>=4.8.0
Pillow>=10.0.0
ultralytics>=8.0.0
face-recognition>=1.3.0
paddleocr>=2.7.0
paddlepaddle
pyttsx3>=2.90
pygame>=2.5.0
matplotlib>=3.7.0
```

Install:
```bash
pip install -r requirements-pi.txt
```

### 5. Set Up Resources

```bash
# Create resource directories
mkdir -p resources/models
mkdir -p resources/known_faces

# Copy models from your development machine
# Use smaller/quantized YOLO models for better Pi performance
scp your-pc:path/to/yolov8n.pt resources/models/object_blind_aide.pt
scp your-pc:path/to/yolov9t-face.pt resources/models/yolov9t-face-lindevs.pt

# Copy known faces
scp -r your-pc:path/to/known_faces/* resources/known_faces/
```

## Performance Optimizations

### 1. Modify Configuration for Pi

Edit `blindaid/core/config.py`:

```python
# Reduce resolution for better FPS
FRAME_WIDTH = 320   # Reduced from 640
FRAME_HEIGHT = 240  # Reduced from 480

# Increase frame skip for smoother performance
OBJECT_DETECTION_FRAME_SKIP = 5  # Process every 5th frame
FACE_PROCESS_EVERY_N_FRAMES = 5
SCENE_PROCESS_EVERY = 3
OCR_FRAME_SKIP = 5

# Lower confidence for faster processing
OBJECT_DETECTION_CONFIDENCE = 0.5  # Lower from 0.6

# Reduce face processing scale
FACE_FRAME_SCALE = 0.15  # More aggressive downscaling
```

### 2. Use Lighter YOLO Models

```bash
# Download YOLOv8 nano model (smallest, fastest)
cd resources/models
wget https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt -O object_blind_aide.pt
```

### 3. Disable Heavy Features Initially

Create a Pi-specific controller that skips caption and depth:

Edit `blindaid/controller.py` to lazy-load only when needed, or run without these features:

```bash
# Run without caption/depth (they need PyTorch which is heavy on Pi)
python -m blindaid --start-mode scene    # Objects + faces only
python -m blindaid --start-mode reading  # OCR only  
```

### 4. Camera Configuration

For Raspberry Pi Camera Module:

```python
# In config.py, camera index might be different
DEFAULT_CAMERA_INDEX = 0  # Usually 0 for Pi Camera
# If using USB camera, try 1, 2, etc.
```

Test camera:
```bash
# Test Pi Camera
libcamera-hello

# Test with Python
python3 -c "import cv2; cap = cv2.VideoCapture(0); print('Camera works!' if cap.isOpened() else 'Camera failed')"
```

### 5. Enable Hardware Acceleration

```bash
# Enable OpenGL driver for better graphics performance
sudo raspi-config
# Advanced Options → GL Driver → GL (Full KMS)

# Add to ~/.bashrc for OpenCV optimization
echo "export OPENBLAS_CORETYPE=ARMV8" >> ~/.bashrc
source ~/.bashrc
```

## Running BlindAid on Pi

### Basic Mode (Recommended Start)

```bash
# Activate venv
cd ~/blindaid
source venv/bin/activate

# Run scene mode only (lightest)
python -m blindaid --start-mode scene
```

### Integrated Mode (If System Can Handle It)

```bash
# Run with all features
python -m blindaid

# Monitor CPU/RAM usage in another terminal
htop
```

### Headless Mode (No Display)

For running without monitor:

```bash
# Disable OpenCV GUI and use audio only
# Modify app to remove cv2.imshow() calls for headless operation
```

## Expected Performance

### Raspberry Pi 5 (8GB)
- **Object Detection**: 5-10 FPS at 320x240
- **Face Recognition**: 3-8 FPS at 320x240
- **OCR**: 1-3 FPS (text detection is CPU intensive)
- **Scene Mode**: 2-5 FPS combined

### Raspberry Pi 5 (4GB)
- **Object Detection**: 3-7 FPS at 320x240
- **Face Recognition**: 2-5 FPS at 320x240
- Avoid running all modes simultaneously

## Troubleshooting

### Camera Issues
```bash
# Check camera detection
v4l2-ctl --list-devices

# Test camera
raspistill -o test.jpg  # For Pi Camera
ffplay /dev/video0       # For USB camera
```

### Memory Issues
```bash
# Increase swap space
sudo dphys-swapfile swapoff
sudo nano /etc/dphys-swapfile
# Set CONF_SWAPSIZE=2048
sudo dphys-swapfile setup
sudo dphys-swapfile swapon
```

### Audio Not Working
```bash
# Test speaker
speaker-test -t wav -c 2

# Configure audio output
sudo raspi-config
# System Options → Audio → Select output device
```

### Performance Too Slow

1. **Use smaller models**: YOLOv8n instead of YOLOv8s/m
2. **Increase frame skip**: Process every 5-10 frames
3. **Lower resolution**: 320x240 or even 160x120
4. **Disable features**: Run single mode only
5. **Overclock**: Safely overclock Pi 5 (advanced users)

## Auto-Start on Boot (Optional)

Create systemd service:

```bash
sudo nano /etc/systemd/system/blindaid.service
```

Add:
```ini
[Unit]
Description=BlindAid Assistive System
After=network.target

[Service]
Type=simple
User=pi
WorkingDirectory=/home/pi/blindaid
ExecStart=/home/pi/blindaid/venv/bin/python -m blindaid --start-mode scene
Restart=on-failure
Environment="DISPLAY=:0"

[Install]
WantedBy=multi-user.target
```

Enable:
```bash
sudo systemctl daemon-reload
sudo systemctl enable blindaid.service
sudo systemctl start blindaid.service

# Check status
sudo systemctl status blindaid.service
```

## Power Consumption Tips

- Use active cooling to prevent thermal throttling
- Consider USB-C power delivery (PD) adapter for sustained performance
- Monitor temperature: `vcgencmd measure_temp`
- If battery-powered, expect ~1-2 hours with standard power bank

## Alternative: Skip Heavy Features

For best Pi performance, modify the code to skip PyTorch-based features:

**Skip Caption & Depth**: Comment out in `requirements.txt`:
```txt
# torch>=2.0.0        # Skip for Pi - too heavy
# transformers>=4.30.0  # Skip for Pi - too heavy
```

This keeps the system lightweight and responsive on Raspberry Pi.

## Remote Development

Develop on PC, deploy to Pi:

```bash
# On your PC
git add .
git commit -m "Pi optimizations"
git push origin main

# On Raspberry Pi
cd ~/blindaid
git pull origin main
source venv/bin/activate
python -m blindaid
```

## Summary

**Quick Setup for Pi 5:**
1. Install Raspberry Pi OS 64-bit
2. Install system dependencies
3. Clone repo and create venv
4. Install lightweight requirements
5. Use smaller YOLO models
6. Lower resolution in config
7. Run single mode first
8. Monitor performance and adjust

**Best Performance**: Object detection mode with YOLOv8n at 320x240 resolution.

For questions or issues specific to Raspberry Pi, check the main README.md or open an issue on GitHub.
