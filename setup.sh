#!/bin/bash
# Setup script for BlindAid (Linux/Mac)

echo -e "\033[0;36mBlindAid Setup Script\033[0m"
echo -e "\033[0;36m=====================\033[0m\n"

# Check Python
echo -e "\033[0;33mChecking Python installation...\033[0m"
python3 --version
if [ $? -ne 0 ]; then
    echo -e "\033[0;31mERROR: Python not found. Please install Python 3.8+\033[0m\n"
    exit 1
fi

# Create venv if it doesn't exist
if [ ! -d "venv" ]; then
    echo -e "\n\033[0;33mCreating virtual environment...\033[0m"
    python3 -m venv venv
    echo -e "\033[0;32mVirtual environment created!\033[0m\n"
else
    echo -e "\n\033[0;32mVirtual environment already exists\033[0m\n"
fi

# Activate venv
echo -e "\033[0;33mTo activate virtual environment, run:\033[0m"
echo -e "\033[0;36m  source venv/bin/activate\033[0m\n"

# Check if venv is active
if [ -n "$VIRTUAL_ENV" ]; then
    echo -e "\033[0;32mVirtual environment is active!\033[0m\n"
    
    # Install requirements
    echo -e "\033[0;33mInstalling dependencies...\033[0m"
    pip install -r requirements.txt
    
    if [ $? -eq 0 ]; then
        echo -e "\n\033[0;32mDependencies installed successfully!\033[0m\n"
    else
        echo -e "\n\033[0;31mERROR: Failed to install dependencies\033[0m\n"
        exit 1
    fi
    
    # Check imports
    echo -e "\033[0;33mVerifying installation...\033[0m"
    python -c "import cv2, ultralytics, face_recognition, paddleocr, torch, transformers; print('All imports OK')"
    
    if [ $? -eq 0 ]; then
        echo -e "\n\033[0;32mSetup complete!\033[0m\n"
        echo -e "\033[0;36mNext steps:\033[0m"
        echo -e "\033[0;37m1. Add YOLO models to resources/models/\033[0m"
        echo -e "\033[0;37m2. Add known faces to resources/known_faces/<name>/\033[0m"
        echo -e "\033[0;37m3. Run: python -m blindaid\033[0m\n"
    else
        echo -e "\n\033[0;33mWARNING: Some imports failed. Check installation.\033[0m\n"
    fi
else
    echo -e "\033[0;33mPlease activate venv first, then run:\033[0m"
    echo -e "\033[0;36m  pip install -r requirements.txt\033[0m\n"
fi
