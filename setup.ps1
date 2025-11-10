# Setup script for BlindAid# BlindAid Setup Script

# Run this after installing requirements.txt# Run this to copy models and resources to the consolidated structure



Write-Host "BlindAid Setup Script" -ForegroundColor CyanWrite-Host "BlindAid Setup Script" -ForegroundColor Cyan

Write-Host "=====================`n" -ForegroundColor CyanWrite-Host "=====================" -ForegroundColor Cyan

Write-Host ""

# Check Python

Write-Host "Checking Python installation..." -ForegroundColor Yellow# Create resources directories if they don't exist

python --versionWrite-Host "Creating resource directories..." -ForegroundColor Yellow

if ($LASTEXITCODE -ne 0) {New-Item -ItemType Directory -Force -Path "resources\models" | Out-Null

    Write-Host "ERROR: Python not found. Please install Python 3.8+`n" -ForegroundColor RedNew-Item -ItemType Directory -Force -Path "resources\known_faces" | Out-Null

    exit 1

}# Copy object detection model

Write-Host "Copying object detection model..." -ForegroundColor Yellow

# Create venv if it doesn't existif (Test-Path "Blind_aide_specs\object_detection\object_blind_aide.pt") {

if (-not (Test-Path "venv")) {    Copy-Item "Blind_aide_specs\object_detection\object_blind_aide.pt" "resources\models\" -Force

    Write-Host "`nCreating virtual environment..." -ForegroundColor Yellow    Write-Host "  ✓ object_blind_aide.pt copied" -ForegroundColor Green

    python -m venv venv} elseif (Test-Path "Blind_aide_specs\object_detection\runs\detect\train\weights\best.pt") {

    Write-Host "Virtual environment created!`n" -ForegroundColor Green    Copy-Item "Blind_aide_specs\object_detection\runs\detect\train\weights\best.pt" "resources\models\object_blind_aide.pt" -Force

} else {    Write-Host "  ✓ best.pt copied as object_blind_aide.pt" -ForegroundColor Green

    Write-Host "`nVirtual environment already exists`n" -ForegroundColor Green} else {

}    Write-Host "  ⚠ Object detection model not found!" -ForegroundColor Red

    Write-Host "    Please manually copy your YOLO model to: resources\models\object_blind_aide.pt" -ForegroundColor Red

# Activate venv}

Write-Host "To activate virtual environment, run:" -ForegroundColor Yellow

Write-Host "  venv\Scripts\activate`n" -ForegroundColor Cyan# Copy face detection model

Write-Host "Copying face detection model..." -ForegroundColor Yellow

# Check if venv is activeif (Test-Path "sec-1\weights\yolov9t-face-lindevs.pt") {

$venvActive = $env:VIRTUAL_ENV    Copy-Item "sec-1\weights\yolov9t-face-lindevs.pt" "resources\models\" -Force

if ($venvActive) {    Write-Host "  ✓ yolov9t-face-lindevs.pt copied" -ForegroundColor Green

    Write-Host "Virtual environment is active!`n" -ForegroundColor Green} else {

        Write-Host "  ⚠ Face detection model not found!" -ForegroundColor Red

    # Install requirements    Write-Host "    Please manually copy your face model to: resources\models\yolov9t-face-lindevs.pt" -ForegroundColor Red

    Write-Host "Installing dependencies..." -ForegroundColor Yellow}

    pip install -r requirements.txt

    # Copy known faces

    if ($LASTEXITCODE -eq 0) {Write-Host "Copying known faces database..." -ForegroundColor Yellow

        Write-Host "`nDependencies installed successfully!`n" -ForegroundColor Greenif (Test-Path "sec-1\known_faces") {

    } else {    Copy-Item "sec-1\known_faces\*" "resources\known_faces\" -Recurse -Force

        Write-Host "`nERROR: Failed to install dependencies`n" -ForegroundColor Red    $faceCount = (Get-ChildItem "resources\known_faces" -Directory).Count

        exit 1    Write-Host "  ✓ Copied faces for $faceCount people" -ForegroundColor Green

    }} else {

        Write-Host "  ⚠ Known faces directory not found!" -ForegroundColor Red

    # Check imports    Write-Host "    Please manually copy faces to: resources\known_faces\<person_name>\*.jpg" -ForegroundColor Red

    Write-Host "Verifying installation..." -ForegroundColor Yellow}

    python -c "import cv2, ultralytics, face_recognition, paddleocr, torch, transformers; print('All imports OK')"

    Write-Host ""

    if ($LASTEXITCODE -eq 0) {Write-Host "Setup complete!" -ForegroundColor Green

        Write-Host "`nSetup complete!`n" -ForegroundColor GreenWrite-Host ""

        Write-Host "Next steps:" -ForegroundColor CyanWrite-Host "Next steps:" -ForegroundColor Cyan

        Write-Host "1. Add YOLO models to resources/models/" -ForegroundColor WhiteWrite-Host "1. Install dependencies: pip install -r requirements.txt" -ForegroundColor White

        Write-Host "2. Add known faces to resources/known_faces/<name>/" -ForegroundColor WhiteWrite-Host "2. Test object detection: python -m blindaid --mode object-detection" -ForegroundColor White

        Write-Host "3. Run: python -m blindaid`n" -ForegroundColor WhiteWrite-Host "3. Test OCR: python -m blindaid --mode ocr" -ForegroundColor White

    } else {Write-Host "4. Test face recognition: python -m blindaid --mode face" -ForegroundColor White

        Write-Host "`nWARNING: Some imports failed. Check installation.`n" -ForegroundColor YellowWrite-Host ""

    }
} else {
    Write-Host "Please activate venv first, then run:" -ForegroundColor Yellow
    Write-Host "  pip install -r requirements.txt`n" -ForegroundColor Cyan
}
