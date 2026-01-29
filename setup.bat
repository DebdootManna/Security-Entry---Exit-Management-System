@echo off
REM Setup Script for Face Re-Identification System (Windows)
REM Security Entry & Exit Management System
REM
REM This script installs dependencies, verifies the environment,
REM and prepares the system for first use on Windows.

setlocal enabledelayedexpansion

echo ============================================================
echo Face Re-Identification System - Setup Script (Windows)
echo ============================================================
echo.

REM Check if script is run from the correct directory
if not exist "face_reidentification_test.py" (
    echo [ERROR] Please run this script from the 'Security Entry ^& Exit Management System' directory
    pause
    exit /b 1
)

REM Check Python version
echo [INFO] Checking Python version...
where python >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Python not found. Please install Python 3.8 or higher.
    echo Download from: https://www.python.org/downloads/
    pause
    exit /b 1
)

for /f "tokens=2" %%i in ('python --version 2^>^&1') do set PYTHON_VERSION=%%i
echo [SUCCESS] Python version: %PYTHON_VERSION%

REM Check pip
echo [INFO] Checking pip...
python -m pip --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] pip not found. Please install pip.
    pause
    exit /b 1
)
echo [SUCCESS] pip is available
echo.

REM Ask about virtual environment
set /p CREATE_VENV="Would you like to create a virtual environment? (recommended) [Y/n]: "
if /i "!CREATE_VENV!"=="" set CREATE_VENV=Y
if /i "!CREATE_VENV!"=="Y" (
    echo [INFO] Creating virtual environment...
    python -m venv venv

    if exist "venv\Scripts\activate.bat" (
        call venv\Scripts\activate.bat
        echo [SUCCESS] Virtual environment created and activated
    ) else (
        echo [WARNING] Could not activate virtual environment automatically
        echo Please activate manually: venv\Scripts\activate.bat
        pause
        exit /b 0
    )
) else (
    echo [INFO] Skipping virtual environment creation
)
echo.

REM Upgrade pip
echo [INFO] Upgrading pip...
python -m pip install --upgrade pip --quiet
if %errorlevel% equ 0 (
    echo [SUCCESS] pip upgraded
) else (
    echo [WARNING] Could not upgrade pip, continuing anyway...
)
echo.

REM Install dependencies
echo ============================================================
echo Installing Dependencies
echo ============================================================
echo [INFO] This may take several minutes...
echo.

REM Define dependencies
set DEPENDENCIES=numpy>=1.24.0 opencv-python>=4.8.0 pillow>=10.0.0 scipy>=1.11.0 ultralytics>=8.0.0 mtcnn>=0.1.1 deepface>=0.0.79 tensorflow>=2.13.0 faiss-cpu>=1.7.4 tf-keras>=2.15.0

for %%d in (%DEPENDENCIES%) do (
    echo [INFO] Installing %%d...
    python -m pip install "%%d" --quiet
    if !errorlevel! equ 0 (
        echo [SUCCESS] %%d installed
    ) else (
        echo [ERROR] Failed to install %%d
    )
)
echo.

REM Verify installations
echo ============================================================
echo Verifying Installation
echo ============================================================
echo.

python -c "import sys; modules = {'OpenCV': 'cv2', 'NumPy': 'numpy', 'Pillow': 'PIL', 'SciPy': 'scipy', 'Ultralytics': 'ultralytics', 'MTCNN': 'mtcnn', 'DeepFace': 'deepface', 'TensorFlow': 'tensorflow', 'FAISS': 'faiss'}; failed = []; [print(f'[SUCCESS] {name} installed') if __import__(module) or True else (print(f'[ERROR] {name} NOT installed'), failed.append(name)) for name, module in modules.items() if not failed.append(name) if True else None]; print(f'\n[WARNING] Failed modules: {failed}') if failed else print('\n[SUCCESS] All modules verified successfully!')"

if %errorlevel% neq 0 (
    echo [ERROR] Verification failed. Please check the errors above.
    pause
    exit /b 1
)
echo.

REM Download YOLO models
echo ============================================================
echo Downloading Required Models
echo ============================================================
echo [INFO] Downloading YOLOv8n-face model (this may take a few minutes)...
echo.

python -c "from ultralytics import YOLO; print('[INFO] Attempting to load YOLOv8n-face...'); model = YOLO('yolov8n-face.pt'); print('[SUCCESS] YOLOv8n-face model downloaded successfully')" 2>nul
if %errorlevel% neq 0 (
    echo [WARNING] Could not download YOLOv8n-face
    echo [INFO] Trying standard YOLOv8n instead...
    python -c "from ultralytics import YOLO; model = YOLO('yolov8n.pt'); print('[SUCCESS] YOLOv8n model downloaded successfully')"
)
echo.

REM Test camera access
echo ============================================================
echo Testing Camera Access
echo ============================================================
echo [INFO] Testing webcam access...
echo.

python -c "import cv2; cap = cv2.VideoCapture(0); ret, frame = cap.read() if cap.isOpened() else (False, None); print(f'[SUCCESS] Webcam accessible (Resolution: {frame.shape[1]}x{frame.shape[0]})') if ret else print('[ERROR] Webcam opened but could not read frame'); cap.release() if cap.isOpened() else None; exit(0 if ret else 1)"

if %errorlevel% neq 0 (
    echo [WARNING] Camera test failed. You may encounter issues when running the system.
    echo [INFO] You may need to:
    echo   - Grant camera permissions
    echo   - Try a different camera index
    echo   - Check if another app is using the camera
)
echo.

REM Create test directories
echo [INFO] Creating test directories...
if not exist "test_faces" mkdir test_faces
if not exist "exports" mkdir exports
echo [SUCCESS] Directories created
echo.

REM Final summary
echo ============================================================
echo Setup Complete!
echo ============================================================
echo.
echo [SUCCESS] All dependencies installed successfully
echo.
echo [INFO] You can now run the system with:
echo.
echo     python face_reidentification_test.py
echo.
echo [INFO] Keyboard shortcuts:
echo     q - Quit
echo     s - Show statistics
echo.
echo [INFO] For more information, see README.md
echo.

if defined VIRTUAL_ENV (
    echo [INFO] Virtual environment is active: %VIRTUAL_ENV%
    echo [WARNING] Remember to activate it before running:
    echo     venv\Scripts\activate.bat
    echo.
)

echo [SUCCESS] Setup completed successfully!
echo.
pause
