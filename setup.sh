#!/bin/bash

# Setup Script for Face Re-Identification System
# Security Entry & Exit Management System
#
# This script installs dependencies, verifies the environment,
# and prepares the system for first use.

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Print colored messages
print_success() {
    echo -e "${GREEN}✓ $1${NC}"
}

print_error() {
    echo -e "${RED}✗ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠ $1${NC}"
}

print_info() {
    echo -e "${BLUE}ℹ $1${NC}"
}

print_header() {
    echo ""
    echo "============================================================"
    echo "$1"
    echo "============================================================"
    echo ""
}

# Check if script is run from the correct directory
if [ ! -f "face_reidentification_test.py" ]; then
    print_error "Please run this script from the 'Security Entry & Exit Management System' directory"
    exit 1
fi

print_header "Face Re-Identification System - Setup Script"

# Check Python version
print_info "Checking Python version..."
if command -v python3 &> /dev/null; then
    PYTHON_CMD=python3
elif command -v python &> /dev/null; then
    PYTHON_CMD=python
else
    print_error "Python not found. Please install Python 3.8 or higher."
    exit 1
fi

PYTHON_VERSION=$($PYTHON_CMD --version 2>&1 | awk '{print $2}')
PYTHON_MAJOR=$(echo $PYTHON_VERSION | cut -d. -f1)
PYTHON_MINOR=$(echo $PYTHON_VERSION | cut -d. -f2)

if [ "$PYTHON_MAJOR" -lt 3 ] || ([ "$PYTHON_MAJOR" -eq 3 ] && [ "$PYTHON_MINOR" -lt 8 ]); then
    print_error "Python 3.8 or higher required. Found: $PYTHON_VERSION"
    exit 1
else
    print_success "Python version: $PYTHON_VERSION"
fi

# Check pip
print_info "Checking pip..."
if ! $PYTHON_CMD -m pip --version &> /dev/null; then
    print_error "pip not found. Please install pip."
    exit 1
else
    print_success "pip is available"
fi

# Create virtual environment (optional but recommended)
print_info "Would you like to create a virtual environment? (recommended) [y/N]"
read -r CREATE_VENV

if [[ $CREATE_VENV =~ ^[Yy]$ ]]; then
    print_info "Creating virtual environment..."
    $PYTHON_CMD -m venv venv

    # Activate virtual environment
    if [ -f "venv/bin/activate" ]; then
        source venv/bin/activate
        print_success "Virtual environment created and activated"
    elif [ -f "venv/Scripts/activate" ]; then
        source venv/Scripts/activate
        print_success "Virtual environment created and activated"
    else
        print_warning "Could not activate virtual environment automatically"
        print_info "Please activate manually:"
        print_info "  On Linux/Mac: source venv/bin/activate"
        print_info "  On Windows: venv\\Scripts\\activate"
        exit 0
    fi
fi

# Upgrade pip
print_info "Upgrading pip..."
$PYTHON_CMD -m pip install --upgrade pip --quiet
print_success "pip upgraded"

# Install dependencies
print_header "Installing Dependencies"

print_info "This may take several minutes..."
echo ""

DEPENDENCIES=(
    "numpy>=1.24.0"
    "opencv-python>=4.8.0"
    "pillow>=10.0.0"
    "scipy>=1.11.0"
    "ultralytics>=8.0.0"
    "mtcnn>=0.1.1"
    "deepface>=0.0.79"
    "tensorflow>=2.13.0"
    "faiss-cpu>=1.7.4"
    "tf-keras>=2.15.0"
)

FAILED_DEPS=()

for dep in "${DEPENDENCIES[@]}"; do
    DEP_NAME=$(echo $dep | cut -d'>' -f1 | cut -d'=' -f1)
    print_info "Installing $DEP_NAME..."

    if $PYTHON_CMD -m pip install "$dep" --quiet; then
        print_success "$DEP_NAME installed"
    else
        print_error "Failed to install $DEP_NAME"
        FAILED_DEPS+=("$DEP_NAME")
    fi
done

# Check for failed installations
if [ ${#FAILED_DEPS[@]} -ne 0 ]; then
    print_warning "Some dependencies failed to install: ${FAILED_DEPS[*]}"
    print_info "You can try installing them manually with:"
    print_info "  pip install -r requirements.txt"
fi

# Verify installations
print_header "Verifying Installation"

$PYTHON_CMD << 'EOF'
import sys

modules = {
    "OpenCV": "cv2",
    "NumPy": "numpy",
    "Pillow": "PIL",
    "SciPy": "scipy",
    "Ultralytics": "ultralytics",
    "MTCNN": "mtcnn",
    "DeepFace": "deepface",
    "TensorFlow": "tensorflow",
    "FAISS": "faiss",
}

failed = []
for name, module in modules.items():
    try:
        __import__(module)
        print(f"✓ {name} installed successfully")
    except ImportError:
        print(f"✗ {name} NOT installed")
        failed.append(name)

if failed:
    print(f"\nWARNING: The following modules failed to import: {', '.join(failed)}")
    sys.exit(1)
else:
    print("\n✓ All modules verified successfully!")
EOF

if [ $? -ne 0 ]; then
    print_error "Verification failed. Please check the errors above."
    exit 1
fi

# Download YOLO models
print_header "Downloading Required Models"

print_info "Downloading YOLOv8n-face model (this may take a few minutes)..."

$PYTHON_CMD << 'EOF'
try:
    from ultralytics import YOLO
    print("Attempting to load YOLOv8n-face...")
    model = YOLO('yolov8n-face.pt')
    print("✓ YOLOv8n-face model downloaded successfully")
except Exception as e:
    print(f"⚠ Could not download YOLOv8n-face: {e}")
    print("  Trying standard YOLOv8n instead...")
    try:
        model = YOLO('yolov8n.pt')
        print("✓ YOLOv8n model downloaded successfully")
    except Exception as e2:
        print(f"✗ Failed to download YOLO model: {e2}")
EOF

# Test camera access
print_header "Testing Camera Access"

print_info "Testing webcam access..."

$PYTHON_CMD << 'EOF'
import cv2
import sys

cap = cv2.VideoCapture(0)
if cap.isOpened():
    ret, frame = cap.read()
    if ret:
        print(f"✓ Webcam accessible (Resolution: {frame.shape[1]}x{frame.shape[0]})")
    else:
        print("✗ Webcam opened but could not read frame")
        sys.exit(1)
    cap.release()
else:
    print("✗ Could not open webcam (index 0)")
    print("  You may need to:")
    print("    - Grant camera permissions")
    print("    - Try a different camera index")
    print("    - Check if another app is using the camera")
    sys.exit(1)
EOF

if [ $? -ne 0 ]; then
    print_warning "Camera test failed. You may encounter issues when running the system."
fi

# Create test directories
print_info "Creating test directories..."
mkdir -p test_faces
mkdir -p exports
print_success "Directories created"

# Final summary
print_header "Setup Complete!"

print_success "All dependencies installed successfully"
print_info "You can now run the system with:"
echo ""
echo "    python face_reidentification_test.py"
echo ""
print_info "Or with the configuration file:"
echo ""
echo "    python face_reidentification_test.py --config config.py"
echo ""
print_info "Keyboard shortcuts:"
echo "    q - Quit"
echo "    s - Show statistics"
echo ""
print_info "For more information, see README.md"

# Check if running in virtual environment
if [[ "$VIRTUAL_ENV" != "" ]]; then
    print_info "Virtual environment is active: $VIRTUAL_ENV"
    print_warning "Remember to activate it before running: source venv/bin/activate"
fi

echo ""
print_success "Setup completed successfully! 🎉"
echo ""
