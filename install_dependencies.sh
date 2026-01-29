#!/bin/bash

# Fixed Installation Script for Face Re-Identification System (macOS)
# This script properly installs all dependencies for Python 3.14.2

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo "================================================================"
echo "Face Re-Identification System - Dependency Installation"
echo "================================================================"
echo ""

# Get Python version
PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}')
echo -e "${BLUE}Using Python: $PYTHON_VERSION${NC}"
echo ""

# Upgrade pip first
echo -e "${BLUE}Step 1: Upgrading pip...${NC}"
python3 -m pip install --upgrade pip
echo -e "${GREEN}✓ pip upgraded${NC}"
echo ""

# Install dependencies one by one with progress
echo -e "${BLUE}Step 2: Installing core dependencies...${NC}"
echo ""

# Core dependencies
echo "Installing numpy..."
python3 -m pip install numpy>=1.24.0
echo -e "${GREEN}✓ numpy installed${NC}"
echo ""

echo "Installing opencv-python..."
python3 -m pip install opencv-python>=4.8.0
echo -e "${GREEN}✓ opencv-python installed${NC}"
echo ""

echo "Installing pillow..."
python3 -m pip install pillow>=10.0.0
echo -e "${GREEN}✓ pillow installed${NC}"
echo ""

echo "Installing scipy..."
python3 -m pip install scipy>=1.11.0
echo -e "${GREEN}✓ scipy installed${NC}"
echo ""

# Deep learning dependencies
echo -e "${BLUE}Step 3: Installing deep learning dependencies...${NC}"
echo ""

echo "Installing tensorflow (this may take a few minutes)..."
python3 -m pip install tensorflow>=2.13.0
echo -e "${GREEN}✓ tensorflow installed${NC}"
echo ""

echo "Installing tf-keras..."
python3 -m pip install tf-keras>=2.15.0
echo -e "${GREEN}✓ tf-keras installed${NC}"
echo ""

# Face recognition dependencies
echo -e "${BLUE}Step 4: Installing face recognition libraries...${NC}"
echo ""

echo "Installing ultralytics (YOLO)..."
python3 -m pip install ultralytics>=8.0.0
echo -e "${GREEN}✓ ultralytics installed${NC}"
echo ""

echo "Installing mtcnn..."
python3 -m pip install mtcnn>=0.1.1
echo -e "${GREEN}✓ mtcnn installed${NC}"
echo ""

echo "Installing deepface..."
python3 -m pip install deepface>=0.0.79
echo -e "${GREEN}✓ deepface installed${NC}"
echo ""

# Similarity search
echo -e "${BLUE}Step 5: Installing FAISS...${NC}"
echo ""

echo "Installing faiss-cpu..."
python3 -m pip install faiss-cpu>=1.7.4
echo -e "${GREEN}✓ faiss-cpu installed${NC}"
echo ""

# Verify installations
echo "================================================================"
echo "Verifying installations..."
echo "================================================================"
echo ""

python3 << 'EOF'
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
    "tf-keras": "tf_keras"
}

print("Verification Results:")
print("-" * 50)

failed = []
for name, module in modules.items():
    try:
        __import__(module)
        print(f"✓ {name} installed successfully")
    except ImportError as e:
        print(f"✗ {name} NOT installed: {e}")
        failed.append(name)

print("-" * 50)

if failed:
    print(f"\n⚠️  WARNING: The following modules failed: {', '.join(failed)}")
    print("\nTry installing them manually:")
    for f in failed:
        module_name = modules.get(f, f).replace('_', '-')
        print(f"  python3 -m pip install {module_name}")
    sys.exit(1)
else:
    print("\n✅ All modules verified successfully!")
    sys.exit(0)
EOF

VERIFY_RESULT=$?

echo ""
echo "================================================================"
if [ $VERIFY_RESULT -eq 0 ]; then
    echo -e "${GREEN}Installation completed successfully!${NC}"
    echo "================================================================"
    echo ""
    echo "You can now run the system:"
    echo "  python3 face_reidentification_test.py"
    echo ""
else
    echo -e "${YELLOW}Installation completed with some warnings.${NC}"
    echo "================================================================"
    echo ""
    echo "Some modules may need manual installation."
    echo "Check the error messages above."
    echo ""
fi

# Test camera access
echo "Testing camera access..."
python3 << 'EOF'
import cv2
import sys

try:
    cap = cv2.VideoCapture(0)
    if cap.isOpened():
        ret, frame = cap.read()
        if ret:
            print(f"✓ Webcam accessible (Resolution: {frame.shape[1]}x{frame.shape[0]})")
        else:
            print("⚠️  Webcam opened but could not read frame")
        cap.release()
    else:
        print("⚠️  Could not open webcam (this may be a permissions issue)")
        print("   Make sure to grant camera permissions in System Settings")
except Exception as e:
    print(f"⚠️  Camera test error: {e}")
EOF

echo ""
echo "================================================================"
echo "Setup complete! Ready to run."
echo "================================================================"
