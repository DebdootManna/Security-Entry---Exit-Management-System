#!/bin/bash

# Virtual Environment Setup for Face Re-Identification System
# For macOS with externally-managed Python (Homebrew)

set -e  # Exit on error

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

echo "================================================================"
echo "Face Re-Identification System - Virtual Environment Setup"
echo "================================================================"
echo ""

# Check Python
echo -e "${BLUE}Checking Python installation...${NC}"
if ! command -v python3 &> /dev/null; then
    echo -e "${RED}✗ Python 3 not found. Please install Python 3.8+${NC}"
    exit 1
fi

PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}')
echo -e "${GREEN}✓ Found Python $PYTHON_VERSION${NC}"
echo ""

# Create virtual environment
echo -e "${BLUE}Creating virtual environment...${NC}"
if [ -d "venv" ]; then
    echo -e "${YELLOW}⚠ Virtual environment already exists${NC}"
    read -p "Do you want to recreate it? (y/N): " RECREATE
    if [[ $RECREATE =~ ^[Yy]$ ]]; then
        echo "Removing existing venv..."
        rm -rf venv
    else
        echo "Using existing virtual environment"
    fi
fi

if [ ! -d "venv" ]; then
    python3 -m venv venv
    echo -e "${GREEN}✓ Virtual environment created${NC}"
else
    echo -e "${GREEN}✓ Virtual environment ready${NC}"
fi
echo ""

# Activate virtual environment
echo -e "${BLUE}Activating virtual environment...${NC}"
source venv/bin/activate
echo -e "${GREEN}✓ Virtual environment activated${NC}"
echo ""

# Upgrade pip
echo -e "${BLUE}Upgrading pip...${NC}"
python -m pip install --upgrade pip --quiet
echo -e "${GREEN}✓ pip upgraded${NC}"
echo ""

# Install dependencies
echo -e "${BLUE}Installing dependencies (this may take 5-10 minutes)...${NC}"
echo ""

DEPENDENCIES=(
    "numpy>=1.24.0"
    "opencv-python>=4.8.0"
    "pillow>=10.0.0"
    "scipy>=1.11.0"
    "tensorflow>=2.13.0"
    "tf-keras>=2.15.0"
    "ultralytics>=8.0.0"
    "mtcnn>=0.1.1"
    "deepface>=0.0.79"
    "faiss-cpu>=1.7.4"
)

FAILED=()

for dep in "${DEPENDENCIES[@]}"; do
    DEP_NAME=$(echo $dep | cut -d'>' -f1 | cut -d'=' -f1)
    echo -ne "Installing ${DEP_NAME}... "

    if python -m pip install "$dep" --quiet 2>/dev/null; then
        echo -e "${GREEN}✓${NC}"
    else
        echo -e "${RED}✗${NC}"
        FAILED+=("$DEP_NAME")
    fi
done

echo ""

# Verify installations
echo "================================================================"
echo "Verifying installations..."
echo "================================================================"
echo ""

python << 'EOF'
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
    print(f"\n⚠️  WARNING: Failed modules: {', '.join(failed)}")
    sys.exit(1)
else:
    print("\n✅ All modules verified successfully!")
    sys.exit(0)
EOF

VERIFY_RESULT=$?

echo ""

# Test camera
if [ $VERIFY_RESULT -eq 0 ]; then
    echo "Testing camera access..."
    python << 'EOF'
import cv2
try:
    cap = cv2.VideoCapture(0)
    if cap.isOpened():
        ret, frame = cap.read()
        if ret:
            print(f"✓ Camera accessible ({frame.shape[1]}x{frame.shape[0]})")
        else:
            print("⚠️  Camera opened but can't read frames")
        cap.release()
    else:
        print("⚠️  Can't open camera (check permissions in System Settings)")
except Exception as e:
    print(f"⚠️  Camera test error: {e}")
EOF
    echo ""
fi

# Final instructions
echo "================================================================"
if [ $VERIFY_RESULT -eq 0 ]; then
    echo -e "${GREEN}Setup completed successfully!${NC}"
else
    echo -e "${YELLOW}Setup completed with warnings.${NC}"
fi
echo "================================================================"
echo ""
echo "To use the system:"
echo ""
echo "1. Activate the virtual environment:"
echo -e "   ${BLUE}source venv/bin/activate${NC}"
echo ""
echo "2. Run the system:"
echo -e "   ${BLUE}python face_reidentification_test.py${NC}"
echo ""
echo "3. When done, deactivate:"
echo -e "   ${BLUE}deactivate${NC}"
echo ""
echo "Next time, just run:"
echo -e "   ${BLUE}source venv/bin/activate && python face_reidentification_test.py${NC}"
echo ""
echo "================================================================"
