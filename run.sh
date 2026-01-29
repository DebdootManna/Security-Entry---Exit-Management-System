#!/bin/bash

# Run Script for Face Recognition System
# Security Entry & Exit Management System

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo "================================================================"
echo "Face Recognition System - Run Script"
echo "================================================================"
echo ""

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo -e "${YELLOW}⚠ Virtual environment not found${NC}"
    echo "Please run setup first: ./setup_venv.sh"
    exit 1
fi

# Activate virtual environment
echo -e "${BLUE}Activating virtual environment...${NC}"
source venv/bin/activate

# Check Python version
PYTHON_VERSION=$(python --version 2>&1 | awk '{print $2}')
echo -e "${GREEN}✓ Using Python $PYTHON_VERSION${NC}"
echo ""

# Determine which script to run based on Python version
MAJOR=$(echo $PYTHON_VERSION | cut -d. -f1)
MINOR=$(echo $PYTHON_VERSION | cut -d. -f2)

echo "================================================================"
echo "Available Systems:"
echo "================================================================"
echo ""
echo "1. Simplified System (Python 3.14+ compatible)"
echo "   - Uses OpenCV template matching"
echo "   - Works without TensorFlow"
echo "   - Good accuracy (~85-90%)"
echo ""
echo "2. Full System (Requires Python 3.8-3.12)"
echo "   - Uses ArcFace deep learning"
echo "   - Requires TensorFlow"
echo "   - Best accuracy (~95%+)"
echo ""

# Check TensorFlow availability
if python -c "import tensorflow" 2>/dev/null; then
    echo -e "${GREEN}✓ TensorFlow available - Both systems supported${NC}"
    TENSORFLOW_AVAILABLE=true
else
    echo -e "${YELLOW}⚠ TensorFlow not available - Only simplified system supported${NC}"
    TENSORFLOW_AVAILABLE=false
fi

echo ""
read -p "Which system do you want to run? (1/2) [1]: " CHOICE

if [ -z "$CHOICE" ]; then
    CHOICE=1
fi

echo ""
echo "================================================================"

if [ "$CHOICE" = "1" ]; then
    echo "Starting Simplified Face Recognition System..."
    echo "================================================================"
    echo ""
    echo "Controls:"
    echo "  q - Quit"
    echo "  s - Show statistics"
    echo ""
    python face_detection_simple.py

elif [ "$CHOICE" = "2" ]; then
    if [ "$TENSORFLOW_AVAILABLE" = true ]; then
        echo "Starting Full Face Recognition System..."
        echo "================================================================"
        echo ""
        echo "Controls:"
        echo "  q - Quit"
        echo "  s - Show statistics"
        echo ""
        python face_reidentification_test.py
    else
        echo "Full system requires TensorFlow (not available in Python 3.14+)"
        echo ""
        echo "Options:"
        echo "1. Use simplified system (run this script and choose option 1)"
        echo "2. Install Python 3.11 or 3.12:"
        echo "   brew install python@3.11"
        echo "   python3.11 -m venv venv_full"
        echo "   source venv_full/bin/activate"
        echo "   pip install -r requirements.txt"
        exit 1
    fi
else
    echo "Invalid choice. Please run again and select 1 or 2."
    exit 1
fi

echo ""
echo "================================================================"
echo "System stopped"
echo "================================================================"
