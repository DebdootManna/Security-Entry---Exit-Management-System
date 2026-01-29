#!/usr/bin/env python3
"""
Face Detection Test - Static Image Version
Security Entry & Exit Management System

This version works with static images for testing without camera access.
Great for testing camera permissions or when webcam is unavailable.

Usage:
    python test_with_image.py
"""

import os
from datetime import datetime

import cv2
import numpy as np

try:
    from ultralytics import YOLO

    YOLO_AVAILABLE = True
except ImportError:
    print("ERROR: Ultralytics not installed. Run: pip install ultralytics")
    YOLO_AVAILABLE = False


def create_test_image():
    """Create a simple test image with a face placeholder."""
    # Create a white canvas
    img = np.ones((480, 640, 3), dtype=np.uint8) * 255

    # Draw a simple face representation
    center_x, center_y = 320, 240

    # Face circle (beige color)
    cv2.circle(img, (center_x, center_y), 80, (220, 200, 180), -1)

    # Eyes
    cv2.circle(img, (center_x - 30, center_y - 20), 10, (50, 50, 50), -1)
    cv2.circle(img, (center_x + 30, center_y - 20), 10, (50, 50, 50), -1)

    # Nose
    cv2.line(
        img, (center_x, center_y), (center_x - 10, center_y + 20), (150, 120, 100), 3
    )

    # Mouth
    cv2.ellipse(img, (center_x, center_y + 30), (30, 15), 0, 0, 180, (150, 100, 100), 2)

    # Add text
    cv2.putText(
        img,
        "Test Image - Grant camera permissions to use webcam",
        (50, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 0, 255),
        2,
    )
    cv2.putText(
        img,
        "System Settings > Privacy & Security > Camera > Terminal",
        (50, 450),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (100, 100, 100),
        1,
    )

    return img


def test_yolo_detection():
    """Test YOLO face detection on a static image."""
    print("=" * 70)
    print("Face Detection Test - Static Image Mode")
    print("=" * 70)
    print()

    if not YOLO_AVAILABLE:
        print("ERROR: Ultralytics not installed")
        return

    # Load YOLO model
    print("Loading YOLO model...")
    try:
        detector = YOLO("yolov8n.pt")  # Using standard YOLOv8n
        print("✓ YOLO model loaded")
    except Exception as e:
        print(f"ERROR loading YOLO: {e}")
        return

    print()
    print("=" * 70)
    print("Test Options:")
    print("=" * 70)
    print()
    print("1. Test with generated placeholder image")
    print("2. Test with your own image file")
    print("3. Instructions to fix camera permissions")
    print("0. Exit")
    print()

    choice = input("Enter your choice (0-3) [1]: ").strip()
    if not choice:
        choice = "1"

    print()

    if choice == "0":
        print("Exiting...")
        return

    elif choice == "1":
        # Use generated test image
        print("Using generated test image...")
        test_img = create_test_image()

    elif choice == "2":
        # Use user's image
        img_path = input("Enter path to image file: ").strip()
        if not os.path.exists(img_path):
            print(f"ERROR: File not found: {img_path}")
            return

        test_img = cv2.imread(img_path)
        if test_img is None:
            print(f"ERROR: Could not read image: {img_path}")
            return
        print(f"✓ Loaded image: {img_path}")

    elif choice == "3":
        print("=" * 70)
        print("How to Fix Camera Permissions on macOS")
        print("=" * 70)
        print()
        print("1. Open 'System Settings' (or System Preferences)")
        print("2. Go to 'Privacy & Security'")
        print("3. Click on 'Camera' in the left sidebar")
        print("4. Find your terminal app in the list:")
        print("   - Terminal")
        print("   - iTerm2")
        print("   - VS Code Terminal")
        print("   - etc.")
        print("5. Toggle it ON to allow camera access")
        print("6. IMPORTANT: Restart your terminal app")
        print("7. Try running the main script again:")
        print()
        print("   python face_detection_simple.py")
        print()
        print("=" * 70)
        return

    else:
        print("Invalid choice")
        return

    print()
    print("Running detection...")
    print()

    # Run YOLO detection
    results = detector(test_img, conf=0.5, verbose=False)

    # Process results
    annotated = test_img.copy()
    detection_count = 0

    for result in results:
        if result.boxes is not None:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                confidence = float(box.conf[0])
                class_id = int(box.cls[0])

                detection_count += 1

                # Draw bounding box
                cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)

                # Draw label
                label = f"Detection {detection_count}: {confidence:.2f}"
                cv2.putText(
                    annotated,
                    label,
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),
                    2,
                )

    print(f"✓ Detection complete: Found {detection_count} object(s)")
    print()

    if detection_count == 0:
        print("Note: No faces detected in this image.")
        print("This is expected for the generated placeholder image.")
        print("Try option 2 with a real photo containing faces.")

    print()
    print("Displaying result...")
    print("Press any key to close the window")

    # Display result
    cv2.imshow("YOLO Detection Test", annotated)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    print()
    print("=" * 70)
    print("Test complete!")
    print()
    print("Once camera permissions are fixed, run:")
    print("  python face_detection_simple.py")
    print("=" * 70)


def test_camera_access():
    """Test if camera can be accessed."""
    print("=" * 70)
    print("Camera Access Test")
    print("=" * 70)
    print()
    print("Attempting to access camera...")
    print()

    cap = cv2.VideoCapture(0)

    if cap.isOpened():
        ret, frame = cap.read()
        if ret:
            print("✓ SUCCESS! Camera is accessible")
            print(f"  Resolution: {frame.shape[1]}x{frame.shape[0]}")
            print()
            print("You can now run the main script:")
            print("  python face_detection_simple.py")

            # Show frame
            cv2.imshow("Camera Test - Press any key to close", frame)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        else:
            print("✗ Camera opened but cannot read frames")
            print()
            print("Possible solutions:")
            print("1. Check if another app is using the camera")
            print("2. Restart your computer")
            print("3. Try a different camera (USB webcam)")

        cap.release()
    else:
        print("✗ FAILED: Cannot access camera")
        print()
        print("Camera permission not granted!")
        print()
        print("=" * 70)
        print("How to Fix:")
        print("=" * 70)
        print()
        print("1. Open 'System Settings'")
        print("2. Go to 'Privacy & Security' → 'Camera'")
        print("3. Enable camera access for your terminal app:")
        print("   ☐ Terminal")
        print("   ☐ iTerm2")
        print("   ☐ VS Code")
        print("   ☐ etc.")
        print("4. Restart your terminal")
        print("5. Run this test again")
        print()
        print("=" * 70)


def main():
    """Main menu."""
    print()
    print("=" * 70)
    print("Face Detection System - Testing Mode")
    print("=" * 70)
    print()
    print("This script helps you test the system and fix camera issues.")
    print()
    print("Options:")
    print("  1. Test YOLO detection with static images")
    print("  2. Test camera access")
    print("  3. View camera permission instructions")
    print("  0. Exit")
    print()

    choice = input("Enter your choice (0-3) [2]: ").strip()
    if not choice:
        choice = "2"

    print()

    if choice == "0":
        print("Exiting...")
        return

    elif choice == "1":
        test_yolo_detection()

    elif choice == "2":
        test_camera_access()

    elif choice == "3":
        print("=" * 70)
        print("macOS Camera Permission Instructions")
        print("=" * 70)
        print()
        print("Step-by-Step Guide:")
        print()
        print("1. Click the Apple menu (🍎) → System Settings")
        print()
        print("2. In the left sidebar, scroll down and click:")
        print("   'Privacy & Security'")
        print()
        print("3. In the right panel, find and click:")
        print("   'Camera'")
        print()
        print("4. You'll see a list of apps. Find your terminal:")
        print("   - Terminal (default macOS terminal)")
        print("   - iTerm2 (if you use iTerm)")
        print("   - Code (if using VS Code terminal)")
        print()
        print("5. Toggle the switch to ON (blue)")
        print()
        print("6. IMPORTANT: Completely quit and restart your terminal:")
        print("   - Press Cmd+Q to quit the terminal")
        print("   - Reopen the terminal")
        print()
        print("7. Navigate back to the project and try again:")
        print("   cd 'Security Entry & Exit Management System'")
        print("   source venv/bin/activate")
        print("   python face_detection_simple.py")
        print()
        print("=" * 70)
        print()
        print("Still not working? Try:")
        print("- Restart your Mac")
        print("- Use a USB webcam instead of built-in camera")
        print("- Run: sudo killall VDCAssistant (resets camera daemon)")
        print()
        print("=" * 70)

    else:
        print("Invalid choice")


if __name__ == "__main__":
    main()
