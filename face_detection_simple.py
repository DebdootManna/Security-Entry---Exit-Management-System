#!/usr/bin/env python3
"""
Simplified Face Detection and Tracking System
Security Entry & Exit Management System

This version works with Python 3.14+ and uses:
- YOLOv8-Face for detection
- Basic OpenCV features for face comparison
- Simple template matching for re-identification

No TensorFlow or DeepFace required!

Usage:
    python face_detection_simple.py
    Press 'q' to quit
"""

import time
import uuid
from collections import defaultdict
from datetime import datetime

import cv2
import numpy as np

try:
    from ultralytics import YOLO

    YOLO_AVAILABLE = True
except ImportError:
    print("ERROR: Ultralytics not installed. Run: pip install ultralytics")
    YOLO_AVAILABLE = False


class SimpleFaceDatabase:
    """
    Simple face database using template matching and histogram comparison.
    Works without deep learning face recognition.
    """

    def __init__(self, similarity_threshold=0.70):
        """
        Initialize simple face database.

        Args:
            similarity_threshold: Minimum similarity score (0-1) for matching
        """
        self.threshold = similarity_threshold
        self.faces = []  # List of stored face templates
        self.metadata = []  # List of visitor metadata

        print(f"✓ Simple face database initialized (threshold: {similarity_threshold})")

    def extract_features(self, face_img):
        """
        Extract simple features from face image using histograms.

        Args:
            face_img: Face image (BGR)

        Returns:
            features: Dictionary with histogram features
        """
        try:
            # Resize to standard size
            face_resized = cv2.resize(face_img, (128, 128))

            # Convert to grayscale and color spaces
            gray = cv2.cvtColor(face_resized, cv2.COLOR_BGR2GRAY)
            hsv = cv2.cvtColor(face_resized, cv2.COLOR_BGR2HSV)

            # Calculate histograms
            hist_gray = cv2.calcHist([gray], [0], None, [256], [0, 256])
            hist_hue = cv2.calcHist([hsv], [0], None, [180], [0, 180])
            hist_sat = cv2.calcHist([hsv], [1], None, [256], [0, 256])

            # Normalize histograms
            hist_gray = cv2.normalize(hist_gray, hist_gray).flatten()
            hist_hue = cv2.normalize(hist_hue, hist_hue).flatten()
            hist_sat = cv2.normalize(hist_sat, hist_sat).flatten()

            return {
                "template": face_resized,
                "gray": gray,
                "hist_gray": hist_gray,
                "hist_hue": hist_hue,
                "hist_sat": hist_sat,
            }
        except Exception as e:
            print(f"[ERROR] Feature extraction failed: {e}")
            return None

    def compare_faces(self, features1, features2):
        """
        Compare two face features and return similarity score.

        Args:
            features1: First face features
            features2: Second face features

        Returns:
            similarity: Score between 0 and 1 (1 = identical)
        """
        try:
            # Template matching
            result = cv2.matchTemplate(
                features1["gray"], features2["gray"], cv2.TM_CCOEFF_NORMED
            )
            template_score = np.max(result)

            # Histogram comparison
            gray_score = cv2.compareHist(
                features1["hist_gray"], features2["hist_gray"], cv2.HISTCMP_CORREL
            )
            hue_score = cv2.compareHist(
                features1["hist_hue"], features2["hist_hue"], cv2.HISTCMP_CORREL
            )
            sat_score = cv2.compareHist(
                features1["hist_sat"], features2["hist_sat"], cv2.HISTCMP_CORREL
            )

            # Combined score (weighted average)
            similarity = (
                template_score * 0.4
                + gray_score * 0.2
                + hue_score * 0.2
                + sat_score * 0.2
            )

            return max(0.0, min(1.0, similarity))  # Clamp to [0, 1]

        except Exception as e:
            print(f"[ERROR] Face comparison failed: {e}")
            return 0.0

    def add_face(self, face_img, visitor_id=None):
        """
        Add a new face to the database.

        Args:
            face_img: Face image (BGR)
            visitor_id: Optional UUID for the visitor

        Returns:
            visitor_id: UUID of the added face
        """
        if visitor_id is None:
            visitor_id = str(uuid.uuid4())

        features = self.extract_features(face_img)
        if features is None:
            return None

        self.faces.append(features)
        self.metadata.append(
            {
                "visitor_id": visitor_id,
                "first_seen": datetime.now(),
                "last_seen": datetime.now(),
                "encounter_count": 1,
            }
        )

        print(f"[DB] Added visitor: {visitor_id[:8]}... (Total: {len(self.faces)})")
        return visitor_id

    def search_face(self, face_img):
        """
        Search for a matching face in the database.

        Args:
            face_img: Face image to search for

        Returns:
            match_found: Boolean
            visitor_id: UUID if match found
            similarity: Best similarity score
        """
        if len(self.faces) == 0:
            return False, None, 0.0

        query_features = self.extract_features(face_img)
        if query_features is None:
            return False, None, 0.0

        best_score = 0.0
        best_idx = -1

        # Compare with all stored faces
        for idx, stored_features in enumerate(self.faces):
            score = self.compare_faces(query_features, stored_features)
            if score > best_score:
                best_score = score
                best_idx = idx

        # Check if best match exceeds threshold
        if best_score >= self.threshold and best_idx >= 0:
            # Match found - update metadata
            visitor_id = self.metadata[best_idx]["visitor_id"]
            self.metadata[best_idx]["last_seen"] = datetime.now()
            self.metadata[best_idx]["encounter_count"] += 1

            print(
                f"[MATCH] Recognized: {visitor_id[:8]}... "
                f"(Score: {best_score:.3f}, Encounters: {self.metadata[best_idx]['encounter_count']})"
            )

            return True, visitor_id, best_score
        else:
            print(
                f"[NO MATCH] New visitor (Best score: {best_score:.3f} < {self.threshold})"
            )
            return False, None, best_score


class SimpleFaceRecognitionSystem:
    """
    Simple face recognition system using YOLO detection and template matching.
    """

    def __init__(self, yolo_model="yolov8n-face.pt", confidence_threshold=0.8):
        """
        Initialize the system.

        Args:
            yolo_model: Path to YOLO model
            confidence_threshold: Minimum detection confidence
        """
        self.confidence_threshold = confidence_threshold

        # Initialize detector
        if YOLO_AVAILABLE:
            try:
                print(f"Loading YOLO model: {yolo_model}...")
                self.detector = YOLO(yolo_model)
                print("✓ YOLO detector loaded")
            except Exception as e:
                print(f"ERROR loading YOLO: {e}")
                print("Trying standard YOLOv8n...")
                try:
                    self.detector = YOLO("yolov8n.pt")
                    print("✓ Using standard YOLOv8n")
                except:
                    self.detector = None
        else:
            self.detector = None

        # Initialize database
        self.database = SimpleFaceDatabase(similarity_threshold=0.70)

        # Performance tracking
        self.frame_times = []
        self.detection_count = 0
        self.recognition_count = 0

    def detect_faces(self, frame):
        """
        Detect faces in frame using YOLO.

        Args:
            frame: Input frame (BGR)

        Returns:
            detections: List of detected faces with bounding boxes
        """
        if self.detector is None:
            return []

        detections = []

        try:
            results = self.detector(
                frame, conf=self.confidence_threshold, verbose=False
            )

            for result in results:
                if result.boxes is not None:
                    for box in result.boxes:
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        confidence = float(box.conf[0])

                        # Ensure valid bounding box
                        x1, y1 = max(0, x1), max(0, y1)
                        x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)

                        if x2 > x1 and y2 > y1:
                            face_crop = frame[y1:y2, x1:x2]

                            detections.append(
                                {
                                    "bbox": (x1, y1, x2, y2),
                                    "confidence": confidence,
                                    "face_img": face_crop,
                                }
                            )

                            self.detection_count += 1

            if detections:
                print(f"[DETECT] Found {len(detections)} face(s)")

        except Exception as e:
            print(f"[ERROR] Detection failed: {e}")

        return detections

    def process_frame(self, frame):
        """
        Process a single frame through the pipeline.

        Args:
            frame: Input frame (BGR)

        Returns:
            annotated_frame: Frame with annotations
            results: List of recognition results
        """
        start_time = time.time()

        # Detect faces
        detections = self.detect_faces(frame)

        results = []

        # Process each detected face
        for detection in detections:
            face_img = detection["face_img"]
            bbox = detection["bbox"]
            confidence = detection["confidence"]

            # Search in database
            is_match, visitor_id, similarity = self.database.search_face(face_img)

            if not is_match:
                # New visitor - add to database
                visitor_id = self.database.add_face(face_img)
            else:
                self.recognition_count += 1

            results.append(
                {
                    "bbox": bbox,
                    "confidence": confidence,
                    "is_match": is_match,
                    "visitor_id": visitor_id,
                    "similarity": similarity,
                }
            )

        # Annotate frame
        annotated_frame = self.annotate_frame(frame, results)

        # Track performance
        elapsed = time.time() - start_time
        self.frame_times.append(elapsed)

        return annotated_frame, results

    def annotate_frame(self, frame, results):
        """
        Draw annotations on frame.

        Args:
            frame: Input frame
            results: Detection and recognition results

        Returns:
            annotated_frame: Frame with annotations
        """
        annotated = frame.copy()

        for result in results:
            x1, y1, x2, y2 = result["bbox"]
            confidence = result["confidence"]
            is_match = result["is_match"]
            visitor_id = result["visitor_id"]
            similarity = result["similarity"]

            # Determine color and label
            if is_match:
                color = (0, 255, 0)  # Green for recognized
                status = "RECOGNIZED"
            else:
                color = (0, 165, 255)  # Orange for new
                status = "NEW VISITOR"

            # Draw bounding box
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)

            # Prepare labels
            visitor_short = visitor_id[:8] if visitor_id else "N/A"
            labels = [
                f"{status}",
                f"ID: {visitor_short}",
                f"Conf: {confidence:.2f}",
                f"Score: {similarity:.3f}",
            ]

            # Draw labels
            y_offset = y1 - 10
            for i, text in enumerate(labels):
                y_pos = y_offset - (len(labels) - 1 - i) * 20

                # Background rectangle
                (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
                cv2.rectangle(
                    annotated, (x1, y_pos - th - 4), (x1 + tw + 4, y_pos + 2), color, -1
                )

                # Text
                cv2.putText(
                    annotated,
                    text,
                    (x1 + 2, y_pos),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.4,
                    (255, 255, 255),
                    1,
                )

        # Add system stats
        if len(self.frame_times) > 0:
            avg_fps = 1.0 / (
                sum(self.frame_times[-30:]) / min(len(self.frame_times), 30)
            )

            stats = [
                f"FPS: {avg_fps:.1f}",
                f"Detections: {self.detection_count}",
                f"Recognitions: {self.recognition_count}",
                f"DB Size: {len(self.database.faces)}",
            ]

            for i, text in enumerate(stats):
                cv2.putText(
                    annotated,
                    text,
                    (10, 30 + i * 25),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 255),
                    2,
                )

        return annotated

    def get_statistics(self):
        """Get system statistics."""
        if len(self.frame_times) > 0:
            avg_time = sum(self.frame_times) / len(self.frame_times)
            avg_fps = 1.0 / avg_time if avg_time > 0 else 0
        else:
            avg_time = 0
            avg_fps = 0

        return {
            "avg_fps": avg_fps,
            "avg_processing_time": avg_time,
            "total_detections": self.detection_count,
            "total_recognitions": self.recognition_count,
            "unique_visitors": len(self.database.faces),
        }


def main():
    """Main function."""
    print("=" * 70)
    print("Simplified Face Recognition System")
    print("Security Entry & Exit Management System")
    print("=" * 70)
    print()

    if not YOLO_AVAILABLE:
        print("ERROR: Ultralytics required. Install with:")
        print("  pip install ultralytics")
        return

    print("Initializing system...")
    system = SimpleFaceRecognitionSystem(
        yolo_model="yolov8n-face.pt", confidence_threshold=0.8
    )

    print()
    print("=" * 70)
    print("Starting webcam...")
    print("Press 'q' to quit, 's' for statistics")
    print("=" * 70)
    print()

    # Open webcam
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("ERROR: Could not open webcam")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    print("✓ Webcam opened")
    print()

    frame_count = 0

    try:
        while True:
            ret, frame = cap.read()

            if not ret:
                print("ERROR: Failed to capture frame")
                break

            frame_count += 1

            # Process frame
            annotated_frame, results = system.process_frame(frame)

            # Display
            cv2.imshow("Simple Face Recognition System", annotated_frame)

            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                print("\nQuitting...")
                break
            elif key == ord("s"):
                stats = system.get_statistics()
                print("\n" + "=" * 70)
                print("STATISTICS")
                print("=" * 70)
                print(f"Average FPS: {stats['avg_fps']:.2f}")
                print(f"Average Processing Time: {stats['avg_processing_time']:.4f}s")
                print(f"Total Detections: {stats['total_detections']}")
                print(f"Total Recognitions: {stats['total_recognitions']}")
                print(f"Unique Visitors: {stats['unique_visitors']}")
                print("=" * 70)
                print()

    except KeyboardInterrupt:
        print("\nInterrupted by user")

    finally:
        cap.release()
        cv2.destroyAllWindows()

        # Final statistics
        print("\n" + "=" * 70)
        print("FINAL STATISTICS")
        print("=" * 70)
        stats = system.get_statistics()
        print(f"Total Frames: {frame_count}")
        print(f"Average FPS: {stats['avg_fps']:.2f}")
        print(f"Total Detections: {stats['total_detections']}")
        print(f"Total Recognitions: {stats['total_recognitions']}")
        print(f"Unique Visitors: {stats['unique_visitors']}")
        print(
            f"Recognition Rate: {(stats['total_recognitions'] / max(stats['total_detections'], 1)) * 100:.1f}%"
        )
        print("=" * 70)
        print("\nVisitor Database:")
        for i, meta in enumerate(system.database.metadata):
            print(f"  Visitor {i + 1}: {meta['visitor_id'][:8]}...")
            print(f"    First: {meta['first_seen'].strftime('%H:%M:%S')}")
            print(f"    Last:  {meta['last_seen'].strftime('%H:%M:%S')}")
            print(f"    Count: {meta['encounter_count']}")
        print("=" * 70)
        print("\nTest completed!")


if __name__ == "__main__":
    main()
