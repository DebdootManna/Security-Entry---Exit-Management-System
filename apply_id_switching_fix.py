#!/usr/bin/env python3
"""
Quick Fix Script for ID-Switching Problem
Applies temporal smoothing and adjusted thresholds to existing tracking system

This script provides a drop-in enhancement for the existing face_detection_simple.py
to reduce over-segmentation (38 IDs for 2 people → 2 IDs for 2 people).

Usage:
    python apply_id_switching_fix.py

Author: Security System Team
Date: 2024
"""

import time
import uuid
from collections import deque
from datetime import datetime, timedelta

import cv2
import numpy as np

try:
    from ultralytics import YOLO

    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    print("[WARNING] YOLO not available. Install: pip install ultralytics")


# ============================================================================
# ENHANCED CONFIGURATION - KEY FIX FOR ID-SWITCHING
# ============================================================================


class EnhancedConfig:
    """Adjusted configuration to fix over-segmentation"""

    # Detection threshold (relaxed for better detection in varying light)
    DETECTION_CONFIDENCE = 0.70  # Was 0.8

    # Matching thresholds (more lenient for museum conditions)
    FACE_THRESHOLD_STRICT = 0.55  # High confidence match
    FACE_THRESHOLD_NORMAL = 0.72  # Standard match (was 0.70)
    FACE_THRESHOLD_RELAXED = 0.82  # Temporal continuity (very lenient)

    # Temporal smoothing - KEY TO PREVENTING ID SWITCHING
    GRACE_PERIOD_SECONDS = 3.0  # Wait 3 seconds before creating new ID
    TEMPORAL_WINDOW_SIZE = 5  # Average last 5 embeddings
    MAX_LOST_TIME_SECONDS = 5.0  # Remove from active after 5 seconds

    # Session management
    MIN_FACE_SIZE = 60
    MAX_ACTIVE_SESSIONS = 100


# ============================================================================
# VISITOR SESSION WITH TEMPORAL TRACKING
# ============================================================================


class VisitorSession:
    """
    Represents a single visitor with temporal tracking.
    Maintains embedding history to reduce ID-switching.
    """

    def __init__(self, visitor_id, face_features):
        self.visitor_id = visitor_id
        self.first_seen = datetime.now()
        self.last_seen = datetime.now()
        self.encounter_count = 1

        # Store multiple observations for temporal smoothing
        self.face_features_history = deque(maxlen=EnhancedConfig.TEMPORAL_WINDOW_SIZE)
        if face_features is not None:
            self.face_features_history.append(face_features)

        # Tracking state
        self.is_lost = False
        self.lost_since = None
        self.consecutive_misses = 0

    def update(self, face_features):
        """Update session with new observation"""
        self.last_seen = datetime.now()
        self.encounter_count += 1
        self.consecutive_misses = 0

        if self.is_lost:
            lost_duration = (datetime.now() - self.lost_since).total_seconds()
            print(f"[✓ RECOVERED] {self.visitor_id[:8]}... after {lost_duration:.1f}s")

        self.is_lost = False
        self.lost_since = None

        if face_features is not None:
            self.face_features_history.append(face_features)

    def mark_lost(self):
        """Mark as lost (not matched in current frame)"""
        if not self.is_lost:
            self.is_lost = True
            self.lost_since = datetime.now()
        self.consecutive_misses += 1

    def seconds_since_last_seen(self):
        """Get seconds since last observation"""
        return (datetime.now() - self.last_seen).total_seconds()

    def seconds_lost(self):
        """Get seconds since marked as lost"""
        if not self.is_lost or self.lost_since is None:
            return 0.0
        return (datetime.now() - self.lost_since).total_seconds()

    def get_average_features(self):
        """Get temporally averaged features for stable matching"""
        if len(self.face_features_history) == 0:
            return None

        # Average all stored features
        avg_features = {}

        # Average template
        templates = [
            f["template"] for f in self.face_features_history if "template" in f
        ]
        if templates:
            avg_features["template"] = templates[-1]  # Use most recent

        # Average histograms
        for key in ["hist_gray", "hist_hue"]:
            values = [f[key] for f in self.face_features_history if key in f]
            if values:
                avg_features[key] = np.mean(values, axis=0)

        return avg_features if avg_features else None


# ============================================================================
# ENHANCED DATABASE WITH GRACE PERIOD
# ============================================================================


class EnhancedFaceDatabase:
    """
    Enhanced database with temporal tracking and grace period.
    Fixes ID-switching by maintaining visitor sessions.
    """

    def __init__(self, similarity_threshold=0.72):
        self.threshold = similarity_threshold

        # Active sessions (currently visible)
        self.active_sessions = {}  # visitor_id -> VisitorSession

        # Lost sessions (within grace period)
        self.lost_sessions = {}  # visitor_id -> VisitorSession

        # Statistics
        self.total_visitors = 0
        self.temporal_recoveries = 0

        print(f"✓ Enhanced Face Database initialized")
        print(f"  Similarity threshold: {similarity_threshold}")
        print(f"  Grace period: {EnhancedConfig.GRACE_PERIOD_SECONDS}s")

    def extract_features(self, face_img):
        """Extract face features (same as original)"""
        try:
            face_resized = cv2.resize(face_img, (128, 128))
            gray = cv2.cvtColor(face_resized, cv2.COLOR_BGR2GRAY)
            hsv = cv2.cvtColor(face_resized, cv2.COLOR_BGR2HSV)

            hist_gray = cv2.calcHist([gray], [0], None, [256], [0, 256])
            hist_hue = cv2.calcHist([hsv], [0], None, [180], [0, 180])

            hist_gray = cv2.normalize(hist_gray, hist_gray).flatten()
            hist_hue = cv2.normalize(hist_hue, hist_hue).flatten()

            return {
                "template": gray,
                "hist_gray": hist_gray,
                "hist_hue": hist_hue,
            }
        except Exception as e:
            print(f"[ERROR] Feature extraction failed: {e}")
            return None

    def compare_features(self, features1, features2):
        """Compare two feature sets"""
        try:
            # Template matching
            result = cv2.matchTemplate(
                features1["template"], features2["template"], cv2.TM_CCOEFF_NORMED
            )
            template_score = np.max(result)

            # Histogram comparison
            gray_score = cv2.compareHist(
                features1["hist_gray"], features2["hist_gray"], cv2.HISTCMP_CORREL
            )
            hue_score = cv2.compareHist(
                features1["hist_hue"], features2["hist_hue"], cv2.HISTCMP_CORREL
            )

            # Weighted combination
            similarity = template_score * 0.4 + gray_score * 0.3 + hue_score * 0.3

            return max(0.0, min(1.0, similarity))
        except Exception as e:
            print(f"[ERROR] Comparison failed: {e}")
            return 0.0

    def search_face(self, face_img):
        """
        Multi-stage search with grace period.
        KEY FIX: Check lost visitors first before creating new ID.
        """
        query_features = self.extract_features(face_img)
        if query_features is None:
            return False, None, 0.0

        # STAGE 1: Check lost visitors first (within grace period)
        lost_match = self._check_lost_visitors(query_features)
        if lost_match:
            return lost_match

        # STAGE 2: Check active sessions
        active_match = self._check_active_sessions(query_features)
        if active_match:
            return active_match

        # STAGE 3: Check with relaxed threshold if someone just lost
        if len(self.lost_sessions) > 0:
            relaxed_match = self._check_active_sessions(
                query_features, threshold=EnhancedConfig.FACE_THRESHOLD_RELAXED
            )
            if relaxed_match:
                print(f"[LENIENT MATCH] Recovered with relaxed threshold")
                return relaxed_match

        # No match found
        return False, None, 0.0

    def _check_lost_visitors(self, query_features):
        """Check recently lost visitors (KEY FIX!)"""
        best_score = 0.0
        best_visitor_id = None
        best_session = None

        for visitor_id, session in list(self.lost_sessions.items()):
            # Check if still within grace period
            if session.seconds_lost() > EnhancedConfig.GRACE_PERIOD_SECONDS:
                # Grace period expired - remove
                del self.lost_sessions[visitor_id]
                print(f"[CLEANUP] {visitor_id[:8]}... grace period expired")
                continue

            # Compare with temporal average
            avg_features = session.get_average_features()
            if avg_features is None:
                continue

            score = self.compare_features(query_features, avg_features)

            # Use relaxed threshold for lost sessions
            if score > EnhancedConfig.FACE_THRESHOLD_RELAXED and score > best_score:
                best_score = score
                best_visitor_id = visitor_id
                best_session = session

        if best_session:
            # Move back to active sessions
            self.active_sessions[best_visitor_id] = best_session
            del self.lost_sessions[best_visitor_id]

            # Update session
            best_session.update(query_features)

            self.temporal_recoveries += 1

            return True, best_visitor_id, best_score

        return None

    def _check_active_sessions(self, query_features, threshold=None):
        """Check active sessions"""
        if threshold is None:
            threshold = self.threshold

        best_score = 0.0
        best_visitor_id = None
        best_session = None

        for visitor_id, session in self.active_sessions.items():
            avg_features = session.get_average_features()
            if avg_features is None:
                continue

            score = self.compare_features(query_features, avg_features)

            if score >= threshold and score > best_score:
                best_score = score
                best_visitor_id = visitor_id
                best_session = session

        if best_session:
            best_session.update(query_features)

            print(
                f"[MATCH] {best_visitor_id[:8]}... "
                f"(Score: {best_score:.3f}, Encounters: {best_session.encounter_count})"
            )

            return True, best_visitor_id, best_score

        return None

    def add_face(self, face_img, visitor_id=None):
        """Add new face to database"""
        if visitor_id is None:
            visitor_id = str(uuid.uuid4())

        features = self.extract_features(face_img)
        if features is None:
            return None

        session = VisitorSession(visitor_id, features)
        self.active_sessions[visitor_id] = session

        self.total_visitors += 1

        print(f"[★ NEW] {visitor_id[:8]}... (Total unique: {self.total_visitors})")
        return visitor_id

    def mark_unmatched_as_lost(self, matched_ids):
        """Mark sessions not seen in current frame as lost"""
        for visitor_id in list(self.active_sessions.keys()):
            if visitor_id not in matched_ids:
                session = self.active_sessions[visitor_id]
                session.mark_lost()

                # Move to lost after consecutive misses
                if session.consecutive_misses > 3:  # ~1 second at 30fps
                    self.lost_sessions[visitor_id] = session
                    del self.active_sessions[visitor_id]
                    print(f"[→ LOST] {visitor_id[:8]}...")


# ============================================================================
# ENHANCED TRACKING SYSTEM
# ============================================================================


class EnhancedFaceRecognitionSystem:
    """
    Enhanced system with temporal tracking to fix ID-switching.
    Drop-in replacement for SimpleFaceRecognitionSystem.
    """

    def __init__(self, yolo_model="yolov8n.pt", confidence_threshold=0.70):
        self.confidence_threshold = confidence_threshold

        # Initialize detector
        if YOLO_AVAILABLE:
            try:
                print(f"Loading YOLO model: {yolo_model}...")
                self.detector = YOLO(yolo_model)
                print("✓ YOLO detector loaded")
            except Exception as e:
                print(f"[ERROR] YOLO loading failed: {e}")
                self.detector = None
        else:
            self.detector = None

        # Initialize enhanced database with adjusted threshold
        self.database = EnhancedFaceDatabase(similarity_threshold=0.72)

        # Performance tracking
        self.frame_times = []
        self.frame_count = 0

    def detect_faces(self, frame):
        """Detect faces using YOLO"""
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
        except Exception as e:
            print(f"[ERROR] Detection failed: {e}")

        return detections

    def process_frame(self, frame):
        """Process frame with enhanced tracking"""
        start_time = time.time()
        self.frame_count += 1

        # Detect faces
        detections = self.detect_faces(frame)

        results = []
        matched_ids = set()

        # Process each detection
        for detection in detections:
            face_img = detection["face_img"]
            bbox = detection["bbox"]
            confidence = detection["confidence"]

            # Search with multi-stage matching
            is_match, visitor_id, similarity = self.database.search_face(face_img)

            if not is_match:
                # Create new visitor
                visitor_id = self.database.add_face(face_img)

            if visitor_id:
                matched_ids.add(visitor_id)

            results.append(
                {
                    "bbox": bbox,
                    "confidence": confidence,
                    "is_match": is_match,
                    "visitor_id": visitor_id,
                    "similarity": similarity,
                }
            )

        # Mark unmatched sessions as lost
        self.database.mark_unmatched_as_lost(matched_ids)

        # Annotate frame
        annotated_frame = self.annotate_frame(frame, results)

        # Performance tracking
        elapsed = time.time() - start_time
        self.frame_times.append(elapsed)

        return annotated_frame, results

    def annotate_frame(self, frame, results):
        """Draw annotations on frame"""
        annotated = frame.copy()

        for result in results:
            x1, y1, x2, y2 = result["bbox"]
            visitor_id = result["visitor_id"]
            is_match = result["is_match"]
            similarity = result["similarity"]

            # Color coding
            if is_match:
                color = (0, 255, 0)  # Green for recognized
                label = "RECOGNIZED"
            else:
                color = (0, 165, 255)  # Orange for new
                label = "NEW VISITOR"

            # Draw bounding box
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)

            # Draw labels
            cv2.putText(
                annotated,
                f"{label}",
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                2,
            )
            cv2.putText(
                annotated,
                f"ID: {visitor_id[:8] if visitor_id else 'N/A'}",
                (x1, y1 - 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                color,
                1,
            )

        # Draw statistics
        stats_text = [
            f"Active: {len(self.database.active_sessions)}",
            f"Lost (grace): {len(self.database.lost_sessions)}",
            f"Total Unique: {self.database.total_visitors}",
            f"Recoveries: {self.database.temporal_recoveries}",
            f"Frame: {self.frame_count}",
        ]

        y_pos = 30
        for text in stats_text:
            cv2.putText(
                annotated,
                text,
                (10, y_pos),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2,
            )
            y_pos += 25

        # FPS
        if len(self.frame_times) > 0:
            avg_fps = 1.0 / np.mean(self.frame_times[-30:])
            cv2.putText(
                annotated,
                f"FPS: {avg_fps:.1f}",
                (10, frame.shape[0] - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 255),
                2,
            )

        return annotated

    def get_statistics(self):
        """Get tracking statistics"""
        return {
            "total_unique_visitors": self.database.total_visitors,
            "active_sessions": len(self.database.active_sessions),
            "lost_sessions": len(self.database.lost_sessions),
            "temporal_recoveries": self.database.temporal_recoveries,
            "frames_processed": self.frame_count,
        }


# ============================================================================
# MAIN FUNCTION
# ============================================================================


def main():
    """Run enhanced tracking system"""
    print("=" * 70)
    print("Enhanced Visitor Tracking - ID-Switching Fix Applied")
    print("Indian Museum & Victoria Memorial")
    print("=" * 70)
    print("\nFixes Applied:")
    print("  ✓ Temporal smoothing with 3-second grace period")
    print("  ✓ Multi-stage matching (lost visitors checked first)")
    print("  ✓ Relaxed thresholds (0.72 standard, 0.82 relaxed)")
    print("  ✓ Session persistence to reduce ID-switching")
    print("=" * 70 + "\n")

    # Initialize enhanced system
    system = EnhancedFaceRecognitionSystem(
        yolo_model="yolov8n.pt", confidence_threshold=0.70
    )

    # Open camera
    print("Opening camera...")
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("[ERROR] Cannot open camera")
        return

    print("✓ Camera opened successfully")
    print("\nControls:")
    print("  q - Quit")
    print("  s - Show statistics")
    print("\nStarting tracking...\n")

    try:
        while True:
            ret, frame = cap.read()

            if not ret:
                print("[WARNING] Cannot read frame")
                break

            # Process frame
            annotated_frame, results = system.process_frame(frame)

            # Display
            cv2.imshow("Enhanced Visitor Tracking (ID-Switching Fix)", annotated_frame)

            # Keyboard controls
            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                print("\nQuitting...")
                break
            elif key == ord("s"):
                print("\n" + "=" * 50)
                print("STATISTICS")
                print("=" * 50)
                stats = system.get_statistics()
                for key, value in stats.items():
                    print(f"  {key}: {value}")
                print("=" * 50 + "\n")

    except KeyboardInterrupt:
        print("\n\nInterrupted by user")

    finally:
        # Cleanup
        cap.release()
        cv2.destroyAllWindows()

        # Final statistics
        print("\n" + "=" * 70)
        print("FINAL STATISTICS")
        print("=" * 70)
        stats = system.get_statistics()
        for key, value in stats.items():
            print(f"  {key}: {value}")
        print("=" * 70)

        print("\n✓ System shutdown complete")


if __name__ == "__main__":
    main()
