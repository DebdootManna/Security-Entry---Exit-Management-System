#!/usr/bin/env python3
"""
Face Detection and Re-Identification Test Script
Security Entry & Exit Management System

This script implements the Detect -> Align -> Encode -> Match pipeline:
1. Detector: YOLOv8-Face for locating faces with >80% confidence
2. Aligner: MTCNN for face alignment and landmark detection
3. Encoder: ArcFace/FaceNet for converting faces to 512D vectors
4. Matcher: FAISS/Cosine Similarity for comparing signatures (threshold: 0.6)

Requirements:
    pip install ultralytics mtcnn opencv-python deepface faiss-cpu scipy numpy pillow

Usage:
    python face_reidentification_test.py
    Press 'q' to quit the video stream
"""

import time
import uuid
import warnings
from collections import defaultdict
from datetime import datetime

import cv2
import numpy as np
from scipy.spatial.distance import cosine

warnings.filterwarnings("ignore")

# Import face detection and recognition libraries
try:
    from ultralytics import YOLO

    YOLO_AVAILABLE = True
except ImportError:
    print("WARNING: Ultralytics not available. Install with: pip install ultralytics")
    YOLO_AVAILABLE = False

try:
    from mtcnn import MTCNN

    MTCNN_AVAILABLE = True
except ImportError:
    print("WARNING: MTCNN not available. Install with: pip install mtcnn")
    MTCNN_AVAILABLE = False

try:
    from deepface import DeepFace

    DEEPFACE_AVAILABLE = True
except ImportError:
    print("WARNING: DeepFace not available. Install with: pip install deepface")
    DEEPFACE_AVAILABLE = False

try:
    import faiss

    FAISS_AVAILABLE = True
except ImportError:
    print("WARNING: FAISS not available. Install with: pip install faiss-cpu")
    FAISS_AVAILABLE = False


class FaceDatabase:
    """
    In-memory database simulating the 'Inside_Now' table.
    Stores face signatures (512D vectors) with associated metadata.
    """

    def __init__(self, use_faiss=True, dimension=512, similarity_threshold=0.6):
        """
        Initialize the face database.

        Args:
            use_faiss: Whether to use FAISS for efficient similarity search
            dimension: Dimension of face embedding vectors (default: 512 for ArcFace)
            similarity_threshold: Cosine distance threshold for matching (default: 0.6)
        """
        self.use_faiss = use_faiss and FAISS_AVAILABLE
        self.dimension = dimension
        self.similarity_threshold = similarity_threshold

        # Storage for signatures and metadata
        self.signatures = []  # List of numpy arrays (512D vectors)
        self.metadata = []  # List of dicts with visitor info

        # Initialize FAISS index for efficient similarity search
        if self.use_faiss:
            # Using FlatL2 index for exact L2 distance search
            # For cosine similarity, we normalize vectors before adding
            self.index = faiss.IndexFlatL2(dimension)
            print(f"✓ FAISS index initialized (dimension: {dimension})")
        else:
            self.index = None
            print("✓ Using cosine similarity fallback (no FAISS)")

    def add_signature(self, signature, visitor_id=None):
        """
        Add a new face signature to the database.

        Args:
            signature: 512D numpy array representing the face
            visitor_id: Optional UUID for the visitor

        Returns:
            visitor_id: UUID of the added signature
        """
        if visitor_id is None:
            visitor_id = str(uuid.uuid4())

        # Normalize signature for cosine similarity
        signature_norm = signature / np.linalg.norm(signature)

        # Store signature and metadata
        self.signatures.append(signature_norm)
        self.metadata.append(
            {
                "visitor_id": visitor_id,
                "first_seen": datetime.now(),
                "last_seen": datetime.now(),
                "encounter_count": 1,
            }
        )

        # Add to FAISS index if available
        if self.use_faiss:
            self.index.add(signature_norm.reshape(1, -1).astype("float32"))

        print(
            f"[DB] Added new visitor: {visitor_id[:8]}... (Total signatures: {len(self.signatures)})"
        )
        return visitor_id

    def search(self, signature):
        """
        Search for a matching signature in the database.

        Args:
            signature: 512D numpy array to search for

        Returns:
            match_found: Boolean indicating if a match was found
            visitor_id: UUID of matched visitor (or None)
            distance: Cosine distance to nearest match
            index: Index of matched signature in database
        """
        if len(self.signatures) == 0:
            return False, None, 1.0, -1

        # Normalize query signature
        signature_norm = signature / np.linalg.norm(signature)

        if self.use_faiss:
            # FAISS search (returns L2 distances)
            # For normalized vectors, L2 distance relates to cosine distance
            # cosine_distance = 1 - cosine_similarity
            # For normalized vectors: L2^2 = 2 * (1 - cosine_similarity)
            distances, indices = self.index.search(
                signature_norm.reshape(1, -1).astype("float32"), k=1
            )

            idx = indices[0][0]
            # Convert L2 distance to cosine distance approximation
            l2_dist = distances[0][0]
            cosine_dist = l2_dist / 2.0  # Approximation for normalized vectors

        else:
            # Fallback: Manual cosine distance calculation
            min_distance = float("inf")
            idx = -1

            for i, stored_sig in enumerate(self.signatures):
                dist = cosine(signature_norm, stored_sig)
                if dist < min_distance:
                    min_distance = dist
                    idx = i

            cosine_dist = min_distance

        # Check if match exceeds threshold
        if cosine_dist <= self.similarity_threshold and idx >= 0:
            # Match found - update metadata
            visitor_id = self.metadata[idx]["visitor_id"]
            self.metadata[idx]["last_seen"] = datetime.now()
            self.metadata[idx]["encounter_count"] += 1

            print(
                f"[MATCH] Recognized visitor: {visitor_id[:8]}... "
                f"(Distance: {cosine_dist:.4f}, Encounters: {self.metadata[idx]['encounter_count']})"
            )

            return True, visitor_id, cosine_dist, idx
        else:
            print(
                f"[NO MATCH] New visitor (Min distance: {cosine_dist:.4f} > threshold: {self.similarity_threshold})"
            )
            return False, None, cosine_dist, -1


class FaceReIdentificationSystem:
    """
    Main system class implementing the full pipeline:
    Detect -> Align -> Encode -> Match
    """

    def __init__(self, yolo_model="yolov8n-face.pt", confidence_threshold=0.8):
        """
        Initialize the face re-identification system.

        Args:
            yolo_model: Path to YOLO model file
            confidence_threshold: Minimum confidence for face detection (default: 0.8)
        """
        self.confidence_threshold = confidence_threshold

        # Initialize detector (YOLO)
        if YOLO_AVAILABLE:
            try:
                print(f"Loading YOLO model: {yolo_model}...")
                self.detector = YOLO(yolo_model)
                print("✓ YOLO detector loaded successfully")
            except Exception as e:
                print(f"ERROR loading YOLO model: {e}")
                print("Trying with standard YOLOv8n model...")
                try:
                    self.detector = YOLO("yolov8n.pt")
                    print("✓ Using standard YOLOv8n (may not be face-specific)")
                except Exception as e2:
                    print(f"ERROR: Could not load any YOLO model: {e2}")
                    self.detector = None
        else:
            self.detector = None

        # Initialize aligner (MTCNN)
        if MTCNN_AVAILABLE:
            try:
                print("Initializing MTCNN aligner...")
                self.aligner = MTCNN()
                print("✓ MTCNN aligner loaded successfully")
            except Exception as e:
                print(f"WARNING: MTCNN initialization failed: {e}")
                self.aligner = None
        else:
            self.aligner = None

        # Initialize encoder (DeepFace with ArcFace)
        if not DEEPFACE_AVAILABLE:
            print("ERROR: DeepFace required for face encoding")
            self.encoder_available = False
        else:
            self.encoder_available = True
            print("✓ DeepFace encoder ready (will use ArcFace/Facenet512)")

        # Initialize face database
        self.database = FaceDatabase(
            use_faiss=FAISS_AVAILABLE, dimension=512, similarity_threshold=0.6
        )

        # Performance tracking
        self.frame_times = []
        self.detection_count = 0
        self.recognition_count = 0

    def detect_faces(self, frame):
        """
        Step 1: Detect faces in the frame using YOLO.

        Args:
            frame: Input image (BGR format from OpenCV)

        Returns:
            detections: List of dicts with 'bbox', 'confidence', 'face_crop'
        """
        if self.detector is None:
            return []

        detections = []

        try:
            # Run YOLO inference
            results = self.detector(
                frame, conf=self.confidence_threshold, verbose=False
            )

            # Process results
            for result in results:
                if result.boxes is not None:
                    for box in result.boxes:
                        # Extract bounding box and confidence
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        confidence = float(box.conf[0])

                        # Ensure valid bounding box
                        x1, y1 = max(0, x1), max(0, y1)
                        x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)

                        if x2 > x1 and y2 > y1:
                            # Crop face region
                            face_crop = frame[y1:y2, x1:x2]

                            detections.append(
                                {
                                    "bbox": (x1, y1, x2, y2),
                                    "confidence": confidence,
                                    "face_crop": face_crop,
                                }
                            )

                            self.detection_count += 1

            print(
                f"[DETECT] Found {len(detections)} face(s) with confidence >{self.confidence_threshold}"
            )

        except Exception as e:
            print(f"[ERROR] Detection failed: {e}")

        return detections

    def align_face(self, face_crop):
        """
        Step 2: Align the detected face using MTCNN landmarks.

        Args:
            face_crop: Cropped face image (BGR)

        Returns:
            aligned_face: Aligned face image (RGB) or original if alignment fails
        """
        if self.aligner is None:
            # Fallback: Just resize and convert to RGB
            aligned = cv2.resize(face_crop, (160, 160))
            return cv2.cvtColor(aligned, cv2.COLOR_BGR2RGB)

        try:
            # Convert to RGB for MTCNN
            face_rgb = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)

            # Detect face landmarks
            detections = self.aligner.detect_faces(face_rgb)

            if len(detections) > 0:
                # Get the first detection (primary face)
                detection = detections[0]
                keypoints = detection["keypoints"]

                # Calculate angle for alignment based on eyes
                left_eye = keypoints["left_eye"]
                right_eye = keypoints["right_eye"]

                # Calculate rotation angle
                dY = right_eye[1] - left_eye[1]
                dX = right_eye[0] - left_eye[0]
                angle = np.degrees(np.arctan2(dY, dX))

                # Rotate image to align eyes horizontally
                center = tuple(np.array(face_rgb.shape[1::-1]) / 2)
                rot_mat = cv2.getRotationMatrix2D(center, angle, 1.0)
                aligned = cv2.warpAffine(
                    face_rgb, rot_mat, face_rgb.shape[1::-1], flags=cv2.INTER_LINEAR
                )

                # Resize to standard size
                aligned = cv2.resize(aligned, (160, 160))

                print(f"[ALIGN] Face aligned (rotation: {angle:.2f}°)")
                return aligned
            else:
                # No landmarks detected, return resized original
                print("[ALIGN] No landmarks found, using original")
                aligned = cv2.resize(face_rgb, (160, 160))
                return aligned

        except Exception as e:
            print(f"[ERROR] Alignment failed: {e}")
            # Fallback
            face_rgb = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
            aligned = cv2.resize(face_rgb, (160, 160))
            return aligned

    def encode_face(self, aligned_face):
        """
        Step 3: Encode the aligned face into a 512D vector using ArcFace/FaceNet.

        Args:
            aligned_face: Aligned face image (RGB, 160x160)

        Returns:
            embedding: 512D numpy array or None if encoding fails
        """
        if not self.encoder_available:
            return None

        try:
            # Use DeepFace to extract embeddings
            # Try ArcFace first (more accurate), fallback to Facenet512
            try:
                embedding = DeepFace.represent(
                    img_path=aligned_face,
                    model_name="ArcFace",
                    enforce_detection=False,
                    detector_backend="skip",  # Skip detection, we already have aligned face
                )
                model_used = "ArcFace"
            except:
                embedding = DeepFace.represent(
                    img_path=aligned_face,
                    model_name="Facenet512",
                    enforce_detection=False,
                    detector_backend="skip",
                )
                model_used = "Facenet512"

            # Extract embedding vector
            if isinstance(embedding, list) and len(embedding) > 0:
                vector = np.array(embedding[0]["embedding"])
            else:
                vector = np.array(embedding["embedding"])

            print(f"[ENCODE] Generated {vector.shape[0]}D embedding using {model_used}")

            return vector.astype("float32")

        except Exception as e:
            print(f"[ERROR] Encoding failed: {e}")
            return None

    def match_face(self, embedding):
        """
        Step 4: Match the face embedding against the database.

        Args:
            embedding: 512D face vector

        Returns:
            match_info: Dict with 'is_match', 'visitor_id', 'distance'
        """
        if embedding is None:
            return {"is_match": False, "visitor_id": None, "distance": 1.0}

        # Search in database
        is_match, visitor_id, distance, idx = self.database.search(embedding)

        if not is_match:
            # New visitor - add to database
            visitor_id = self.database.add_signature(embedding)
        else:
            self.recognition_count += 1

        return {"is_match": is_match, "visitor_id": visitor_id, "distance": distance}

    def process_frame(self, frame):
        """
        Process a single frame through the full pipeline.

        Args:
            frame: Input video frame (BGR)

        Returns:
            annotated_frame: Frame with annotations
            results: List of detection/recognition results
        """
        start_time = time.time()

        # Step 1: Detect faces
        detections = self.detect_faces(frame)

        results = []

        # Process each detected face
        for detection in detections:
            face_crop = detection["face_crop"]
            bbox = detection["bbox"]
            confidence = detection["confidence"]

            # Step 2: Align face
            aligned_face = self.align_face(face_crop)

            # Step 3: Encode face
            embedding = self.encode_face(aligned_face)

            # Step 4: Match face
            match_info = self.match_face(embedding)

            results.append(
                {"bbox": bbox, "confidence": confidence, "match_info": match_info}
            )

        # Annotate frame
        annotated_frame = self.annotate_frame(frame, results)

        # Track performance
        elapsed = time.time() - start_time
        self.frame_times.append(elapsed)

        return annotated_frame, results

    def annotate_frame(self, frame, results):
        """
        Draw bounding boxes and labels on the frame.

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
            match_info = result["match_info"]

            # Determine color and label
            if match_info["is_match"]:
                color = (0, 255, 0)  # Green for recognized
                status = "RECOGNIZED"
                visitor_id = match_info["visitor_id"][:8]
            else:
                color = (0, 165, 255)  # Orange for new visitor
                status = "NEW VISITOR"
                visitor_id = (
                    match_info["visitor_id"][:8] if match_info["visitor_id"] else "N/A"
                )

            # Draw bounding box
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)

            # Prepare label text
            label = f"{status} | ID:{visitor_id}"
            conf_label = f"Conf: {confidence:.2f}"
            dist_label = f"Dist: {match_info['distance']:.4f}"

            # Draw labels with background
            labels = [label, conf_label, dist_label]
            y_offset = y1 - 10

            for i, text in enumerate(labels):
                y_pos = y_offset - (len(labels) - 1 - i) * 25

                # Get text size for background
                (text_width, text_height), baseline = cv2.getTextSize(
                    text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
                )

                # Draw background rectangle
                cv2.rectangle(
                    annotated,
                    (x1, y_pos - text_height - 5),
                    (x1 + text_width + 5, y_pos + baseline),
                    color,
                    -1,
                )

                # Draw text
                cv2.putText(
                    annotated,
                    text,
                    (x1 + 2, y_pos),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 255, 255),
                    1,
                    cv2.LINE_AA,
                )

        # Add system stats
        if len(self.frame_times) > 0:
            avg_fps = 1.0 / (
                sum(self.frame_times[-30:]) / min(len(self.frame_times), 30)
            )
            stats_text = [
                f"FPS: {avg_fps:.1f}",
                f"Detections: {self.detection_count}",
                f"Recognitions: {self.recognition_count}",
                f"DB Size: {len(self.database.signatures)}",
            ]

            for i, text in enumerate(stats_text):
                cv2.putText(
                    annotated,
                    text,
                    (10, 30 + i * 25),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA,
                )

        return annotated

    def get_statistics(self):
        """Get system performance statistics."""
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
            "unique_visitors": len(self.database.signatures),
        }


def main():
    """
    Main function to run the face re-identification test.
    """
    print("=" * 70)
    print("Face Re-Identification System - Test Script")
    print("Security Entry & Exit Management System")
    print("=" * 70)
    print()

    # Check dependencies
    print("Checking dependencies...")
    dependencies = {
        "YOLO": YOLO_AVAILABLE,
        "MTCNN": MTCNN_AVAILABLE,
        "DeepFace": DEEPFACE_AVAILABLE,
        "FAISS": FAISS_AVAILABLE,
    }

    for lib, available in dependencies.items():
        status = "✓ Available" if available else "✗ Missing"
        print(f"  {lib}: {status}")

    print()

    if not all([YOLO_AVAILABLE, DEEPFACE_AVAILABLE]):
        print("ERROR: Required libraries missing. Please install:")
        print("  pip install ultralytics opencv-python deepface faiss-cpu scipy mtcnn")
        return

    # Initialize system
    print("Initializing Face Re-Identification System...")
    print()

    system = FaceReIdentificationSystem(
        yolo_model="yolov8n-face.pt", confidence_threshold=0.8
    )

    print()
    print("=" * 70)
    print("Starting webcam capture...")
    print("Press 'q' to quit, 's' to show statistics")
    print("=" * 70)
    print()

    # Open webcam
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("ERROR: Could not open webcam")
        return

    # Set camera properties for better performance
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)

    print("✓ Webcam opened successfully")
    print()

    frame_count = 0

    try:
        while True:
            # Capture frame
            ret, frame = cap.read()

            if not ret:
                print("ERROR: Failed to capture frame")
                break

            frame_count += 1

            # Process every frame (can be changed to process every N frames for performance)
            if frame_count % 1 == 0:  # Process every frame
                annotated_frame, results = system.process_frame(frame)
            else:
                annotated_frame = frame

            # Display frame
            cv2.imshow("Face Re-Identification System", annotated_frame)

            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                print("\nQuitting...")
                break
            elif key == ord("s"):
                # Show statistics
                stats = system.get_statistics()
                print("\n" + "=" * 70)
                print("SYSTEM STATISTICS")
                print("=" * 70)
                print(f"Average FPS: {stats['avg_fps']:.2f}")
                print(f"Average Processing Time: {stats['avg_processing_time']:.4f}s")
                print(f"Total Detections: {stats['total_detections']}")
                print(f"Total Recognitions: {stats['total_recognitions']}")
                print(f"Unique Visitors in DB: {stats['unique_visitors']}")
                print("=" * 70)
                print()

    except KeyboardInterrupt:
        print("\nInterrupted by user")

    finally:
        # Cleanup
        cap.release()
        cv2.destroyAllWindows()

        # Final statistics
        print("\n" + "=" * 70)
        print("FINAL STATISTICS")
        print("=" * 70)
        stats = system.get_statistics()
        print(f"Total Frames Processed: {frame_count}")
        print(f"Average FPS: {stats['avg_fps']:.2f}")
        print(f"Average Processing Time: {stats['avg_processing_time']:.4f}s")
        print(f"Total Face Detections: {stats['total_detections']}")
        print(f"Total Re-identifications: {stats['total_recognitions']}")
        print(f"Unique Visitors Registered: {stats['unique_visitors']}")
        print(
            f"Recognition Rate: {(stats['total_recognitions'] / max(stats['total_detections'], 1)) * 100:.1f}%"
        )
        print("=" * 70)
        print("\nDatabase Contents:")
        for i, metadata in enumerate(system.database.metadata):
            print(f"  Visitor {i + 1}:")
            print(f"    ID: {metadata['visitor_id']}")
            print(
                f"    First Seen: {metadata['first_seen'].strftime('%Y-%m-%d %H:%M:%S')}"
            )
            print(
                f"    Last Seen: {metadata['last_seen'].strftime('%Y-%m-%d %H:%M:%S')}"
            )
            print(f"    Encounters: {metadata['encounter_count']}")
        print("=" * 70)
        print("\nTest completed successfully!")


if __name__ == "__main__":
    main()
