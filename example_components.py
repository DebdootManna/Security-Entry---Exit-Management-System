#!/usr/bin/env python3
"""
Example Components Script
Face Re-Identification System - Component-Level Demonstrations

This script demonstrates how to use individual pipeline components separately
for educational purposes and testing. Each section can be run independently
to understand how each component works.

Components demonstrated:
1. Face Detection (YOLO)
2. Face Alignment (MTCNN)
3. Face Encoding (ArcFace/FaceNet)
4. Similarity Matching (FAISS/Cosine)

Usage:
    python example_components.py
"""

import cv2
import numpy as np
from scipy.spatial.distance import cosine

print("=" * 70)
print("Face Re-Identification System - Component Examples")
print("=" * 70)
print()

# ============================================================================
# EXAMPLE 1: Face Detection with YOLO
# ============================================================================


def example_1_face_detection():
    """
    Demonstrate face detection using YOLOv8-Face.
    Shows how to load model, detect faces, and draw bounding boxes.
    """
    print("=" * 70)
    print("EXAMPLE 1: Face Detection with YOLO")
    print("=" * 70)
    print()

    try:
        from ultralytics import YOLO

        print("Loading YOLO model...")
        model = YOLO("yolov8n-face.pt")
        print("✓ Model loaded successfully")
        print()

        # Open webcam
        print("Opening webcam...")
        cap = cv2.VideoCapture(0)

        if not cap.isOpened():
            print("✗ Could not open webcam")
            return

        print("✓ Webcam opened")
        print()
        print("Press 'q' to quit this example and move to next")
        print()

        frame_count = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1

            # Run detection
            results = model(frame, conf=0.8, verbose=False)

            # Draw results
            annotated = frame.copy()
            detection_count = 0

            for result in results:
                if result.boxes is not None:
                    for box in result.boxes:
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        confidence = float(box.conf[0])
                        detection_count += 1

                        # Draw bounding box
                        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)

                        # Draw label
                        label = f"Face {detection_count}: {confidence:.2f}"
                        cv2.putText(
                            annotated,
                            label,
                            (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5,
                            (0, 255, 0),
                            2,
                        )

            # Show info
            info_text = f"Frame: {frame_count} | Faces: {detection_count}"
            cv2.putText(
                annotated,
                info_text,
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
            )

            cv2.imshow("Example 1: Face Detection", annotated)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        cap.release()
        cv2.destroyAllWindows()
        print("✓ Example 1 completed")
        print()

    except ImportError:
        print("✗ Ultralytics not installed. Run: pip install ultralytics")
    except Exception as e:
        print(f"✗ Error: {e}")


# ============================================================================
# EXAMPLE 2: Face Alignment with MTCNN
# ============================================================================


def example_2_face_alignment():
    """
    Demonstrate face alignment using MTCNN.
    Shows landmark detection and face rotation for normalization.
    """
    print("=" * 70)
    print("EXAMPLE 2: Face Alignment with MTCNN")
    print("=" * 70)
    print()

    try:
        from mtcnn import MTCNN

        print("Initializing MTCNN...")
        detector = MTCNN()
        print("✓ MTCNN initialized")
        print()

        # Open webcam
        print("Opening webcam...")
        cap = cv2.VideoCapture(0)

        if not cap.isOpened():
            print("✗ Could not open webcam")
            return

        print("✓ Webcam opened")
        print()
        print("Press 'q' to quit this example and move to next")
        print()

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Convert to RGB for MTCNN
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Detect faces and landmarks
            detections = detector.detect_faces(frame_rgb)

            annotated = frame.copy()

            for detection in detections:
                # Draw bounding box
                x, y, w, h = detection["box"]
                cv2.rectangle(annotated, (x, y), (x + w, y + h), (255, 0, 0), 2)

                # Draw landmarks
                keypoints = detection["keypoints"]
                for name, point in keypoints.items():
                    cv2.circle(annotated, point, 3, (0, 255, 0), -1)
                    cv2.putText(
                        annotated,
                        name[:4],
                        (point[0] + 5, point[1]),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.3,
                        (0, 255, 0),
                        1,
                    )

                # Calculate rotation angle from eyes
                left_eye = keypoints["left_eye"]
                right_eye = keypoints["right_eye"]
                dY = right_eye[1] - left_eye[1]
                dX = right_eye[0] - left_eye[0]
                angle = np.degrees(np.arctan2(dY, dX))

                # Show angle
                cv2.putText(
                    annotated,
                    f"Rotation: {angle:.1f} deg",
                    (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 0, 0),
                    2,
                )

                # Show aligned face in corner
                try:
                    face_crop = frame_rgb[y : y + h, x : x + w]
                    if face_crop.size > 0:
                        # Rotate to align
                        center = (w // 2, h // 2)
                        rot_mat = cv2.getRotationMatrix2D(center, angle, 1.0)
                        aligned = cv2.warpAffine(face_crop, rot_mat, (w, h))
                        aligned = cv2.resize(aligned, (100, 100))
                        aligned_bgr = cv2.cvtColor(aligned, cv2.COLOR_RGB2BGR)

                        # Place in corner
                        annotated[10:110, 10:110] = aligned_bgr
                        cv2.putText(
                            annotated,
                            "Aligned Face",
                            (10, 125),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.4,
                            (255, 255, 255),
                            1,
                        )
                except:
                    pass

            info_text = f"Detected: {len(detections)} face(s)"
            cv2.putText(
                annotated,
                info_text,
                (10, annotated.shape[0] - 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2,
            )

            cv2.imshow("Example 2: Face Alignment", annotated)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        cap.release()
        cv2.destroyAllWindows()
        print("✓ Example 2 completed")
        print()

    except ImportError:
        print("✗ MTCNN not installed. Run: pip install mtcnn")
    except Exception as e:
        print(f"✗ Error: {e}")


# ============================================================================
# EXAMPLE 3: Face Encoding with ArcFace/FaceNet
# ============================================================================


def example_3_face_encoding():
    """
    Demonstrate face encoding using DeepFace (ArcFace/FaceNet).
    Shows how to convert face images to 512D vectors.
    """
    print("=" * 70)
    print("EXAMPLE 3: Face Encoding with ArcFace")
    print("=" * 70)
    print()

    try:
        from deepface import DeepFace
        from ultralytics import YOLO

        print("Loading models...")
        yolo = YOLO("yolov8n-face.pt")
        print("✓ Models loaded")
        print()

        # Open webcam
        print("Opening webcam...")
        cap = cv2.VideoCapture(0)

        if not cap.isOpened():
            print("✗ Could not open webcam")
            return

        print("✓ Webcam opened")
        print()
        print("Press 'q' to quit this example and move to next")
        print("Press 's' to save current face embedding")
        print()

        saved_embeddings = []

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            annotated = frame.copy()

            # Detect face
            results = yolo(frame, conf=0.8, verbose=False)

            for result in results:
                if result.boxes is not None and len(result.boxes) > 0:
                    box = result.boxes[0]
                    x1, y1, x2, y2 = map(int, box.xyxy[0])

                    # Crop face
                    face_crop = frame[y1:y2, x1:x2]

                    if face_crop.size > 0:
                        try:
                            # Convert to RGB
                            face_rgb = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
                            face_rgb = cv2.resize(face_rgb, (160, 160))

                            # Encode face
                            embedding = DeepFace.represent(
                                img_path=face_rgb,
                                model_name="ArcFace",
                                enforce_detection=False,
                                detector_backend="skip",
                            )

                            if isinstance(embedding, list):
                                vector = np.array(embedding[0]["embedding"])
                            else:
                                vector = np.array(embedding["embedding"])

                            # Draw bounding box
                            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)

                            # Show vector info
                            info = [
                                f"Vector Dim: {vector.shape[0]}D",
                                f"Vector Norm: {np.linalg.norm(vector):.2f}",
                                f"Saved: {len(saved_embeddings)}",
                                "",
                                "Press 's' to save",
                            ]

                            y_offset = 30
                            for i, text in enumerate(info):
                                cv2.putText(
                                    annotated,
                                    text,
                                    (10, y_offset + i * 25),
                                    cv2.FONT_HERSHEY_SIMPLEX,
                                    0.6,
                                    (255, 255, 255),
                                    2,
                                )

                            # Show first few values
                            values_text = f"First 5 values: [{', '.join([f'{v:.2f}' for v in vector[:5]])}...]"
                            cv2.putText(
                                annotated,
                                values_text,
                                (10, annotated.shape[0] - 20),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.4,
                                (255, 255, 255),
                                1,
                            )

                        except Exception as e:
                            cv2.putText(
                                annotated,
                                f"Encoding error: {str(e)[:30]}",
                                (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.5,
                                (0, 0, 255),
                                1,
                            )

            cv2.imshow("Example 3: Face Encoding", annotated)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("s"):
                if "vector" in locals():
                    saved_embeddings.append(vector.copy())
                    print(
                        f"✓ Saved embedding #{len(saved_embeddings)} (shape: {vector.shape})"
                    )

        cap.release()
        cv2.destroyAllWindows()

        print()
        print(f"✓ Example 3 completed (Saved {len(saved_embeddings)} embeddings)")
        print()

        return saved_embeddings

    except ImportError as e:
        print(f"✗ Required library not installed: {e}")
    except Exception as e:
        print(f"✗ Error: {e}")

    return []


# ============================================================================
# EXAMPLE 4: Similarity Matching with FAISS and Cosine Distance
# ============================================================================


def example_4_similarity_matching(saved_embeddings=None):
    """
    Demonstrate similarity matching using FAISS and cosine distance.
    Shows how to compare face embeddings and find matches.
    """
    print("=" * 70)
    print("EXAMPLE 4: Similarity Matching")
    print("=" * 70)
    print()

    try:
        import faiss
        from deepface import DeepFace
        from ultralytics import YOLO

        # Initialize database
        print("Initializing FAISS index...")
        dimension = 512
        index = faiss.IndexFlatL2(dimension)
        metadata = []

        # Add saved embeddings from previous example
        if saved_embeddings and len(saved_embeddings) > 0:
            print(f"Loading {len(saved_embeddings)} saved embeddings...")
            for i, emb in enumerate(saved_embeddings):
                emb_norm = emb / np.linalg.norm(emb)
                index.add(emb_norm.reshape(1, -1).astype("float32"))
                metadata.append(f"Saved_{i + 1}")
            print(f"✓ Loaded {len(saved_embeddings)} embeddings into database")
        else:
            print("No saved embeddings from previous example")

        print("✓ FAISS index ready")
        print()

        # Open webcam
        print("Opening webcam...")
        cap = cv2.VideoCapture(0)
        yolo = YOLO("yolov8n-face.pt")

        if not cap.isOpened():
            print("✗ Could not open webcam")
            return

        print("✓ Webcam opened")
        print()
        print("Press 'q' to quit")
        print("Press 'a' to add current face to database")
        print()

        threshold = 0.6

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            annotated = frame.copy()

            # Detect face
            results = yolo(frame, conf=0.8, verbose=False)

            for result in results:
                if result.boxes is not None and len(result.boxes) > 0:
                    box = result.boxes[0]
                    x1, y1, x2, y2 = map(int, box.xyxy[0])

                    # Crop and encode face
                    face_crop = frame[y1:y2, x1:x2]

                    if face_crop.size > 0:
                        try:
                            face_rgb = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
                            face_rgb = cv2.resize(face_rgb, (160, 160))

                            embedding = DeepFace.represent(
                                img_path=face_rgb,
                                model_name="ArcFace",
                                enforce_detection=False,
                                detector_backend="skip",
                            )

                            if isinstance(embedding, list):
                                vector = np.array(embedding[0]["embedding"])
                            else:
                                vector = np.array(embedding["embedding"])

                            vector_norm = vector / np.linalg.norm(vector)

                            # Search in database
                            if index.ntotal > 0:
                                distances, indices = index.search(
                                    vector_norm.reshape(1, -1).astype("float32"), k=1
                                )

                                distance = (
                                    distances[0][0] / 2.0
                                )  # Convert L2 to cosine approx
                                idx = indices[0][0]

                                if distance < threshold:
                                    # Match found
                                    color = (0, 255, 0)
                                    label = f"MATCH: {metadata[idx]}"
                                    status = "RECOGNIZED"
                                else:
                                    # No match
                                    color = (0, 165, 255)
                                    label = "NEW FACE"
                                    status = "NEW"

                                # Draw result
                                cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
                                cv2.putText(
                                    annotated,
                                    label,
                                    (x1, y1 - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX,
                                    0.6,
                                    color,
                                    2,
                                )

                                # Show info
                                info = [
                                    f"Status: {status}",
                                    f"Distance: {distance:.4f}",
                                    f"Threshold: {threshold}",
                                    f"DB Size: {index.ntotal}",
                                    "",
                                    "Press 'a' to add",
                                ]

                                y_offset = 30
                                for i, text in enumerate(info):
                                    cv2.putText(
                                        annotated,
                                        text,
                                        (10, y_offset + i * 25),
                                        cv2.FONT_HERSHEY_SIMPLEX,
                                        0.6,
                                        (255, 255, 255),
                                        2,
                                    )
                            else:
                                cv2.rectangle(
                                    annotated, (x1, y1), (x2, y2), (255, 255, 0), 2
                                )
                                cv2.putText(
                                    annotated,
                                    "Database empty - press 'a'",
                                    (10, 30),
                                    cv2.FONT_HERSHEY_SIMPLEX,
                                    0.7,
                                    (255, 255, 255),
                                    2,
                                )

                        except Exception as e:
                            cv2.putText(
                                annotated,
                                f"Error: {str(e)[:40]}",
                                (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.5,
                                (0, 0, 255),
                                1,
                            )

            cv2.imshow("Example 4: Similarity Matching", annotated)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("a"):
                if "vector_norm" in locals():
                    index.add(vector_norm.reshape(1, -1).astype("float32"))
                    metadata.append(f"User_{index.ntotal}")
                    print(f"✓ Added face to database (Total: {index.ntotal})")

        cap.release()
        cv2.destroyAllWindows()

        print()
        print(f"✓ Example 4 completed (Database size: {index.ntotal})")
        print()

    except ImportError as e:
        print(f"✗ Required library not installed: {e}")
    except Exception as e:
        print(f"✗ Error: {e}")


# ============================================================================
# EXAMPLE 5: Complete Pipeline Demo
# ============================================================================


def example_5_complete_pipeline():
    """
    Demonstrate the complete pipeline: Detect → Align → Encode → Match
    This is a simplified version of the full system.
    """
    print("=" * 70)
    print("EXAMPLE 5: Complete Pipeline Demo")
    print("=" * 70)
    print()

    try:
        import faiss
        from deepface import DeepFace
        from mtcnn import MTCNN
        from ultralytics import YOLO

        print("Loading all models...")
        yolo = YOLO("yolov8n-face.pt")
        mtcnn = MTCNN()
        index = faiss.IndexFlatL2(512)
        database = []
        print("✓ All models loaded")
        print()

        print("Opening webcam...")
        cap = cv2.VideoCapture(0)

        if not cap.isOpened():
            print("✗ Could not open webcam")
            return

        print("✓ Webcam opened")
        print()
        print("Complete Pipeline: Detect → Align → Encode → Match")
        print("Press 'q' to quit")
        print()

        frame_count = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1
            annotated = frame.copy()

            # Step 1: Detect
            results = yolo(frame, conf=0.8, verbose=False)

            for result in results:
                if result.boxes is not None and len(result.boxes) > 0:
                    box = result.boxes[0]
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    confidence = float(box.conf[0])

                    face_crop = frame[y1:y2, x1:x2]

                    if face_crop.size > 0:
                        try:
                            # Step 2: Align
                            face_rgb = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
                            detections = mtcnn.detect_faces(face_rgb)

                            if detections:
                                keypoints = detections[0]["keypoints"]
                                left_eye = keypoints["left_eye"]
                                right_eye = keypoints["right_eye"]
                                dY = right_eye[1] - left_eye[1]
                                dX = right_eye[0] - left_eye[0]
                                angle = np.degrees(np.arctan2(dY, dX))

                                center = tuple(np.array(face_rgb.shape[1::-1]) / 2)
                                rot_mat = cv2.getRotationMatrix2D(center, angle, 1.0)
                                aligned = cv2.warpAffine(
                                    face_rgb, rot_mat, face_rgb.shape[1::-1]
                                )
                                aligned = cv2.resize(aligned, (160, 160))
                            else:
                                aligned = cv2.resize(face_rgb, (160, 160))

                            # Step 3: Encode
                            embedding = DeepFace.represent(
                                img_path=aligned,
                                model_name="ArcFace",
                                enforce_detection=False,
                                detector_backend="skip",
                            )

                            if isinstance(embedding, list):
                                vector = np.array(embedding[0]["embedding"])
                            else:
                                vector = np.array(embedding["embedding"])

                            vector_norm = vector / np.linalg.norm(vector)

                            # Step 4: Match
                            if index.ntotal > 0:
                                distances, indices = index.search(
                                    vector_norm.reshape(1, -1).astype("float32"), k=1
                                )
                                distance = distances[0][0] / 2.0

                                if distance < 0.6:
                                    color = (0, 255, 0)
                                    label = f"RECOGNIZED (ID: {indices[0][0] + 1})"
                                else:
                                    # New person
                                    index.add(
                                        vector_norm.reshape(1, -1).astype("float32")
                                    )
                                    database.append(vector_norm)
                                    color = (0, 165, 255)
                                    label = f"NEW (ID: {index.ntotal})"
                            else:
                                # First person
                                index.add(vector_norm.reshape(1, -1).astype("float32"))
                                database.append(vector_norm)
                                color = (0, 165, 255)
                                label = "FIRST PERSON (ID: 1)"

                            # Draw results
                            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
                            cv2.putText(
                                annotated,
                                label,
                                (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.6,
                                color,
                                2,
                            )

                            # Show pipeline stages
                            stages = [
                                f"1. DETECT: {confidence:.2f}",
                                f"2. ALIGN: {angle:.1f}deg"
                                if "angle" in locals()
                                else "2. ALIGN: N/A",
                                f"3. ENCODE: {vector.shape[0]}D",
                                f"4. MATCH: {distance:.4f}"
                                if "distance" in locals()
                                else "4. MATCH: First",
                                "",
                                f"Database: {index.ntotal} people",
                            ]

                            y_offset = 30
                            for i, text in enumerate(stages):
                                cv2.putText(
                                    annotated,
                                    text,
                                    (10, y_offset + i * 25),
                                    cv2.FONT_HERSHEY_SIMPLEX,
                                    0.6,
                                    (255, 255, 255),
                                    2,
                                )

                        except Exception as e:
                            cv2.putText(
                                annotated,
                                f"Error: {str(e)[:40]}",
                                (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.5,
                                (0, 0, 255),
                                1,
                            )

            cv2.imshow("Example 5: Complete Pipeline", annotated)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        cap.release()
        cv2.destroyAllWindows()

        print()
        print(f"✓ Example 5 completed")
        print(f"  Total frames: {frame_count}")
        print(f"  Unique people: {index.ntotal}")
        print()

    except ImportError as e:
        print(f"✗ Required library not installed: {e}")
    except Exception as e:
        print(f"✗ Error: {e}")


# ============================================================================
# Main Menu
# ============================================================================


def main():
    """Main menu for running examples."""

    print("Select an example to run:")
    print()
    print("1. Face Detection (YOLO)")
    print("2. Face Alignment (MTCNN)")
    print("3. Face Encoding (ArcFace)")
    print("4. Similarity Matching (FAISS)")
    print("5. Complete Pipeline Demo")
    print("6. Run All Examples Sequentially")
    print("0. Exit")
    print()

    while True:
        choice = input("Enter your choice (0-6): ").strip()

        if choice == "0":
            print("Exiting...")
            break
        elif choice == "1":
            example_1_face_detection()
        elif choice == "2":
            example_2_face_alignment()
        elif choice == "3":
            embeddings = example_3_face_encoding()
            if embeddings:
                use_embeddings = (
                    input("Use saved embeddings for matching example? (y/n): ")
                    .strip()
                    .lower()
                )
                if use_embeddings == "y":
                    example_4_similarity_matching(embeddings)
        elif choice == "4":
            example_4_similarity_matching()
        elif choice == "5":
            example_5_complete_pipeline()
        elif choice == "6":
            print("\nRunning all examples sequentially...\n")
            example_1_face_detection()
            example_2_face_alignment()
            embeddings = example_3_face_encoding()
            example_4_similarity_matching(embeddings)
            example_5_complete_pipeline()
            print("All examples completed!")
            break
        else:
            print("Invalid choice. Please enter 0-6.")

        print()
        cont = input("Run another example? (y/n): ").strip().lower()
        if cont != "y":
            break

    print()
    print("=" * 70)
    print("Thank you for exploring the Face Re-Identification components!")
    print("=" * 70)


if __name__ == "__main__":
    main()
