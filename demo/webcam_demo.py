"""
Dwarpala Demo — Webcam Verification

Run: python demo/webcam_demo.py
"""

import cv2
import numpy as np
import sys
import time
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from dwarpala.kavach import FaceDetector, FaceAligner, QualityAssessor
from dwarpala.prana.texture_analyzer import TextureAnalyzer
from dwarpala.utils.logger import get_logger

logger = get_logger("demo.webcam")


def draw_fancy_box(frame, bbox, color, label, thickness=2):
    """Draw a stylish bounding box with corner accents."""
    x1, y1, x2, y2 = [int(v) for v in bbox]
    w, h = x2 - x1, y2 - y1
    corner_len = min(30, w // 4, h // 4)

    # Main rectangle (thin)
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 1)

    # Corner accents (thick)
    for cx, cy, dx, dy in [
        (x1, y1, 1, 1), (x2, y1, -1, 1),
        (x1, y2, 1, -1), (x2, y2, -1, -1),
    ]:
        cv2.line(frame, (cx, cy), (cx + dx * corner_len, cy), color, thickness)
        cv2.line(frame, (cx, cy), (cx, cy + dy * corner_len), color, thickness)

    # Label background
    text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)[0]
    cv2.rectangle(
        frame,
        (x1, y1 - text_size[1] - 10),
        (x1 + text_size[0] + 10, y1),
        color, -1,
    )
    cv2.putText(
        frame, label,
        (x1 + 5, y1 - 5),
        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1,
    )


def main():
    """Run live webcam face detection and liveness demo."""
    print("\n" + "=" * 50)
    print("  🛕 DWARPALA — Live Webcam Demo")
    print("  Press 'q' to quit, 's' for liveness snapshot")
    print("=" * 50 + "\n")

    # Initialize modules
    detector = FaceDetector(backend="opencv", max_faces=1)
    aligner = FaceAligner(output_size=(112, 112))
    quality = QualityAssessor()
    texture = TextureAnalyzer()

    # Open webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Cannot open webcam!")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    fps_counter = 0
    fps_start = time.time()
    display_fps = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # FPS calculation
        fps_counter += 1
        if time.time() - fps_start >= 1.0:
            display_fps = fps_counter
            fps_counter = 0
            fps_start = time.time()

        # Detect face
        detection = detector.detect_largest(rgb)

        if detection is not None:
            # Quality check
            aligned = aligner.align(rgb, detection.landmarks)
            report = quality.assess(aligned, detection)

            color = (0, 255, 0) if report.is_acceptable else (0, 165, 255)
            label = (
                f"Face | blur={report.blur_score:.0f} "
                f"bright={report.brightness:.0f}"
            )

            draw_fancy_box(frame, detection.bbox, color, label)

            # Draw landmarks
            for lm in detection.landmarks:
                cv2.circle(frame, (int(lm[0]), int(lm[1])), 3, (0, 255, 255), -1)

            # Quality issues overlay
            if not report.is_acceptable:
                for i, issue in enumerate(report.issues):
                    cv2.putText(
                        frame, f"⚠ {issue}",
                        (10, frame.shape[0] - 30 - i * 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1,
                    )
        else:
            cv2.putText(
                frame, "No face detected",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2,
            )

        # HUD overlay
        cv2.putText(
            frame, f"DWARPALA | FPS: {display_fps}",
            (10, frame.shape[0] - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1,
        )

        cv2.imshow("Dwarpala - Live Demo", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("s") and detection is not None:
            # Texture liveness check on current frame
            aligned_bgr = cv2.cvtColor(aligned, cv2.COLOR_RGB2BGR)
            t_result = texture.analyze(aligned)
            print(f"\n📸 Liveness Snapshot: {t_result}")

    cap.release()
    cv2.destroyAllWindows()
    print("\n👋 Demo ended.")


if __name__ == "__main__":
    main()
