#!/usr/bin/env python3
"""Quick demo of the Ethical Avatar Tracker on webcam."""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import cv2
from src.face_tracker import FaceTracker

def main():
    model_path = "models/face_landmarker.task"
    if not os.path.exists(model_path):
        print(f"Model not found at {model_path}")
        print("Download it with:")
        print("  mkdir -p models")
        print("  wget -O models/face_landmarker.task \\")
        print("    https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task")
        return

    tracker = FaceTracker(model_path=model_path)
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Cannot open webcam")
        return

    print("Running Ethical Avatar Tracker demo. Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        data = tracker.process(frame)

        # Overlay FPS
        fps_text = f"FPS: {data['fps']:.1f}"
        cv2.putText(frame, fps_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        if data["blendshapes"]:
            active = sum(1 for v in data["blendshapes"].values() if v > 0.4)
            cv2.putText(frame, f"Active expressions: {active}", (10, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 200, 0), 2)

        cv2.imshow("Ethical Avatar Tracker", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    tracker.close()
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
