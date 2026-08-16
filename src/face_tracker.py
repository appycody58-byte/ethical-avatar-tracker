import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
import time
from collections import deque


class OneEuroFilter:
    """High-quality temporal smoother for landmarks and blendshapes."""
    def __init__(self, min_cutoff=1.0, beta=0.007, d_cutoff=1.0):
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff
        self.x_prev = None
        self.dx_prev = None
        self.t_prev = None

    def __call__(self, x, t=None):
        if t is None:
            t = time.time()
        if self.x_prev is None:
            self.x_prev = x
            self.dx_prev = np.zeros_like(x)
            self.t_prev = t
            return x

        t_e = t - self.t_prev
        if t_e <= 0:
            return self.x_prev

        dx = (x - self.x_prev) / t_e
        alpha_d = self._smoothing_factor(self.d_cutoff, t_e)
        edx = alpha_d * dx + (1.0 - alpha_d) * self.dx_prev

        cutoff = self.min_cutoff + self.beta * np.abs(edx)
        alpha = self._smoothing_factor(cutoff, t_e)
        x_hat = alpha * x + (1.0 - alpha) * self.x_prev

        self.x_prev = x_hat
        self.dx_prev = edx
        self.t_prev = t
        return x_hat

    def _smoothing_factor(self, cutoff, t_e):
        r = 2 * np.pi * cutoff * t_e
        return r / (r + 1.0)


class FaceTracker:
    """Production-ready MediaPipe Face Landmarker wrapper with smoothing + FPS."""

    def __init__(self, model_path="models/face_landmarker.task", max_faces=1):
        base_options = python.BaseOptions(model_asset_path=model_path)
        options = vision.FaceLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.VIDEO,
            num_faces=max_faces,
            output_face_blendshapes=True,
            output_facial_transformation_matrixes=True,
            min_face_detection_confidence=0.6,
            min_face_presence_confidence=0.6,
            min_tracking_confidence=0.6,
        )
        self.landmarker = vision.FaceLandmarker.create_from_options(options)

        self.landmark_filters = None
        self.blendshape_filters = None
        self.fps_queue = deque(maxlen=30)
        self.prev_time = time.time()

    def process(self, frame_bgr):
        t = time.time()
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        result = self.landmarker.detect_for_video(mp_image, int(t * 1000))

        output = {
            "landmarks": None,
            "blendshapes": None,
            "matrix": None,
            "fps": 0.0,
        }

        if result.face_landmarks:
            raw_landmarks = np.array(
                [[lm.x, lm.y, lm.z] for lm in result.face_landmarks[0]]
            )

            if self.landmark_filters is None:
                self.landmark_filters = [
                    [OneEuroFilter(min_cutoff=1.5, beta=0.01) for _ in range(3)]
                    for _ in range(478)
                ]
                self.blendshape_filters = [
                    OneEuroFilter(min_cutoff=1.0, beta=0.05) for _ in range(52)
                ]

            smoothed = np.zeros_like(raw_landmarks)
            for i in range(min(478, len(raw_landmarks))):
                for j in range(3):
                    smoothed[i, j] = self.landmark_filters[i][j](raw_landmarks[i, j], t)
            output["landmarks"] = smoothed

            if result.face_blendshapes:
                raw_bs = np.array([cat.score for cat in result.face_blendshapes[0]])
                smoothed_bs = np.array(
                    [
                        self.blendshape_filters[i](raw_bs[i], t)
                        for i in range(min(52, len(raw_bs)))
                    ]
                )
                output["blendshapes"] = {
                    result.face_blendshapes[0][i].category_name: float(smoothed_bs[i])
                    for i in range(min(52, len(raw_bs)))
                }

            if result.facial_transformation_matrixes:
                output["matrix"] = result.facial_transformation_matrixes[0]

        # FPS calculation
        dt = t - self.prev_time
        self.fps_queue.append(dt)
        self.prev_time = t
        if len(self.fps_queue) > 1:
            avg_dt = sum(self.fps_queue) / len(self.fps_queue)
            output["fps"] = 1.0 / avg_dt if avg_dt > 0 else 0.0

        return output

    def close(self):
        self.landmarker.close()
