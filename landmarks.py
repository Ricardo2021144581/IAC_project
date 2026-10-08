"""Landmark extraction for the IML project application.

Reads a video, finds one face per frame with MediaPipe, keeps the 68 points of the
scheme used by the RAVDESS Facial Landmark Tracking dataset (OpenFace / iBUG 68),
resamples the trajectories to 30 frames per second and returns a table in the same
format as the CSV files of the dataset:

    frame, timestamp, confidence, x_0 ... x_67, y_0 ... y_67

Coordinates are in pixels, with the origin at the top-left corner of the image and
y increasing downwards, as in the dataset. Do not change this file: the application
calls `predict` in classifier.py with the table returned by `extract_landmarks`.

Two MediaPipe interfaces are supported, chosen automatically:
  * the legacy Face Mesh solution (mediapipe <= 0.10.14), whose model is installed
    with the package;
  * the Face Landmarker task (recent versions of mediapipe), whose model file is
    downloaded once to models/face_landmarker.task.
Both return the 468-point canonical face mesh; MP68 maps it to the 68-point scheme.
"""
from __future__ import annotations

import os
import urllib.request

import cv2
import numpy as np
import pandas as pd

TARGET_FPS = 30.0

# Index of the MediaPipe face-mesh point used for each of the 68 points (0..67) of Figure 2.
MP68 = [127, 234, 93, 132, 58, 172, 136, 150, 152, 379, 365, 397, 288, 361, 323, 454, 356,  # 0-16 jaw
        70, 63, 105, 66, 107,                                                                 # 17-21 brow (image left)
        336, 296, 334, 293, 300,                                                              # 22-26 brow (image right)
        168, 197, 5, 4,                                                                       # 27-30 nose bridge
        75, 97, 2, 326, 305,                                                                  # 31-35 nose base
        33, 160, 158, 133, 153, 144,                                                          # 36-41 eye (image left)
        362, 385, 387, 263, 373, 380,                                                         # 42-47 eye (image right)
        61, 39, 37, 0, 267, 269, 291, 405, 314, 17, 84, 181,                                  # 48-59 outer lips
        78, 82, 13, 312, 308, 317, 14, 87]                                                    # 60-67 inner lips
assert len(MP68) == 68

COLUMNS = ["frame", "timestamp", "confidence"] + [f"x_{k}" for k in range(68)] + [f"y_{k}" for k in range(68)]

MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/face_landmarker/"
             "face_landmarker/float16/1/face_landmarker.task")
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "face_landmarker.task")


class _Detector:
    """One face, 468+ mesh points per RGB frame, with whichever MediaPipe interface is installed."""

    def __init__(self):
        import mediapipe as mp
        self.mp = mp
        if hasattr(mp, "solutions"):
            self.kind = "face_mesh"
            self.impl = mp.solutions.face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1,
                                                        refine_landmarks=False, min_detection_confidence=0.5,
                                                        min_tracking_confidence=0.5)
        else:
            from mediapipe.tasks.python import BaseOptions, vision
            if not os.path.exists(MODEL_PATH):
                os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
                print(f"Downloading the face landmarker model to {MODEL_PATH} ...")
                urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
            self.kind = "face_landmarker"
            opts = vision.FaceLandmarkerOptions(base_options=BaseOptions(model_asset_path=MODEL_PATH),
                                                running_mode=vision.RunningMode.VIDEO, num_faces=1)
            self.impl = vision.FaceLandmarker.create_from_options(opts)

    def __call__(self, rgb: np.ndarray, t_ms: int):
        """Return a (468+, 2) array of normalised (x, y), or None if no face was found."""
        if self.kind == "face_mesh":
            res = self.impl.process(rgb)
            if not res.multi_face_landmarks:
                return None
            pts = res.multi_face_landmarks[0].landmark
        else:
            image = self.mp.Image(image_format=self.mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb))
            res = self.impl.detect_for_video(image, t_ms)
            if not res.face_landmarks:
                return None
            pts = res.face_landmarks[0]
        return np.array([[p.x, p.y] for p in pts])

    def close(self):
        self.impl.close()


def extract_landmarks(video_path: str):
    """Return (frames, info, middle_image).

    frames: DataFrame in the format of the dataset, resampled to 30 frames per second;
    info: dict describing the recording; middle_image: RGB frame with the 68 points drawn.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"cannot open video {video_path}")
    fps_reported = cap.get(cv2.CAP_PROP_FPS) or 0.0
    det = _Detector()
    times, points, images = [], [], []
    n_read, last_t = 0, -1
    while True:
        ok, bgr = cap.read()
        if not ok:
            break
        t_ms = cap.get(cv2.CAP_PROP_POS_MSEC)
        if not np.isfinite(t_ms) or t_ms <= last_t:          # some containers report no usable timestamps
            t_ms = last_t + (1000.0 / fps_reported if 0 < fps_reported < 240 else 1000.0 / TARGET_FPS)
        last_t = t_ms
        n_read += 1
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        h, w = rgb.shape[:2]
        mesh = det(rgb, int(round(t_ms)))
        if mesh is None:
            continue
        p = mesh[MP68] * np.array([w, h])                     # normalised -> pixels
        times.append(t_ms / 1000.0)
        points.append(p)
        images.append(rgb)
    cap.release()
    det.close()

    info = {"frames_read": n_read, "frames_with_face": len(points), "fps_reported": round(fps_reported, 2),
            "resolution": f"{images[0].shape[1]} x {images[0].shape[0]}" if images else "unknown", "detector": det.kind}
    if len(points) < 2:
        return pd.DataFrame(columns=COLUMNS), info, None

    times = np.array(times) - times[0]
    P = np.stack(points)                                       # (T, 68, 2)
    info["duration_s"] = round(float(times[-1]), 2)
    info["fps_measured"] = round(float((len(times) - 1) / times[-1]), 2) if times[-1] > 0 else None

    # resample every coordinate to a regular grid at 30 frames per second
    grid = np.arange(0.0, times[-1] + 1e-9, 1.0 / TARGET_FPS)
    X = np.column_stack([np.interp(grid, times, P[:, k, 0]) for k in range(68)])
    Y = np.column_stack([np.interp(grid, times, P[:, k, 1]) for k in range(68)])
    frames = pd.DataFrame(np.hstack([X, Y]), columns=COLUMNS[3:])
    frames.insert(0, "confidence", 1.0)
    frames.insert(0, "timestamp", np.round(grid, 3))
    frames.insert(0, "frame", np.arange(1, len(grid) + 1))
    info["frames_at_30fps"] = len(frames)

    return frames, info, _preview(images[len(images) // 2], P[len(images) // 2])


_REGIONS = [range(0, 17), range(17, 22), range(22, 27), range(27, 31), range(31, 36), list(range(36, 42)) + [36],
            list(range(42, 48)) + [42], list(range(48, 60)) + [48], list(range(60, 68)) + [60]]


def _preview(rgb: np.ndarray, p: np.ndarray) -> np.ndarray:
    """The face of one frame, cropped, with the 68 points and the regions of Figure 2 drawn on it."""
    x0, y0 = p.min(axis=0); x1, y1 = p.max(axis=0)
    m = 0.35 * max(x1 - x0, y1 - y0)
    h, w = rgb.shape[:2]
    a, b = int(max(0, x0 - m)), int(min(w, x1 + m)); c, d = int(max(0, y0 - m)), int(min(h, y1 + m))
    img = np.ascontiguousarray(rgb[c:d, a:b])
    scale = 480.0 / max(img.shape[:2])
    img = cv2.resize(img, (int(img.shape[1] * scale), int(img.shape[0] * scale)))
    q = (p - [a, c]) * scale
    for region in _REGIONS:
        pts = np.round(q[list(region)]).astype(np.int32).reshape(-1, 1, 2)
        cv2.polylines(img, [pts], False, (230, 230, 230), 1, cv2.LINE_AA)
    for (x, y) in q:
        cv2.circle(img, (int(round(x)), int(round(y))), 3, (26, 163, 160), -1, cv2.LINE_AA)
    return img
