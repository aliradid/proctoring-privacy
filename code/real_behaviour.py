"""Real-face behavioural-cue benchmark.

We collect 10 CC0/Pexels face photographs (Pexels license permits academic
use), run MediaPipe FaceMesh on every image to obtain the actual 478-landmark
mesh, then simulate exam-like behaviour by applying *geometric transforms* to
the real anchor landmarks. The transforms reproduce the four conditions of
interest:

  neutral             — small Gaussian jitter, no global rotation
  subtle_movement     — brief glance away that should NOT trigger
  off_screen_gaze     — sustained head yaw beyond the threshold
  head_rotation       — periodic head rotation, simulating a candidate
                        repeatedly checking a secondary screen
  lip_movement        — modulation of the upper/lower lip distance

Because the anchor mesh is real, the geometric descriptors (gaze deviation,
yaw, lip variance) are computed on the same shape distributions a deployed
system would see; the *motion* is procedural and labelled so we can compute
detection metrics.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import DATA_DIR, FACEMESH_LANDMARK_COUNT


def _load_face_landmarker():
    import mediapipe as mp
    from mediapipe.tasks import python
    from mediapipe.tasks.python import vision

    base = python.BaseOptions(model_asset_path=str(DATA_DIR / "face_landmarker.task"))
    opts = vision.FaceLandmarkerOptions(base_options=base, num_faces=1)
    return vision.FaceLandmarker.create_from_options(opts)


def extract_real_landmarks() -> list[np.ndarray]:
    import mediapipe as mp

    landmarker = _load_face_landmarker()
    out = []
    for path in sorted((DATA_DIR / "faces").glob("*.jpg")):
        img = cv2.imread(str(path))
        if img is None:
            continue
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        res = landmarker.detect(mp_img)
        if not res.face_landmarks:
            continue
        lms = res.face_landmarks[0]
        h, w = img.shape[:2]
        # Convert to pixel coordinates so the geometric thresholds in
        # visual_module.BehaviourAnalyzer apply unchanged.
        pts = np.array([[l.x * w, l.y * h, l.z * w] for l in lms[:FACEMESH_LANDMARK_COUNT]], dtype=np.float32)
        out.append(pts)
    return out


def apply_yaw(pts: np.ndarray, yaw_deg: float) -> np.ndarray:
    """Rotate landmarks around the nose-bridge axis by the requested yaw."""
    nose_bridge = pts[168] if pts.shape[0] > 168 else pts.mean(axis=0)
    centred = pts - nose_bridge
    a = np.radians(yaw_deg)
    R = np.array([
        [np.cos(a), 0, np.sin(a)],
        [0, 1, 0],
        [-np.sin(a), 0, np.cos(a)],
    ], dtype=np.float32)
    rotated = centred @ R.T
    return rotated + nose_bridge


def apply_lip_motion(pts: np.ndarray, amplitude: float, t: float) -> np.ndarray:
    upper_idx = 13
    lower_idx = 14
    pts = pts.copy()
    delta = amplitude * (1 + np.sin(2 * np.pi * 3.0 * t))
    pts[upper_idx, 1] -= delta * 1.5
    pts[lower_idx, 1] += delta * 1.5
    return pts


def jitter(pts: np.ndarray, rng: np.random.Generator, sigma: float = 1.2) -> np.ndarray:
    out = pts.copy()
    out[:, :2] += rng.normal(0, sigma, (pts.shape[0], 2)).astype(np.float32)
    return out


def simulate_real_trial(anchor: np.ndarray, label: str, seed: int,
                         analyzer, dur: float = 4.0, fps: int = 25):
    g = np.random.default_rng(seed)
    n = int(dur * fps)
    gaze_devs, yaws, lip_vars, sust = [], [], [], []
    sigma = float(g.uniform(0.8, 2.8))
    for i in range(n):
        t = i / fps
        pts = anchor.copy()
        if label == "off_screen":
            target = g.uniform(20, 35)
            if t > 0.5:
                pts = apply_yaw(pts, target)
            else:
                pts = apply_yaw(pts, g.normal(0, 1.5))
        elif label == "head_rotation":
            if 0.8 < t < 3.4:
                pts = apply_yaw(pts, g.uniform(22, 32) * np.sin(2 * np.pi * 0.5 * t))
            else:
                pts = apply_yaw(pts, g.normal(0, 1.5))
        elif label == "lip_movement":
            pts = apply_lip_motion(pts, g.uniform(0.4, 0.7), t)
            pts = apply_yaw(pts, g.normal(0, 1.5))
        elif label == "subtle_movement":
            if 1.0 < t < 1.3:
                pts = apply_yaw(pts, 12)
            else:
                pts = apply_yaw(pts, g.normal(0, 1.5))
        elif label == "neutral":
            pts = apply_yaw(pts, g.normal(0, 1.5))
        pts = jitter(pts, g, sigma=sigma)
        ev = analyzer.update(pts, timestamp=t)
        gaze_devs.append(ev.gaze_deviation_deg)
        yaws.append(abs(ev.head_yaw))
        lip_vars.append(ev.lip_variance)
        sust.append(ev.sustained_off_screen)
    return float(np.max(gaze_devs)), bool(np.any(sust)), float(np.max(yaws)), float(np.max(lip_vars))


if __name__ == "__main__":
    pts = extract_real_landmarks()
    print(f"extracted {len(pts)} real face anchor meshes")
    for i, p in enumerate(pts):
        print(i, p.shape, "centroid:", p.mean(axis=0)[:2])
