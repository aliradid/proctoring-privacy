"""Visual subsystem: YOLOv8-s object detection + MediaPipe FaceMesh landmarks.

The module exposes two analyzers:
    * ObjectDetector – wraps a YOLOv8 model and returns risk-class metadata.
    * BehaviourAnalyzer – consumes MediaPipe FaceMesh landmarks and computes
      gaze deviation, head pose, and lip-movement variance.

Only numerical metadata is exposed externally; no raw frames or facial imagery
are retained, in line with the privacy-by-design constraints of the framework.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Iterable

import cv2
import numpy as np

from config import (
    FACEMESH_LANDMARK_COUNT,
    GAZE_DEVIATION_THRESHOLD_DEG,
    HEAD_POSE_PITCH_THRESHOLD,
    HEAD_POSE_YAW_THRESHOLD,
    LIP_MOVEMENT_VARIANCE_THRESHOLD,
    SUSTAINED_GAZE_SECONDS,
    YOLO_CONF_THRESHOLD,
    YOLO_HIGH_RISK_CLASSES,
    YOLO_IOU_THRESHOLD,
    YOLO_MODEL,
)


@dataclass
class DetectionEvent:
    label: str
    confidence: float
    timestamp: float
    bbox: tuple[float, float, float, float]
    class_id: int


@dataclass
class BehaviourEvent:
    gaze_deviation_deg: float
    head_pitch: float
    head_yaw: float
    lip_variance: float
    sustained_off_screen: bool
    timestamp: float


class ObjectDetector:
    """YOLOv8 wrapper restricted to proctoring risk classes."""

    def __init__(self, model_path: str = YOLO_MODEL, device: str | None = None):
        from ultralytics import YOLO

        self.model = YOLO(model_path)
        self.device = device
        self.conf = YOLO_CONF_THRESHOLD
        self.iou = YOLO_IOU_THRESHOLD
        self.risk_classes = YOLO_HIGH_RISK_CLASSES

    def infer(self, frame: np.ndarray, timestamp: float | None = None) -> list[DetectionEvent]:
        ts = time.time() if timestamp is None else timestamp
        kw = {"conf": self.conf, "iou": self.iou, "verbose": False}
        if self.device:
            kw["device"] = self.device
        results = self.model(frame, **kw)
        events: list[DetectionEvent] = []
        for r in results:
            if r.boxes is None:
                continue
            for cls, conf, box in zip(
                r.boxes.cls.cpu().numpy().astype(int),
                r.boxes.conf.cpu().numpy(),
                r.boxes.xyxy.cpu().numpy(),
            ):
                if cls in self.risk_classes:
                    events.append(
                        DetectionEvent(
                            label=self.risk_classes[cls],
                            confidence=float(conf),
                            timestamp=ts,
                            bbox=tuple(float(x) for x in box),
                            class_id=int(cls),
                        )
                    )
        return events


_LEFT_EYE = [33, 133, 159, 145]
_RIGHT_EYE = [263, 362, 386, 374]
_NOSE_TIP = 1
_CHIN = 152
_LEFT_EAR = 234
_RIGHT_EAR = 454
_UPPER_LIP = 13
_LOWER_LIP = 14
_LEFT_LIP = 78
_RIGHT_LIP = 308


class BehaviourAnalyzer:
    """Consumes facial landmarks and emits non-identifying behavioural cues."""

    def __init__(self):
        self._gaze_off_start: float | None = None
        self._lip_history: list[float] = []
        self._lip_window = 30

    @staticmethod
    def _angle(a: np.ndarray, b: np.ndarray) -> float:
        v = b - a
        return math.degrees(math.atan2(v[1], v[0]))

    def _gaze_deviation(self, landmarks: np.ndarray) -> float:
        left_center = landmarks[_LEFT_EYE].mean(axis=0)
        right_center = landmarks[_RIGHT_EYE].mean(axis=0)
        eye_axis = right_center - left_center
        nose_proj = landmarks[_NOSE_TIP] - (left_center + right_center) / 2
        normal = np.array([-eye_axis[1], eye_axis[0]])
        normal_norm = normal / (np.linalg.norm(normal) + 1e-8)
        cos = np.dot(nose_proj[:2], normal_norm) / (np.linalg.norm(nose_proj[:2]) + 1e-8)
        cos = float(np.clip(cos, -1.0, 1.0))
        return math.degrees(math.acos(abs(cos)))

    def _head_pose(self, landmarks: np.ndarray) -> tuple[float, float]:
        left_ear = landmarks[_LEFT_EAR]
        right_ear = landmarks[_RIGHT_EAR]
        nose = landmarks[_NOSE_TIP]
        chin = landmarks[_CHIN]
        yaw = math.degrees(
            math.atan2(nose[0] - (left_ear[0] + right_ear[0]) / 2, abs(right_ear[0] - left_ear[0]) + 1e-6)
        )
        pitch = math.degrees(math.atan2(nose[1] - chin[1], abs(chin[1] - nose[1]) + 1e-6))
        return float(pitch), float(yaw)

    def _lip_variance(self, landmarks: np.ndarray) -> float:
        upper = landmarks[_UPPER_LIP]
        lower = landmarks[_LOWER_LIP]
        gap = float(np.linalg.norm(upper - lower))
        self._lip_history.append(gap)
        if len(self._lip_history) > self._lip_window:
            self._lip_history.pop(0)
        return float(np.var(self._lip_history)) if len(self._lip_history) >= 5 else 0.0

    def update(self, landmarks: np.ndarray, timestamp: float) -> BehaviourEvent:
        if landmarks.ndim != 2 or landmarks.shape[0] < FACEMESH_LANDMARK_COUNT:
            return BehaviourEvent(0.0, 0.0, 0.0, 0.0, False, timestamp)
        gaze = self._gaze_deviation(landmarks)
        pitch, yaw = self._head_pose(landmarks)
        lip_var = self._lip_variance(landmarks)
        off_screen = gaze > GAZE_DEVIATION_THRESHOLD_DEG or abs(yaw) > HEAD_POSE_YAW_THRESHOLD
        if off_screen:
            if self._gaze_off_start is None:
                self._gaze_off_start = timestamp
            sustained = (timestamp - self._gaze_off_start) >= SUSTAINED_GAZE_SECONDS
        else:
            self._gaze_off_start = None
            sustained = False
        return BehaviourEvent(
            gaze_deviation_deg=gaze,
            head_pitch=pitch,
            head_yaw=yaw,
            lip_variance=lip_var,
            sustained_off_screen=sustained,
            timestamp=timestamp,
        )


def detection_event_to_dict(ev: DetectionEvent) -> dict:
    return {
        "label": ev.label,
        "confidence": ev.confidence,
        "timestamp": ev.timestamp,
        "bbox": list(ev.bbox),
        "class_id": ev.class_id,
    }


def behaviour_event_to_dict(ev: BehaviourEvent) -> dict:
    return {
        "gaze_deviation_deg": ev.gaze_deviation_deg,
        "head_pitch": ev.head_pitch,
        "head_yaw": ev.head_yaw,
        "lip_variance": ev.lip_variance,
        "sustained_off_screen": ev.sustained_off_screen,
        "timestamp": ev.timestamp,
    }
