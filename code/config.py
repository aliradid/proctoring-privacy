"""Configuration for the privacy-preserving proctoring experimental framework.

All hyperparameters, thresholds and weights used throughout the study are
centralized here so that every experiment is reproducible from a single file.
"""
from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"
FIGURES_DIR = ROOT / "figures"
MODELS_DIR = ROOT / "models"

for d in (DATA_DIR, RESULTS_DIR, FIGURES_DIR, MODELS_DIR):
    d.mkdir(parents=True, exist_ok=True)

RANDOM_SEED = 42

YOLO_MODEL = "yolov8s.pt"
YOLO_CONF_THRESHOLD = 0.35
YOLO_IOU_THRESHOLD = 0.45
YOLO_HIGH_RISK_CLASSES = {
    67: "cell phone",
    73: "book",
    63: "laptop",
    62: "tv",
    66: "keyboard",
    64: "mouse",
    0:  "person",
}

FACEMESH_LANDMARK_COUNT = 468
GAZE_DEVIATION_THRESHOLD_DEG = 18.0
HEAD_POSE_PITCH_THRESHOLD = 22.0
HEAD_POSE_YAW_THRESHOLD = 25.0
SUSTAINED_GAZE_SECONDS = 2.5
LIP_MOVEMENT_VARIANCE_THRESHOLD = 0.0021
CALIBRATION_SECONDS = 9.0

WHISPER_MODEL = "base"
AUDIO_SAMPLE_RATE = 16000
AUDIO_BUFFER_SECONDS = 5.0
ACOUSTIC_ANOMALY_THRESHOLD = 0.55

FUSION_ALPHA = 0.45
FUSION_BETA  = 0.30
FUSION_GAMMA = 0.25
SI_WARN_THRESHOLD = 0.35
SI_FLAG_THRESHOLD = 0.65

SCENARIO_DURATION_SEC = 60
SCENARIO_FPS = 5
NUM_SCENARIOS_PER_CLASS = 80

SCENARIO_CLASSES = [
    "normal",
    "smartphone_use",
    "off_screen_gaze",
    "secondary_speaker",
    "multi_anomaly",
]

PLOT_DPI = 220
PLOT_STYLE = "whitegrid"
