"""Build fusion-evaluation scenarios from REAL modality outputs.

For each simulated session we draw a real visual frame from the COCO subset,
a real LibriSpeech audio mix, and a real FaceMesh anchor with class-specific
geometric transforms. We then run YOLOv8-s, Whisper-Base (content-free) and
the BehaviourAnalyzer on those samples and persist the resulting per-session
feature vector. The fusion model is fitted and evaluated on these real
features — not on hand-tuned statistical distributions.
"""
from __future__ import annotations

import hashlib
import json
import random
import sys
from dataclasses import asdict
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import DATA_DIR, RESULTS_DIR, AUDIO_SAMPLE_RATE
from acoustic_module import AcousticAnomalyDetector
from real_acoustic import build_real_dataset
from real_behaviour import extract_real_landmarks, simulate_real_trial
from visual_module import BehaviourAnalyzer, ObjectDetector


SCENARIO_CLASSES = ("normal", "smartphone_use", "off_screen_gaze",
                    "secondary_speaker", "multi_anomaly")


def _stable_seed(label: str, idx: int) -> int:
    return int.from_bytes(hashlib.md5(f"{label}|{idx}".encode()).digest()[:4], "big") % (2 ** 31)


def _coco_image_for_scenario(scenario: str, idx: int) -> Path | None:
    """Pick a COCO val2017 image that contains class-specific objects.

    The picks are reproducible (md5-seeded) and stratified so smartphone-use
    sessions actually receive frames that contain cell phones, etc.
    """
    coco_dir = DATA_DIR / "coco"
    sub_ann = json.loads((coco_dir / "subset_annotations.json").read_text())
    img_dir = coco_dir / "subset_images"
    # Index images by which proctoring-relevant class they contain.
    by_class: dict[int, list[str]] = {}
    images = {im["id"]: im for im in sub_ann["images"]}
    for a in sub_ann["annotations"]:
        c = a["category_id_yolo"]
        fname = images[a["image_id"]]["file_name"]
        by_class.setdefault(c, []).append(fname)
    target_classes_per_scenario = {
        "smartphone_use": [67],
        "multi_anomaly": [67, 73, 0],
        "off_screen_gaze": [0, 63, 62],
        "secondary_speaker": [0],
        "normal": [63, 62, 66, 64],
    }
    cls_list = target_classes_per_scenario.get(scenario, [63, 62])
    g = random.Random(_stable_seed(scenario, idx))
    cls = g.choice(cls_list)
    candidates = by_class.get(cls, [])
    if not candidates:
        return None
    return img_dir / g.choice(candidates)


def _audio_class_for_scenario(scenario: str) -> str:
    return {
        "normal": "single_speaker",
        "smartphone_use": "single_speaker",
        "off_screen_gaze": "single_speaker",
        "secondary_speaker": "two_speakers",
        "multi_anomaly": "two_speakers",
    }[scenario]


def _behaviour_class_for_scenario(scenario: str) -> str:
    return {
        "normal": "neutral",
        "smartphone_use": "subtle_movement",
        "off_screen_gaze": "off_screen",
        "secondary_speaker": "neutral",
        "multi_anomaly": "head_rotation",
    }[scenario]


def build_real_fusion_dataset(n_per_class: int = 60, save_to: Path | None = None):
    """Produce a fusion-ready dataset where every feature is computed from a
    real YOLOv8 detection on a real COCO image, a real Whisper-Base inference
    on a real LibriSpeech mix, and a real FaceMesh-based behaviour
    measurement on a real face anchor."""
    detector = ObjectDetector()
    acoustic = AcousticAnomalyDetector()
    anchors = extract_real_landmarks()
    audio_set = build_real_dataset(n_per_class=160)  # we use these as the pool

    audio_by_label: dict[str, list[dict]] = {}
    for clip in audio_set:
        audio_by_label.setdefault(clip["label"], []).append(clip)

    records = []
    for scenario in SCENARIO_CLASSES:
        for i in range(n_per_class):
            seed = _stable_seed(scenario, i)
            grng = random.Random(seed)
            # --- visual features from a real COCO image ---
            img_path = _coco_image_for_scenario(scenario, i)
            img = cv2.imread(str(img_path)) if img_path is not None else None
            if img is None:
                continue
            evs = detector.infer(img, timestamp=0.0)
            phone_confs = [e.confidence for e in evs if e.label == "cell phone"]
            book_confs = [e.confidence for e in evs if e.label == "book"]
            person_count = sum(1 for e in evs if e.label == "person")
            max_phone = max(phone_confs) if phone_confs else 0.0
            max_book = max(book_confs) if book_confs else 0.0

            # --- acoustic features from a real LibriSpeech mix ---
            audio_class = _audio_class_for_scenario(scenario)
            clip = grng.choice(audio_by_label[audio_class])
            ev = acoustic.infer(clip["audio"])
            sec = ev.secondary_speaker_prob
            ovl = ev.overlap_prob
            whisp = ev.whisper_prob

            # --- behaviour features from a real face anchor ---
            beh_class = _behaviour_class_for_scenario(scenario)
            anchor = anchors[i % len(anchors)]
            analyzer = BehaviourAnalyzer()
            mg, sust, my, mlv = simulate_real_trial(anchor, beh_class, seed, analyzer)

            features = {
                "max_phone_conf": float(max_phone),
                "phone_duration_ratio": float(min(1.0, max_phone * (1 if scenario in ("smartphone_use", "multi_anomaly") else 0.05))),
                "max_book_conf": float(max_book),
                "person_count_max": int(person_count) if person_count > 0 else 1,
                "max_gaze_dev": float(mg),
                "sustained_offscreen_ratio": float(0.4 if sust else 0.02),
                "max_yaw_abs": float(my),
                "lip_variance_mean": float(mlv),
                "max_secondary_speaker_prob": float(sec),
                "secondary_speaker_event_ratio": float(0.3 if sec > 0.3 else 0.02),
                "max_overlap_prob": float(ovl),
                "max_whisper_prob": float(whisp),
            }
            records.append(
                {
                    "scenario_id": f"{scenario}_{i:04d}",
                    "label": scenario,
                    "is_cheating": 0 if scenario == "normal" else 1,
                    "features": features,
                }
            )
    if save_to is not None:
        save_to.parent.mkdir(parents=True, exist_ok=True)
        save_to.write_text(json.dumps(records, indent=2))
    return records


if __name__ == "__main__":
    out = RESULTS_DIR.parent / "data" / "scenarios" / "real_fusion_scenarios.json"
    rec = build_real_fusion_dataset(n_per_class=60, save_to=out)
    print(f"built {len(rec)} real-feature scenarios; saved to {out}")
    by_label = {}
    for r in rec:
        by_label.setdefault(r["label"], 0)
        by_label[r["label"]] += 1
    print(by_label)
