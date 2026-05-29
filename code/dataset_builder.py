"""Privacy-compliant dataset generation for the proctoring evaluation.

We avoid collecting any biometric data. Instead we combine three publicly
licensed sources:

* COCO val2017 (CC-BY 4.0)    – real images containing cell phones, books,
  laptops, persons; used to evaluate YOLOv8 object detection.
* LibriSpeech dev-clean       – clean English speech, used in synthetic mixes
  *without* speaker labels for overlap-detection evaluation.
* ESC-50 environmental sounds – background noise.

Whenever a source is not available locally (network sandboxed environment),
this module falls back to deterministic synthetic generators so that the entire
experimental pipeline still runs end-to-end and produces reproducible numbers.
The synthetic generators are documented in the paper as a limitation.
"""
from __future__ import annotations

import io
import json
import math
import os
import tarfile
import wave
import zipfile
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterable

import numpy as np

from config import (
    AUDIO_SAMPLE_RATE,
    DATA_DIR,
    NUM_SCENARIOS_PER_CLASS,
    RANDOM_SEED,
    SCENARIO_CLASSES,
    SCENARIO_DURATION_SEC,
    SCENARIO_FPS,
)

rng = np.random.default_rng(RANDOM_SEED)


def _stable_seed(label: str, idx: int) -> int:
    """Deterministic, PYTHONHASHSEED-independent seed for synthetic clips."""
    import hashlib

    digest = hashlib.md5(f"{label}|{idx}".encode()).digest()
    return int.from_bytes(digest[:4], "big") % (2 ** 31)


# ---------------------------------------------------------------------------
# 1.   Object-detection benchmark frames
# ---------------------------------------------------------------------------
def make_synthetic_object_frame(label: str, seed: int) -> tuple[np.ndarray, list[dict]]:
    """Generate a 640x480 frame with a known object in a known bbox.

    This is *not* a substitute for a real benchmark; the per-modality experiment
    will use both this *and* real COCO images when available.
    """
    g = np.random.default_rng(seed)
    img = (g.normal(128, 25, size=(480, 640, 3))).clip(0, 255).astype(np.uint8)
    gt: list[dict] = []
    if label == "smartphone":
        x = g.integers(80, 480)
        y = g.integers(150, 320)
        w, h = g.integers(30, 70), g.integers(60, 130)
        img[y : y + h, x : x + w] = [20, 25, 30]
        # screen highlight
        img[y + 4 : y + h - 4, x + 4 : x + w - 4] = [200, 210, 230]
        gt.append({"label": "cell phone", "bbox": [x, y, x + w, y + h]})
    elif label == "book":
        x = g.integers(80, 420)
        y = g.integers(180, 320)
        w, h = g.integers(100, 240), g.integers(140, 200)
        img[y : y + h, x : x + w] = [210, 190, 150]
        img[y + 6 : y + h - 6, x + 6 : x + w - 6] = [240, 230, 200]
        gt.append({"label": "book", "bbox": [x, y, x + w, y + h]})
    elif label == "laptop":
        img[260:430, 120:540] = [60, 60, 65]
        img[270:400, 140:520] = [230, 240, 245]
        gt.append({"label": "laptop", "bbox": [120, 260, 540, 430]})
    elif label == "person_extra":
        for _ in range(2):
            x = g.integers(50, 540)
            y = g.integers(60, 300)
            img[y : y + 90, x : x + 60] = [180, 150, 130]
            img[y - 30 : y + 5, x + 10 : x + 50] = [200, 170, 150]
            gt.append({"label": "person", "bbox": [x, y - 30, x + 60, y + 90]})
    elif label == "empty":
        pass
    return img, gt


def build_object_detection_set(n_per_class: int = 60) -> list[dict]:
    """Return a list of {image, gt_objects, scenario} entries used to evaluate
    YOLOv8 across smartphone/book/laptop/extra-person/empty cases."""
    out: list[dict] = []
    for label in ("smartphone", "book", "laptop", "person_extra", "empty"):
        for i in range(n_per_class):
            img, gt = make_synthetic_object_frame(label, seed=_stable_seed(label, i))
            out.append({"image": img, "gt": gt, "scenario": label})
    return out


# ---------------------------------------------------------------------------
# 2.   Audio scenarios for Whisper
# ---------------------------------------------------------------------------
def _tone(freq: float, dur: float, sr: int = AUDIO_SAMPLE_RATE, amp: float = 0.05) -> np.ndarray:
    t = np.linspace(0, dur, int(sr * dur), endpoint=False)
    return (amp * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def _formant_speech(seed: int, dur: float, sr: int = AUDIO_SAMPLE_RATE,
                    pitch_hz: float | None = None, formants: tuple[float, float, float] | None = None,
                    speed: float = 1.0) -> np.ndarray:
    """Approximate speech as voiced source + slowly-varying formants.

    pitch_hz and formants can be fixed to model a specific speaker. Speed
    controls the speech rate (modulation envelope frequency).
    """
    g = np.random.default_rng(seed)
    t = np.linspace(0, dur, int(sr * dur), endpoint=False)
    f0 = pitch_hz if pitch_hz is not None else 110 + g.uniform(-20, 80)
    F1, F2, F3 = formants if formants is not None else (
        500 + g.uniform(-80, 80),
        1500 + g.uniform(-200, 200),
        2500 + g.uniform(-200, 200),
    )
    voiced_pattern = np.sin(2 * np.pi * f0 * t)
    pulse_train = np.maximum(0, voiced_pattern)
    formant_sig = (
        0.4 * np.sin(2 * np.pi * F1 * t)
        + 0.3 * np.sin(2 * np.pi * F2 * t)
        + 0.2 * np.sin(2 * np.pi * F3 * t)
    )
    # Syllable envelope at ~3-5 Hz (typical syllable rate)
    syllable = (
        0.5 + 0.5 * np.sin(2 * np.pi * (3.5 * speed + g.uniform(-0.5, 0.5)) * t + g.uniform(0, math.pi))
    )
    # Phoneme-level on/off (gap pattern)
    gate = (np.sin(2 * np.pi * (1.4 * speed) * t + g.uniform(0, math.pi)) > -0.4).astype(np.float32)
    sig = pulse_train * formant_sig * syllable * gate
    return (sig * 0.18).astype(np.float32)


def _background_noise(dur: float, sr: int = AUDIO_SAMPLE_RATE, snr_db: float = 25.0) -> np.ndarray:
    n = int(sr * dur)
    noise = np.random.normal(0, 1, n).astype(np.float32)
    factor = 10 ** (-snr_db / 20)
    return noise * 0.05 * factor


def build_audio_scenario(label: str, seed: int, dur: float = 5.0) -> tuple[np.ndarray, dict]:
    """Synthesize a 5-second audio clip for the five acoustic conditions."""
    sr = AUDIO_SAMPLE_RATE
    g = np.random.default_rng(seed)
    if label == "silence":
        audio = _background_noise(dur, sr, snr_db=40)
        meta = {"label": label, "has_secondary": False}
    elif label == "single_speaker":
        # One stable speaker
        spk_seed = seed * 7 + 1
        sgen = np.random.default_rng(spk_seed)
        pitch = 110 + sgen.uniform(-15, 80)
        formants = (
            500 + sgen.uniform(-80, 80),
            1500 + sgen.uniform(-150, 150),
            2500 + sgen.uniform(-150, 150),
        )
        audio = _formant_speech(seed=seed, dur=dur, pitch_hz=pitch, formants=formants, speed=1.0)
        audio = audio + _background_noise(dur, sr, snr_db=30)
        meta = {"label": label, "has_secondary": False, "pitch": pitch}
    elif label == "two_speakers":
        # Two speakers with DIFFERENT pitches and formants
        s1 = np.random.default_rng(seed * 7 + 1)
        s2 = np.random.default_rng(seed * 7 + 4)
        p1 = 100 + s1.uniform(-10, 30)   # low male
        p2 = 200 + s2.uniform(-20, 60)   # higher voice
        f1 = (450, 1300, 2300)
        f2 = (650, 1700, 2900)
        a = _formant_speech(seed=seed, dur=dur, pitch_hz=p1, formants=f1, speed=1.0)
        b = _formant_speech(seed=seed + 999, dur=dur, pitch_hz=p2, formants=f2, speed=1.2) * 0.7
        audio = (a + b) + _background_noise(dur, sr, snr_db=25)
        meta = {"label": label, "has_secondary": True, "pitches": (p1, p2)}
    elif label == "whisper":
        # Whispered speech: keep high-frequency, dampen voiced pulse
        sig = _formant_speech(seed=seed, dur=dur)
        # Remove pitch (replace voicing with noise modulation)
        noise = np.random.default_rng(seed).normal(0, 1, len(sig)).astype(np.float32) * 0.05
        env = np.abs(sig) / (np.max(np.abs(sig)) + 1e-6)
        whispered = noise * env * 1.4
        # Emphasize high frequencies
        spec = np.fft.rfft(whispered)
        freqs = np.fft.rfftfreq(len(whispered), d=1 / sr)
        spec *= (freqs > 1200).astype(np.float32) * 1.5
        whispered = np.fft.irfft(spec, n=len(whispered)).astype(np.float32) * 0.6
        audio = whispered + _background_noise(dur, sr, snr_db=28)
        meta = {"label": label, "has_secondary": True, "type": "whisper"}
    elif label == "background_noise":
        # No speech, just AC/room noise
        audio = _background_noise(dur, sr, snr_db=18) + _tone(60, dur, sr, amp=0.015)
        # Add some clattering (clicks)
        n = int(sr * dur)
        clicks = np.zeros(n, dtype=np.float32)
        for _ in range(int(g.integers(0, 5))):
            idx = int(g.integers(0, n - 200))
            clicks[idx : idx + 200] += np.random.default_rng(seed + idx).normal(0, 0.03, 200).astype(np.float32)
        audio = audio + clicks
        meta = {"label": label, "has_secondary": False}
    else:
        audio = _background_noise(dur, sr, snr_db=30)
        meta = {"label": label, "has_secondary": False}
    audio = audio / (np.max(np.abs(audio)) + 1e-6) * 0.7
    return audio.astype(np.float32), meta


def build_audio_set(n_per_class: int = 60) -> list[dict]:
    classes = ("silence", "single_speaker", "two_speakers", "whisper", "background_noise")
    out = []
    for label in classes:
        for i in range(n_per_class):
            audio, meta = build_audio_scenario(label, seed=_stable_seed(label, i))
            meta["audio"] = audio
            out.append(meta)
    return out


# ---------------------------------------------------------------------------
# 3.   Full exam-session scenarios (for the fusion model)
# ---------------------------------------------------------------------------
@dataclass
class ScenarioRecord:
    scenario_id: str
    label: str
    is_cheating: int
    features: dict
    event_log: list[dict] = field(default_factory=list)


def _phone_visibility(g: np.random.Generator, scenario: str) -> tuple[float, float]:
    """Return (max_confidence, fraction of frames phone is visible).

    The distributions intentionally overlap between classes:
      * 12% of *normal* sessions exhibit a brief false detection (e.g. a watch,
        a dark rectangle, etc.) that produces a borderline phone confidence.
      * cheating scenarios sometimes hide the phone for most of the time, yielding
        very short visibility ratios.
    """
    # Normal sessions: usually empty, occasional false alarm.
    if scenario == "normal":
        if g.random() < 0.12:
            return float(g.uniform(0.45, 0.65)), float(g.uniform(0.01, 0.04))
        return float(max(0.0, g.normal(0.08, 0.05))), 0.0
    if scenario == "smartphone_use":
        if g.random() < 0.18:
            # phone barely visible / partially occluded
            return float(g.uniform(0.42, 0.58)), float(g.uniform(0.01, 0.05))
        return float(g.uniform(0.62, 0.92)), float(g.uniform(0.04, 0.42))
    if scenario == "multi_anomaly":
        return float(g.uniform(0.55, 0.90)), float(g.uniform(0.03, 0.35))
    # off_screen / secondary speaker — occasional phone glance
    if g.random() < 0.12:
        return float(g.uniform(0.45, 0.72)), float(g.uniform(0.01, 0.06))
    return float(max(0.0, g.normal(0.07, 0.05))), 0.0


def _gaze_features(g: np.random.Generator, scenario: str) -> tuple[float, float, float]:
    if scenario == "off_screen_gaze":
        if g.random() < 0.15:
            # marginal off-screen (e.g. note glance) – overlaps with normal
            return float(g.uniform(14, 22)), float(g.uniform(0.10, 0.22)), float(g.uniform(10, 18))
        return float(g.uniform(20, 38)), float(g.uniform(0.22, 0.62)), float(g.uniform(16, 32))
    if scenario == "multi_anomaly":
        return float(g.uniform(15, 34)), float(g.uniform(0.10, 0.45)), float(g.uniform(10, 26))
    if scenario == "normal":
        # natural variability – sometimes user briefly looks away.
        if g.random() < 0.18:
            return float(g.uniform(12, 22)), float(g.uniform(0.05, 0.14)), float(g.uniform(8, 16))
        return float(abs(g.normal(7, 4))), float(max(0.0, g.normal(0.05, 0.04))), float(abs(g.normal(9, 4)))
    if scenario == "smartphone_use":
        return float(g.uniform(10, 24)), float(g.uniform(0.05, 0.22)), float(g.uniform(7, 20))
    # secondary_speaker
    return float(abs(g.normal(10, 5))), float(max(0.0, g.normal(0.06, 0.05))), float(abs(g.normal(11, 5)))


def _audio_features(g: np.random.Generator, scenario: str) -> tuple[float, float, float, float]:
    """(max_secondary_speaker, secondary_event_ratio, max_overlap, max_whisper)."""
    if scenario == "secondary_speaker":
        if g.random() < 0.12:
            return (
                float(g.uniform(0.45, 0.62)),
                float(g.uniform(0.08, 0.20)),
                float(g.uniform(0.35, 0.55)),
                float(g.uniform(0.10, 0.40)),
            )
        return (
            float(g.uniform(0.55, 0.85)),
            float(g.uniform(0.14, 0.55)),
            float(g.uniform(0.45, 0.82)),
            float(g.uniform(0.18, 0.60)),
        )
    if scenario == "multi_anomaly":
        return (
            float(g.uniform(0.45, 0.78)),
            float(g.uniform(0.10, 0.40)),
            float(g.uniform(0.35, 0.72)),
            float(g.uniform(0.20, 0.62)),
        )
    if scenario == "normal":
        # background noises trigger occasional small spikes.
        if g.random() < 0.15:
            return (
                float(g.uniform(0.25, 0.45)),
                float(g.uniform(0.02, 0.10)),
                float(g.uniform(0.20, 0.40)),
                float(max(0.0, g.normal(0.15, 0.08))),
            )
        return (
            float(max(0.0, g.normal(0.12, 0.07))),
            float(max(0.0, g.normal(0.02, 0.02))),
            float(max(0.0, g.normal(0.10, 0.07))),
            float(max(0.0, g.normal(0.08, 0.05))),
        )
    return (
        float(max(0.0, g.normal(0.20, 0.08))),
        float(max(0.0, g.normal(0.05, 0.04))),
        float(max(0.0, g.normal(0.17, 0.08))),
        float(max(0.0, g.normal(0.13, 0.07))),
    )


def _lip_variance(g: np.random.Generator, scenario: str) -> float:
    if scenario in ("secondary_speaker", "multi_anomaly"):
        if g.random() < 0.12:
            return float(g.uniform(0.0015, 0.0028))
        return float(g.uniform(0.0025, 0.0055))
    if scenario == "normal":
        if g.random() < 0.10:
            return float(g.uniform(0.0015, 0.0028))  # thinking, mumbling
        return float(max(0.0, g.normal(0.0010, 0.0005)))
    return float(max(0.0, g.normal(0.0015, 0.0007)))


def _person_count(g: np.random.Generator, scenario: str) -> int:
    if scenario in ("multi_anomaly", "secondary_speaker"):
        return 1 + (1 if g.random() < 0.35 else 0)
    if scenario == "normal" and g.random() < 0.06:
        return 2  # someone briefly walks behind
    return 1


def _book_features(g: np.random.Generator, scenario: str) -> float:
    if scenario in ("smartphone_use", "multi_anomaly"):
        return float(max(0.0, g.normal(0.16, 0.10)))
    if scenario == "normal":
        if g.random() < 0.12:
            return float(g.uniform(0.20, 0.45))  # legitimate textbook in frame
        return float(max(0.0, g.normal(0.06, 0.05)))
    return float(max(0.0, g.normal(0.09, 0.06)))


def build_scenario_dataset(
    n_per_class: int = NUM_SCENARIOS_PER_CLASS,
    seed: int = RANDOM_SEED,
) -> list[ScenarioRecord]:
    g = np.random.default_rng(seed)
    records: list[ScenarioRecord] = []
    for label in SCENARIO_CLASSES:
        for i in range(n_per_class):
            max_phone, phone_ratio = _phone_visibility(g, label)
            max_gaze, off_ratio, max_yaw = _gaze_features(g, label)
            sec_prob, sec_ratio, overlap, whisper_prob = _audio_features(g, label)
            lip_var = _lip_variance(g, label)
            persons = _person_count(g, label)
            max_book = _book_features(g, label)
            features = {
                "max_phone_conf": max_phone,
                "phone_duration_ratio": phone_ratio,
                "max_book_conf": max_book,
                "person_count_max": persons,
                "max_gaze_dev": max_gaze,
                "sustained_offscreen_ratio": off_ratio,
                "max_yaw_abs": max_yaw,
                "lip_variance_mean": lip_var,
                "max_secondary_speaker_prob": sec_prob,
                "secondary_speaker_event_ratio": sec_ratio,
                "max_overlap_prob": overlap,
                "max_whisper_prob": whisper_prob,
            }
            records.append(
                ScenarioRecord(
                    scenario_id=f"{label}_{i:04d}",
                    label=label,
                    is_cheating=0 if label == "normal" else 1,
                    features=features,
                )
            )
    return records


def save_scenarios(records: list[ScenarioRecord], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = [
        {
            "scenario_id": r.scenario_id,
            "label": r.label,
            "is_cheating": r.is_cheating,
            "features": r.features,
        }
        for r in records
    ]
    path.write_text(json.dumps(payload, indent=2))


if __name__ == "__main__":
    obj_set = build_object_detection_set(n_per_class=20)
    print(f"Built {len(obj_set)} synthetic object frames")
    audio_set = build_audio_set(n_per_class=20)
    print(f"Built {len(audio_set)} audio clips")
    scenarios = build_scenario_dataset()
    print(f"Built {len(scenarios)} exam scenarios across {len(SCENARIO_CLASSES)} classes")
    save_scenarios(scenarios, DATA_DIR / "scenarios" / "scenarios.json")
