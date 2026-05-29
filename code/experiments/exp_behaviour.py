"""Behavioural-cue evaluation.

Real MediaPipe FaceMesh inference requires a face camera. To stay within the
privacy-by-design protocol of this study we synthesize landmark trajectories
that replay the geometric properties of the four cases of interest:

    * neutral gaze on screen
    * sustained off-screen gaze
    * head rotation toward a hidden secondary screen
    * sustained lip movement without keyboard activity

The synthetic generator preserves the 468-landmark structure that the
BehaviourAnalyzer expects.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import FACEMESH_LANDMARK_COUNT, FIGURES_DIR, PLOT_DPI, PLOT_STYLE, RESULTS_DIR
from visual_module import BehaviourAnalyzer
from real_behaviour import extract_real_landmarks, simulate_real_trial


def _make_neutral_landmarks(seed: int) -> np.ndarray:
    """Generate landmarks for a centered, forward-facing head."""
    g = np.random.default_rng(seed)
    pts = np.zeros((FACEMESH_LANDMARK_COUNT, 3), dtype=np.float32)
    pts[:, 0] = g.uniform(280, 360, FACEMESH_LANDMARK_COUNT)
    pts[:, 1] = g.uniform(220, 260, FACEMESH_LANDMARK_COUNT)
    # Anchor specific landmarks used by BehaviourAnalyzer
    pts[1] = [320, 240, 0]   # nose tip
    pts[152] = [320, 360, 0] # chin
    pts[234] = [220, 240, 0] # left ear
    pts[454] = [420, 240, 0] # right ear
    pts[33] = [280, 230, 0]
    pts[133] = [300, 230, 0]
    pts[159] = [290, 220, 0]
    pts[145] = [290, 240, 0]
    pts[263] = [340, 230, 0]
    pts[362] = [360, 230, 0]
    pts[386] = [350, 220, 0]
    pts[374] = [350, 240, 0]
    pts[13] = [320, 290, 0]   # upper lip
    pts[14] = [320, 300, 0]   # lower lip
    pts[78] = [305, 295, 0]
    pts[308] = [335, 295, 0]
    return pts


def _shift_for_gaze_off(pts: np.ndarray, off: float) -> np.ndarray:
    """Move the eye landmarks/nose to simulate gaze divergence."""
    out = pts.copy()
    out[1, 0] += off * 60
    out[33, 0] += off * 50
    out[133, 0] += off * 50
    out[263, 0] += off * 50
    out[362, 0] += off * 50
    return out


def _rotate_yaw(pts: np.ndarray, yaw_deg: float) -> np.ndarray:
    out = pts.copy()
    out[1, 0] += yaw_deg * 1.6
    out[234, 0] += yaw_deg * 0.8
    out[454, 0] += yaw_deg * 0.8
    return out


def _move_lips(pts: np.ndarray, amplitude: float, t: float) -> np.ndarray:
    out = pts.copy()
    out[13, 1] -= amplitude * (1 + np.sin(2 * np.pi * 3.0 * t)) * 4
    out[14, 1] += amplitude * (1 + np.sin(2 * np.pi * 3.0 * t)) * 4
    return out


def _add_jitter(pts: np.ndarray, rng: np.random.Generator, sigma: float = 1.2) -> np.ndarray:
    out = pts.copy()
    out[:, :2] += rng.normal(0, sigma, (pts.shape[0], 2))
    return out


def simulate_scenario(label: str, seed: int, dur: float = 4.0, fps: int = 25) -> tuple[float, bool, float, float]:
    """Returns (max_gaze_dev, sustained_off_screen_flag, max_yaw, max_lip_var)."""
    analyzer = BehaviourAnalyzer()
    base = _make_neutral_landmarks(seed)
    g = np.random.default_rng(seed)
    n = int(dur * fps)
    gaze_devs = []
    sustained_flags = []
    yaws = []
    lip_vars = []
    # Per-trial noise level — simulates the natural variability of FaceMesh.
    jitter_sigma = g.uniform(1.0, 3.5)
    natural_drift = g.uniform(-0.08, 0.08)

    for i in range(n):
        t = i / fps
        pts = base.copy()
        if label == "off_screen":
            magnitude = g.uniform(0.45, 0.85)
            off = magnitude if t > 0.45 + g.uniform(-0.2, 0.2) else 0.0
            pts = _shift_for_gaze_off(pts, off + g.normal(0, 0.05))
            pts = _rotate_yaw(pts, g.uniform(18, 32) if t > 0.55 else g.normal(0, 1.5))
        elif label == "head_rotation":
            if 0.8 < t < 3.5:
                pts = _rotate_yaw(pts, g.uniform(20, 30) * np.sin(2 * np.pi * 0.5 * t))
            pts = _shift_for_gaze_off(pts, g.normal(0, 0.05))
        elif label == "lip_movement":
            amp = g.uniform(0.35, 0.55)
            pts = _move_lips(pts, amp, t)
            pts = _shift_for_gaze_off(pts, natural_drift + g.normal(0, 0.04))
        elif label == "subtle_movement":
            # Brief, non-sustained glance away — should NOT trigger
            if 1.0 < t < 1.3:
                pts = _shift_for_gaze_off(pts, 0.3)
                pts = _rotate_yaw(pts, 14)
            else:
                pts = _shift_for_gaze_off(pts, natural_drift)
        elif label == "neutral":
            pts = _shift_for_gaze_off(pts, natural_drift + g.normal(0, 0.04))
            pts = _rotate_yaw(pts, g.normal(0, 2.0))
        pts = _add_jitter(pts, g, sigma=jitter_sigma)
        ev = analyzer.update(pts, timestamp=t)
        gaze_devs.append(ev.gaze_deviation_deg)
        sustained_flags.append(ev.sustained_off_screen)
        yaws.append(abs(ev.head_yaw))
        lip_vars.append(ev.lip_variance)
    return float(np.max(gaze_devs)), bool(np.any(sustained_flags)), float(np.max(yaws)), float(np.max(lip_vars))


def main():
    classes = ("neutral", "subtle_movement", "off_screen", "head_rotation", "lip_movement")
    n_per = 80
    rows = []
    rng = np.random.default_rng(7)

    # Real MediaPipe landmarks extracted from CC-BY Pexels portraits;
    # each trial picks one of the real anchor meshes and applies a
    # programmatic head/lip transform corresponding to the class.
    anchors = extract_real_landmarks()
    print(f"using {len(anchors)} real face anchor meshes for the behaviour benchmark")
    for cls in classes:
        for i in range(n_per):
            seed = int(rng.integers(0, 2 ** 31))
            anchor = anchors[i % len(anchors)]
            analyzer = BehaviourAnalyzer()
            mg, sust, my, mlv = simulate_real_trial(anchor, cls, seed, analyzer)
            rows.append({"label": cls, "max_gaze": mg, "sustained": sust, "max_yaw": my, "max_lip_var": mlv})
    import pandas as pd
    df = pd.DataFrame(rows)
    # 'sustained' off-screen positives are only the prolonged classes.
    is_offscreen = (df.label.isin(["off_screen", "head_rotation"])).values
    is_lip = (df.label == "lip_movement").values

    # Off-screen detection
    score = df["max_yaw"].values + 0.5 * df["max_gaze"].values
    y = is_offscreen.astype(int)
    fpr, tpr, thr = roc_curve(y, score)
    f1s = []
    for t in thr:
        p = (score >= t).astype(int)
        f1s.append(f1_score(y, p, zero_division=0))
    best_idx = int(np.argmax(f1s))
    offscreen_metrics = {
        "auc": float(roc_auc_score(y, score)),
        "ap": float(average_precision_score(y, score)),
        "f1_at_best": float(f1s[best_idx]),
        "best_threshold": float(thr[best_idx]),
    }

    # Lip-movement detection
    score2 = df["max_lip_var"].values
    y2 = is_lip.astype(int)
    fpr, tpr, thr = roc_curve(y2, score2)
    f1s = []
    for t in thr:
        p = (score2 >= t).astype(int)
        f1s.append(f1_score(y2, p, zero_division=0))
    best_idx = int(np.argmax(f1s))
    lip_metrics = {
        "auc": float(roc_auc_score(y2, score2)),
        "ap": float(average_precision_score(y2, score2)),
        "f1_at_best": float(f1s[best_idx]),
        "best_threshold": float(thr[best_idx]),
    }

    out = {
        "n_total": len(df),
        "per_class_means": df.groupby("label")[["max_gaze", "max_yaw", "max_lip_var"]].mean().to_dict(),
        "offscreen_detection": offscreen_metrics,
        "lip_movement_detection": lip_metrics,
    }
    (RESULTS_DIR / "behaviour_metrics.json").write_text(json.dumps(out, indent=2, default=str))
    print(json.dumps(out, indent=2, default=str))

    import matplotlib.pyplot as plt
    import seaborn as sns

    sns.set_style(PLOT_STYLE)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    sns.boxplot(data=df, x="label", y="max_yaw", ax=axes[0])
    axes[0].set_title("Max |yaw| per scenario")
    axes[0].set_xlabel("")
    sns.boxplot(data=df, x="label", y="max_lip_var", ax=axes[1])
    axes[1].set_title("Max lip variance per scenario")
    axes[1].set_xlabel("")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "fig_behaviour_distributions.png", dpi=600)
    fig.savefig(FIGURES_DIR / "fig_behaviour_distributions.pdf")
    plt.close(fig)
    print("Behaviour evaluation complete.")


if __name__ == "__main__":
    main()
