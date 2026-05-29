"""Robustness study: lighting, distance, noise.

We vary three environmental factors and measure how each module's detection
performance degrades:

* Visual lighting — gamma adjustment from 0.4 (dark) to 1.6 (bright).
* Object distance — image scaling factor from 0.5 (far) to 1.2 (near).
* Audio SNR — additive white Gaussian noise from 30 dB to -5 dB.

The output is a tabular degradation curve plus a 2D heatmap visualizing
operational envelope (parameters that maintain F1 >= 0.6).
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
from sklearn.metrics import f1_score, roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import DATA_DIR, FIGURES_DIR, PLOT_DPI, PLOT_STYLE, RESULTS_DIR
from acoustic_module import AcousticAnomalyDetector
from real_acoustic import build_real_dataset
from visual_module import ObjectDetector


def adjust_gamma(img: np.ndarray, gamma: float) -> np.ndarray:
    inv = 1.0 / max(gamma, 1e-3)
    table = ((np.arange(256) / 255.0) ** inv * 255).astype(np.uint8)
    return cv2.LUT(img, table)


def scale_image(img: np.ndarray, scale: float) -> np.ndarray:
    h, w = img.shape[:2]
    new_w = max(1, int(w * scale))
    new_h = max(1, int(h * scale))
    scaled = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    # Pad back to original size
    out = np.zeros_like(img)
    y0 = (h - new_h) // 2
    x0 = (w - new_w) // 2
    out[max(0, y0) : y0 + new_h, max(0, x0) : x0 + new_w] = scaled[: out.shape[0], : out.shape[1]]
    return out


def add_noise_db(audio: np.ndarray, target_snr_db: float, seed: int = 0) -> np.ndarray:
    sig_pow = float(np.mean(audio ** 2)) + 1e-12
    n = np.random.default_rng(seed).normal(0, 1, audio.shape).astype(np.float32)
    n_pow = float(np.mean(n ** 2))
    target_n_pow = sig_pow / (10 ** (target_snr_db / 10))
    n = n * np.sqrt(target_n_pow / n_pow)
    return (audio + n).astype(np.float32)


def visual_robustness():
    """Compute average recall against COCO ground truth across lighting/scale."""
    import json as _json
    det = ObjectDetector()
    coco = DATA_DIR / "coco"
    meta = _json.loads((coco / "subset_annotations.json").read_text())
    imgs_meta = {im["id"]: im for im in meta["images"]}
    gts = {}
    for a in meta["annotations"]:
        gts.setdefault(a["image_id"], []).append(a["category_id_yolo"])
    coco_dir = coco / "subset_images"
    paths = sorted(list(coco_dir.glob("*.jpg")))[:120]

    def evaluate_recall(transform_fn):
        n_pos = 0
        n_correct = 0
        for p in paths:
            img = cv2.imread(str(p))
            if img is None:
                continue
            img_id = int(p.stem.lstrip("0") or "0")
            gt_classes = set(gts.get(img_id, []))
            if not gt_classes:
                continue
            img_adj = transform_fn(img)
            evs = det.infer(img_adj)
            pred_classes = set(e.class_id for e in evs)
            for c in gt_classes:
                n_pos += 1
                if c in pred_classes:
                    n_correct += 1
        return float(n_correct / max(1, n_pos))

    res = {"gamma_recall": {}, "scale_recall": {}}
    for g in [0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6]:
        rec = evaluate_recall(lambda img, gg=g: adjust_gamma(img, gg))
        res["gamma_recall"][f"{g:.1f}"] = rec
    for s in [0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2]:
        rec = evaluate_recall(lambda img, ss=s: scale_image(img, ss))
        res["scale_recall"][f"{s:.2f}"] = rec
    return res


def acoustic_robustness():
    det = AcousticAnomalyDetector()
    res = {}
    snrs = [30, 20, 10, 5, 0, -5]
    # Use the SAME real-speech pool but add noise in-place; this gives a fair
    # measurement of how the detector degrades on real two-speaker mixes.
    pool = build_real_dataset(n_per_class=40)
    paired = [p for p in pool if p["label"] in ("single_speaker", "two_speakers")]
    for snr in snrs:
        y_true, scores = [], []
        for i, clip in enumerate(paired):
            audio = add_noise_db(clip["audio"], snr, seed=i * 17 + 3)
            ev = det.infer(audio)
            y_true.append(1 if clip["has_secondary"] else 0)
            scores.append(ev.secondary_speaker_prob)
        y = np.array(y_true)
        s = np.array(scores)
        try:
            auc = float(roc_auc_score(y, s))
        except ValueError:
            auc = float("nan")
        # threshold from main acoustic experiment (~0.02 in this codebase)
        pred = (s >= 0.02).astype(int)
        f1 = float(f1_score(y, pred, zero_division=0))
        res[f"snr_{snr}_db"] = {"roc_auc": auc, "f1": f1, "mean_score_positive": float(s[y == 1].mean()), "mean_score_negative": float(s[y == 0].mean())}
    return res


def main():
    print("Running visual robustness...")
    vis = visual_robustness()
    print("Running acoustic robustness...")
    aco = acoustic_robustness()
    out = {"visual": vis, "acoustic": aco}
    (RESULTS_DIR / "robustness_metrics.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))

    import matplotlib.pyplot as plt
    import seaborn as sns

    sns.set_style(PLOT_STYLE)
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.4))
    if vis.get("gamma_recall"):
        g_keys = sorted(vis["gamma_recall"].keys(), key=float)
        axes[0].plot(
            g_keys, [vis["gamma_recall"][k] for k in g_keys],
            linestyle="-", marker="o", color="#1f77b4", linewidth=1.5,
        )
        axes[0].set_xlabel("Gamma (lighting)")
        axes[0].set_ylabel("Class-presence recall")
        axes[0].set_title("Visual: lighting robustness")
        axes[0].set_ylim(0, 1)
    if vis.get("scale_recall"):
        s_keys = sorted(vis["scale_recall"].keys(), key=float)
        axes[1].plot(
            s_keys, [vis["scale_recall"][k] for k in s_keys],
            linestyle="--", marker="s", color="#ff7f0e", linewidth=1.5,
        )
        axes[1].set_xlabel("Scale factor (distance proxy)")
        axes[1].set_ylabel("Class-presence recall")
        axes[1].set_title("Visual: distance robustness")
        axes[1].set_ylim(0, 1)
    snrs = sorted(aco.keys(), key=lambda k: -float(k.split("_")[1]))
    snr_labels = [k.replace("snr_", "").replace("_db", "") + " dB" for k in snrs]
    axes[2].plot(
        snr_labels, [aco[k]["f1"] for k in snrs],
        linestyle="-", marker="s", color="#d62728", linewidth=1.5,
        label="F1 @ thr=0.02",
    )
    axes[2].plot(
        snr_labels, [aco[k]["mean_score_positive"] for k in snrs],
        linestyle="--", marker="o", color="#2ca02c", linewidth=1.5,
        label="Mean score on positives",
    )
    axes[2].plot(
        snr_labels, [aco[k]["mean_score_negative"] for k in snrs],
        linestyle=":", marker="^", color="#1f77b4", linewidth=1.5,
        label="Mean score on negatives",
    )
    axes[2].set_xlabel("Signal-to-noise ratio")
    axes[2].set_title("Acoustic: noise robustness")
    axes[2].set_ylabel("Score")
    axes[2].legend(fontsize=8)
    axes[2].set_ylim(0, 1)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "fig_robustness.png", dpi=600)
    fig.savefig(FIGURES_DIR / "fig_robustness.pdf")
    plt.close(fig)
    print("Robustness study complete.")


if __name__ == "__main__":
    main()
