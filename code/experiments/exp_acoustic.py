"""Acoustic subsystem evaluation using synthetic 5-second clips.

We synthesize 5 audio classes with 80 clips each, run the Whisper-Base encoder
in content-free mode, and report the resulting secondary-speaker probabilities.

Metrics: precision, recall, F1, ROC AUC, PR AUC for the binary task
"has_secondary_speaker_or_whisper". Multi-class clip metrics are also reported.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import FIGURES_DIR, PLOT_DPI, PLOT_STYLE, RESULTS_DIR
from acoustic_module import AcousticAnomalyDetector
from real_acoustic import build_real_dataset


def main():
    detector = AcousticAnomalyDetector()
    audio_set = build_real_dataset(n_per_class=80)
    print(f"Evaluating Whisper-Base encoder on {len(audio_set)} REAL LibriSpeech clips")
    y_true = []
    scores = {"secondary_speaker": [], "overlap": [], "whisper": []}
    latencies = []
    per_class = {}
    for clip in audio_set:
        t0 = time.perf_counter()
        ev = detector.infer(clip["audio"])
        latencies.append(time.perf_counter() - t0)
        y_true.append(1 if clip["has_secondary"] else 0)
        scores["secondary_speaker"].append(ev.secondary_speaker_prob)
        scores["overlap"].append(ev.overlap_prob)
        scores["whisper"].append(ev.whisper_prob)
        per_class.setdefault(clip["label"], {"sec": [], "overlap": [], "whisper": []})
        per_class[clip["label"]]["sec"].append(ev.secondary_speaker_prob)
        per_class[clip["label"]]["overlap"].append(ev.overlap_prob)
        per_class[clip["label"]]["whisper"].append(ev.whisper_prob)

    y_true = np.array(y_true)
    out = {
        "n_clips": len(audio_set),
        "mean_inference_ms": float(np.mean(latencies) * 1000),
        "per_class_means": {
            cls: {k: float(np.mean(v)) for k, v in d.items()}
            for cls, d in per_class.items()
        },
    }
    for score_name, s in scores.items():
        s = np.array(s)
        auc = float(roc_auc_score(y_true, s))
        ap = float(average_precision_score(y_true, s))
        # Choose threshold maximizing F1 on the score
        fpr, tpr, thr = roc_curve(y_true, s)
        f1_at = []
        for t in thr:
            pred = (s >= t).astype(int)
            tp = ((pred == 1) & (y_true == 1)).sum()
            fp = ((pred == 1) & (y_true == 0)).sum()
            fn = ((pred == 0) & (y_true == 1)).sum()
            p = tp / max(1, tp + fp)
            r = tp / max(1, tp + fn)
            f1_at.append(2 * p * r / max(1e-6, p + r))
        best_idx = int(np.argmax(f1_at))
        best_thr = float(thr[best_idx])
        pred = (s >= best_thr).astype(int)
        cm = confusion_matrix(y_true, pred).tolist()
        out[score_name] = {
            "roc_auc": auc,
            "pr_auc": ap,
            "best_threshold": best_thr,
            "f1_at_best": float(f1_at[best_idx]),
            "confusion_matrix": cm,
        }
    (RESULTS_DIR / "acoustic_metrics.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))

    # ROC/PR figures
    import matplotlib.pyplot as plt
    import seaborn as sns

    sns.set_style(PLOT_STYLE)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))
    _styles = ["-", "--", "-."]
    _markers = ["o", "s", "^"]
    for idx, (name, s) in enumerate(scores.items()):
        s = np.array(s)
        ls = _styles[idx % len(_styles)]
        mk = _markers[idx % len(_markers)]
        fpr, tpr, _ = roc_curve(y_true, s)
        ax1.plot(fpr, tpr, label=f"{name} (AUC={roc_auc_score(y_true, s):.3f})",
                 linestyle=ls, marker=mk, markevery=max(1, len(fpr) // 8), markersize=5, linewidth=1.3)
        prec, rec, _ = precision_recall_curve(y_true, s)
        ax2.plot(rec, prec, label=f"{name} (AP={average_precision_score(y_true, s):.3f})",
                 linestyle=ls, marker=mk, markevery=max(1, len(rec) // 8), markersize=5, linewidth=1.3)
    ax1.plot([0, 1], [0, 1], "k:", alpha=0.5)
    ax1.set_xlabel("False positive rate")
    ax1.set_ylabel("True positive rate")
    ax1.set_title("ROC – content-free acoustic anomaly")
    ax1.legend(loc="lower right", fontsize=9)
    ax2.set_xlabel("Recall")
    ax2.set_ylabel("Precision")
    ax2.set_title("PR – content-free acoustic anomaly")
    ax2.legend(loc="lower left", fontsize=9)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "fig_acoustic_roc_pr.png", dpi=600)
    fig.savefig(FIGURES_DIR / "fig_acoustic_roc_pr.pdf")
    plt.close(fig)

    # Per-class probability distribution
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    labels = list(per_class.keys())
    data = [per_class[l]["sec"] for l in labels]
    ax.boxplot(data, labels=labels, showmeans=True)
    ax.set_ylabel("Secondary-speaker probability")
    ax.set_title("Per-class output distribution of the acoustic module")
    ax.axhline(out["secondary_speaker"]["best_threshold"], color="red", linestyle="--", label=f"Best threshold = {out['secondary_speaker']['best_threshold']:.2f}")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "fig_acoustic_per_class.png", dpi=600)
    fig.savefig(FIGURES_DIR / "fig_acoustic_per_class.pdf")
    plt.close(fig)

    print("Acoustic evaluation complete.")


if __name__ == "__main__":
    main()
