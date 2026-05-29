"""Generate supplementary figures: architecture diagram, fusion confusion matrix,
suspicion-index timeline, summary table.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import DATA_DIR, FIGURES_DIR, PLOT_DPI, PLOT_STYLE, RESULTS_DIR
from fusion_model import feature_vector, make_rf_fusion


def _load_real_fusion_records():
    """Load the real-modality fusion scenarios produced by
    real_fusion_features.py (the same file exp_fusion.py evaluates)."""
    path = DATA_DIR / "scenarios" / "real_fusion_scenarios.json"
    return json.loads(path.read_text())


def make_architecture_figure():
    import matplotlib.patches as mpatches
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(11, 5.6))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 60)
    ax.axis("off")
    boxes = [
        (3, 36, 18, 12, "Webcam stream\n(volatile, never saved)", "#cce5ff"),
        (3, 18, 18, 12, "Microphone stream\n(volatile, never saved)", "#cce5ff"),
        (28, 47, 20, 10, "YOLOv8-s\nobject detection", "#fff2cc"),
        (28, 32, 20, 10, "MediaPipe FaceMesh\nbehavioural cues", "#fff2cc"),
        (28, 18, 20, 10, "Whisper-Base encoder\n(content-free)", "#fff2cc"),
        (54, 30, 18, 14, "Suspicion Index\nFusion Model\n(Random Forest)", "#d5e8d4"),
        (76, 36, 20, 10, "Real-time alert\n(warn / flag)", "#f8cecc"),
        (76, 18, 20, 10, "Event log\n(metadata only,\nno biometric data)", "#f8cecc"),
    ]
    for x, y, w, h, txt, color in boxes:
        rect = mpatches.FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.4",
            linewidth=1.0,
            edgecolor="#444",
            facecolor=color,
        )
        ax.add_patch(rect)
        ax.text(x + w / 2, y + h / 2, txt, ha="center", va="center", fontsize=9)
    arrows = [
        (21, 42, 28, 52),
        (21, 42, 28, 37),
        (21, 24, 28, 23),
        (48, 52, 54, 41),
        (48, 37, 54, 38),
        (48, 23, 54, 33),
        (72, 37, 76, 41),
        (72, 37, 76, 23),
    ]
    for x1, y1, x2, y2 in arrows:
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1), arrowprops=dict(arrowstyle="->", color="#444"))
    # Legend mapping the box colours to pipeline stages (greyscale-safe: the
    # labels make the categories explicit even if colour is lost in print).
    legend_handles = [
        mpatches.Patch(facecolor="#cce5ff", edgecolor="#444", label="Volatile sensor stream"),
        mpatches.Patch(facecolor="#fff2cc", edgecolor="#444", label="Modality encoder"),
        mpatches.Patch(facecolor="#d5e8d4", edgecolor="#444", label="Fusion model"),
        mpatches.Patch(facecolor="#f8cecc", edgecolor="#444", label="Output (alert / metadata log)"),
    ]
    ax.legend(handles=legend_handles, loc="lower center", ncol=4, fontsize=8, frameon=True, bbox_to_anchor=(0.5, -0.04))
    ax.set_title("End-to-end privacy-preserving proctoring pipeline")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "fig_architecture.png", dpi=600, bbox_inches="tight")
    fig.savefig(FIGURES_DIR / "fig_architecture.pdf", bbox_inches="tight")
    print("Saved fig_architecture.png")


def make_fusion_confusion():
    from sklearn.metrics import confusion_matrix
    from sklearn.model_selection import StratifiedKFold
    import matplotlib.pyplot as plt
    import seaborn as sns

    sns.set_style(PLOT_STYLE)
    # Use the SAME real-modality feature set the fusion experiment uses, so
    # the confusion matrix is consistent with Table 5 / Figure 8 / Figure 9.
    rec = _load_real_fusion_records()
    feats = [r["features"] for r in rec]
    y = np.array([r["is_cheating"] for r in rec])
    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    preds = np.zeros_like(y)
    for tr, te in kf.split(np.zeros(len(y)), y):
        m = make_rf_fusion()
        m.fit([feats[i] for i in tr], y[tr])
        preds[te] = m.predict([feats[i] for i in te])
    cm = confusion_matrix(y, preds)
    fig, ax = plt.subplots(figsize=(5, 4.2))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        xticklabels=["Predicted normal", "Predicted cheating"],
        yticklabels=["Actual normal", "Actual cheating"],
        cmap="Blues",
        ax=ax,
    )
    ax.set_title("Random-Forest fusion: 5-fold CV confusion matrix")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "fig_fusion_confusion.png", dpi=600)
    fig.savefig(FIGURES_DIR / "fig_fusion_confusion.pdf")
    plt.close(fig)
    print("Saved fig_fusion_confusion.png")


def make_si_timeline():
    """Simulate a 60-second session and plot the Suspicion Index trajectory."""
    import matplotlib.pyplot as plt
    import seaborn as sns

    sns.set_style(PLOT_STYLE)
    rng = np.random.default_rng(11)
    fps = 5
    dur = 60
    t = np.linspace(0, dur, dur * fps)
    visual = np.abs(rng.normal(0.07, 0.04, size=t.size))
    acoustic = np.abs(rng.normal(0.05, 0.04, size=t.size))
    behaviour = np.abs(rng.normal(0.07, 0.04, size=t.size))
    # Inject anomalies
    visual[100:140] += np.linspace(0.4, 0.8, 40)  # smartphone visible
    behaviour[200:260] += np.linspace(0.3, 0.7, 60)  # head turn
    acoustic[220:240] += np.linspace(0.3, 0.7, 20)  # secondary voice
    si = np.clip(0.45 * visual + 0.30 * acoustic + 0.25 * behaviour, 0, 1)
    fig, ax = plt.subplots(figsize=(11, 4.5))
    # Greyscale-safe: distinguish the three modality traces by marker shape and
    # line weight, not colour, so they remain separable in a B&W printout.
    ax.plot(t, visual, color="0.15", linestyle="-", linewidth=1.2,
            marker="o", markevery=9, markersize=4, label="visual risk")
    ax.plot(t, acoustic, color="0.45", linestyle="--", linewidth=1.2,
            marker="s", markevery=9, markersize=4, label="acoustic risk")
    ax.plot(t, behaviour, color="0.6", linestyle=":", linewidth=1.6,
            marker="^", markevery=9, markersize=4, label="behaviour risk")
    ax.plot(t, si, color="black", linewidth=2.6, label="Suspicion Index")
    ax.axhline(0.35, color="black", linestyle=(0, (4, 2)), linewidth=1.0, alpha=0.6, label="warn threshold (0.35)")
    ax.axhline(0.65, color="black", linestyle=(0, (1, 1)), linewidth=1.2, alpha=0.8, label="flag threshold (0.65)")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Risk score")
    ax.set_title("Simulated session: Suspicion Index trajectory")
    ax.legend(loc="upper left", fontsize=9)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "fig_si_timeline.png", dpi=600)
    fig.savefig(FIGURES_DIR / "fig_si_timeline.pdf")
    plt.close(fig)
    print("Saved fig_si_timeline.png")


def main():
    make_architecture_figure()
    make_fusion_confusion()
    make_si_timeline()


if __name__ == "__main__":
    main()
