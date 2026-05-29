"""Fusion model ablation, comparison and statistical testing.

Compares five fusion strategies (rule-based weighted sum, logistic regression,
random forest, MLP, naive Bayes) and runs leave-one-modality-out ablations.

Stratified 5-fold cross-validation is used; for each fold we report:
  - precision / recall / F1
  - ROC AUC
  - PR AUC
  - 95% bootstrap CI on F1
McNemar's chi-square test is run pair-wise between the best fusion model and
each baseline to test for statistically significant differences.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from copy import deepcopy

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import DATA_DIR, FIGURES_DIR, PLOT_DPI, PLOT_STYLE, RANDOM_SEED, RESULTS_DIR
from fusion_model import (
    FEATURE_ORDER,
    WeightedSumFusion,
    feature_vector,
    make_bayes_fusion,
    make_logistic_fusion,
    make_mlp_fusion,
    make_rf_fusion,
)
from dataclasses import dataclass


@dataclass
class ScenarioRecord:
    scenario_id: str
    label: str
    is_cheating: int
    features: dict


def load_real_fusion_dataset(path: Path | None = None) -> list[ScenarioRecord]:
    path = path or (DATA_DIR / "scenarios" / "real_fusion_scenarios.json")
    payload = json.loads(path.read_text())
    return [ScenarioRecord(**r) for r in payload]


def _bootstrap_ci(y_true: np.ndarray, y_pred: np.ndarray, metric, n_boot: int = 1000, alpha: float = 0.05):
    rng = np.random.default_rng(123)
    scores = []
    n = len(y_true)
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        scores.append(metric(y_true[idx], y_pred[idx]))
    lo, hi = np.quantile(scores, [alpha / 2, 1 - alpha / 2])
    return float(lo), float(hi)


def _eval_model(model_factory, X, y, feats, n_splits: int = 5):
    """Run stratified CV and return predictions + metrics."""
    kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_SEED)
    y_all_true = []
    y_all_pred = []
    y_all_prob = []
    fold_metrics = []
    for fold_idx, (tr, te) in enumerate(kf.split(X, y)):
        model = model_factory()
        model.fit([feats[i] for i in tr], y[tr])
        prob = model.predict_proba([feats[i] for i in te])[:, 1]
        pred = (prob >= 0.5).astype(int) if hasattr(model, "model") else model.predict([feats[i] for i in te])
        y_all_true.extend(y[te])
        y_all_pred.extend(pred)
        y_all_prob.extend(prob)
        fold_metrics.append(
            {
                "fold": fold_idx,
                "f1": float(f1_score(y[te], pred, zero_division=0)),
                "precision": float(precision_score(y[te], pred, zero_division=0)),
                "recall": float(recall_score(y[te], pred, zero_division=0)),
                "roc_auc": float(roc_auc_score(y[te], prob)) if len(np.unique(y[te])) > 1 else float("nan"),
                "pr_auc": float(average_precision_score(y[te], prob)),
            }
        )
    yt, yp, ypr = np.array(y_all_true), np.array(y_all_pred), np.array(y_all_prob)
    summary = {
        "f1_mean": float(np.mean([m["f1"] for m in fold_metrics])),
        "f1_std": float(np.std([m["f1"] for m in fold_metrics])),
        "precision_mean": float(np.mean([m["precision"] for m in fold_metrics])),
        "recall_mean": float(np.mean([m["recall"] for m in fold_metrics])),
        "roc_auc_mean": float(np.nanmean([m["roc_auc"] for m in fold_metrics])),
        "pr_auc_mean": float(np.mean([m["pr_auc"] for m in fold_metrics])),
        "f1_ci_95": _bootstrap_ci(yt, yp, lambda a, b: f1_score(a, b, zero_division=0)),
        "fold_metrics": fold_metrics,
    }
    return yt, yp, ypr, summary


def mcnemar(y_true, pred_a, pred_b) -> tuple[float, float]:
    b = int(((pred_a == y_true) & (pred_b != y_true)).sum())
    c = int(((pred_a != y_true) & (pred_b == y_true)).sum())
    if b + c == 0:
        return 0.0, 1.0
    stat = (abs(b - c) - 1) ** 2 / (b + c)
    p = 1 - stats.chi2.cdf(stat, df=1)
    return float(stat), float(p)


def main():
    records = load_real_fusion_dataset()
    print(f"Loaded {len(records)} real-feature scenarios from disk")
    feats = [r.features for r in records]
    y = np.array([r.is_cheating for r in records])
    X = np.array([feature_vector(f) for f in feats])

    models = {
        "weighted_sum": lambda: WeightedSumFusion(),
        "logistic": make_logistic_fusion,
        "random_forest": make_rf_fusion,
        "mlp": make_mlp_fusion,
        "naive_bayes": make_bayes_fusion,
    }

    full_results = {}
    preds_by_model = {}
    probs_by_model = {}
    # The fold-ordered ground-truth labels are identical across models because
    # every model uses StratifiedKFold with the same random_state; we capture
    # them once and use them for McNemar and for the pooled ROC plot so the
    # arrays are never misaligned with the original record order.
    yt_fold_order = None
    for name, factory in models.items():
        print(f"  -> fitting {name}")
        yt, yp, ypr, summary = _eval_model(factory, X, y, feats)
        full_results[name] = summary
        preds_by_model[name] = yp
        probs_by_model[name] = ypr
        if yt_fold_order is None:
            yt_fold_order = yt
        else:
            assert np.array_equal(yt_fold_order, yt), "fold ordering diverged across models"

    # Pairwise McNemar of the best vs. others — uses fold-ordered labels.
    best = max(full_results.items(), key=lambda kv: kv[1]["f1_mean"])[0]
    mcnemar_table = {}
    yt_ref = yt_fold_order
    for name in models:
        if name == best:
            continue
        stat, p = mcnemar(yt_ref, preds_by_model[best], preds_by_model[name])
        mcnemar_table[name] = {"chi2": stat, "p_value": p, "vs": best}

    # Leave-one-modality-out ablation using the best model. The three modality
    # groups are mutually exclusive: visual = object-detection signals,
    # behaviour = FaceMesh geometry signals, acoustic = Whisper signals.
    ablation = {}
    visual_keys = ["max_phone_conf", "phone_duration_ratio", "max_book_conf",
                   "person_count_max"]
    behaviour_keys = ["max_gaze_dev", "sustained_offscreen_ratio",
                      "max_yaw_abs", "lip_variance_mean"]
    acoustic_keys = ["max_secondary_speaker_prob", "secondary_speaker_event_ratio",
                     "max_overlap_prob", "max_whisper_prob"]
    for omitted, keys in [
        ("no_visual", visual_keys),
        ("no_acoustic", acoustic_keys),
        ("no_behaviour", behaviour_keys),
        ("only_visual", [k for k in FEATURE_ORDER if k not in visual_keys]),
        ("only_acoustic", [k for k in FEATURE_ORDER if k not in acoustic_keys]),
    ]:
        feats_ab = [
            {k: 0.0 if k in keys else v for k, v in f.items()} for f in feats
        ]
        factory = models[best]
        _, yp, ypr, summary = _eval_model(factory, X, y, feats_ab)
        ablation[omitted] = summary

    out = {
        "fusion": full_results,
        "best_model": best,
        "mcnemar_vs_best": mcnemar_table,
        "ablation_with_best_model": ablation,
        "n_records": len(records),
        "feature_order": FEATURE_ORDER,
        "class_balance": {"positive": int(y.sum()), "negative": int(len(y) - y.sum())},
    }
    (RESULTS_DIR / "fusion_metrics.json").write_text(json.dumps(out, indent=2))
    print(json.dumps({k: out[k] for k in ["best_model", "mcnemar_vs_best"]}, indent=2))

    # Summary plots
    import matplotlib.pyplot as plt
    import seaborn as sns

    sns.set_style(PLOT_STYLE)
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    names = list(full_results.keys())
    f1_means = [full_results[n]["f1_mean"] for n in names]
    f1_stds = [full_results[n]["f1_std"] for n in names]
    pal = sns.color_palette("muted", len(names))
    hatches = ["//", "\\\\", "xx", "..", "++"]
    bars = ax.bar(
        names, f1_means, yerr=f1_stds, capsize=4,
        color=pal, alpha=0.85, edgecolor="black", linewidth=0.8,
    )
    for bar, h in zip(bars, hatches):
        bar.set_hatch(h)
    ax.set_ylabel("F1 (stratified 5-fold CV)")
    ax.set_title("Fusion strategy comparison")
    ax.set_ylim(0.0, 1.05)
    for i, (m, s) in enumerate(zip(f1_means, f1_stds)):
        ax.text(i, m + s + 0.01, f"{m:.3f}", ha="center", fontsize=9)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "fig_fusion_comparison.png", dpi=600)
    fig.savefig(FIGURES_DIR / "fig_fusion_comparison.pdf")
    plt.close(fig)

    # Ablation plot
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    keys = list(ablation.keys())
    vals = [ablation[k]["f1_mean"] for k in keys]
    stds = [ablation[k]["f1_std"] for k in keys]
    full_f1 = full_results[best]["f1_mean"]
    ax.axhline(full_f1, color="black", linestyle="--", label=f"Full {best} ({full_f1:.3f})")
    pal = sns.color_palette("Set2", len(keys))
    abl_hatches = ["//", "\\\\", "xx", "..", "++"]
    bars = ax.bar(
        keys, vals, yerr=stds, capsize=4,
        color=pal, alpha=0.85, edgecolor="black", linewidth=0.8,
    )
    for bar, h in zip(bars, abl_hatches):
        bar.set_hatch(h)
    ax.set_ylabel("F1")
    ax.set_title("Leave-one-modality-out ablation (best model)")
    ax.set_ylim(0.0, 1.05)
    ax.legend()
    for i, (m, s) in enumerate(zip(vals, stds)):
        ax.text(i, m + s + 0.01, f"{m:.3f}", ha="center", fontsize=9)
    plt.setp(ax.get_xticklabels(), rotation=15, ha="right")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "fig_fusion_ablation.png", dpi=600)
    fig.savefig(FIGURES_DIR / "fig_fusion_ablation.pdf")
    plt.close(fig)

    # ROC curves — use distinct line styles + markers so the curves remain
    # distinguishable when printed in black and white.
    fig, ax = plt.subplots(figsize=(7, 5))
    line_styles = ["-", "--", "-.", ":", (0, (3, 1, 1, 1))]
    markers = ["o", "s", "^", "D", "v"]
    for idx, (name, prob) in enumerate(probs_by_model.items()):
        fpr, tpr, _ = roc_curve(yt_ref, prob)
        auc = roc_auc_score(yt_ref, prob)
        ax.plot(
            fpr, tpr,
            label=f"{name} (AUC={auc:.3f})",
            linestyle=line_styles[idx % len(line_styles)],
            marker=markers[idx % len(markers)],
            markevery=int(max(1, len(fpr) / 12)),
            linewidth=1.4,
        )
    ax.plot([0, 1], [0, 1], "k:", alpha=0.5)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("Fusion model ROC curves on stratified 5-fold CV")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "fig_fusion_roc.png", dpi=600)
    fig.savefig(FIGURES_DIR / "fig_fusion_roc.pdf")
    plt.close(fig)

    print("Fusion experiment complete.")


if __name__ == "__main__":
    main()
