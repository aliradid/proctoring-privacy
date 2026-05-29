"""Visual subsystem evaluation on the real COCO subset.

For every proctoring-relevant class (cell phone, book, laptop, person, tv,
keyboard, mouse) we compute:

* per-class precision, recall, F1, AP@0.5;
* image-level confusion matrix (predicted vs. expected risk class);
* aggregate macro/micro F1, mean average precision;
* ROC and PR curves using detection confidence as the score.

All results are written to ``results/visual_metrics.json`` and figures to
``figures/``.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import (
    DATA_DIR,
    FIGURES_DIR,
    PLOT_DPI,
    PLOT_STYLE,
    RESULTS_DIR,
    YOLO_CONF_THRESHOLD,
    YOLO_HIGH_RISK_CLASSES,
)
from visual_module import ObjectDetector


def iou(a, b):
    xa1, ya1, xa2, ya2 = a
    xb1, yb1, xb2, yb2 = b
    ix1, iy1 = max(xa1, xb1), max(ya1, yb1)
    ix2, iy2 = min(xa2, xb2), min(ya2, yb2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    union = (xa2 - xa1) * (ya2 - ya1) + (xb2 - xb1) * (yb2 - yb1) - inter
    return inter / (union + 1e-6)


def load_subset():
    coco = DATA_DIR / "coco"
    meta = json.loads((coco / "subset_annotations.json").read_text())
    imgs = {im["id"]: im for im in meta["images"]}
    gts: dict[int, list[dict]] = {}
    for a in meta["annotations"]:
        gts.setdefault(a["image_id"], []).append(a)
    return imgs, gts


def detect_all(detector: ObjectDetector, imgs: dict, gts: dict):
    coco_dir = DATA_DIR / "coco" / "subset_images"
    all_preds = []
    all_gts = []
    per_img_results = []
    for img_id, im_meta in imgs.items():
        p = coco_dir / im_meta["file_name"]
        if not p.exists():
            continue
        frame = cv2.imread(str(p))
        if frame is None:
            continue
        t0 = time.perf_counter()
        evs = detector.infer(frame, timestamp=0.0)
        dt = time.perf_counter() - t0
        preds = [
            {
                "class_id": e.class_id,
                "label": e.label,
                "confidence": e.confidence,
                "bbox": list(e.bbox),
            }
            for e in evs
        ]
        gt = [
            {
                "class_id": g["category_id_yolo"],
                "bbox_xyxy": [
                    g["bbox_xywh"][0],
                    g["bbox_xywh"][1],
                    g["bbox_xywh"][0] + g["bbox_xywh"][2],
                    g["bbox_xywh"][1] + g["bbox_xywh"][3],
                ],
            }
            for g in gts.get(img_id, [])
        ]
        per_img_results.append({"image_id": img_id, "infer_sec": dt, "preds": preds, "gt": gt})
        all_preds.extend([(p["class_id"], p["confidence"], p["bbox"], img_id) for p in preds])
        all_gts.extend([(g["class_id"], g["bbox_xyxy"], img_id) for g in gt])
    return all_preds, all_gts, per_img_results


def per_class_metrics(all_preds, all_gts, iou_thr: float = 0.5):
    """Compute precision/recall/F1/AP per class using greedy matching."""
    classes = sorted({c for c, _, _, _ in all_preds} | {c for c, _, _ in all_gts})
    metrics = {}
    for cls in classes:
        cls_preds = sorted([p for p in all_preds if p[0] == cls], key=lambda x: -x[1])
        cls_gts = [g for g in all_gts if g[0] == cls]
        gt_by_img: dict[int, list] = {}
        for g in cls_gts:
            gt_by_img.setdefault(g[2], []).append({"box": g[1], "matched": False})
        tp_arr, fp_arr, scores = [], [], []
        for _, conf, box, img_id in cls_preds:
            best_iou, best_g = 0.0, None
            for g in gt_by_img.get(img_id, []):
                if g["matched"]:
                    continue
                i = iou(box, g["box"])
                if i > best_iou:
                    best_iou, best_g = i, g
            scores.append(conf)
            if best_g is not None and best_iou >= iou_thr:
                best_g["matched"] = True
                tp_arr.append(1)
                fp_arr.append(0)
            else:
                tp_arr.append(0)
                fp_arr.append(1)
        if not cls_preds:
            metrics[cls] = {"precision": 0, "recall": 0, "f1": 0, "ap": 0, "n_gt": len(cls_gts), "n_pred": 0}
            continue
        tp = np.cumsum(tp_arr)
        fp = np.cumsum(fp_arr)
        recall = tp / max(1, len(cls_gts))
        precision = tp / np.maximum(1, tp + fp)
        # 101-point interpolation AP
        if len(cls_gts) == 0:
            ap = 0.0
        else:
            interp = []
            for t in np.linspace(0, 1, 101):
                mask = recall >= t
                interp.append(float(precision[mask].max()) if mask.any() else 0.0)
            ap = float(np.mean(interp))
        f1 = float(2 * precision[-1] * recall[-1] / max(1e-6, precision[-1] + recall[-1]))
        metrics[cls] = {
            "precision": float(precision[-1]),
            "recall": float(recall[-1]),
            "f1": f1,
            "ap": ap,
            "n_gt": len(cls_gts),
            "n_pred": len(cls_preds),
            "scores": list(map(float, scores)),
            "tp_arr": tp_arr,
        }
    return metrics


def plot_pr_curves(metrics, out_path: Path):
    import matplotlib.pyplot as plt
    import seaborn as sns

    sns.set_style(PLOT_STYLE)
    fig, ax = plt.subplots(figsize=(7, 5))
    # Distinct line styles + markers so the 7 curves remain separable in greyscale.
    styles = ["-", "--", "-.", ":", (0, (3, 1, 1, 1)), (0, (5, 1)), (0, (1, 1))]
    markers = ["o", "s", "^", "D", "v", "P", "X"]
    for idx, (cls, m) in enumerate(metrics.items()):
        if not m.get("scores") or not m.get("tp_arr"):
            continue
        order = np.argsort(-np.array(m["scores"]))
        tp_sorted = np.array(m["tp_arr"])[order]
        fp_sorted = 1 - tp_sorted
        tp_cum = np.cumsum(tp_sorted)
        fp_cum = np.cumsum(fp_sorted)
        recall = tp_cum / max(1, m["n_gt"])
        precision = tp_cum / np.maximum(1, tp_cum + fp_cum)
        name = YOLO_HIGH_RISK_CLASSES.get(cls, str(cls))
        ax.plot(
            recall, precision, label=f"{name} (AP={m['ap']:.3f})",
            linestyle=styles[idx % len(styles)],
            marker=markers[idx % len(markers)],
            markevery=max(1, len(recall) // 8), markersize=5, linewidth=1.3,
        )
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("YOLOv8-s precision–recall on the proctoring subset of COCO val2017")
    ax.legend(loc="lower left", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=600)
    fig.savefig(out_path.with_suffix(".pdf"))
    plt.close(fig)


def plot_confusion_matrix(metrics, out_path: Path):
    """Per-class TP / FP / FN bar chart (greyscale-safe).

    TP, FP and FN are derived exactly from the matched detections:
      TP = round(recall * n_gt), FN = n_gt - TP, FP = n_pred - TP.
    Showing FN as well as FP makes the missed-detection problem (e.g. the
    low book recall) visible, which a TP/FP-only plot hid.
    """
    import matplotlib.pyplot as plt
    import seaborn as sns

    sns.set_style(PLOT_STYLE)
    classes = sorted(metrics.keys(), key=lambda c: -metrics[c]["n_gt"])
    names = [YOLO_HIGH_RISK_CLASSES.get(c, str(c)) for c in classes]
    tp, fp, fn = [], [], []
    for c in classes:
        m = metrics[c]
        t = int(round(m["recall"] * m["n_gt"]))
        tp.append(t)
        fp.append(max(0, m["n_pred"] - t))
        fn.append(max(0, m["n_gt"] - t))
    x = np.arange(len(classes))
    w = 0.27
    fig, ax = plt.subplots(figsize=(8, 4.2))
    b1 = ax.bar(x - w, tp, w, label="True positive", color="#4c72b0", edgecolor="black", hatch="//")
    b2 = ax.bar(x, fp, w, label="False positive", color="#dd8452", edgecolor="black", hatch="..")
    b3 = ax.bar(x + w, fn, w, label="False negative", color="#c44e52", edgecolor="black", hatch="xx")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=20, ha="right")
    ax.set_ylabel("Detections (IoU ≥ 0.5)")
    ax.set_title("Per-class TP / FP / FN after greedy IoU matching")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=600)
    fig.savefig(out_path.with_suffix(".pdf"))
    plt.close(fig)


def main():
    imgs, gts = load_subset()
    detector = ObjectDetector()
    print(f"Running YOLOv8-s on {len(imgs)} images ...")
    all_preds, all_gts, per_img = detect_all(detector, imgs, gts)
    metrics = per_class_metrics(all_preds, all_gts, iou_thr=0.5)

    macro_f1 = float(np.mean([m["f1"] for m in metrics.values() if m["n_gt"] > 0]))
    macro_ap = float(np.mean([m["ap"] for m in metrics.values() if m["n_gt"] > 0]))
    mean_lat_ms = float(np.mean([r["infer_sec"] for r in per_img]) * 1000)

    out = {
        "n_images": len(per_img),
        "per_class": {
            YOLO_HIGH_RISK_CLASSES.get(c, str(c)): {
                k: v for k, v in m.items() if k not in ("scores", "tp_arr")
            }
            for c, m in metrics.items()
        },
        "macro_f1@0.5": macro_f1,
        "mean_AP@0.5": macro_ap,
        "mean_inference_ms": mean_lat_ms,
    }
    (RESULTS_DIR / "visual_metrics.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))

    plot_pr_curves(metrics, FIGURES_DIR / "fig_visual_pr_curves.png")
    plot_confusion_matrix(metrics, FIGURES_DIR / "fig_visual_tp_fp.png")
    print("Visual evaluation complete.")


if __name__ == "__main__":
    main()
