"""Build a privacy-compliant COCO subset for the proctoring object-detection
benchmark.

We keep only images that contain at least one proctoring-relevant class
(cell phone, book, laptop, tv, keyboard, mouse, person). The COCO category IDs
are remapped to the COCO IDs that the YOLOv8 model uses (cell phone = 67, etc.)
so they can be matched directly against the model predictions.

The selected images are downloaded individually from the COCO image server so
we don't need the 19 GB train2017 zip. Total subset size is < 100 MB.
"""
from __future__ import annotations

import json
import shutil
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import urllib.request

from config import DATA_DIR

COCO_TO_YOLO = {
    1: 0,    # person
    77: 67,  # cell phone
    84: 73,  # book
    73: 63,  # laptop
    72: 62,  # tv
    76: 66,  # keyboard
    74: 64,  # mouse
}

TARGET_CATS = set(COCO_TO_YOLO.keys())
# Upper cap on the stratified selection. In practice the cap is not the binding
# constraint: only 217 val2017 images contain the seven target categories under
# the per-category slice below, so the realised subset is 217 images (pinned by
# the committed data/coco/subset_annotations.json).
MAX_IMAGES = 400


def select_images(ann_path: Path):
    with ann_path.open() as f:
        d = json.load(f)
    img_meta = {im["id"]: im for im in d["images"]}
    by_img: dict[int, list[dict]] = {}
    for a in d["annotations"]:
        if a["category_id"] in TARGET_CATS and a.get("iscrowd", 0) == 0:
            by_img.setdefault(a["image_id"], []).append(a)
    # Stratify by which target class is present so we get a balanced subset.
    # Image ids and categories are visited in sorted order so the selection is
    # fully deterministic across machines (independent of dict/JSON ordering).
    by_cat: dict[int, list[int]] = {c: [] for c in TARGET_CATS}
    for img_id in sorted(by_img):
        for a in by_img[img_id]:
            by_cat[a["category_id"]].append(img_id)
    chosen = set()
    per_cat = MAX_IMAGES // len(TARGET_CATS) + 5
    for c in sorted(TARGET_CATS):
        for img_id in by_cat[c][:per_cat]:
            chosen.add(img_id)
            if len(chosen) >= MAX_IMAGES:
                break
        if len(chosen) >= MAX_IMAGES:
            break
    sub_anns = [a for img_id in sorted(chosen) for a in by_img[img_id]]
    sub_imgs = [img_meta[i] for i in sorted(chosen)]
    return sub_imgs, sub_anns


def _download_one(im, out_dir: Path) -> bool:
    url = im.get("coco_url") or im.get("flickr_url")
    out = out_dir / im["file_name"]
    if out.exists() and out.stat().st_size > 0:
        return True
    try:
        with urllib.request.urlopen(url, timeout=15) as r:
            data = r.read()
        out.write_bytes(data)
        return True
    except Exception as e:
        print(f"download fail {url}: {e}")
        return False


def download_subset():
    coco_dir = DATA_DIR / "coco"
    ann_path = coco_dir / "annotations" / "instances_val2017.json"
    out_imgs = coco_dir / "subset_images"
    out_imgs.mkdir(parents=True, exist_ok=True)
    out_json = coco_dir / "subset_annotations.json"
    sub_imgs, sub_anns = select_images(ann_path)
    print(f"Selected {len(sub_imgs)} images with {len(sub_anns)} annotations")

    with ThreadPoolExecutor(max_workers=16) as ex:
        successes = list(ex.map(lambda im: _download_one(im, out_imgs), sub_imgs))
    print(f"Downloaded {sum(successes)} / {len(sub_imgs)}")

    # Build YOLO-style annotations referencing the YOLOv8 class ids.
    yolo_id_to_name = {
        0: "person", 67: "cell phone", 73: "book", 63: "laptop",
        62: "tv", 66: "keyboard", 64: "mouse",
    }
    yolo_anns = []
    for a in sub_anns:
        if a["category_id"] not in COCO_TO_YOLO:
            continue
        yolo_id = COCO_TO_YOLO[a["category_id"]]
        yolo_anns.append(
            {
                "image_id": a["image_id"],
                "category_id_yolo": yolo_id,
                "category_name": yolo_id_to_name.get(yolo_id, str(yolo_id)),
                "bbox_xywh": a["bbox"],
            }
        )
    out_json.write_text(
        json.dumps(
            {
                "images": [{"id": im["id"], "file_name": im["file_name"], "width": im["width"], "height": im["height"]} for im in sub_imgs],
                "annotations": yolo_anns,
            },
            indent=2,
        )
    )
    print(f"Wrote {out_json}")


if __name__ == "__main__":
    download_subset()
