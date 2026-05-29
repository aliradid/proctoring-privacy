#!/usr/bin/env bash
# Bootstrap script: downloads every external asset required to reproduce the
# experiments. The script is idempotent — already-downloaded files are skipped.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

mkdir -p data/coco data/librispeech data/faces

# ----------------------------------------------------------------------------
# 1. COCO val2017 annotations + image subset (selected via code/coco_subset.py)
# ----------------------------------------------------------------------------
if [ ! -f data/coco/annotations/instances_val2017.json ]; then
  echo "[1/3] downloading COCO annotations ..."
  curl -L --fail --max-time 600 -o data/coco/ann.zip \
    "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
  (cd data/coco && unzip -q ann.zip)
fi

if [ ! -d data/coco/subset_images ] || [ -z "$(ls -A data/coco/subset_images 2>/dev/null)" ]; then
  echo "[1/3] selecting and downloading the proctoring subset of COCO val2017 ..."
  python3 code/coco_subset.py
fi

# ----------------------------------------------------------------------------
# 2. LibriSpeech dev-clean (real read English speech, CC-BY 4.0)
# ----------------------------------------------------------------------------
if [ ! -d data/librispeech/LibriSpeech/dev-clean ]; then
  echo "[2/3] downloading LibriSpeech dev-clean ..."
  curl -L --fail --max-time 900 -o data/librispeech/dev-clean.tar.gz \
    "https://www.openslr.org/resources/12/dev-clean.tar.gz"
  (cd data/librispeech && tar xzf dev-clean.tar.gz)
fi

# ----------------------------------------------------------------------------
# 3. MediaPipe FaceLandmarker checkpoint + 10 CC-BY Pexels face anchors
# ----------------------------------------------------------------------------
if [ ! -f data/face_landmarker.task ]; then
  echo "[3/3] downloading MediaPipe FaceLandmarker checkpoint ..."
  curl -L --fail -o data/face_landmarker.task \
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
fi

if [ -z "$(ls -A data/faces 2>/dev/null | grep -E '\.jpg$' || true)" ]; then
  echo "[3/3] downloading 10 CC-BY face anchors from Pexels ..."
  urls=(
    "https://images.pexels.com/photos/1239291/pexels-photo-1239291.jpeg?w=600&auto=compress"
    "https://images.pexels.com/photos/762020/pexels-photo-762020.jpeg?w=600&auto=compress"
    "https://images.pexels.com/photos/733872/pexels-photo-733872.jpeg?w=600&auto=compress"
    "https://images.pexels.com/photos/2381069/pexels-photo-2381069.jpeg?w=600&auto=compress"
    "https://images.pexels.com/photos/2613260/pexels-photo-2613260.jpeg?w=600&auto=compress"
    "https://images.pexels.com/photos/415829/pexels-photo-415829.jpeg?w=600&auto=compress"
    "https://images.pexels.com/photos/1844547/pexels-photo-1844547.jpeg?w=600&auto=compress"
    "https://images.pexels.com/photos/736716/pexels-photo-736716.jpeg?w=600&auto=compress"
    "https://images.pexels.com/photos/220453/pexels-photo-220453.jpeg?w=600&auto=compress"
    "https://images.pexels.com/photos/3206165/pexels-photo-3206165.jpeg?w=600&auto=compress"
  )
  i=1
  for u in "${urls[@]}"; do
    curl -s -L "$u" -o "data/faces/face_${i}.jpg"
    i=$((i+1))
  done
fi

echo
echo "Bootstrap complete. Run 'bash reproduce.sh' to regenerate every result."
