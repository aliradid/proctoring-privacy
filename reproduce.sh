#!/usr/bin/env bash
# Reproduce every result reported in the paper.
# Requires: bootstrap.sh to have been run first.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/code"

echo "==> visual subsystem (YOLOv8 on COCO subset)"
python3 -m experiments.exp_visual

echo "==> acoustic subsystem (Whisper-Base on LibriSpeech)"
python3 -m experiments.exp_acoustic

echo "==> behavioural cues (MediaPipe on real face anchors)"
python3 -m experiments.exp_behaviour

echo "==> building real-modality fusion features"
python3 real_fusion_features.py

echo "==> fusion model comparison + ablation"
python3 -m experiments.exp_fusion

echo "==> latency / memory benchmark"
python3 -m experiments.exp_latency

echo "==> adversarial privacy verification"
python3 -m experiments.exp_privacy

echo "==> robustness study"
python3 -m experiments.exp_robustness

echo "==> supplementary figures"
python3 -m experiments.exp_figures

echo "==> building manuscript DOCX"
python3 build_informatica.py

echo
echo "All experiments reproduced. Check results/ for JSON metrics,"
echo "figures/ for plots, and manuscript/ for the final DOCX."
