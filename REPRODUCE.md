# Reproducing the experiments

This document explains how to regenerate every numerical result in the paper
from scratch on a fresh machine.

## 1. System requirements

* macOS, Linux, or Windows (WSL works)
* Python 3.9+ (3.10–3.12 recommended)
* 20 GB free disk space (the LibriSpeech dev-clean tarball is 337 MB, COCO val2017 annotations 250 MB, plus model checkpoints)
* Internet access for the bootstrap step
* Optional: an Apple-Silicon GPU exposed through MPS, or any CUDA-capable GPU

## 2. Install Python dependencies

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 3. Download the external assets

```bash
bash bootstrap.sh
```

The script downloads:

* **COCO val2017** annotations (CC-BY 4.0) and selects a 217-image proctoring
  subset that is downloaded image-by-image (≈ 100 MB).
* **LibriSpeech dev-clean** (CC-BY 4.0, ≈ 337 MB) — real read English speech.
* **MediaPipe FaceLandmarker** float16 checkpoint (≈ 3.6 MB).
* **10 CC-BY Pexels portraits** used as real-face anchors for the behavioural
  benchmark.

The COCO image subset selection and the LibriSpeech utterance picks are
deterministic (md5-seeded) so two clones of the repository will produce the
same evaluation files.

## 4. Run every experiment

```bash
bash reproduce.sh
```

This sequentially executes all eight experiment scripts:

| Step | Script | Outputs |
|------|--------|---------|
| 1 | `experiments/exp_visual.py` | `results/visual_metrics.json`, `figures/fig_visual_pr_curves.png`, `figures/fig_visual_tp_fp.png` |
| 2 | `experiments/exp_acoustic.py` | `results/acoustic_metrics.json`, `figures/fig_acoustic_roc_pr.png`, `figures/fig_acoustic_per_class.png` |
| 3 | `experiments/exp_behaviour.py` | `results/behaviour_metrics.json`, `figures/fig_behaviour_distributions.png` |
| 4 | `real_fusion_features.py` | `data/scenarios/real_fusion_scenarios.json` |
| 5 | `experiments/exp_fusion.py` | `results/fusion_metrics.json`, `figures/fig_fusion_*` |
| 6 | `experiments/exp_latency.py` | `results/latency_metrics.json`, `figures/fig_latency.png` |
| 7 | `experiments/exp_privacy.py` | `results/privacy_metrics.json`, `figures/fig_privacy.png` |
| 8 | `experiments/exp_robustness.py` | `results/robustness_metrics.json`, `figures/fig_robustness.png` |
| 9 | `experiments/exp_figures.py` | architecture diagram, SI timeline, summary table |
| 10 | `build_informatica.py` | `manuscript/Article_Informatica.docx` |

Total wall time on an Apple M4 Max (Mac Studio, 36 GB unified memory) is
approximately **15 minutes**.

## 5. Headline numbers

After `reproduce.sh` exits, the JSON files in `results/` should match:

| Metric | Expected value |
|---|---|
| YOLOv8-s macro F1 on COCO subset | 0.606 |
| YOLOv8-s mAP@0.5 on COCO subset | 0.519 |
| Whisper-Base secondary-speaker ROC AUC (LibriSpeech) | 0.774 |
| Whisper-Base whisper ROC AUC (LibriSpeech) | 0.935 |
| Random-forest fusion F1 (real features, 5-fold CV) | 0.905 ± 0.041 |
| End-to-end pipeline FPS on M4 Max (MPS) | 29.5 |
| Privacy raw-audio re-ID AUC | 0.994 |
| Privacy metadata-only re-ID AUC | 0.478 |

Small per-run variation (≤ ±0.02 in AUC, ≤ ±1 FPS) is expected from numerical
non-determinism of GPU kernels and SciPy / NumPy version differences.

## 6. Reusing only one modality

Each module is importable on its own:

```python
from code.visual_module import ObjectDetector, BehaviourAnalyzer
from code.acoustic_module import AcousticAnomalyDetector
from code.fusion_model import make_rf_fusion

det = ObjectDetector(device="mps")
events = det.infer(frame, timestamp=0.0)
```

See `code/config.py` for every threshold and hyperparameter.

## 7. License and data attribution

* Code: MIT (see `LICENSE`).
* COCO val2017: CC-BY 4.0 (cocodataset.org).
* LibriSpeech: CC-BY 4.0 (openslr.org/12).
* Pexels portraits: free under the [Pexels license](https://www.pexels.com/license/).
* MediaPipe model and YOLOv8-s checkpoint: distributed by their respective
  upstream projects under their own licenses.
