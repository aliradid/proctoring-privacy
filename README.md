# Privacy-Preserving Multi-Modal Proctoring

Code and experimental package accompanying the paper
**"A Privacy-Preserving Multi-Modal Proctoring Framework Using Deep Learning for Secure Online Examinations"**
by Radid Ali, Ghazouani Mohamed and El habib Benlahmar (Hassan II University, Casablanca).

The framework combines **YOLOv8-s** (object detection), **MediaPipe FaceMesh**
(non-identifying geometric behavioural cues), and a **Whisper-Base encoder in
content-free mode** (acoustic anomaly detection) into a learned **Suspicion
Index Fusion Model**. No biometric template, raw video, or raw audio is ever
persisted; the only artefact written to disk is a metadata event stream that
an adversarial attacker cannot invert to re-identify candidates.

## Headline results

| Component | Metric | Value |
|---|---|---|
| Visual (YOLOv8-s on 217 real COCO val2017 images) | macro F1 @ IoU 0.5 | **0.606** |
| Acoustic (Whisper-Base on 400 LibriSpeech clips) | secondary-speaker ROC AUC | **0.774** |
| Acoustic (Whisper-Base on 400 LibriSpeech clips) | whisper-detection ROC AUC | **0.935** |
| Behaviour (FaceMesh on real Pexels face anchors) | off-screen ROC AUC | **0.938** |
| Fusion (Random Forest, 5-fold CV, real features) | F1 | **0.905 ± 0.041** |
| Fusion vs. weighted-sum baseline | McNemar | **not significant (p = 0.37)** |
| End-to-end pipeline (Apple M4 Max, MPS / CPU) | FPS | **23.7 / 19.4** |
| End-to-end pipeline | peak resident memory | **1,102 MB** |
| Privacy on 30 real LibriSpeech speakers | raw-audio re-ID AUC | **0.994** |
| Privacy: undefended acoustic metadata re-ID AUC | (content-free, not identity-free) | **0.816** |
| Privacy: epsilon-DP acoustic metadata re-ID AUC | (epsilon = 4/value; detection 0.774 -> 0.625) | **0.547** |

## Repository layout

```
.
├── LICENSE                     MIT
├── CITATION.cff                citation metadata
├── requirements.txt            pinned Python dependencies
├── bootstrap.sh                downloads COCO subset, LibriSpeech, Pexels anchors, MediaPipe model
├── reproduce.sh                runs every experiment end-to-end
├── REPRODUCE.md                step-by-step reproducibility guide
├── code/
│   ├── config.py               hyperparameters and thresholds (seed 42)
│   ├── visual_module.py        YOLOv8-s + MediaPipe FaceMesh wrappers
│   ├── acoustic_module.py      Whisper-Base encoder-only inference
│   ├── fusion_model.py         five fusion strategies (weighted sum, LR, RF, MLP, NB)
│   ├── dataset_builder.py      synthetic generators (used for sanity checks only)
│   ├── coco_subset.py          downloads a proctoring subset of COCO val2017
│   ├── real_acoustic.py        builds the LibriSpeech acoustic benchmark
│   ├── real_behaviour.py       runs MediaPipe on real Pexels face anchors
│   ├── real_fusion_features.py wires real modality outputs into fusion features
│   ├── build_informatica.py    Markdown -> DOCX manuscript builder (journal format)
│   └── experiments/            one file per experiment (visual, acoustic, behaviour,
│                               fusion, latency, privacy, robustness, figures)
├── data/                       (re-populated by bootstrap.sh)
├── results/                    JSON metrics produced by each experiment
├── figures/                    publication-quality figures (vector PDF committed; PNG previews regenerated locally)
└── manuscript/
    ├── manuscript.md           source markdown
    └── Article_Informatica.docx   built from manuscript.md
```

## Quick start

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
bash bootstrap.sh        # ~ 5 minutes, ~ 600 MB download
bash reproduce.sh        # ~ 15 minutes on an Apple M4 Max
```

See `REPRODUCE.md` for full instructions and expected per-experiment outputs.

## Privacy contract

The system writes **only** records of the form
`(event_type, probability, timestamp, module)` to disk. Sections 3.3 and 6.7
of the manuscript formalise the threat model (honest-but-curious server,
internal observer, external observer, curious operator) and report an
adversarial verification on 30 real LibriSpeech speakers. A random forest
trained on raw MFCC features re-identifies speakers at AUC 0.994; the same
classifier trained on the *real* content-free acoustic metadata our system
emits still reaches 0.816, showing that a content-free detector output is not
automatically identity-free. Releasing that metadata through an
epsilon-differentially-private Laplace mechanism (epsilon = 4 per value) lowers
re-identification to 0.547 (near chance), at the cost of reducing
secondary-speaker detection AUC from 0.774 to 0.625, a privacy-utility
trade-off characterised across the full budget range in Section 6.7.

## Licence and data attribution

* **Code**: MIT (see `LICENSE`).
* **COCO val2017**: Creative Commons Attribution 4.0 ([cocodataset.org](https://cocodataset.org)).
* **LibriSpeech dev-clean**: Creative Commons Attribution 4.0 ([openslr.org/12](https://www.openslr.org/12)).
* **Pexels portraits**: free under the [Pexels licence](https://www.pexels.com/license/).
* **YOLOv8-s checkpoint** and **MediaPipe FaceLandmarker checkpoint**: distributed
  by Ultralytics and Google under their own licences.

## Contact

For questions about the paper or this implementation, please contact the
corresponding author, Radid Ali (Department of Mathematics and Computer Science,
Hassan II University, Faculty of Sciences Ben M'sik, Casablanca, Morocco) at
`ali.radid-etu@etu.univh2c.ma`.
