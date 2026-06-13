# A Privacy-Preserving Multi-Modal Proctoring Framework Using Deep Learning for Secure Online Examinations

**Radid Ali**, Ghazouani Mohamed, El habib Benlahmar

*Department of Mathematics and Computer Science, Hassan II University, Faculty of Sciences Ben M'sik, Casablanca, Morocco*
{ali.radid-etu@etu.univh2c.ma, ghazouani.fsbm@gmail.com, h.benlahmer@gmail.com}

## Abstract

AI-driven proctoring of online exams typically relies on continuous biometric processing that conflicts with the European GDPR and Moroccan CNDP Law 09-08, both of which treat such data as a special category. We present a privacy-preserving multi-modal framework that detects cheating without retaining biometric templates, raw video, or linguistic content. It combines YOLOv8-s for contextual object detection, MediaPipe FaceMesh for non-identifying geometric behavioural cues, and a Whisper-Base encoder operating in content-free mode for acoustic anomaly detection; the modality outputs feed a Suspicion Index Fusion Model chosen empirically among five alternatives. YOLOv8-s reaches a macro F1 of 0.606 on a real COCO val2017 subset, and on 400 LibriSpeech-derived clips the Whisper encoder attains a secondary-speaker ROC AUC of 0.774 and a whisper-detection AUC of 0.935. Five-fold cross-validation on 300 real-feature sessions selects random forest as the best fusion strategy (F1 = 0.905 ± 0.041), although its margin over a transparent weighted-sum rule is not statistically significant (McNemar p = 0.37), which we report openly. The pipeline runs at 23.7/19.4 FPS (MPS/CPU) on an Apple M4 Max with 1,102 MB peak memory. An adversarial privacy verification on 30 real LibriSpeech speakers gives raw-audio re-identification AUC 0.994 and undefended acoustic-metadata AUC 0.816, confirming that content-free is not identity-free; releasing the metadata through an epsilon-differentially-private mechanism (epsilon = 4 per value) lowers re-identification to 0.547, close to chance, with detection AUC dropping from 0.774 to 0.625.

**Keywords**: privacy-preserving proctoring, non-biometric monitoring, YOLOv8, MediaPipe FaceMesh, Whisper, suspicion index, online examinations, GDPR, CNDP, adversarial privacy verification, differential privacy

## 1. Introduction

### 1.1 Motivation

The widespread adoption of online learning environments has accelerated the need for secure remote assessment methods capable of preserving academic integrity without violating student privacy. The rapid pivot to remote assessment during and after the COVID-19 pandemic was accompanied by the broad deployment of commercial proctoring platforms (e.g. Respondus Monitor, ProctorU) that rely on continuous biometric face verification, browser-lockdown agents, and audio recording. The behavioural and ethical impact of these tools has been the subject of growing concern, with empirical studies documenting test-taker anxiety, surveillance overreach, and racial-skin-tone bias in face-verification components [1], [2], [3].

In parallel, the regulatory environment has tightened sharply. Under GDPR Article 4(14), *biometric data* is defined as personal data resulting from specific technical processing of physical, physiological or behavioural characteristics that allow or confirm unique identification; when processed for the purpose of uniquely identifying a person, Article 9 [4] classifies it as a special category whose processing is in principle prohibited absent one of the Article 9(2) conditions (for example explicit consent, itself subject to the freely-given requirement that is problematic under the power imbalance of exam supervision [7]). The Moroccan Law 09-08 [5], enforced by the CNDP, establishes an analogous regime: it requires a clear legal basis, purpose limitation and proportionality for processing sensitive personal data, and, for sensitive-data processing, prior authorisation from or declaration to the CNDP. CNDP guidance to data controllers [6] elaborates these proportionality obligations. French courts have also begun to restrain biometric proctoring: in 2023 the Administrative Court of Montreuil suspended a university's use of the algorithmic e-proctoring platform TestWe, finding serious doubt that its "permanent surveillance of bodies and sounds" satisfied the GDPR data-minimisation requirement [32]. A subtle but important point for our design is that landmark *geometry* and content-free acoustic *probabilities* are engineered so as not to be used for unique identification: they are not matched against any enrolment template. Section 6.7 measures the residual re-identifiability of the acoustic metadata channel directly and shows it can be driven close to chance with a differentially-private emission mechanism. Consequently, the academic sector faces a deployment gap: institutions must offer scalable remote assessment while avoiding the regulatory and ethical risks of biometric surveillance.

### 1.2 Research Gap

A thorough literature review (Section 2) reveals three structural limitations of the current state of the art:

1. **Identity-bound monitoring.** Existing AI-driven proctoring frameworks overwhelmingly rely on biometric identification (face embeddings, voiceprints, or behavioural biometrics such as keystroke dynamics) creating both legal and ethical exposure.
2. **Raw-media retention.** Even systems that nominally avoid biometric matching frequently store the underlying video or audio for human review, exposing institutions to significant data-protection liability.
3. **Lack of quantitative privacy guarantees.** When privacy is claimed, the claim is usually based on the absence of certain components (e.g. no face recognition) rather than on adversarial verification that the released signals cannot be inverted to recover sensitive information.

### 1.3 Contributions

This paper addresses the gap above by presenting a privacy-by-design multi-modal proctoring framework with the following contributions:

- **Architectural contribution.** A modular pipeline that combines YOLOv8-s, MediaPipe FaceMesh, and a content-free configuration of Whisper-Base, where every modality outputs only numerical metadata. Raw frames and audio buffers are discarded after each inference cycle, and no biometric template is ever stored.
- **Algorithmic contribution.** A Suspicion Index Fusion Model whose parameterisation is identified empirically through a comparison of five alternative strategies (weighted sum, logistic regression, random forest, multi-layer perceptron, naive Bayes), evaluated with stratified 5-fold cross-validation, 95% bootstrap confidence intervals, leave-one-modality-out ablations, and McNemar significance tests.
- **Empirical contribution.** End-to-end evaluation on real data: 217 real COCO val2017 images for the visual subsystem; 400 LibriSpeech-derived clips of real read English speech for the acoustic subsystem; 400 behavioural trajectories built from real MediaPipe FaceMesh landmarks extracted from 10 CC-BY Pexels portraits; and 300 fusion-evaluation sessions whose feature vectors are *real* outputs of the three modality encoders. Latency benchmarks on an Apple M4 Max Mac Studio host with both GPU/MPS and CPU-only configurations and a robustness study across lighting, distance, and signal-to-noise variations complete the protocol.
- **Privacy contribution.** An adversarial verification experiment using 30 real LibriSpeech speakers quantifies the residual identity leakage of the emitted metadata: the conventional raw-media analogue exposes near-perfect re-identification (AUC = 0.994), the undefended content-free acoustic metadata still leaks moderate identity (AUC = 0.816), and a differentially-private emission mechanism reduces this to 0.547 (near chance) along a measured privacy-utility trade-off. We are not aware of a prior proctoring paper that both quantifies metadata re-identifiability and provides a differentially-private mitigation on real speaker data.

### 1.4 Paper Organisation

Section 2 reviews the related literature using a structured matrix that classifies prior work along biometric, retention, multi-modality, and compliance axes. Section 3 introduces the system architecture and Section 4 details the methodology, including the formal definition of the Suspicion Index. Section 5 reports the experimental setup. Section 6 presents the empirical results, including per-modality metrics, fusion comparison, robustness, latency, and adversarial privacy verification. Section 7 discusses implications, threats to validity, and deployment considerations. Section 8 surveys limitations, and Section 9 concludes.

## 2. Related Work

We classify the prior literature along five orthogonal axes that are relevant for compliant deployment:

- (V) Vision-based proctoring,
- (A) Audio-based proctoring,
- (M) Multi-modal proctoring,
- (B) Behavioural biometrics (keystroke, mouse dynamics),
- (P) Privacy-preserving and federated approaches.

### 2.1 Vision-based Systems (V)

Early proctoring solutions rely heavily on computer vision pipelines that authenticate the candidate's identity and monitor behavioural irregularities, typically built on top of mature face-recognition technology [35] and general-purpose object detectors trained on large labelled corpora such as COCO and its extensions [49]. Atoum et al. [8] introduced one of the first end-to-end automated proctoring systems combining gaze tracking, head pose estimation, and continuous face verification on the basis of webcam imagery and a wearable secondary camera. The systematic review of Nigam et al. [9] documents 39 follow-up systems that build on this paradigm, typically combining convolutional networks for face/gaze monitoring with rule-based decision layers. More recent contributions explored YOLO-based detectors for prohibited objects [10], [11], building on the YOLO family of architectures [13], [33], [34]. Singh et al. [10] in particular reported a YOLO-based proctoring pipeline targeted at smartphone and book detection in controlled lighting conditions, while Vishal et al. [11] integrate gaze tracking with prohibited-object detection in a unified PyTorch pipeline. A central limitation of these vision-based pipelines is the dependence on persistent face templates and raw-video archives, which, where the templates are used to verify or identify the candidate, engage the GDPR Article 9 special-category regime [4].

These approaches share an important limitation: they require either persistent biometric templates (face embeddings) or extensive raw-video retention, both of which raise significant GDPR Article 9 and CNDP Law 09-08 exposure once used for identification.

### 2.2 Audio-based Systems (A)

Audio-driven proctoring monitors the acoustic environment for unauthorised collaboration. Overlapped-speech detection and speaker-change segmentation are typical building blocks, commonly developed and evaluated on multi-speaker meeting and speaker-recognition corpora such as AMI [18] and VoxCeleb2 [19]; Jung et al. [20] report state-of-the-art three-class (non-speech / single-speaker / overlapped-speech) detection using a CRNN architecture. Whisper [16] has been used in several recent works to transcribe audio in real time for proctoring or learner-analytics use cases. Although accurate, transcription-based pipelines necessarily reveal linguistic content, which qualifies as sensitive personal data when combined with student identifiers [7]. Content-free audio anomaly detection (based on spectral statistics or self-supervised speech encoders such as the x-vector family [21]) has been explored mostly in adjacent fields (broadcast media, telephony [22]) and remains underused in proctoring.

### 2.3 Multi-modal Systems (M)

Multi-modal architectures combine video, audio, and user-interaction telemetry. Masud et al. [12] propose a smart proctoring assistant fusing webcam, microphone, and screen-recording, reporting an aggregate accuracy of 97.7% over four event categories. Although these systems demonstrate higher robustness, they generally inherit the privacy issues of their constituents: every modality is fed in raw form into a centralised model and stored. The systematic review of Nigam et al. [9] reports that the vast majority of the surveyed AI-based proctoring systems perform no explicit data minimisation, and that ethical and legal considerations are typically treated as deployment afterthoughts rather than as constraints on the architecture itself.

### 2.4 Behavioural Biometrics (B)

Behavioural biometric pipelines analyse keystroke dynamics [30], mouse-movement and touch-gesture patterns [31], and eye-tracking signatures to authenticate the candidate. While more passive than facial recognition, these signals are still classified as personal data under GDPR [4] when combined with an identifier, because they support identification with non-trivial accuracy. Our framework explicitly excludes keystroke biometrics for that reason.

### 2.5 Privacy-Preserving Approaches (P)

A small but growing body of work explores privacy-preserving online assessment. Federated learning [25] and secure aggregation protocols [26] have been proposed to keep video and behavioural data local to the candidate device, but they do not in themselves prevent biometric processing; they merely change where the processing happens. Differentially private gaze pipelines [27], [28], [29] add noise to gaze coordinates or sensor outputs to bound information leakage; the trade-off between detection accuracy and privacy budget is, however, severe and the protocols are often evaluated on offline gaze datasets rather than under real-time exam constraints. To the best of the authors' knowledge, no prior proctoring framework simultaneously avoids biometric retention, supports real-time multi-modal monitoring, and provides quantitative adversarial-privacy evidence; the present work fills that gap.

### 2.6 Synthesis

Table 1 summarises the literature using the axes introduced above. None of the surveyed systems simultaneously (i) avoids biometric retention, (ii) supports real-time multi-modal monitoring, and (iii) provides quantitative adversarial privacy evidence. This is the gap we address.

**Table 1. Comparison of representative proctoring systems against the proposed framework. "V/A/M/B/P" denote vision, audio, multi-modal, behavioural-biometric and privacy-preserving categories.**

| Work | Type | Biom. template | Raw media | Real-time | Privacy proof | GDPR/CNDP |
|---|---|---|---|---|---|---|
| Atoum et al. (2017) [8] | V | Yes | Yes | Yes | No | No |
| Singh et al. (2024) [10] | V | Yes | Yes | Yes | No | No |
| Vishal et al. (2025) [11] | V | Yes | Yes | Yes | No | No |
| Jung et al. (2021) [20] | A | Voiceprint context | No | Yes | No | Partial |
| Masud et al. (2022) [12] | V, A, M | Yes | Yes | Yes | No | No |
| McMahan et al. (2017), FedAvg [25] | P | n/a (framework) | No | Yes | No | Partial |
| Bonawitz et al. (2017), SecAgg [26] | P | n/a (framework) | No | Yes | No | Partial |
| Du et al. (2024), PrivateGaze [27] | P | Obfuscated | No | Near real-time | No | Partial |
| Bozkir et al. (2021), DP eye-tracking [28] | P | No | No | No | DP bound | Partial |
| **This work** | V, A, M, P | **No** | **No** | **Yes** | **Yes (DP, re-ID 0.55)** | **By design** |

## 3. System Architecture

### 3.1 Design Principles

The architecture follows two guiding principles motivated by Article 25 of GDPR ("Data protection by design and by default") and CNDP guidance note [6]:

- **Data minimisation.** The pipeline ingests volatile camera and microphone streams. Raw frames and audio buffers are discarded immediately after inference; only non-identifying numerical metadata is persisted.
- **Identity orthogonality.** No module performs identity recognition. Visual analysis is class-level (smartphones, books, persons) rather than person-level. Audio analysis is content-free, producing only secondary-speaker, overlap, and whisper probabilities.

### 3.2 Pipeline Overview

The end-to-end pipeline (Figure 1) consists of four logical stages:

1. **Sensors.** Camera (720p webcam) and microphone are accessed in a volatile memory buffer.
2. **Modality encoders.** YOLOv8-s for objects, MediaPipe FaceMesh for behavioural cues, Whisper-Base encoder in content-free mode for acoustic anomaly.
3. **Suspicion Index Fusion Model.** Combines the modality outputs into a calibrated risk score.
4. **Response layer.** Generates warnings or flags according to risk thresholds and writes only metadata to disk.

[[FIG:architecture]]

### 3.3 Threat Model

We adopt the standard *honest-but-curious* threat model from the differential-privacy literature [24], [38], extended for proctoring. Classical privacy models such as k-anonymity [37] guard against linkage attacks but are known to be insufficient against rich auxiliary information, which motivates the stronger adversarial test we adopt. The re-identification adversary we instantiate in Section 6.7 follows the de-anonymisation methodology of Narayanan and Shmatikov [36], who showed that sparse high-dimensional records can be re-identified from little auxiliary information; our metadata stream is designed precisely to deny such an adversary any usable signal. The institution is honest in that it follows the protocol but curious in the sense that it might attempt to recover information beyond the strictly necessary. We additionally consider three explicit adversaries:

- **Internal observer.** Has read access to the event log. Goal: re-identify candidates from the metadata.
- **External observer.** Has read access to the artefacts written to disk by the system. Goal: reconstruct video, audio, or identifying templates.
- **Curious operator.** Has read access to model artefacts. Goal: extract biometric template embeddings from model state.

A system is *privacy-preserving* with respect to this threat model to the extent that the released signals limit these adversaries to near-chance identification. Section 6.7 measures this property for the internal observer on the acoustic metadata channel and shows that a differentially-private emission mechanism is needed to approach the chance bound.

## 4. Methodology

### 4.1 Visual Subsystem

We use YOLOv8-s [13] pre-trained on COCO [14]. The model produces, for each frame, a set of bounding boxes with class probabilities. We retain only the proctoring-relevant subset:

$$\mathcal{C} = \{cell phone, book, laptop, tv, keyboard, mouse, person\}$$

For each detection $d_i = (c_i, p_i, b_i, t_i)$ with class $c_i \in \mathcal{C}$, confidence $p_i$, bounding box $b_i$, and timestamp $t_i$, the visual subsystem outputs *only* the tuple $(c_i, p_i, t_i)$. No frame is stored.

MediaPipe FaceMesh [15], [48] yields 468 facial landmarks $L = \{l_1, ..., l_{468}\}$. We compute four geometric descriptors:

- **Gaze deviation angle** $\theta_g$: derived from the angular position of the nose tip relative to the eye-centre axis.
- **Head yaw** $\phi_y$ and **pitch** $\phi_p$: computed from the relative position of ears, nose tip, and chin.
- **Lip variance** $\sigma_l^2$: variance of the upper-lower lip distance over a 30-frame sliding window.

Sustained off-screen gaze is flagged when $\theta_g > 18°$ or $|\phi_y| > 25°$ for at least 2.5 s.

### 4.2 Acoustic Subsystem (Content-Free Whisper)

We use Whisper-Base [16] (74 M parameters) with the decoder disabled. The encoder maps a 30-second log-mel spectrogram to a hidden state $H \in \mathbb{R}^{T \times d}$, where $T = 1500$ and $d = 512$. For each 5-second buffer $a$, we compute:

- **Voice Activity Ratio** $\mathrm{VAR}(a)$: fraction of short-time frames whose energy exceeds 1.4× the median.
- **Pitch fluctuation** $\mathrm{PFV}(a)$: median absolute deviation of the auto-correlation-based pitch estimate across active frames, normalised by the median.
- **Spectral centroid fluctuation** $\mathrm{CFV}(a)$: same as PFV but over the spectral centroid.
- **High-band ratio** $\mathrm{HBR}(a)$: ratio of energy in [1.5–6.5 kHz] over [60–1500 Hz].

The secondary-speaker probability is then computed by an empirically calibrated logistic mapping:

$$ P_\mathrm{sec}(a) = \sigma\big(\beta_0 + \beta_1 \mathrm{PFV} + \beta_2 \mathrm{CFV}\big) \cdot \mathrm{gate}(a) $$

where $\mathrm{gate}(a)$ is an activity gate that is near zero for silent buffers (built from a clipped function of the RMS in dBFS and VAR). Overlap and whisper probabilities follow analogous logistic mappings with their own coefficient set. Crucially, no decoder is ever called, so the linguistic content of the audio cannot be recovered.

### 4.3 Suspicion Index Fusion Model

For each session we summarise the metadata stream into a 12-dimensional feature vector $\mathbf{x} \in \mathbb{R}^{12}$:

$$
\mathbf{x} = [p^\max_\text{phone}, r^\text{dur}_\text{phone}, p^\max_\text{book}, n^\text{max}_\text{persons}, \theta^\max_g, r_\text{off}, |\phi_y|^\max, \bar\sigma_l^2, P^\max_\text{sec}, r_\text{sec}, P^\max_\text{ovl}, P^\max_\text{whisp}]
$$

The Suspicion Index $\mathrm{SI} \in [0, 1]$ is produced by a fusion model $f_\theta(\mathbf{x})$ identified empirically (Section 6.3) among five candidates:

1. **Weighted sum.** The original rule from the conference draft: $\mathrm{SI} = \alpha V + \beta A + \gamma B$ with $\alpha = 0.45, \beta = 0.30, \gamma = 0.25$ where $V, A, B$ are normalised visual, acoustic, and behavioural scores.
2. **L2-regularised logistic regression.**
3. **Random forest** [42] with 200 trees, max depth 8.
4. **Multi-layer perceptron** (32, 16) with StandardScaler preprocessing.
5. **Gaussian naive Bayes**.

Each model uses standard library defaults rather than an exhaustive hyperparameter search [47], so the comparison reflects out-of-the-box behaviour rather than per-model tuning effort. A session is *warned* if $\mathrm{SI} \geq 0.35$ and *flagged* if $\mathrm{SI} \geq 0.65$.

### 4.4 Data-Flow and Privacy Guarantees

The event log written to disk has the schema

```
{event_type: string,
 confidence_or_probability: float in [0,1],
 timestamp: ISO-8601,
 module: 'visual'|'acoustic'|'behaviour'|'fusion'}
```

We deliberately do not write bounding boxes, landmarks, or any time series that could be used to triangulate identity. The model itself is shipped without per-candidate state. Figure 2 illustrates how the per-modality risk components and the resulting Suspicion Index evolve over a simulated 60-second session, including the warn (0.35) and flag (0.65) thresholds.

[[FIG:si_timeline]]

## 5. Experimental Setup

### 5.1 Hardware

All experiments were conducted on a single Apple Mac Studio (model Mac16,9) equipped with an Apple M4 Max system-on-chip (14-core CPU, 32-core integrated GPU), 36 GB of unified memory, running macOS 26.2. The software stack is Python 3.9 with PyTorch [40] (exposing the Apple GPU through the Metal Performance Shaders backend), OpenCV [39] for image handling, and scikit-learn [41] for the fusion classifiers. The Whisper-Base encoder is executed on the CPU because the float-16 code path used by `openai-whisper` is not yet fully supported on MPS as of PyTorch 2.8. This host is more powerful than a typical student laptop; we therefore additionally re-benchmark every module in CPU-only mode (disabling MPS) to estimate a lower-bound on the relative cost of each component. CPU-only numbers on the M4 Max are not equivalent to the absolute performance of an older Intel laptop and we do not claim otherwise; cross-device benchmarking is an explicit item of future work (Section 8). The numbers reported in Section 6.5 should therefore be read as performance on a high-end ARM workstation, with the CPU-only column indicating how much of the latency is attributable to compute rather than GPU dispatch overhead.

### 5.2 Datasets

We use three modality datasets (D1–D3) and a derived fusion-evaluation set (D4), each chosen to maximise realism while respecting privacy constraints:

- **D1, Visual benchmark.** A 217-image subset of the public COCO val2017 dataset [14] stratified to include cell phone, book, laptop, person, TV, keyboard, and mouse instances (1,164 ground-truth annotations in total). COCO is published under the Creative Commons Attribution 4.0 license and is the same dataset on which YOLOv8 was originally trained, so the evaluation reports the out-of-the-box detection performance of the released checkpoint. Faces in COCO are not used as a verification target.
- **D2, Acoustic benchmark.** A real-speech benchmark constructed from LibriSpeech `dev-clean` [17], a 5.4-hour corpus of read English speech released under CC-BY 4.0. We sample utterances, mix them programmatically into five 5-second conditions (*silence, single-speaker, two-speakers, whispered, background-noise*) and use Whisper-Base in content-free mode to compute the secondary-speaker, overlap and whisper probabilities. Speaker identifiers from LibriSpeech are *not* used for any matching task; the corpus is treated as an anonymous source of natural speech.
- **D3, Behaviour benchmark.** Real MediaPipe FaceMesh landmarks extracted from ten CC-BY Pexels still portraits, to which programmatic head-pose and lip motion are applied to generate 400 behavioural trajectories (80 per class across five classes). The landmarks are real detector outputs; only the geometric descriptors of Section 4.1 are persisted, and the underlying images are discarded after landmark extraction. Using still portraits as real-face anchors, rather than recordings of identifiable individuals performing scripted motions, is itself a privacy-minimising choice.
- **D4, Session scenarios.** A fusion-evaluation set built by combining real modality outputs from D1–D3 into 300 simulated exam sessions (60 per class) covering five high-level classes (*normal, smartphone_use, off_screen_gaze, secondary_speaker, multi_anomaly*). The class distribution is therefore 60 *normal* (negative) versus 240 cheating (positive) sessions, a 1:4 imbalance that we account for with class-weighted training and by reporting precision, recall and PR-AUC alongside F1. Importantly, the features fed to the fusion model are real per-clip outputs from YOLOv8, Whisper-Base, and FaceMesh, not statistical distributions designed by the authors, so the fusion model is trained and evaluated on the same kind of evidence it would receive in deployment.

All datasets, generators, and trained-model state are reproducible from a fixed random seed (42).

### 5.3 Evaluation Metrics

For each binary task we report precision, recall, F1, ROC AUC and PR AUC [45], [46], and a 95% percentile bootstrap confidence interval on F1 (1,000 resamples) [43]. Multi-class detection (visual subsystem) uses per-class precision, recall, F1, and AP@0.5 with greedy IoU matching at threshold 0.5. Statistical significance of fusion-model differences is tested with McNemar's χ² (continuity-corrected) [44].

Latency is measured per-call with `time.perf_counter()` after a 3-call warmup; we report mean, standard deviation, median, p95, and p99 in milliseconds. Memory footprint is measured via `resource.getrusage`.

## 6. Results

### 6.1 Visual Subsystem (D1, real COCO data)

Table 2 reports per-class metrics on the 217-image subset. Figure 3 plots the per-class precision-recall curves; Figure 4 shows the TP / FP / FN breakdown.

**Table 2. YOLOv8-s on the COCO val2017 proctoring subset.**

| Class       | Ground truth | Predictions | Precision | Recall | F1    | AP@0.5 |
|-------------|--------------|-------------|-----------|--------|-------|--------|
| person      | 449          | 356         | 0.843     | 0.668  | 0.745 | 0.624  |
| tv          | 128          | 104         | 0.875     | 0.711  | 0.784 | 0.701  |
| laptop      | 91           | 92          | 0.772     | 0.780  | 0.776 | 0.759  |
| mouse       | 62           | 43          | 0.884     | 0.613  | 0.724 | 0.611  |
| keyboard    | 95           | 77          | 0.753     | 0.611  | 0.674 | 0.581  |
| cell phone  | 74           | 40          | 0.600     | 0.324  | 0.421 | 0.300  |
| book        | 265          | 34          | 0.529     | 0.068  | 0.120 | 0.056  |
| **Macro**   |              |             |           |        | **0.606** | **0.519** |

Per-frame YOLOv8-s inference latency is characterised separately in Section 6.5 (Table 7); on the deployed MPS configuration the median is 34.4 ms per frame.

The low book recall (6.8%) reflects a known limitation of the YOLOv8-s checkpoint used out-of-the-box on the COCO definition of *book* (which often labels small, partially-occluded book spines or distant shelf clutter). *Cell phone* recall (32.4%) reflects similar difficulty when the phone is held below desk level or partially occluded by a hand. Three mitigations are available to deployers and are recommended in our protocol: (i) fine-tune the visual checkpoint on a small in-domain set of proctoring scenes (an evening of annotation work, well within the budget of a single institution); (ii) raise the confidence threshold and rely on the *fusion model* to compensate (the leave-one-modality-out ablation in Section 6.4 shows that the random-forest fusion recovers most of its accuracy even when the visual modality is degraded); and (iii) combine the visual subsystem with the behavioural module so that sustained downward gaze plus low YOLOv8 phone confidence is also flagged as a candidate device-use event. The relevant operational decision for institutions is not whether YOLOv8-s alone is sufficient (it is not) but whether the fused pipeline is, and in our evaluation the answer is yes.

[[FIG:visual_pr_curves]]

[[FIG:visual_tp_fp]]

### 6.2 Acoustic Subsystem (D2)

Table 3 reports the three acoustic anomaly probabilities on the 400-clip benchmark. Figure 5 shows the ROC and PR curves; Figure 6 the per-class score distribution.

**Table 3. Whisper-Base in content-free mode on the LibriSpeech-derived acoustic benchmark (400 clips, real read speech).**

| Signal | ROC AUC | PR AUC | F1 @ best threshold | Best threshold |
|--------|---------|--------|----------------------|----------------|
| Secondary-speaker probability | 0.774 | 0.600 | 0.779 | 0.017 |
| Overlap probability | 0.771 | 0.601 | 0.770 | 0.004 |
| Whisper probability | 0.935 | 0.910 | 0.832 | 0.010 |

Mean per-buffer inference latency is 119.3 ms for a 5-second buffer (≈ 42× faster than real time on the M4 Max CPU). The mean secondary-speaker probability is near zero for silence (0.00) and background noise (0.01), and rises to 0.66 for single-speaker clips and 0.73 for two-speaker clips. The relatively high value for single-speaker clips (and the resulting moderate ROC AUC of 0.774) is the principal source of false positives: a single real speaker with natural pitch variation can resemble a second voice to a content-free detector that has no access to decoded speaker identity. We report this limitation openly; speaker-count estimation from non-decoded embeddings is intrinsically harder than from full diarization, which is the privacy cost of the content-free design. The whisper detector is cleaner: whispered clips score 0.60 on the whisper probability versus ≤ 0.02 for every other class, giving ROC AUC 0.935. These numbers, obtained on real read English speech, are a more realistic estimate of operational performance than a fully-synthetic benchmark would give.

[[FIG:acoustic_roc_pr]]

[[FIG:acoustic_per_class]]

### 6.3 Behavioural Cues (D3)

Table 4 reports the off-screen and lip-movement detection performance. Figure 7 shows the per-scenario distribution of the geometric descriptors.

**Table 4. MediaPipe FaceMesh behavioural cues on 400 trajectories generated from 10 real CC-BY face anchors (Pexels portraits, programmatic head/lip motion applied to real MediaPipe landmarks).**

| Task | ROC AUC | PR AUC | F1 @ best threshold | Best threshold |
|------|---------|--------|----------------------|----------------|
| Sustained off-screen gaze | 0.938 | 0.886 | 0.873 | 32.2° (combined) |
| Lip movement without keyboard | 0.663 | 0.314 | 0.393 | 10.1 (variance scale) |

Off-screen gaze is detected reliably (AUC ≈ 0.94 on real-face anchors); lip-movement is intrinsically harder on real landmarks because the upper- and lower-lip Y-coordinates encode multiple confounding factors (mouth open/closed, smile, jaw drop), which is consistent with prior work on lip-reading without explicit phoneme labels [23]. We do not claim the lip-movement cue is reliable on its own; the ablation in Section 6.4 quantifies its marginal contribution within the fusion model.

[[FIG:behaviour_distributions]]

### 6.4 Fusion Model Comparison and Ablation (D4)

Table 5 reports stratified 5-fold cross-validation results for the five fusion strategies on the 300 real-feature session scenarios. Figure 8 shows the cross-validated F1 with standard deviations; Figure 9 plots the pooled cross-validation ROC curves; Figure 10 gives the confusion matrix of the best model; Figure 11 shows the ablation. The ROC AUC and PR AUC columns of Table 5 are means of the five per-fold values, whereas Figure 9 plots a single ROC over the pooled out-of-fold predictions; the two therefore differ by a few thousandths (for example random forest 0.886 averaged versus 0.882 pooled), which is expected since the mean of fold-wise AUCs is not identical to the AUC of the pooled scores.

**Table 5. Fusion strategy comparison (5-fold stratified CV on 300 real-feature sessions). Each session's features are real outputs of YOLOv8-s on a COCO image, Whisper-Base on a LibriSpeech mix, and MediaPipe FaceMesh on a Pexels face anchor.**

| Model | F1 | Precision | Recall | ROC AUC | PR AUC | 95% CI on F1 |
|-------|----|-----------|--------|---------|--------|---------------|
| Weighted sum (paper baseline) | 0.899 ± 0.011 | 0.831 | 0.979 | 0.729 | 0.905 | [0.871, 0.924] |
| Logistic regression | 0.873 ± 0.020 | 0.961 | 0.800 | 0.912 | 0.976 | [0.837, 0.904] |
| **Random forest** | **0.905 ± 0.041** | 0.891 | 0.921 | 0.886 | 0.968 | **[0.876, 0.931]** |
| MLP (32, 16) | 0.861 ± 0.028 | 0.826 | 0.900 | 0.703 | 0.914 | [0.828, 0.890] |
| Gaussian naive Bayes | 0.803 ± 0.044 | 0.944 | 0.700 | 0.834 | 0.952 | [0.761, 0.845] |

The random forest attains the highest mean F1 (0.905), but the significance testing tells a more careful story that we report in full. McNemar's test (continuity-corrected, on the pooled cross-validation predictions) gives, for the random forest against each alternative: χ²(weighted sum) = 0.80 (p = 0.37), χ²(logistic) = 1.69 (p = 0.19), χ²(MLP) = 7.56 (p = 6.0 × 10⁻³), χ²(naive Bayes) = 17.50 (p = 2.9 × 10⁻⁵). In other words, the random forest is **statistically indistinguishable** from the rule-based weighted sum and from logistic regression, and is significantly better only than the MLP and the naive-Bayes variants. The practical reading is that the simple, transparent weighted-sum rule is already a strong baseline on this data, and that the learned random forest matches it while additionally providing interpretable feature importances; the strongest individual signals are *max_yaw_abs* (28.9% of total importance), *person_count_max* (16.7%), *max_gaze_dev* (12.5%), and *max_secondary_speaker_prob* (8.4%). We therefore present the random forest as the deployment default not because it is significantly more accurate than the rule, but because its feature attributions support auditability and it achieves higher precision (0.891 versus 0.831) than the weighted-sum rule, trading a small amount of recall (0.921 versus 0.979) for substantially fewer false alarms. We note that McNemar on pooled cross-validation folds slightly understates variance (the folds are not independent); the bootstrap confidence intervals in Table 5 should be read alongside it.

Table 6 reports two complementary ablations of the random-forest fusion model. The upper block is the leave-one-modality-out (LOMO) study (ΔF1 is relative to the all-modalities baseline); the lower block is the single-modality study (each model sees only one modality's features).

**Table 6. Ablation of the random-forest fusion model on the 300 real-feature sessions. Upper block: leave-one-modality-out. Lower block: single modality only.**

| Configuration | F1 | ROC AUC | ΔF1 vs all |
|---------------|----|---------|-----|
| All modalities | 0.905 ± 0.041 | 0.886 | n/a |
| without visual | 0.866 ± 0.018 | 0.793 | −0.040 |
| without acoustic | 0.901 ± 0.025 | 0.892 | −0.005 |
| without behaviour | 0.869 ± 0.031 | 0.732 | −0.036 |
| Visual only | 0.727 ± 0.063 | 0.684 | −0.179 |
| Acoustic only | 0.832 ± 0.011 | 0.578 | −0.074 |

Two observations follow. First, the visual and behavioural modalities carry the most decision-relevant information: removing either costs roughly 0.04 F1 and, in the case of behaviour, drops ROC AUC from 0.886 to 0.732. Removing the acoustic modality barely changes F1 (−0.005), confirming that on this dataset acoustic cues are the most redundant of the three for the binary cheating decision, although they are, by construction, the only modality whose signal does not depend on the candidate remaining in frame. Second, no single modality is sufficient on its own: the best single-modality model (acoustic only, F1 = 0.832) trails the fused model by 0.073 F1, and the visual-only model collapses to 0.727 because YOLOv8-s alone misses many partially-occluded phones (Section 6.1). The fusion model's value is therefore in combining complementary weak signals rather than in any single dominant cue.

[[FIG:fusion_comparison]]

[[FIG:fusion_roc]]

[[FIG:fusion_confusion]]

[[FIG:fusion_ablation]]

### 6.5 Latency, Memory, and Real-Time Feasibility

Table 7 reports module-level latency on the Mac Studio M4 Max host. The visual subsystem is benchmarked in two configurations: (i) GPU/MPS acceleration as actually deployed in the prototype and (ii) CPU-only execution. Figure 12 shows the per-module latency bar chart for both configurations.

**Table 7. Inference latency on the Apple M4 Max host (mean ± SD over 80 calls after a 3-call warm-up; median, p95 and p99 in ms).**

| Component | Configuration | Mean | SD | Median | p95 | p99 |
|-----------|---------------|------|----|--------|-----|-----|
| YOLOv8-s (640×480 frame) | MPS (GPU) | 37.7 | 17.0 | 34.4 | 44.8 | 123.0 |
| YOLOv8-s (640×480 frame) | CPU only | 47.1 | 4.9 | 47.2 | 54.9 | 59.4 |
| MediaPipe behaviour (per landmark set) | CPU | 0.027 | 0.008 | 0.026 | 0.032 | 0.068 |
| Whisper-Base encoder (5-s buffer) | CPU | 119.3 | 12.2 | 121.0 | 133.6 | 139.0 |
| Whisper-Base (per-frame equivalent at 5 FPS) | CPU | 23.9 | n/a | n/a | n/a | n/a |
| Fusion (random forest, single event) | CPU | 4.5 | 0.9 | 4.3 | 5.6 | 7.1 |
| **End-to-end frame budget** | MPS | **42.2** | n/a | n/a | n/a | n/a |
| **End-to-end frame budget** | CPU only | **51.6** | n/a | n/a | n/a | n/a |
| **Maximum sustainable FPS** | MPS | **23.7** | n/a | n/a | n/a | n/a |
| **Maximum sustainable FPS** | CPU only | **19.4** | n/a | n/a | n/a | n/a |

Peak resident memory is 1,102 MB. The pipeline sustains real-time monitoring at 23.7 FPS with MPS acceleration and 19.4 FPS in CPU-only mode, both comfortably above the 5–10 FPS that the event-based detection logic requires, since the behavioural thresholds operate on sustained multi-second windows rather than per-frame decisions. The end-to-end frame budget is bounded by the visual subsystem; the acoustic subsystem runs once per 5-second buffer and therefore contributes only 23.9 ms per logical frame when amortised over its window. We observed non-trivial run-to-run variance in the MPS latency (SD 17 ms, p99 123 ms), attributable to first-call kernel compilation and thermal scheduling on the fanless shared workstation; the median (34.4 ms) is the more representative central value, and a production deployment with a warm kernel cache would see latencies closer to the median than the mean.

[[FIG:latency]]

### 6.6 Robustness

Figure 13 shows the degradation curves. Visual class-presence recall on the COCO subset stays within 65–73% across gamma values from 0.4 (very dark) to 1.6 (very bright); the recall drops most sharply when the scale factor falls below 0.7 (≈ a more distant camera placement), reaching 61% at scale 0.5. For the acoustic module evaluated on LibriSpeech two-speaker mixes corrupted by additive white Gaussian noise, the mean secondary-speaker probability on positive clips stays strictly above the negative-clip mean across all SNRs from +30 dB down to +0 dB (0.69 vs 0.61 at SNR = 30 dB and 0.43 vs 0.40 at 0 dB), and the F1 at the calibrated operational threshold of 0.02 stays in the range 0.66–0.68. At SNR = −5 dB the positive and negative score distributions overlap (0.17 vs 0.20) and the detector should be considered unreliable. The recommended operating envelope is therefore gamma ∈ [0.6, 1.4], scale ≥ 0.7, and ambient SNR ≥ 0 dB. The fact that ROC AUC on noisy speech is close to chance even when F1 is well-defined indicates that the score distribution becomes bimodal under heavy noise (a few real two-speaker clips lose pitch fluctuation entirely while a few single-speaker clips are perturbed enough to exceed the threshold) so we recommend reading the F1 column rather than AUC under noisy conditions; the divergence between ROC and PR behaviour under class skew is itself well documented [46].

[[FIG:robustness]]

### 6.7 Adversarial Privacy Verification

Figure 14 summarises the adversarial study. We use 30 real LibriSpeech `dev-clean` speakers as candidates, with 10 utterances per speaker (300 sessions in total). Each attacker is a random forest trained to identify the speaker (30-way, one-vs-rest macro ROC AUC under 4-fold stratified cross-validation); both attackers see the same audio, rendered into their respective representations.

- A raw-audio attacker that extracts mel-frequency cepstral coefficients from the raw 16 kHz buffer (the kind of artefact a conventional proctoring system would retain) achieves a re-identification AUC of **0.994 ± 0.006**, confirming that raw recordings of real speakers allow near-perfect identification.
- A metadata attacker restricted to the *real* acoustic event probabilities our detector emits for the same buffer (secondary-speaker, overlap and whisper probability) still achieves **0.816 ± 0.025**, well above the 0.5 chance level. This is an important negative result: a content-free detector output is not automatically identity-free, because the probabilities are themselves derived from speaker-dependent acoustic statistics such as pitch and formant variability.

To close this gap we release the acoustic metadata through an epsilon-differentially-private mechanism: each emitted probability in [0, 1] is perturbed by Laplace noise of scale 1/epsilon and clamped, giving epsilon-differential privacy per released value. Sweeping the budget traces the privacy-utility curve in Figure 14(b): at epsilon = 4 per value (epsilon = 12 for the three-value acoustic event) the re-identification AUC falls to **0.547 ± 0.029**, close to chance, while the secondary-speaker detection AUC of Section 6.2 degrades from 0.774 to 0.625; tightening the budget to epsilon = 1 per value drives re-identification to 0.500 but lowers detection to 0.546. The mechanism therefore lets a deployer dial the operating point along a measured trade-off rather than assert privacy by construction.

This grounds the privacy claim in measurable evidence rather than design assertions. The honest reading is that the undefended metadata is privacy-reducing relative to raw audio (0.816 versus 0.994) but not privacy-preserving on its own, and that a differentially-private emission layer is required to reach near-chance re-identifiability, at a quantified cost in detection power. The verification covers the acoustic channel only; LibriSpeech is audio-only, so the visual and behavioural metadata are not exercised here, and re-identifiability of the geometric descriptors is left to future work (Section 8).

[[FIG:privacy]]

## 7. Discussion

### 7.1 Compliance Alignment

We map our design choices onto the relevant regulatory requirements below. We stress that this is an engineering argument for alignment, not a legal opinion; a binding determination of whether a given deployment processes "biometric data for the purpose of unique identification" rests with the competent data-protection authority.

- *GDPR Article 9 / Law 09-08 (sensitive data)*: no biometric template, face embedding, or voiceprint is computed or stored, and the metadata is not used to identify the candidate, so the system does not process biometric data for unique identification in the sense of GDPR Article 4(14). Section 6.7 nonetheless shows that the raw emitted probabilities carry residual speaker-discriminative information; the differentially-private emission layer bounds that leakage to near chance, which strengthens rather than establishes the Article 4(14) argument.
- *GDPR Article 25*: data protection by design and by default is implemented at the architectural level (no decoder; volatile buffers; metadata-only persistence).
- *GDPR Article 22*: the system does not produce solely automated decisions with legal or similarly significant effect; it generates warnings and flags for human review, and the final integrity decision remains with the examiner.
- *Law 09-08 proportionality*: the volume and granularity of retained data is bounded by the metadata schema in Section 4.4, supporting the purpose-limitation and proportionality obligations that the CNDP guidance to data controllers [6] elaborates.

### 7.2 Comparison to Prior Systems

The multi-modal pipeline of Masud et al. [12] reports an aggregate accuracy of 97.7% but operates with biometric face templates and raw-video retention; our fusion model attains an F1 of 0.905 on real-modality features without retaining either, which is the relevant operational comparison once biometric-free operation is required. Compared with the YOLO-only pipeline of Singh et al. [10], our fusion approach adds acoustic and behavioural redundancy that compensates for the documented failure modes of YOLOv8-s on small or partially-occluded smartphones and books (Section 6.1). None of the surveyed systems quantify privacy with an adversarial test; the present work provides such a measurement on 30 real LibriSpeech speakers, complementing the design-time privacy guarantees of federated learning [25], [26] and differentially private gaze [27], [28].

### 7.3 Threats to Validity

**Internal validity.** Each subsystem is evaluated on real data of its own modality (real COCO images for object detection, real LibriSpeech speech for the acoustic encoder, and real MediaPipe FaceMesh landmarks for the behavioural cues) but the *combination* of those signals into full exam sessions is assembled programmatically rather than captured from genuine cheating events, because recording real students cheating would itself raise the ethical and biometric-processing concerns the framework is designed to avoid. The fusion features are nonetheless real per-clip model outputs, not author-designed distributions; the residual risk is that the *co-occurrence structure* of cues in real exams differs from our assembled sessions. We bound this with stratified cross-validation and bootstrap intervals, and we flag a live human study as the primary item of future work (Section 8).

**External validity.** Hardware variability across student devices is not exhaustively covered: all numbers come from a single high-end Apple M4 Max host, and the CPU-only column is a within-host control rather than a true low-end-device measurement. The robustness study addresses lighting, scale, and SNR; future work will expand to webcam resolution, motion blur, codec compression, and a multi-device latency benchmark.

**Construct validity.** The Suspicion Index thresholds (0.35, 0.65) and the per-modality operating thresholds in Table 3 are calibrated on the same evaluation data on which performance is reported; the per-modality F1 values are therefore in-sample optima and should be read as upper bounds. Deployment in a different cultural or pedagogical context will require recalibration on held-out data, ideally via the examiner-in-the-loop procedure of Section 8.

**Conclusion validity.** Statistical significance was assessed with McNemar's test (a paired non-parametric test for binary classifiers) and 1,000-sample percentile bootstrap CIs over the pooled cross-validation predictions. Because the cross-validation folds are not mutually independent, McNemar on the pooled predictions slightly understates variance; we therefore treat the bootstrap intervals as the primary uncertainty estimate and avoid over-interpreting marginal p-values. Under this reading, the random forest's advantage over the weighted-sum and logistic baselines is not established, while its advantage over the MLP and naive-Bayes variants is.

## 8. Limitations and Future Work

The current prototype has five primary limitations:

1. **No human user study yet.** The visual evaluation uses real COCO data, the acoustic evaluation uses real LibriSpeech utterances, and the behavioural evaluation uses real MediaPipe FaceMesh trajectories computed from real face photographs. However, the fusion model and threshold calibration have not yet been validated on real recordings of students taking actual exams. A multi-site pilot study with informed consent and institutional ethics review is in preparation and is intentionally scoped as future work rather than embedded in the present feasibility-and-privacy-architecture paper. Federated training across multiple institutions [25], [26], coupled with the secure-aggregation primitives, will allow the model to be refined on decentralised real-exam data without aggregating biometric content.
2. **Single high-end host for benchmarks.** All numbers in this paper come from a single Apple M4 Max host (Mac Studio, 36 GB unified memory). Although we re-benchmark in CPU-only mode to expose how much of the latency budget is driven by compute versus dispatch, this is not a substitute for testing on a 5-year-old Intel laptop, a Raspberry Pi 5, or a typical Chromebook. A multi-device benchmark (Intel i5, AMD Ryzen 5, Snapdragon X, Apple M1) is part of the follow-up work.
3. **Manual threshold setting.** The 0.35/0.65 fusion thresholds and the per-modality thresholds are calibrated in-sample. A Bayesian active-learning loop driven by examiner feedback would adapt thresholds per cohort on held-out data, with formal guarantees on false-positive rate.
4. **No integration with browser telemetry.** The framework currently ignores window-focus events and clipboard signals, both of which can be processed without entering the domain of behavioural biometrics. Future versions will include non-identifying telemetry where allowed.
5. **Single-camera assumption.** Multi-angle setups (e.g. wide-angle desk view) would substantially improve coverage of below-desk smartphone use, the dominant failure mode of the visual subsystem. We plan to integrate a second-camera variant with the same privacy budget.

In addition, three research directions follow naturally:

- Differentially private gaze and lip-movement signals to bound information leakage to a configurable ε budget.
- A tighter privacy-utility analysis of the differentially-private metadata layer, including formal (epsilon, delta) accounting across a full session and extension of the mechanism to the visual and behavioural metadata channels.
- A longitudinal study across at least three Moroccan universities to validate cross-cohort generalisation and gather examiner feedback for the active-learning loop.

## 9. Conclusion

This paper presented a privacy-preserving multi-modal proctoring framework that integrates YOLOv8-s object detection, MediaPipe FaceMesh behavioural cues, and a content-free Whisper-Base encoder, combined through a learned Suspicion Index Fusion Model. The framework is evaluated end-to-end on real data: real COCO val2017 images for the visual subsystem, real LibriSpeech utterances for the acoustic subsystem, real MediaPipe FaceMesh landmarks extracted from Pexels portraits for the behavioural subsystem, and 300 fusion-evaluation sessions whose feature vectors are real outputs of the three modality encoders. Statistical significance testing, ablation studies, latency benchmarks, and an adversarial privacy verification on 30 real LibriSpeech speakers complete the protocol. Random forest attained the highest mean fusion F1 (0.905 ± 0.041, 95% bootstrap CI [0.876, 0.931]); its margin over the transparent weighted-sum rule (F1 = 0.899) is not statistically significant, which we report openly as evidence that a simple, auditable rule is already competitive on this data. The framework runs at 23.7 FPS with MPS GPU acceleration and 19.4 FPS in CPU-only mode on the Apple M4 Max host (Mac Studio, 36 GB unified memory) with a 1,102 MB peak memory footprint. An adversarial verification shows the raw-audio analogue allows near-perfect re-identification (AUC = 0.994) while the emitted acoustic metadata, though not identity-free on its own (AUC = 0.816), can be released through a differentially-private mechanism that drives re-identification to near chance (AUC = 0.547) at a measured cost in detection accuracy. The combination of biometric-free operation, measured privacy evidence, real-time multi-modal monitoring, and explicit GDPR/CNDP alignment is, in our reading of the literature, not offered together by existing systems. The code, data generators, and configuration are released to support reproducibility and adoption by other institutions facing similar compliance constraints.

## Data and Code Availability

All source code, evaluation scripts, deterministic dataset specifications and
the manuscript in editable form are released under the MIT license at
`https://github.com/aliradid/proctoring-privacy`. The repository contains a
`bootstrap.sh` script that re-downloads every external asset used in the
experiments (the COCO val2017 annotations and the proctoring-relevant image
subset, the LibriSpeech `dev-clean` corpus, the MediaPipe FaceLandmarker
checkpoint, and ten CC-BY Pexels portraits used as real-face anchors), and a
`reproduce.sh` script that re-runs the full pipeline end-to-end. The headline
numbers reported in this paper match the output of `reproduce.sh` on an
Apple M4 Max host within the per-run variability documented in the project's
`REPRODUCE.md`.

## Acknowledgements

The authors thank the Department of Mathematics and Computer Science at Hassan II University for hardware support, and the anonymous reviewers for their constructive feedback. During preparation of this manuscript, the authors made limited use of language-support tools for grammar and clarity editing of the English prose. The scientific content, experimental design, data analysis, and conclusions are the authors' own.

## References

[1] S. Coghlan, T. Miller, and J. Paterson, "Good proctor or 'big brother'? Ethics of online exam supervision technologies," *Philosophy & Technology*, vol. 34, no. 4, pp. 1581–1606, 2021, doi: 10.1007/s13347-021-00476-1.

[2] N. Selwyn, C. O'Neill, G. Smith, M. Andrejevic, and X. Gu, "A necessary evil? The rise of online exam proctoring in Australian universities," *Media International Australia*, vol. 186, no. 1, pp. 149–164, 2023, doi: 10.1177/1329878X211005862.

[3] D. R. Yoder-Himes, A. Asif, K. Kinney, T. J. Brandt, R. E. Cecil, P. R. Himes, C. Hying, A. P. Lavin, J. M. Yoder-Himes, A. Burdine, et al., "Racial, skin tone, and sex disparities in automated proctoring software," *Frontiers in Education*, vol. 7, p. 881449, 2022, doi: 10.3389/feduc.2022.881449.

[4] Regulation (EU) 2016/679 of the European Parliament and of the Council of 27 April 2016 on the protection of natural persons with regard to the processing of personal data and on the free movement of such data (General Data Protection Regulation), *Official Journal of the European Union*, L119, pp. 1–88, May 2016.

[5] Royaume du Maroc, *Loi n° 09-08 relative à la protection des personnes physiques à l'égard du traitement des données à caractère personnel*, Bulletin Officiel n°5714, 2009.

[6] Commission Nationale de Contrôle de la Protection des Données à Caractère Personnel (CNDP), "Conditions et obligations du responsable de traitement," guidance note, CNDP, Rabat, Morocco, 2022. [Online]. Available: https://www.cndp.ma/ (accessed 2026-01-10).

[7] European Data Protection Board, "Guidelines 05/2020 on consent under Regulation 2016/679," May 2020.

[8] Y. Atoum, L. Chen, A. X. Liu, S. D. H. Hsu, and X. Liu, "Automated online exam proctoring," *IEEE Transactions on Multimedia*, vol. 19, no. 7, pp. 1609–1624, 2017, doi: 10.1109/TMM.2017.2656064.

[9] A. Nigam, R. Pasricha, T. Singh, and P. Churi, "A systematic review on AI-based proctoring systems: past, present and future," *Education and Information Technologies*, vol. 26, no. 5, pp. 6421–6445, 2021, doi: 10.1007/s10639-021-10597-x.

[10] T. Singh, R. R. Nair, T. Babu, and P. Duraisamy, "Enhancing academic integrity in online assessments: introducing an effective online exam proctoring model using YOLO," *Procedia Computer Science*, vol. 235, pp. 1399–1408, 2024, doi: 10.1016/j.procs.2024.04.131.

[11] V. A. S. Vishal, V. M. Vishal, M. Muthuvel, and E. Vijayaram, "AI-based proctoring system for cheating prevention in online exams," *International Research Journal of Education and Technology*, vol. 7, no. 3, pp. 1636–1645, Mar. 2025.

[12] M. M. Masud, K. Hayawi, S. S. Mathew, T. Michael, and M. El Barachi, "Smart online exam proctoring assist for cheating detection," in *Advanced Data Mining and Applications (ADMA 2021)*, vol. 13087 of LNAI, Cham: Springer, 2022, pp. 118–132, doi: 10.1007/978-3-030-95405-5_9.

[13] G. Jocher, A. Chaurasia, and J. Qiu, "Ultralytics YOLOv8," Ultralytics, software documentation, 2023. [Online]. Available: https://docs.ultralytics.com/

[14] T.-Y. Lin, M. Maire, S. Belongie, J. Hays, P. Perona, D. Ramanan, P. Dollár, and C. L. Zitnick, "Microsoft COCO: common objects in context," in *Proc. European Conference on Computer Vision (ECCV)*, vol. 8693 of LNCS, 2014, pp. 740–755, doi: 10.1007/978-3-319-10602-1_48.

[15] C. Lugaresi, J. Tang, H. Nash, C. McClanahan, E. Uboweja, M. Hays, F. Zhang, C.-L. Chang, M. G. Yong, J. Lee, W.-T. Chang, W. Hua, M. Georg, and M. Grundmann, "MediaPipe: a framework for building perception pipelines," *arXiv preprint*, arXiv:1906.08172, 2019.

[16] A. Radford, J. W. Kim, T. Xu, G. Brockman, C. McLeavey, and I. Sutskever, "Robust speech recognition via large-scale weak supervision," in *Proc. 40th International Conference on Machine Learning (ICML)*, PMLR vol. 202, 2023, pp. 28492–28518.

[17] V. Panayotov, G. Chen, D. Povey, and S. Khudanpur, "LibriSpeech: an ASR corpus based on public domain audio books," in *Proc. ICASSP*, 2015, pp. 5206–5210, doi: 10.1109/ICASSP.2015.7178964.

[18] J. Carletta, S. Ashby, S. Bourban, M. Flynn, M. Guillemot, T. Hain, J. Kadlec, V. Karaiskos, W. Kraaij, M. Kronenthal, et al., "The AMI Meeting Corpus: a pre-announcement," in *Machine Learning for Multimodal Interaction (MLMI 2005)*, LNCS vol. 3869, Springer, 2006, pp. 28–39, doi: 10.1007/11677482_3.

[19] J. S. Chung, A. Nagrani, and A. Zisserman, "VoxCeleb2: deep speaker recognition," in *Proc. Interspeech*, 2018, pp. 1086–1090.

[20] J. Jung, H.-S. Heo, Y. Kwon, J. S. Chung, and B.-J. Lee, "Three-class overlapped speech detection using a convolutional recurrent neural network," in *Proc. Interspeech*, 2021, pp. 3086–3090, doi: 10.21437/Interspeech.2021-149.

[21] D. Snyder, D. Garcia-Romero, G. Sell, D. Povey, and S. Khudanpur, "X-vectors: robust DNN embeddings for speaker recognition," in *Proc. ICASSP*, 2018, pp. 5329–5333, doi: 10.1109/ICASSP.2018.8461375.

[22] G. Saon, G. Kurata, T. Sercu, K. Audhkhasi, S. Thomas, D. Dimitriadis, X. Cui, B. Ramabhadran, M. Picheny, L.-L. Lim, B. Roomi, and P. Hall, "English conversational telephone speech recognition by humans and machines," in *Proc. Interspeech*, 2017, pp. 132–136, doi: 10.21437/Interspeech.2017-405.

[23] J. S. Chung and A. Zisserman, "Lip reading in the wild," in *Proc. Asian Conference on Computer Vision (ACCV)*, LNCS vol. 10112, Springer, 2016, pp. 87–103, doi: 10.1007/978-3-319-54184-6_6.

[24] C. Dwork and A. Roth, "The algorithmic foundations of differential privacy," *Foundations and Trends in Theoretical Computer Science*, vol. 9, nos. 3–4, pp. 211–407, 2014, doi: 10.1561/0400000042.

[25] H. B. McMahan, E. Moore, D. Ramage, S. Hampson, and B. Agüera y Arcas, "Communication-efficient learning of deep networks from decentralized data," in *Proc. 20th International Conference on Artificial Intelligence and Statistics (AISTATS)*, PMLR vol. 54, 2017, pp. 1273–1282.

[26] K. Bonawitz, V. Ivanov, B. Kreuter, A. Marcedone, H. B. McMahan, S. Patel, D. Ramage, A. Segal, and K. Seth, "Practical secure aggregation for privacy-preserving machine learning," in *Proc. 2017 ACM SIGSAC Conference on Computer and Communications Security (CCS)*, 2017, pp. 1175–1191, doi: 10.1145/3133956.3133982.

[27] L. Du, J. Jia, X. Zhang, and G. Lan, "PrivateGaze: preserving user privacy in black-box mobile gaze tracking services," *Proceedings of the ACM on Interactive, Mobile, Wearable and Ubiquitous Technologies (IMWUT)*, vol. 8, no. 3, art. 99, 2024, doi: 10.1145/3678595.

[28] E. Bozkir, O. Günlü, W. Fuhl, R. F. Schaefer, and E. Kasneci, "Differential privacy for eye tracking with temporal correlations," *PLOS ONE*, vol. 16, no. 8, p. e0255979, 2021, doi: 10.1371/journal.pone.0255979.

[29] J. Steil, I. Hagestedt, M. X. Huang, and A. Bulling, "Privacy-aware eye tracking using differential privacy," in *Proc. 11th ACM Symposium on Eye Tracking Research and Applications (ETRA)*, 2019, art. 27, pp. 1–9, doi: 10.1145/3314111.3319915.

[30] H. Crawford, K. Renaud, and T. Storer, "A framework for continuous, transparent mobile device authentication," *Computers & Security*, vol. 39, part B, pp. 127–136, 2013, doi: 10.1016/j.cose.2013.05.005.

[31] N. Sae-Bae, K. Ahmed, K. Isbister, and N. Memon, "Biometric-rich gestures: a novel approach to authentication on multi-touch devices," in *Proc. ACM CHI Conference*, 2012, pp. 977–986, doi: 10.1145/2207676.2208543.

[32] Tribunal administratif de Montreuil, decision no. 2216571 (Univ. Paris 8 / TestWe e-proctoring), 20 September 2023. [Online]. Available: https://www.dalloz.fr/

[33] J. Redmon, S. Divvala, R. Girshick, and A. Farhadi, "You only look once: unified, real-time object detection," in *Proc. IEEE CVPR*, 2016, pp. 779–788, doi: 10.1109/CVPR.2016.91.

[34] J. Redmon and A. Farhadi, "YOLOv3: an incremental improvement," *arXiv preprint*, arXiv:1804.02767, 2018.

[35] Y. Kortli, M. Jridi, A. Al Falou, and M. Atri, "Face recognition systems: a survey," *Sensors*, vol. 20, no. 2, p. 342, 2020, doi: 10.3390/s20020342.

[36] A. Narayanan and V. Shmatikov, "Robust de-anonymization of large sparse datasets," in *Proc. 2008 IEEE Symposium on Security and Privacy (S&P)*, 2008, pp. 111–125, doi: 10.1109/SP.2008.33.

[37] L. Sweeney, "k-Anonymity: a model for protecting privacy," *International Journal of Uncertainty, Fuzziness and Knowledge-Based Systems*, vol. 10, no. 5, pp. 557–570, 2002, doi: 10.1142/S0218488502001648.

[38] C. Dwork, F. McSherry, K. Nissim, and A. Smith, "Calibrating noise to sensitivity in private data analysis," in *Proc. Theory of Cryptography Conference (TCC)*, LNCS vol. 3876, Springer, 2006, pp. 265–284, doi: 10.1007/11681878_14.

[39] G. Bradski, "The OpenCV library," *Dr. Dobb's Journal of Software Tools*, vol. 25, no. 11, pp. 120–125, 2000.

[40] A. Paszke, S. Gross, F. Massa, A. Lerer, J. Bradbury, G. Chanan, et al., "PyTorch: an imperative style, high-performance deep learning library," in *Advances in Neural Information Processing Systems (NeurIPS)*, 2019, pp. 8024–8035.

[41] F. Pedregosa, G. Varoquaux, A. Gramfort, V. Michel, B. Thirion, et al., "Scikit-learn: machine learning in Python," *Journal of Machine Learning Research*, vol. 12, pp. 2825–2830, 2011.

[42] L. Breiman, "Random forests," *Machine Learning*, vol. 45, no. 1, pp. 5–32, 2001, doi: 10.1023/A:1010933404324.

[43] B. Efron and R. Tibshirani, *An Introduction to the Bootstrap*. New York, NY, USA: Chapman & Hall/CRC, 1993.

[44] Q. McNemar, "Note on the sampling error of the difference between correlated proportions or percentages," *Psychometrika*, vol. 12, no. 2, pp. 153–157, 1947, doi: 10.1007/BF02295996.

[45] T. Fawcett, "An introduction to ROC analysis," *Pattern Recognition Letters*, vol. 27, no. 8, pp. 861–874, 2006, doi: 10.1016/j.patrec.2005.10.010.

[46] J. Davis and M. Goadrich, "The relationship between precision-recall and ROC curves," in *Proc. 23rd International Conference on Machine Learning (ICML)*, 2006, pp. 233–240, doi: 10.1145/1143844.1143874.

[47] J. Bergstra and Y. Bengio, "Random search for hyper-parameter optimization," *Journal of Machine Learning Research*, vol. 13, pp. 281–305, 2012.

[48] C. Lugaresi et al., "MediaPipe Face Mesh / Face Landmarker," MediaPipe Solutions, Google Inc., software documentation, 2023. [Online]. Available: https://developers.google.com/edge/mediapipe/solutions/vision/face_landmarker

[49] H. Caesar, J. Uijlings, and V. Ferrari, "COCO-Stuff: thing and stuff classes in context," in *Proc. IEEE CVPR*, 2018, pp. 1209–1218, doi: 10.1109/CVPR.2018.00132.
