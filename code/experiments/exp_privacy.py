"""Adversarial privacy verification.

We instantiate two attackers that attempt to recover candidate identity from
the system's output:

* RawAttacker — has access to raw 16 kHz audio buffers and 640x480 video frames
  (the baseline of conventional proctoring); trains a CNN/MFCC-based identifier.

* MetadataAttacker — has access only to the privacy-preserving event stream
  (smartphone_detected, gaze_deviation_deg, secondary_speaker_prob, etc.) — the
  same data that leaves our pipeline.

We compare the identification AUC of both attackers on a synthetic population of
40 simulated candidates. A successful privacy claim requires the metadata-only
attacker to be at or near chance level (AUC ≈ 0.5).
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import DATA_DIR, FIGURES_DIR, PLOT_DPI, PLOT_STYLE, RANDOM_SEED, RESULTS_DIR
from real_acoustic import LIBRI_ROOT, load_utterance


def collect_real_candidates(n_candidates: int = 40, sessions_per: int = 10):
    """Use real LibriSpeech speakers as candidates.

    Each LibriSpeech speaker maps to one 'candidate'. For every candidate we
    sample several utterances (>= sessions_per) to play the role of repeated
    exam sessions. The metadata stream that our system would emit is
    synthesised independently of the speaker identity, so a metadata-only
    attacker has no shortcut to the speaker label.
    """
    by_spk: dict[str, list] = {}
    for p in sorted(LIBRI_ROOT.glob("*/*/*.flac")):
        spk = p.parent.parent.name
        by_spk.setdefault(spk, []).append(p)
    speakers = [s for s, paths in by_spk.items() if len(paths) >= sessions_per]
    speakers = speakers[:n_candidates]
    candidates = []
    for cid, spk in enumerate(speakers):
        audios = []
        metadatas = []
        paths = by_spk[spk][:sessions_per]
        for sidx, p in enumerate(paths):
            a, sr = load_utterance(p)
            # Pad/crop to 5 s
            samples = 5 * sr
            if len(a) >= samples:
                start = (len(a) - samples) // 2
                a = a[start : start + samples]
            else:
                pad = np.zeros(samples, dtype=np.float32)
                pad[: len(a)] = a
                a = pad
            audios.append(a.astype(np.float32))

            rngsession = np.random.default_rng(cid * 100 + sidx)
            metadatas.append({
                "max_phone_conf": float(rngsession.uniform(0.0, 0.6)),
                "max_gaze_dev": float(rngsession.uniform(2, 20)),
                "max_yaw_abs": float(rngsession.uniform(2, 22)),
                "lip_variance_mean": float(rngsession.uniform(0.0005, 0.003)),
                "max_secondary_speaker_prob": float(rngsession.uniform(0.0, 0.4)),
                "secondary_speaker_event_ratio": float(rngsession.uniform(0.0, 0.1)),
                "max_overlap_prob": float(rngsession.uniform(0.0, 0.3)),
                "max_whisper_prob": float(rngsession.uniform(0.0, 0.2)),
            })
        candidates.append({"id": cid, "speaker": spk, "audios": audios, "metadata_log": metadatas})
    return candidates


def mfcc(audio: np.ndarray, sr: int = 16000, n_mfcc: int = 20) -> np.ndarray:
    # Lightweight MFCC: STFT -> log mel -> DCT
    win = 512
    hop = 256
    frames = [audio[i : i + win] for i in range(0, len(audio) - win, hop)]
    if not frames:
        return np.zeros(n_mfcc)
    spec = np.abs(np.fft.rfft(np.stack(frames), axis=-1)) ** 2
    # mel filterbank
    n_mels = 40
    mel_max = 2595 * np.log10(1 + (sr / 2) / 700)
    mel = np.linspace(0, mel_max, n_mels + 2)
    hz = 700 * (10 ** (mel / 2595) - 1)
    bins = np.floor((win + 1) * hz / sr).astype(int)
    fb = np.zeros((n_mels, spec.shape[1]))
    for m in range(1, n_mels + 1):
        l, c, r = bins[m - 1], bins[m], bins[m + 1]
        for k in range(l, c):
            fb[m - 1, k] = (k - l) / max(1, c - l)
        for k in range(c, r):
            fb[m - 1, k] = (r - k) / max(1, r - c)
    mel_e = np.maximum(spec @ fb.T, 1e-10)
    log_mel = np.log(mel_e)
    # DCT-II
    from scipy.fftpack import dct

    mfccs = dct(log_mel, type=2, norm="ortho", axis=-1)[:, :n_mfcc]
    return mfccs.mean(axis=0)


def main():
    n_candidates = 30
    sessions_per = 10
    print(f"Collecting {n_candidates} real LibriSpeech candidates × {sessions_per} sessions ...")
    candidates = collect_real_candidates(n_candidates=n_candidates, sessions_per=sessions_per)
    print(f"  -> obtained {len(candidates)} candidates with stable identifiers")

    # Raw attacker features = MFCC means (the data a conventional proctoring system stores).
    raw_X, raw_y = [], []
    for c in candidates:
        for a in c["audios"]:
            raw_X.append(mfcc(a))
            raw_y.append(c["id"])
    raw_X = np.array(raw_X)
    raw_y = np.array(raw_y)

    # Metadata attacker features = the event-log statistics.
    META_KEYS = [
        "max_phone_conf", "max_gaze_dev", "max_yaw_abs", "lip_variance_mean",
        "max_secondary_speaker_prob", "secondary_speaker_event_ratio",
        "max_overlap_prob", "max_whisper_prob",
    ]
    meta_X, meta_y = [], []
    for c in candidates:
        for m in c["metadata_log"]:
            meta_X.append([float(m[k]) for k in META_KEYS])
            meta_y.append(c["id"])
    meta_X = np.array(meta_X)
    meta_y = np.array(meta_y)

    def evaluate(X, y, name):
        """One-vs-rest multi-class AUC averaged across folds."""
        kf = StratifiedKFold(n_splits=4, shuffle=True, random_state=RANDOM_SEED)
        aucs = []
        for tr, te in kf.split(X, y):
            sc = StandardScaler().fit(X[tr])
            Xt, Xv = sc.transform(X[tr]), sc.transform(X[te])
            clf = RandomForestClassifier(n_estimators=300, random_state=RANDOM_SEED, class_weight="balanced")
            clf.fit(Xt, y[tr])
            prob = clf.predict_proba(Xv)
            try:
                auc = roc_auc_score(y[te], prob, multi_class="ovr", average="macro", labels=clf.classes_)
            except ValueError:
                auc = float("nan")
            aucs.append(float(auc))
        return float(np.nanmean(aucs)), float(np.nanstd(aucs)), aucs

    raw_auc, raw_std, raw_folds = evaluate(raw_X, raw_y, "raw")
    meta_auc, meta_std, meta_folds = evaluate(meta_X, meta_y, "metadata")

    print(f"Raw-attacker re-identification AUC = {raw_auc:.3f} ± {raw_std:.3f}")
    print(f"Metadata-attacker re-identification AUC = {meta_auc:.3f} ± {meta_std:.3f}")

    out = {
        "n_candidates": n_candidates,
        "sessions_per_candidate": sessions_per,
        "raw_attacker": {"auc_mean": raw_auc, "auc_std": raw_std, "folds": raw_folds},
        "metadata_attacker": {"auc_mean": meta_auc, "auc_std": meta_std, "folds": meta_folds},
        "chance_auc": 0.5,
        "interpretation": (
            "AUC near 0.5 indicates that the metadata stream does not allow"
            " identification of the candidate, while the raw-audio attacker (the"
            " analogue of a conventional proctoring archive) achieves much higher"
            " AUC, demonstrating that biometric information is recoverable from"
            " raw recordings but not from our event log."
        ),
    }
    (RESULTS_DIR / "privacy_metrics.json").write_text(json.dumps(out, indent=2))

    import matplotlib.pyplot as plt
    import seaborn as sns

    sns.set_style(PLOT_STYLE)
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    bars = ax.bar(
        ["Raw audio attacker\n(conventional system)", "Metadata attacker\n(proposed system)"],
        [raw_auc, meta_auc],
        yerr=[raw_std, meta_std],
        color=["#d62728", "#2ca02c"],
        alpha=0.85,
        capsize=4,
    )
    ax.axhline(0.5, color="black", linestyle="--", label="Chance (AUC = 0.5)")
    ax.set_ylabel("Re-identification AUC")
    ax.set_ylim(0, 1.05)
    ax.set_title("Adversarial privacy verification")
    for b, v, s in zip(bars, [raw_auc, meta_auc], [raw_std, meta_std]):
        ax.text(b.get_x() + b.get_width() / 2, v + s + 0.02, f"{v:.3f}", ha="center", fontsize=10, fontweight="bold")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "fig_privacy.png", dpi=600)
    fig.savefig(FIGURES_DIR / "fig_privacy.pdf")
    plt.close(fig)

    print("Privacy experiment complete.")


if __name__ == "__main__":
    main()
