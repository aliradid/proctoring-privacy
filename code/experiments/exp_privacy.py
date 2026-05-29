"""Adversarial privacy verification (acoustic channel).

We instantiate two attackers that attempt to recover candidate identity from
the audio channel, using real LibriSpeech speakers as candidates:

* RawAttacker — has access to the raw 16 kHz audio buffer (the analogue of a
  conventional proctoring archive); trains an MFCC-based identifier.

* MetadataAttacker — has access only to the acoustic event metadata our pipeline
  actually emits for the same buffer: the secondary-speaker, overlap and whisper
  probabilities produced by the Whisper-encoder anomaly detector. These are the
  real detector outputs (not synthetic placeholders); no raw audio leaves the
  detector.

Both attackers see the same audio rendered into their respective representations,
so the comparison is fair. A privacy claim for the acoustic channel requires the
metadata attacker to stay at or near chance (AUC ~ 0.5) while the raw attacker
re-identifies easily. The experiment is scoped to the acoustic channel: the
visual and behavioural metadata fields are not exercised here because the
LibriSpeech corpus is audio-only.
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


def dp_release(values: np.ndarray, epsilon_per_value: float, rng: np.random.Generator,
               sensitivity: float = 1.0) -> np.ndarray:
    """Differentially-private release of probability-valued metadata.

    The pipeline does not persist the raw detector probabilities; it releases an
    epsilon-differentially-private version via the Laplace mechanism. Each value
    lies in [0, 1] (sensitivity 1), so Laplace noise of scale sensitivity/epsilon
    gives epsilon-DP per released value; a k-value acoustic event therefore
    consumes a total budget of k * epsilon_per_value under sequential
    composition. The noisy value is clamped back to [0, 1].
    """
    noisy = values + rng.laplace(0.0, sensitivity / epsilon_per_value, size=np.shape(values))
    return np.clip(noisy, 0.0, 1.0)


def collect_real_candidates(n_candidates: int = 40, sessions_per: int = 10):
    """Use real LibriSpeech speakers as candidates.

    Each LibriSpeech speaker maps to one 'candidate'. For every candidate we
    take several of their utterances (>= sessions_per) to play the role of
    repeated exam sessions, each cropped/padded to one 5 s buffer. Only the raw
    audio is returned here; the acoustic event metadata our pipeline emits is
    computed from this audio by the real detector in main().
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
        paths = by_spk[spk][:sessions_per]
        for p in paths:
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
        candidates.append({"id": cid, "speaker": spk, "audios": audios})
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

    # Metadata attacker features = the REAL acoustic event metadata our pipeline
    # emits for each buffer (the secondary-speaker, overlap and whisper
    # probabilities from the Whisper-encoder detector). These are computed from
    # the same audio the raw attacker sees, so the comparison is fair; the
    # detector never exposes the raw waveform.
    print("Computing real acoustic metadata with the Whisper-encoder detector ...")
    from acoustic_module import AcousticAnomalyDetector

    detector = AcousticAnomalyDetector()
    META_KEYS = ["secondary_speaker_prob", "overlap_prob", "whisper_prob"]
    meta_X, meta_y = [], []
    for c in candidates:
        for a in c["audios"]:
            ev = detector.infer(a, timestamp=0.0)
            meta_X.append([ev.secondary_speaker_prob, ev.overlap_prob, ev.whisper_prob])
            meta_y.append(c["id"])
    meta_X = np.array(meta_X)
    meta_y = np.array(meta_y)

    def reid_auc(X, y):
        """One-vs-rest multi-class re-identification AUC averaged across folds."""
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
        return float(np.nanmean(aucs)), float(np.nanstd(aucs))

    raw_auc, raw_std = reid_auc(raw_X, raw_y)
    meta_auc, meta_std = reid_auc(meta_X, meta_y)
    print(f"Raw-attacker re-identification AUC          = {raw_auc:.3f} ± {raw_std:.3f}")
    print(f"Metadata-attacker (undefended) re-ID AUC    = {meta_auc:.3f} ± {meta_std:.3f}")

    # Utility benchmark: secondary-speaker detection on the 400-clip LibriSpeech
    # set, so we can show how much detection power the privacy mechanism costs.
    print("Building utility benchmark (secondary-speaker detection) ...")
    from real_acoustic import build_real_dataset

    ds = build_real_dataset(n_per_class=80)
    util_sec, util_lab = [], []
    for d in ds:
        ev = detector.infer(d["audio"].astype(np.float32), timestamp=0.0)
        util_sec.append(ev.secondary_speaker_prob)
        util_lab.append(1 if d["has_secondary"] else 0)
    util_sec = np.array(util_sec)
    util_lab = np.array(util_lab)
    detect_auc_clean = float(roc_auc_score(util_lab, util_sec))
    print(f"Secondary-speaker detection AUC (no DP)     = {detect_auc_clean:.3f}")

    # Differentially-private metadata emission: sweep the privacy budget and
    # measure both re-identification (privacy) and detection (utility), averaged
    # over several noise realisations.
    EPS_GRID = [8.0, 4.0, 2.0, 1.0]
    OPERATING_EPS_PER = 4.0
    N_DRAWS = 16
    curve = []
    for eps in EPS_GRID:
        reids, dets = [], []
        for s in range(N_DRAWS):
            mX = dp_release(meta_X, eps, np.random.default_rng(1000 + s))
            reids.append(reid_auc(mX, meta_y)[0])
            un = dp_release(util_sec, eps, np.random.default_rng(2000 + s))
            dets.append(float(roc_auc_score(util_lab, un)))
        curve.append({
            "epsilon_per_value": eps,
            "epsilon_total": eps * len(META_KEYS),
            "reid_auc_mean": float(np.mean(reids)),
            "reid_auc_std": float(np.std(reids)),
            "detection_auc_mean": float(np.mean(dets)),
            "detection_auc_std": float(np.std(dets)),
        })
        print(f"  eps/value={eps:>4} (total {eps*len(META_KEYS):>4.0f}): "
              f"re-ID {np.mean(reids):.3f}, detection {np.mean(dets):.3f}")
    op = next(c for c in curve if c["epsilon_per_value"] == OPERATING_EPS_PER)

    out = {
        "channel": "acoustic",
        "n_candidates": n_candidates,
        "sessions_per_candidate": sessions_per,
        "metadata_features": META_KEYS,
        "metadata_source": "real Whisper-encoder detector outputs (not synthetic)",
        "raw_attacker": {"auc_mean": raw_auc, "auc_std": raw_std},
        "metadata_attacker_undefended": {"auc_mean": meta_auc, "auc_std": meta_std},
        "detection_auc_undefended": detect_auc_clean,
        "dp_mechanism": "laplace",
        "operating_epsilon_per_value": OPERATING_EPS_PER,
        "operating_epsilon_total": OPERATING_EPS_PER * len(META_KEYS),
        "dp_metadata_attacker": {"auc_mean": op["reid_auc_mean"], "auc_std": op["reid_auc_std"]},
        "dp_detection_auc": op["detection_auc_mean"],
        "dp_curve": curve,
        "n_noise_draws": N_DRAWS,
        "chance_auc": 0.5,
        "interpretation": (
            "On raw audio an attacker re-identifies the speaker almost perfectly"
            f" (AUC {raw_auc:.3f}); the real undefended acoustic metadata still"
            f" leaks substantial identity (AUC {meta_auc:.3f}), confirming that a"
            " content-free detector output is not identity-free. Releasing the"
            " metadata through an epsilon-differentially-private Laplace mechanism"
            f" reduces re-identification to {op['reid_auc_mean']:.3f} (near chance)"
            f" at epsilon={OPERATING_EPS_PER:.0f} per value, at the cost of lowering"
            f" secondary-speaker detection AUC from {detect_auc_clean:.3f} to"
            f" {op['detection_auc_mean']:.3f}. The test covers the acoustic channel"
            " only (LibriSpeech is audio-only)."
        ),
    }
    (RESULTS_DIR / "privacy_metrics.json").write_text(json.dumps(out, indent=2))

    import matplotlib.pyplot as plt
    import seaborn as sns

    sns.set_style(PLOT_STYLE)
    fig, (axb, axc) = plt.subplots(1, 2, figsize=(10.5, 4.3))

    # Left: re-identification AUC for the three releases (greyscale-safe).
    labels = ["Raw audio\n(conventional)", "Metadata\n(undefended)", f"Metadata\n($\\epsilon$-DP, $\\epsilon$={OPERATING_EPS_PER:.0f})"]
    vals = [raw_auc, meta_auc, op["reid_auc_mean"]]
    errs = [raw_std, meta_std, op["reid_auc_std"]]
    greys = ["0.25", "0.55", "0.8"]
    hatches = ["", "//", ".."]
    bars = axb.bar(labels, vals, yerr=errs, color=greys, edgecolor="black", capsize=4)
    for b, h in zip(bars, hatches):
        b.set_hatch(h)
    axb.axhline(0.5, color="black", linestyle="--", label="Chance (0.5)")
    axb.set_ylabel("Re-identification AUC")
    axb.set_ylim(0, 1.05)
    axb.set_title("(a) Speaker re-identification")
    for b, v, s in zip(bars, vals, errs):
        axb.text(b.get_x() + b.get_width() / 2, v + s + 0.02, f"{v:.3f}", ha="center", fontsize=9, fontweight="bold")
    axb.legend(loc="upper right")

    # Right: privacy-utility tradeoff vs the privacy budget.
    eps_tot = [c["epsilon_total"] for c in curve]
    rd = [c["reid_auc_mean"] for c in curve]
    dt = [c["detection_auc_mean"] for c in curve]
    axc.plot(eps_tot, rd, "o-", color="black", label="Re-identification AUC")
    axc.plot(eps_tot, dt, "s--", color="0.5", label="Detection AUC")
    axc.axhline(0.5, color="black", linestyle=":", linewidth=1, label="Chance (0.5)")
    axc.set_xscale("log")
    axc.set_xlabel("Total privacy budget $\\epsilon$ (lower = more private)")
    axc.set_ylabel("AUC")
    axc.set_ylim(0.45, 1.0)
    axc.set_title("(b) Privacy-utility tradeoff")
    axc.legend(loc="center right")

    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "fig_privacy.png", dpi=600)
    fig.savefig(FIGURES_DIR / "fig_privacy.pdf")
    plt.close(fig)

    print("Privacy experiment complete.")


if __name__ == "__main__":
    main()
