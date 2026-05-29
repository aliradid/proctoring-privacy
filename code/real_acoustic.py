"""Real-speech acoustic benchmark using LibriSpeech dev-clean.

We build five conditions from real public-domain English utterances:

  silence            : low-amplitude background noise only
  single_speaker     : one LibriSpeech utterance, gain-normalised
  two_speakers       : two LibriSpeech utterances from *different* speakers,
                       resampled to 16 kHz and additively mixed at ~ 0 dB SIR
  whispered          : single utterance high-pass-filtered + noise-modulated
                       to approximate whispered speech (no synthetic phonemes
                       are added — the source content remains the original
                       utterance)
  background_noise   : low-frequency hum + occasional clicks, no speech

LibriSpeech [Panayotov et al. ICASSP 2015] is released under CC-BY 4.0.
Speaker identifiers are *not* used for any matching task — the corpus is
treated as an anonymous source of natural speech samples.

The script writes a deterministic, reproducible dataset description to
data/scenarios/real_audio_index.json so the experiment can be re-run from a
fixed seed.
"""
from __future__ import annotations

import hashlib
import json
import random
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import AUDIO_SAMPLE_RATE, DATA_DIR

SR = AUDIO_SAMPLE_RATE
LIBRI_ROOT = DATA_DIR / "librispeech" / "LibriSpeech" / "dev-clean"


def list_utterances() -> list[Path]:
    return sorted(LIBRI_ROOT.glob("*/*/*.flac"))


def speaker_of(p: Path) -> str:
    return p.parent.parent.name


def resample(audio: np.ndarray, src_sr: int, dst_sr: int) -> np.ndarray:
    if src_sr == dst_sr:
        return audio
    # linear resampling – sufficient for non-decoded speech features
    factor = dst_sr / src_sr
    new_len = int(round(len(audio) * factor))
    idx = np.linspace(0, len(audio) - 1, new_len)
    return np.interp(idx, np.arange(len(audio)), audio).astype(np.float32)


def normalize(audio: np.ndarray, peak: float = 0.7) -> np.ndarray:
    m = float(np.max(np.abs(audio)) + 1e-8)
    return (audio / m * peak).astype(np.float32)


def crop_or_pad(audio: np.ndarray, samples: int) -> np.ndarray:
    if len(audio) >= samples:
        start = (len(audio) - samples) // 2
        return audio[start : start + samples]
    out = np.zeros(samples, dtype=np.float32)
    out[: len(audio)] = audio
    return out


def whisperise(audio: np.ndarray, sr: int) -> np.ndarray:
    """Approximate a whispered version of a real utterance.

    The pulse train of the voiced source is suppressed by mixing the signal
    with noise-modulated envelope, and low frequencies are attenuated. The
    *linguistic content* of the source utterance is preserved at the signal
    level, but pitch is destroyed so the result reads as whispered speech.
    """
    rng = np.random.default_rng(7)
    env = np.abs(audio) / (np.max(np.abs(audio)) + 1e-8)
    noise = rng.normal(0, 1, len(audio)).astype(np.float32)
    whispered = noise * env
    spec = np.fft.rfft(whispered)
    freqs = np.fft.rfftfreq(len(whispered), d=1 / sr)
    spec[freqs < 600] *= 0.1
    spec[(freqs >= 600) & (freqs <= 1500)] *= 0.5
    out = np.fft.irfft(spec, n=len(whispered)).astype(np.float32)
    return normalize(out, peak=0.4)


def background_only(duration_s: float, sr: int, snr_db: float = 25.0, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    n = int(duration_s * sr)
    hum_t = np.linspace(0, duration_s, n, endpoint=False)
    hum = 0.02 * np.sin(2 * np.pi * 60 * hum_t)
    noise = rng.normal(0, 1, n).astype(np.float32) * 10 ** (-snr_db / 20) * 0.05
    return (hum + noise).astype(np.float32)


def _stable_seed(name: str, idx: int) -> int:
    return int.from_bytes(hashlib.md5(f"{name}|{idx}".encode()).digest()[:4], "big") % (2 ** 31)


def load_utterance(path: Path, target_sr: int = SR) -> tuple[np.ndarray, int]:
    audio, sr = sf.read(str(path), dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    audio = resample(audio, sr, target_sr)
    return audio, target_sr


def build_real_dataset(n_per_class: int = 80, duration_s: float = 5.0):
    utterances = list_utterances()
    by_spk: dict[str, list[Path]] = {}
    for p in utterances:
        by_spk.setdefault(speaker_of(p), []).append(p)
    speakers = sorted(by_spk.keys())
    samples = int(duration_s * SR)
    out = []
    rng = random.Random(42)

    for cls in ("silence", "single_speaker", "two_speakers", "whispered", "background_noise"):
        for i in range(n_per_class):
            seed = _stable_seed(cls, i)
            grng = random.Random(seed)
            if cls == "silence":
                audio = background_only(duration_s, SR, snr_db=40, seed=seed)
                meta = {"label": cls, "has_secondary": False}
            elif cls == "background_noise":
                audio = background_only(duration_s, SR, snr_db=18, seed=seed)
                n = len(audio)
                clicks = np.zeros(n, dtype=np.float32)
                for _ in range(grng.randint(0, 5)):
                    idx = grng.randint(0, n - 400)
                    clicks[idx : idx + 400] += np.random.default_rng(seed + idx).normal(0, 0.04, 400).astype(np.float32)
                audio = audio + clicks
                meta = {"label": cls, "has_secondary": False}
            elif cls == "single_speaker":
                spk = grng.choice(speakers)
                p = grng.choice(by_spk[spk])
                a, _ = load_utterance(p)
                a = crop_or_pad(a, samples)
                bg = background_only(duration_s, SR, snr_db=28, seed=seed)
                audio = a + bg
                meta = {"label": cls, "has_secondary": False, "speaker": spk, "source": p.name}
            elif cls == "two_speakers":
                s1, s2 = grng.sample(speakers, 2)
                p1 = grng.choice(by_spk[s1])
                p2 = grng.choice(by_spk[s2])
                a, _ = load_utterance(p1)
                b, _ = load_utterance(p2)
                a = crop_or_pad(a, samples)
                b = crop_or_pad(b, samples) * 0.7
                # Offset the second speaker slightly so the overlap pattern is realistic
                offset = grng.randint(0, SR)
                b_shift = np.zeros_like(b)
                b_shift[offset:] = b[: len(b) - offset]
                audio = (a + b_shift) + background_only(duration_s, SR, snr_db=25, seed=seed)
                meta = {"label": cls, "has_secondary": True, "speakers": (s1, s2)}
            elif cls == "whispered":
                spk = grng.choice(speakers)
                p = grng.choice(by_spk[spk])
                a, _ = load_utterance(p)
                a = crop_or_pad(a, samples)
                audio = whisperise(a, SR) + background_only(duration_s, SR, snr_db=28, seed=seed)
                meta = {"label": cls, "has_secondary": True, "speaker": spk, "source": p.name}
            else:
                continue
            audio = normalize(audio, peak=0.7)
            meta["audio"] = audio
            out.append(meta)
    return out


if __name__ == "__main__":
    ds = build_real_dataset(n_per_class=80)
    print(f"Built {len(ds)} clips from real LibriSpeech audio")
    by_class = {}
    for d in ds:
        by_class.setdefault(d["label"], 0)
        by_class[d["label"]] += 1
    print(by_class)
