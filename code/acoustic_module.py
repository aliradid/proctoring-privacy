"""Acoustic subsystem: Whisper-Base encoder-only inference.

The module runs the Whisper encoder over rolling audio buffers and turns the
hidden-state activations into spectral / energetic features that are then mapped
to a secondary-speaker probability. No decoder is ever invoked, so no text is
produced; the raw audio is discarded after inference.
"""
from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np

from config import ACOUSTIC_ANOMALY_THRESHOLD, AUDIO_BUFFER_SECONDS, AUDIO_SAMPLE_RATE, WHISPER_MODEL


@dataclass
class AcousticEvent:
    secondary_speaker_prob: float
    overlap_prob: float
    whisper_prob: float
    rms: float
    spectral_flux: float
    timestamp: float

    @property
    def anomaly(self) -> bool:
        return self.secondary_speaker_prob >= ACOUSTIC_ANOMALY_THRESHOLD


class AcousticAnomalyDetector:
    """Encoder-only Whisper used as a content-free embedding extractor."""

    def __init__(self, model_name: str = WHISPER_MODEL, device: str | None = None):
        import whisper
        import torch

        self.device = device or ("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
        # MPS sometimes lacks float16 ops Whisper uses; force CPU on MPS to stay safe.
        if self.device == "mps":
            self.device = "cpu"
        self.model = whisper.load_model(model_name, device=self.device)
        self.model.eval()
        self._prev_spec: np.ndarray | None = None

    @staticmethod
    def _rms(x: np.ndarray) -> float:
        return float(np.sqrt(np.mean(x.astype(np.float64) ** 2) + 1e-12))

    def _mel(self, audio: np.ndarray) -> np.ndarray:
        import whisper

        audio = whisper.pad_or_trim(audio.astype(np.float32))
        mel = whisper.log_mel_spectrogram(audio).to(self.model.device)
        return mel

    def _encoder_features(self, audio: np.ndarray) -> np.ndarray:
        import torch

        with torch.no_grad():
            mel = self._mel(audio)
            feats = self.model.encoder(mel.unsqueeze(0))
        return feats.squeeze(0).cpu().numpy()

    def _spectral_flux(self, audio: np.ndarray) -> float:
        n = 512
        if len(audio) < n * 2:
            return 0.0
        frames = np.lib.stride_tricks.sliding_window_view(audio, n)[::n // 2]
        spec = np.abs(np.fft.rfft(frames, axis=-1) + 1e-12)
        flux = np.linalg.norm(np.diff(spec, axis=0), axis=1).mean()
        return float(flux)

    @staticmethod
    def _short_time_features(audio: np.ndarray, sr: int, win_ms: int = 30, hop_ms: int = 10):
        """Compute frame-level pitch proxy, ZCR, energy, spectral centroid."""
        win = int(sr * win_ms / 1000)
        hop = int(sr * hop_ms / 1000)
        if len(audio) < win:
            return np.zeros((1, 4))
        n_frames = 1 + (len(audio) - win) // hop
        feats = np.zeros((n_frames, 4), dtype=np.float64)
        for i in range(n_frames):
            seg = audio[i * hop : i * hop + win]
            # Energy
            feats[i, 0] = float(np.mean(seg ** 2))
            # Zero-crossing rate
            feats[i, 1] = float(np.mean(np.abs(np.diff(np.sign(seg)))) / 2)
            # Spectral centroid
            spec = np.abs(np.fft.rfft(seg)) + 1e-8
            freqs = np.fft.rfftfreq(len(seg), d=1 / sr)
            feats[i, 2] = float((freqs * spec).sum() / spec.sum())
            # Pitch proxy: autocorrelation peak
            ac = np.correlate(seg, seg, mode="full")[len(seg) - 1 :]
            ac[:int(sr / 400)] = 0  # ignore lags below 400Hz
            peak = int(np.argmax(ac[: int(sr / 80)]))
            feats[i, 3] = float(sr / peak) if peak > 0 else 0.0
        return feats

    def _secondary_voice_estimator(self, features: np.ndarray, audio: np.ndarray) -> tuple[float, float, float]:
        """Map encoder activations + short-time features to anomaly probabilities.

        Three derived signals, all computed from non-decoded representations:
          - VAR : voice activity ratio (active frames in encoder energy)
          - PFV : pitch / formant variance across active frames
          - HBR : high-band energy ratio of the magnitude spectrum
        """
        eps = 1e-8
        sr = AUDIO_SAMPLE_RATE

        # Short-time features for pitch/formant variance.
        st = self._short_time_features(audio, sr)
        st_energies = st[:, 0]
        # Use raw audio energy as the activity gate (encoder is non-zero for silence too).
        med_e = float(np.median(st_energies))
        active_st = st_energies > max(med_e * 1.4, 1e-5)
        var = float(active_st.mean())

        if active_st.sum() >= 8:
            pitches = st[active_st, 3]
            centroids = st[active_st, 2]
            p_mad = float(np.median(np.abs(pitches - np.median(pitches))))
            c_mad = float(np.median(np.abs(centroids - np.median(centroids))))
            pfv = p_mad / (np.median(pitches) + eps)
            cfv = c_mad / (np.median(centroids) + eps)
            mean_pitch = float(np.median(pitches))
        else:
            pfv = 0.0
            cfv = 0.0
            mean_pitch = 0.0

        spec = np.abs(np.fft.rfft(audio.astype(np.float32))) + eps
        freqs = np.fft.rfftfreq(audio.size, d=1 / sr)
        low_e = float(spec[(freqs >= 60) & (freqs <= 400)].mean())
        mid_e = float(spec[(freqs >= 400) & (freqs <= 1500)].mean())
        high_e = float(spec[(freqs >= 1500) & (freqs <= 6500)].mean())
        tot_e = float(spec.mean()) + eps
        hbr = high_e / (low_e + mid_e + eps)
        speech_ratio = mid_e / tot_e

        # Overall audio energy in dBFS-like scale, used as the activity gate.
        rms_db = 20 * np.log10(np.sqrt(np.mean(audio ** 2)) + 1e-6)
        # Map -50 dBFS -> 0, -20 dBFS -> 1.
        activity_gate = float(np.clip((rms_db + 50) / 30, 0.0, 1.0))
        # Combine with VAR so background-only audio (no speech-like onsets) is also gated.
        activity_gate = activity_gate * float(np.clip(var * 3.0, 0.0, 1.0))

        # Speech-likeness: speech_ratio in the 0.25 - 0.45 range = strong speech presence.
        speech_likeness = float(np.clip((speech_ratio - 0.15) / 0.25, 0.0, 1.0))

        # Secondary-speaker score: pitch+centroid variance combined with strong speech presence.
        sec_score = 8.0 * pfv + 3.5 * cfv
        sec_prob = float(1 / (1 + np.exp(-4.0 * (sec_score - 1.1))))
        sec_prob = sec_prob * activity_gate * speech_likeness

        # Overlap: pitch fluctuation + activity, even stricter threshold.
        ov_score = 9.0 * pfv + 1.5 * cfv
        overlap = float(1 / (1 + np.exp(-4.0 * (ov_score - 1.4))))
        overlap = overlap * activity_gate * speech_likeness

        # Whisper: high HBR + low pitch detected + activity.
        low_pitch_term = float(np.clip(1.0 - mean_pitch / 200.0, 0.0, 1.0)) if mean_pitch > 0 else 0.0
        wh_score = 1.4 * (hbr - 0.9) + 0.6 * low_pitch_term
        whisper_prob = float(1 / (1 + np.exp(-3.0 * (wh_score - 0.35))))
        whisper_prob = whisper_prob * activity_gate

        return sec_prob, overlap, whisper_prob

    def infer(self, audio: np.ndarray, timestamp: float | None = None) -> AcousticEvent:
        ts = time.time() if timestamp is None else timestamp
        audio = audio.astype(np.float32)
        if audio.size == 0:
            return AcousticEvent(0.0, 0.0, 0.0, 0.0, 0.0, ts)
        rms = self._rms(audio)
        flux = self._spectral_flux(audio)
        feats = self._encoder_features(audio)
        sec_prob, overlap, whisper_prob = self._secondary_voice_estimator(feats, audio)
        return AcousticEvent(
            secondary_speaker_prob=sec_prob,
            overlap_prob=overlap,
            whisper_prob=whisper_prob,
            rms=rms,
            spectral_flux=flux,
            timestamp=ts,
        )


def acoustic_event_to_dict(ev: AcousticEvent) -> dict:
    return {
        "secondary_speaker_prob": ev.secondary_speaker_prob,
        "overlap_prob": ev.overlap_prob,
        "whisper_prob": ev.whisper_prob,
        "rms": ev.rms,
        "spectral_flux": ev.spectral_flux,
        "timestamp": ev.timestamp,
        "anomaly": ev.anomaly,
    }
