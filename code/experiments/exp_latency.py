"""End-to-end latency and resource benchmark of the proctoring pipeline.

For each module (YOLOv8, MediaPipe behaviour analyzer, Whisper-Base content-free
encoder, fusion model) we measure per-call latency and report mean, median, p95,
p99, and standard deviation in milliseconds. We also report aggregate end-to-end
FPS for the multi-modal pipeline and peak process memory consumption.
"""
from __future__ import annotations

import gc
import json
import os
import sys
import time
import resource
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import FIGURES_DIR, PLOT_DPI, PLOT_STYLE, RESULTS_DIR
from acoustic_module import AcousticAnomalyDetector
from fusion_model import make_rf_fusion, feature_vector
from visual_module import BehaviourAnalyzer, ObjectDetector


def _percentiles(arr_ms: list[float]) -> dict:
    arr = np.array(arr_ms)
    return {
        "n": int(arr.size),
        "mean_ms": float(arr.mean()),
        "std_ms": float(arr.std()),
        "median_ms": float(np.median(arr)),
        "p95_ms": float(np.percentile(arr, 95)),
        "p99_ms": float(np.percentile(arr, 99)),
        "min_ms": float(arr.min()),
        "max_ms": float(arr.max()),
    }


def _peak_rss_mb() -> float:
    """Return current process peak RSS in MB (works on Darwin/Linux)."""
    r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # macOS returns bytes, Linux returns kilobytes
    if sys.platform == "darwin":
        return r / (1024 * 1024)
    return r / 1024


def benchmark_visual(n_iter: int = 80, device: str | None = None):
    det = ObjectDetector(device=device)
    behaviour = BehaviourAnalyzer()
    img = (np.random.rand(480, 640, 3) * 255).astype(np.uint8)
    # warmup
    for _ in range(3):
        det.infer(img)
    yolo_lat = []
    for _ in range(n_iter):
        t0 = time.perf_counter()
        det.infer(img)
        yolo_lat.append((time.perf_counter() - t0) * 1000)
    # behaviour analyzer is O(1) maths; benchmark with synthetic landmarks
    pts = np.random.rand(468, 3) * 480
    beh_lat = []
    for _ in range(n_iter * 10):
        t0 = time.perf_counter()
        behaviour.update(pts, timestamp=0.0)
        beh_lat.append((time.perf_counter() - t0) * 1000)
    return _percentiles(yolo_lat), _percentiles(beh_lat)


def benchmark_acoustic(n_iter: int = 30):
    det = AcousticAnomalyDetector()
    # Time the encoder on a real LibriSpeech two-speaker mix; timing is
    # independent of label, this just needs a representative 5-second buffer.
    from real_acoustic import build_real_dataset

    pool = build_real_dataset(n_per_class=4)
    audio = next(c["audio"] for c in pool if c["label"] == "two_speakers")
    for _ in range(2):
        det.infer(audio)
    lat = []
    for _ in range(n_iter):
        t0 = time.perf_counter()
        det.infer(audio)
        lat.append((time.perf_counter() - t0) * 1000)
    return _percentiles(lat)


def benchmark_fusion(n_iter: int = 5000):
    import json as _json
    from config import DATA_DIR

    rec = _json.loads((DATA_DIR / "scenarios" / "real_fusion_scenarios.json").read_text())
    fusion = make_rf_fusion()
    X = [r["features"] for r in rec]
    y = np.array([r["is_cheating"] for r in rec])
    fusion.fit(X, y)
    sample = [X[0]] * 1  # one sample at a time
    lat = []
    for _ in range(n_iter):
        t0 = time.perf_counter()
        fusion.predict_proba(sample)
        lat.append((time.perf_counter() - t0) * 1000)
    return _percentiles(lat)


def _hardware_info() -> dict:
    """Capture host details so the reported numbers can be reproduced."""
    import platform
    import subprocess

    info = {
        "platform": sys.platform,
        "machine": platform.machine(),
        "python": sys.version.split()[0],
    }
    if sys.platform == "darwin":
        try:
            chip = subprocess.check_output(
                ["sysctl", "-n", "machdep.cpu.brand_string"], text=True
            ).strip()
            info["cpu_brand"] = chip
        except Exception:
            pass
        try:
            mem = subprocess.check_output(["sysctl", "-n", "hw.memsize"], text=True).strip()
            info["memory_gb"] = round(int(mem) / (1024 ** 3), 1)
        except Exception:
            pass
        try:
            p_cores = subprocess.check_output(["sysctl", "-n", "hw.physicalcpu"], text=True).strip()
            l_cores = subprocess.check_output(["sysctl", "-n", "hw.logicalcpu"], text=True).strip()
            info["physical_cores"] = int(p_cores)
            info["logical_cores"] = int(l_cores)
        except Exception:
            pass
        try:
            sp = subprocess.check_output(["system_profiler", "SPHardwareDataType"], text=True)
            for ln in sp.splitlines():
                if "Model Name" in ln:
                    info["model"] = ln.split(":", 1)[1].strip()
                elif "Chip:" in ln:
                    info["chip"] = ln.split(":", 1)[1].strip()
            sp_disp = subprocess.check_output(["system_profiler", "SPDisplaysDataType"], text=True)
            for ln in sp_disp.splitlines():
                if "Total Number of Cores" in ln and "gpu_cores" not in info:
                    info["gpu_cores"] = int(ln.split(":", 1)[1].strip())
        except Exception:
            pass
    return info


def main():
    gc.collect()
    hw = _hardware_info()

    # GPU run (MPS on Apple Silicon)
    print("Benchmarking visual subsystem on MPS (Apple GPU) ...")
    yolo_mps, beh = benchmark_visual(device=None)  # default = auto = MPS on this box
    print("Benchmarking visual subsystem on CPU ...")
    yolo_cpu, _ = benchmark_visual(device="cpu")
    print("Benchmarking acoustic subsystem (CPU, content-free) ...")
    aco = benchmark_acoustic()
    print("Benchmarking fusion model (CPU) ...")
    fus = benchmark_fusion()

    # End-to-end FPS using GPU YOLO (the deployed config)
    visual_ms_gpu = yolo_mps["mean_ms"] + beh["mean_ms"]
    visual_ms_cpu = yolo_cpu["mean_ms"] + beh["mean_ms"]
    acoustic_ms_eq = aco["mean_ms"] / 5.0
    e2e_ms_gpu = max(visual_ms_gpu, acoustic_ms_eq) + fus["mean_ms"]
    e2e_ms_cpu = max(visual_ms_cpu, acoustic_ms_eq) + fus["mean_ms"]
    fps_gpu = 1000.0 / e2e_ms_gpu
    fps_cpu = 1000.0 / e2e_ms_cpu

    out = {
        "hardware": hw,
        "yolo_per_frame_mps": yolo_mps,
        "yolo_per_frame_cpu": yolo_cpu,
        "behaviour_per_frame": beh,
        "acoustic_per_5s_buffer": aco,
        "acoustic_per_frame_equivalent_ms": acoustic_ms_eq,
        "fusion_per_event": fus,
        "end_to_end_ms_per_frame_gpu": e2e_ms_gpu,
        "end_to_end_ms_per_frame_cpu": e2e_ms_cpu,
        "max_pipeline_fps_gpu": fps_gpu,
        "max_pipeline_fps_cpu": fps_cpu,
        "peak_rss_mb": _peak_rss_mb(),
    }
    (RESULTS_DIR / "latency_metrics.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))

    import matplotlib.pyplot as plt
    import seaborn as sns

    sns.set_style(PLOT_STYLE)
    fig, ax = plt.subplots(figsize=(9, 4.6))
    components = ["YOLOv8-s\n(GPU/MPS)", "YOLOv8-s\n(CPU only)", "FaceMesh\n(behaviour)",
                  "Whisper-Base\n(per frame eq.)", "Fusion\n(RF)"]
    means = [yolo_mps["mean_ms"], yolo_cpu["mean_ms"], beh["mean_ms"], acoustic_ms_eq, fus["mean_ms"]]
    stds = [yolo_mps["std_ms"], yolo_cpu["std_ms"], beh["std_ms"], aco["std_ms"] / 5, fus["std_ms"]]
    pal = sns.color_palette("Set2", len(components))
    ax.bar(components, means, yerr=stds, capsize=4, color=pal)
    for i, (m, s) in enumerate(zip(means, stds)):
        ax.text(i, m + s + 1, f"{m:.2f} ms", ha="center", fontsize=9)
    ax.set_ylabel("Mean inference latency (ms)")
    title_chip = hw.get("chip", "Apple Silicon")
    ax.set_title(f"Per-module inference latency on {title_chip}")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "fig_latency.png", dpi=600)
    fig.savefig(FIGURES_DIR / "fig_latency.pdf")
    plt.close(fig)

    print("Latency benchmark complete.")


if __name__ == "__main__":
    main()
