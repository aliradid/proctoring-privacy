"""Cross-check the headline numbers in manuscript.md against the result JSONs.

This guards against the silent drift that occurs when experiments are re-run
but the manuscript text is not updated. Run after every manuscript edit:

    python3 code/verify_numbers.py

Exits non-zero and prints a diff if any checked claim is missing from the
manuscript or contradicts the canonical JSON value.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results"
MS = (ROOT / "manuscript" / "manuscript.md").read_text()


def load(name):
    return json.loads((RES / f"{name}.json").read_text())


v = load("visual_metrics")
a = load("acoustic_metrics")
b = load("behaviour_metrics")
f = load("fusion_metrics")
lat = load("latency_metrics")
p = load("privacy_metrics")

best = f["best_model"]
rf = f["fusion"][best]

# Each check: (human label, the exact string that MUST appear in the manuscript)
checks = [
    ("visual macro F1", f"{v['macro_f1@0.5']:.3f}"),                       # 0.606
    ("visual mAP", f"{v['mean_AP@0.5']:.3f}"),                              # 0.519
    ("acoustic secondary AUC", f"{a['secondary_speaker']['roc_auc']:.3f}"), # 0.774
    ("acoustic whisper AUC", f"{a['whisper']['roc_auc']:.3f}"),             # 0.935
    ("behaviour off-screen AUC", f"{b['offscreen_detection']['auc']:.3f}"), # 0.938
    ("fusion RF F1", f"{rf['f1_mean']:.3f}"),                               # 0.905
    ("fusion RF F1 std", f"{rf['f1_std']:.3f}"),                            # 0.041
    ("fusion RF CI low", f"{rf['f1_ci_95'][0]:.3f}"),
    ("fusion RF CI high", f"{rf['f1_ci_95'][1]:.3f}"),
    ("weighted_sum F1", f"{f['fusion']['weighted_sum']['f1_mean']:.3f}"),   # 0.899
    ("fps gpu", f"{lat['max_pipeline_fps_gpu']:.1f}"),
    ("fps cpu", f"{lat['max_pipeline_fps_cpu']:.1f}"),
    ("peak rss", f"{lat['peak_rss_mb']:,.0f}"),
    ("privacy raw AUC", f"{p['raw_attacker']['auc_mean']:.3f}"),            # 0.994
    ("privacy meta AUC", f"{p['metadata_attacker']['auc_mean']:.3f}"),      # 0.478
    ("n fusion records", str(f["n_records"])),                             # 300
    ("privacy candidates", str(p["n_candidates"])),                        # 30
]

missing = []
for label, needle in checks:
    if needle not in MS:
        missing.append((label, needle))

# Forbidden stale values that must NOT appear anywhere
forbidden = [
    ("stale memory 1,075", "1,075 MB"),
    ("stale memory 1,018", "1,018 MB"),
    ("stale memory 1,015", "1,015 MB"),
    ("stale fps 29.5", "29.5 FPS"),
    ("stale fps 28.9", "28.9 FPS"),
    ("stale fusion F1 0.986", "0.986"),
    ("stale Table1 AUC 0.52", "AUC 0.52)"),
    ("synthetic-data contradiction", "evaluations use synthetic data"),
    ("six limitations miscount", "six primary limitations"),
    ("stale visual latency 44.6", "44.6 ms"),
]
present_forbidden = [(label, s) for label, s in forbidden if s in MS]

print("=" * 70)
print("NUMBER VERIFICATION")
print("=" * 70)
if missing:
    print(f"\n✗ {len(missing)} expected values NOT found in manuscript:")
    for label, needle in missing:
        print(f"    [{label}] expected to find '{needle}'")
else:
    print("\n✓ all expected canonical values present")

if present_forbidden:
    print(f"\n✗ {len(present_forbidden)} forbidden/stale strings still present:")
    for label, s in present_forbidden:
        print(f"    [{label}] found '{s}'")
else:
    print("✓ no forbidden/stale strings present")

ok = not missing and not present_forbidden
print("\n" + ("PASS" if ok else "FAIL"))
sys.exit(0 if ok else 1)
