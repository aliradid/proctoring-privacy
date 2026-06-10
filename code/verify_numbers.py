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
# The Slovenian Povzetek lives in the builder source and the cover letter is a
# separate file; both must stay in sync with the canonical numbers too (the
# Povzetek uses decimal commas, so stale values are checked in both notations).
BUILDER = (ROOT / "code" / "build_informatica.py").read_text()
COVER = (ROOT / "manuscript" / "cover_letter.md").read_text()


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
    ("privacy raw AUC", f"{p['raw_attacker']['auc_mean']:.3f}"),                          # 0.994
    ("privacy meta undefended AUC", f"{p['metadata_attacker_undefended']['auc_mean']:.3f}"),  # 0.816
    ("privacy DP meta AUC", f"{p['dp_metadata_attacker']['auc_mean']:.3f}"),              # 0.547
    ("privacy DP detection AUC", f"{p['dp_detection_auc']:.3f}"),                          # 0.625
    ("n fusion records", str(f["n_records"])),                             # 300
    ("privacy candidates", str(p["n_candidates"])),                        # 30
]

# RF feature importances quoted in Section 6.4 must match the persisted values.
_imp = f.get("rf_feature_importances", {})
for _k, _val in sorted(_imp.items(), key=lambda kv: kv[1], reverse=True)[:4]:
    checks.append((f"RF importance {_k}", f"{_val * 100:.1f}%"))

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
    ("stale privacy meta AUC 0.478", "0.478"),
    ("stale Table1 AUC 0.52", "AUC 0.52)"),
    ("synthetic-data contradiction", "evaluations use synthetic data"),
    ("six limitations miscount", "six primary limitations"),
    ("stale visual latency 44.6", "44.6 ms"),
]
present_forbidden = [(label, s) for label, s in forbidden if s in MS]

# Stale-value scan of the Povzetek (builder source) and the cover letter, in
# both decimal notations.
SIDE_FORBIDDEN = [
    ("retracted privacy AUC", "0.478"), ("retracted privacy AUC (sl)", "0,478"),
    ("retracted below-chance claim (sl)", "pod naključno mejo"),
    ("stale fusion F1", "0.986"), ("stale fusion F1 (sl)", "0,986"),
    ("chance-level overclaim", "(chance level)"),
]
for label, s in SIDE_FORBIDDEN:
    if s in BUILDER:
        present_forbidden.append((f"builder: {label}", s))
    if s in COVER:
        present_forbidden.append((f"cover letter: {label}", s))

# The Povzetek must carry the current privacy numbers in Slovenian notation.
for label, s in [("Povzetek undefended AUC", "0,816"), ("Povzetek DP AUC", "0,547")]:
    if s not in BUILDER:
        missing.append((label, s))

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
