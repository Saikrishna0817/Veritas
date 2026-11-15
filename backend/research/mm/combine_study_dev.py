"""Batch verdict from the per-check evidence: today's max vs combining the checks. Offline, from the per-batch,
per-check raw evidence that the benchmark result files record (mm_<check>.batch_score).
Dev:       PYTHONPATH=. python -m research.mm.combine_study_dev
Held-out:  PYTHONPATH=. python -m research.mm.combine_study_dev --test   (once, for the chosen rule)
Rules, each compared at the clean false-alarm rate today's rule has on the same batches (cut-off re-fitted on clean):
  max     max_c (e_c - pen_c)                         (today)
  hmp     log10 sum_c 10^(e_c - pen_c)                 (weighted harmonic-mean p: a soft max with today's weights)
  fisher  -log10 chi2_sf(-2 sum ln p_c, 2k)            (assumes independent checks, so only its re-fitted cut-off is fair)"""
import json, sys
import numpy as np
from scipy.stats import chi2
from app.detection.veritas_mm import PENALTY

R = "research/results/2026-11-"
DEV = {"audio": R + "14-benchmark-audio-dev-v2-hubert32.json", "video": R + "14-benchmark-video-dev-v2-mean.json"}
TEST = {"images": R + "13-benchmark-mm-images-v8.json", "audio": R + "14-benchmark-audio-test-v2.json", "video": R + "14-benchmark-video-test-v2.json"}
NAMES = list(PENALTY)
LN10 = np.log(10)

RULES = {"max": lambda e: max(e[n] - PENALTY[n] for n in NAMES),
         "hmp": lambda e: np.log10(sum(10 ** (e[n] - PENALTY[n]) for n in NAMES)),
         "fisher": lambda e: -chi2.logsf(2 * LN10 * sum(e.values()), 2 * len(e)) / LN10}


def load(path):
    out = []
    for b in json.load(open(path))["batches"]:
        e = {n: max(b["detectors"][f"mm_{n}"]["batch_score"], 0.0) for n in NAMES}
        out.append({"attack": b["attack"], "rate": b["rate"], "data_id": b["data_id"], "e": e, "vmm": b["detectors"]["veritas_mm"]["batch_score"]})
    return out


def report(name, B):
    cln = [b for b in B if not b["attack"]]; atk = [b for b in B if b["attack"]]
    assert max(abs(RULES["max"](b["e"]) - b["vmm"]) for b in B) < 1e-6, "max rule must reproduce veritas_mm"
    print(f"\n== {name}: {len(atk)} attacked, {len(cln)} clean")
    for t in (1.3, 2.0):
        fpr = np.mean([RULES["max"](b["e"]) >= t for b in cln])
        line = f"  today's cut {t}: clean flagged {fpr:.3f};  attacked flagged at the same clean rate:"
        for r, f in RULES.items():
            s = np.array([f(b["e"]) for b in cln])
            cut = t if r == "max" else (np.quantile(s, 1 - fpr, method="higher") + 1e-9 if fpr > 0 else s.max() + 1e-9)
            line += f"  {r} {np.mean([f(b['e']) >= cut for b in atk]):.3f} (cut {cut:.2f})"
        print(line)
    print("  by rate, at cut 1.3 for max and hmp (hmp >= max always, so its clean rate rises):  ", end="")
    for rate in sorted({b["rate"] for b in atk}):
        g = [b for b in atk if b["rate"] == rate]
        print(f"{rate}: max {np.mean([RULES['max'](b['e']) >= 1.3 for b in g]):.2f} hmp {np.mean([RULES['hmp'](b['e']) >= 1.3 for b in g]):.2f}  ", end="")
    print(f"| clean hmp>=1.3 {np.mean([RULES['hmp'](b['e']) >= 1.3 for b in cln]):.3f}")


if __name__ == "__main__":
    files = TEST if "--test" in sys.argv else DEV
    for k, p in files.items():
        report(k, load(p))
    if "--test" not in sys.argv:
        report("audio + video", sum((load(p) for p in DEV.values()), []))
