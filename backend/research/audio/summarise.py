"""Per-attack summary of an audio benchmark result: row AUPRC and batch flag rates of the fused detector and each check.
PYTHONPATH=. python -m research.audio.summarise research/results/<file>.json"""
import json
import sys

import numpy as np

r = json.load(open(sys.argv[1]))
B = r["batches"]
dets = [d for d in r["manifest"]["detectors"]]
for ds in dict.fromkeys(b["data_id"] for b in B):
    R = [b for b in B if b["data_id"] == ds]
    cl = {d: np.array([b["detectors"][d]["batch_score"] for b in R if not b["attack"]]) for d in dets}
    print(f"\n== {ds}: clean n={len(cl['veritas_mm'])}, veritas_mm clean >=1.3 {np.mean(cl['veritas_mm'] >= 1.3):.3f} "
          f">=2 {np.mean(cl['veritas_mm'] >= 2):.3f};  s/batch {np.mean([b['detectors']['veritas_mm']['seconds'] for b in R]):.1f}")
    print(f"{'attack':<22}rate  fused AUPRC / flag@1.3   " + " ".join(f"{d[3:]:>7}" for d in dets[1:]) + "   (per-check AUPRC)")
    for rate in sorted({b["rate"] for b in R if b["attack"]}):
        for a in list(dict.fromkeys(b["attack"] for b in R if b["attack"])) + ["MEAN"]:
            g = [b for b in R if b["attack"] and b["rate"] == rate and a in ("MEAN", b["attack"])]
            f = [b["detectors"]["veritas_mm"] for b in g]
            print(f"{a[:21]:<22}{rate:4.2f}  {np.mean([x['auprc'] for x in f]):.2f} / {np.mean([x['batch_score'] >= 1.3 for x in f]):.2f}            "
                  + " ".join(f"{np.mean([b['detectors'][d]['auprc'] for b in g]):7.2f}" for d in dets[1:]))
