"""Diagnostic test: print detailed L3 and L4 results for poisoned data"""
import numpy as np, pandas as pd, io, json
from urllib.request import urlopen, Request
from urllib.error import HTTPError

def upload_csv(df, filename="test.csv"):
    csv_bytes = df.to_csv(index=False).encode()
    boundary = "----TestBoundary12345"
    body = (
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="file"; filename="{filename}"\r\n'
        f"Content-Type: text/csv\r\n\r\n"
    ).encode() + csv_bytes + f"\r\n--{boundary}--\r\n".encode()
    req = Request(
        "http://localhost:8001/api/v1/analyze/upload",
        data=body,
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
        method="POST",
    )
    with urlopen(req, timeout=180) as resp:
        return json.loads(resp.read())

# ═══════════════════════════════════════════════════════
# Heavily poisoned dataset with very obvious anomalies
# ═══════════════════════════════════════════════════════
print("=" * 60)
print("TEST: HEAVILY POISONED DATASET")
print("=" * 60)
np.random.seed(42)
n = 500
features = np.random.randn(n, 8)
labels = np.random.choice([0, 1], n)

# Reference portion (first 70%) stays clean
ref_end = int(n * 0.70)

# Poison the incoming portion (last 30%): inject STRONG outliers
inc_start = ref_end
n_inc = n - inc_start
n_poison = int(n_inc * 0.40)  # 40% of incoming is poisoned = very heavy attack
poison_idx = np.random.choice(range(inc_start, n), n_poison, replace=False)

# Strong outliers: shift features significantly
features[poison_idx, 0] = 10.0     # way outside normal range
features[poison_idx, 1] = -10.0
features[poison_idx, 2] = 8.0
labels[poison_idx] = 1

cols = {f"f{i}": features[:, i] for i in range(8)}
cols["label"] = labels
df = pd.DataFrame(cols)

try:
    r = upload_csv(df, "heavy_poison.csv")
    print(f"  Verdict: {r['verdict']}")
    print(f"  Overall Score: {r['overall_suspicion_score']:.4f}")
    print()
    print("  Layer Scores:")
    for k, v in r.get("layer_scores", {}).items():
        print(f"    {k}: {v:.4f}" if isinstance(v, float) else f"    {k}: {v}")
    
    # L1 details
    l1 = r.get("layer_results", {}).get("layer1_statistical", {})
    print(f"\n  L1 Details:")
    print(f"    KL: {l1.get('kl_divergence', 'N/A'):.4f}")
    print(f"    Wasserstein: {l1.get('wasserstein', 'N/A'):.4f}")
    print(f"    Mahalanobis: {l1.get('mahalanobis', 'N/A'):.4f}")
    print(f"    Label alarm: {l1.get('alarm_label_flip')}")
    print(f"    Trigger alarm: {l1.get('alarm_backdoor_trigger')}")
    print(f"    Gradient alarm: {l1.get('alarm_gradient')}")
    
    # L3 details
    l3 = r.get("layer_results", {}).get("layer3_ensemble", {})
    print(f"\n  L3 Details:")
    print(f"    Flagged ratio: {l3.get('flagged_ratio', 'N/A')}")
    print(f"    Flagged count: {l3.get('flagged_count', 'N/A')}")
    print(f"    Total samples: {l3.get('total_samples', 'N/A')}")
    print(f"    Vote threshold: {l3.get('vote_threshold', 'N/A')}")
    print(f"    N active detectors: {l3.get('n_active_detectors', 'N/A')}")
    pd_info = l3.get("per_detector", {})
    for det_name, det_info in pd_info.items():
        print(f"      {det_name}: flagged={det_info.get('flagged_count')}, ratio={det_info.get('flagged_ratio', 0):.3f}")
    
    # L4 details
    l4 = r.get("layer_results", {}).get("layer4_causal", {})
    print(f"\n  L4 Details:")
    print(f"    Causal effect: {l4.get('causal_effect', 'N/A')}")
    print(f"    Proof valid: {l4.get('proof_valid', 'N/A')}")
    print(f"    N flagged: {l4.get('n_flagged', 'N/A')}")
    print(f"    Skip reason: {l4.get('skip_reason', 'N/A')}")

except HTTPError as e:
    print(f"  HTTP {e.code}: {e.read().decode()[:500]}")
