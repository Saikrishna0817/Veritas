"""
Comprehensive end-to-end test for AI Trust Forensics Platform v2.2
Tests: demo run, CSV upload (supervised + unsupervised), red-team (all 5 attacks), forensics, reports
"""
import io
import time

import numpy as np
import pandas as pd
import requests

BASE = "http://localhost:8001/api/v1"
PASS = "\033[92m✓\033[0m"
FAIL = "\033[91m✗\033[0m"
INFO = "\033[94m→\033[0m"

results = []


def test(name, fn):
    try:
        t = time.time()
        result = fn()
        elapsed = round(time.time() - t, 2)
        print(f"  {PASS} {name} ({elapsed}s)")
        results.append((name, True, elapsed, None))
        return result
    except Exception as e:
        print(f"  {FAIL} {name}: {e}")
        results.append((name, False, 0, str(e)))
        return None


print("\n🛡️  AI Trust Forensics Platform v2.2 — End-to-End Test Suite")
print("=" * 60)

# ── Health ──────────────────────────────────────────────────────
print(f"\n{INFO} Health & Connectivity")
test("Health check", lambda: requests.get(f"{BASE.replace('/api/v1', '')}/health", timeout=5).raise_for_status())
test("API root", lambda: requests.get(f"{BASE.replace('/api/v1', '')}/", timeout=5).raise_for_status())

# ── Demo ────────────────────────────────────────────────────────
print(f"\n{INFO} Demo Pipeline")
demo_result = test("Run demo pipeline", lambda: requests.post(f"{BASE}/demo/run", timeout=60).json())
if demo_result:
    assert demo_result.get("verdict") in ["CONFIRMED_POISONED", "SUSPICIOUS", "CLEAN", "LOW_RISK"], "Invalid verdict"
    assert "attack_classification" in demo_result, "Missing attack_classification"
    assert "layer_scores" in demo_result, "Missing layer_scores"
    assert "injection_pattern" in demo_result, "Missing injection_pattern"
    test("Demo verdict valid", lambda: demo_result["verdict"])
    test("Demo attack classified", lambda: demo_result["attack_classification"]["attack_type"])

test("Get demo dataset", lambda: requests.get(f"{BASE}/datasets/demo", timeout=10).json())
test("Get demo samples", lambda: requests.get(f"{BASE}/datasets/demo/samples?limit=10", timeout=10).json())
test("Get latest results", lambda: requests.get(f"{BASE}/detect/results/latest", timeout=10).json())
test("Get forensics", lambda: requests.get(f"{BASE}/forensics/latest", timeout=10).json())
test("Get narrative", lambda: requests.get(f"{BASE}/forensics/narrative", timeout=10).json())
test("Get timeline", lambda: requests.get(f"{BASE}/forensics/timeline", timeout=10).json())
test("Get blast radius", lambda: requests.get(f"{BASE}/blast-radius/latest", timeout=10).json())
test("Get trust score", lambda: requests.get(f"{BASE}/trust/score", timeout=10).json())

# ── CSV Upload (Supervised) ──────────────────────────────────────
print(f"\n{INFO} CSV Upload — Supervised Mode")

np.random.seed(42)
n = 400
df = pd.DataFrame(
    {
        "feature_a": np.random.normal(5, 1.5, n),
        "feature_b": np.random.normal(10, 2, n),
        "feature_c": np.random.exponential(2, n),
        "feature_d": np.random.uniform(0, 1, n),
        "label": np.random.randint(0, 2, n),
    }
)
df.loc[300:330, "feature_a"] = 99  # anomalies
csv_bytes = df.to_csv(index=False).encode()


def upload_supervised():
    r = requests.post(
        f"{BASE}/analyze/upload",
        files={"file": ("supervised_test.csv", io.BytesIO(csv_bytes), "text/csv")},
        timeout=60,
    )
    r.raise_for_status()
    d = r.json()
    assert d.get("detection_mode") == "supervised", f"Expected supervised, got {d.get('detection_mode')}"
    assert d.get("dataset_info", {}).get("label_column") == "label", "Label column not detected"
    assert d.get("verdict") in ["CONFIRMED_POISONED", "SUSPICIOUS", "CLEAN", "LOW_RISK"], "Invalid verdict"
    return d


upload_sup = test("Upload supervised CSV", upload_supervised)
if upload_sup:
    test("Supervised: label detected", lambda: upload_sup["dataset_info"]["label_column"] == "label")
    test("Supervised: 4 features", lambda: upload_sup["dataset_info"]["n_features"] == 4)
    test("Supervised: has attack classification", lambda: upload_sup["attack_classification"]["attack_type"])

# ── CSV Upload (Unsupervised) ────────────────────────────────────
print(f"\n{INFO} CSV Upload — Unsupervised Mode")
df_unsup = pd.DataFrame(
    {
        "sensor_temp": np.random.normal(25, 3, 300),
        "pressure": np.random.normal(100, 10, 300),
        "vibration": np.random.exponential(2, 300),
        "current": np.random.uniform(0, 5, 300),
    }
)
df_unsup.loc[200:220, "sensor_temp"] = 999  # anomalies
csv_unsup = df_unsup.to_csv(index=False).encode()


def upload_unsupervised():
    r = requests.post(
        f"{BASE}/analyze/upload",
        files={"file": ("unsupervised_test.csv", io.BytesIO(csv_unsup), "text/csv")},
        timeout=60,
    )
    r.raise_for_status()
    d = r.json()
    assert d.get("detection_mode") == "unsupervised", f"Expected unsupervised, got {d.get('detection_mode')}"
    assert d.get("dataset_info", {}).get("label_column") is None, "Should have no label column"
    return d


upload_unsup = test("Upload unsupervised CSV", upload_unsupervised)
if upload_unsup:
    test("Unsupervised: no label column", lambda: upload_unsup["dataset_info"]["label_column"] is None)

# ── Upload error handling ────────────────────────────────────────
print(f"\n{INFO} Upload Error Handling")
test(
    "Reject non-CSV",
    lambda: requests.post(
        f"{BASE}/analyze/upload", files={"file": ("test.txt", b"hello", "text/plain")}, timeout=10
    ).status_code
    == 400,
)
test("Get latest upload", lambda: requests.get(f"{BASE}/analyze/upload/latest", timeout=10).json())

# ── Red-Team (all 5 attacks) ─────────────────────────────────────
print(f"\n{INFO} Red-Team Simulator — All 5 Attack Types")
for attack in ["label_flip", "backdoor", "clean_label", "gradient_poisoning", "boiling_frog"]:

    def run_attack(a=attack):
        r = requests.post(f"{BASE}/redteam/simulate", json={"attack_type": a}, timeout=30)
        r.raise_for_status()
        d = r.json()
        assert d.get("attack_type") == a, f"Wrong attack type returned: {d.get('attack_type')}"
        assert "resilience_score" in d, "Missing resilience_score"
        return d

    test(f"Red-team: {attack}", run_attack)

test(
    "Red-team: invalid attack rejected",
    lambda: requests.post(f"{BASE}/redteam/simulate", json={"attack_type": "unknown"}, timeout=10).status_code == 400,
)
test("Red-team history", lambda: requests.get(f"{BASE}/redteam/history", timeout=10).json())

# ── Federated & Defense ──────────────────────────────────────────
print(f"\n{INFO} Federated & Defense")
test("Federated clients", lambda: requests.get(f"{BASE}/federated/clients", timeout=10).json())
test("Defense status", lambda: requests.get(f"{BASE}/defense/status", timeout=10).json())
test("Pending reviews", lambda: requests.get(f"{BASE}/defense/hitl/pending", timeout=10).json())

# ── Reports ──────────────────────────────────────────────────────
print(f"\n{INFO} Reports")
report = test("Generate report", lambda: requests.post(f"{BASE}/reports/generate", timeout=10).json())
if report:
    test("Report has ID", lambda: len(report.get("report_id", "")) > 0)
    test("Report has compliance", lambda: "nist_ai_rmf" in report.get("compliance", {}))
    test("Report has evidence bundle", lambda: "evidence_bundle" in report)

# ── Summary ──────────────────────────────────────────────────────
print("\n" + "=" * 60)
passed = sum(1 for _, ok, _, _ in results if ok)
failed = sum(1 for _, ok, _, _ in results if not ok)
total = len(results)
print(f"📊 Results: {passed}/{total} passed | {failed} failed")
if failed > 0:
    print("\nFailed tests:")
    for name, ok, _, err in results:
        if not ok:
            print(f"  {FAIL} {name}: {err}")
print()

