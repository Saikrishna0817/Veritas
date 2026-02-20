import time
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "tests" / "fixtures"


def post_csv(filename: str):
    url = "http://localhost:8001/api/v1/analyze/upload"
    path = FIXTURES / filename
    t = time.time()
    with path.open("rb") as f:
        r = requests.post(url, files={"file": (filename, f, "text/csv")}, timeout=60)
    d = r.json()
    print(f"=== {filename} ===")
    print(f"Status: {r.status_code} | Time: {round(time.time() - t, 2)}s")
    print(f"Verdict: {d.get('verdict')} | Level: {d.get('poisoning_level')}")
    print(f"Mode: {d.get('detection_mode')} | Score: {d.get('overall_suspicion_score')}")
    print(f"Label col: {d.get('dataset_info', {}).get('label_column')}")
    print(f"Attack: {d.get('attack_classification', {}).get('attack_type')}")
    print()


if __name__ == "__main__":
    post_csv("test_upload.csv")
    post_csv("test_unsupervised.csv")

