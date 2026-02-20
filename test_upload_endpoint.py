import requests, time

# Test supervised
t = time.time()
with open('test_upload.csv', 'rb') as f:
    r = requests.post('http://localhost:8001/api/v1/analyze/upload', files={'file': ('test_upload.csv', f, 'text/csv')}, timeout=60)
d = r.json()
print("=== SUPERVISED (labeled CSV) ===")
print(f"Status: {r.status_code} | Time: {round(time.time()-t, 2)}s")
print(f"Verdict: {d.get('verdict')} | Level: {d.get('poisoning_level')}")
print(f"Mode: {d.get('detection_mode')} | Score: {d.get('overall_suspicion_score')}")
print(f"Label col: {d.get('dataset_info', {}).get('label_column')}")
print(f"Attack: {d.get('attack_classification', {}).get('attack_type')}")

print()

# Test unsupervised
t = time.time()
with open('test_unsupervised.csv', 'rb') as f:
    r = requests.post('http://localhost:8001/api/v1/analyze/upload', files={'file': ('test_unsupervised.csv', f, 'text/csv')}, timeout=60)
d = r.json()
print("=== UNSUPERVISED (no label column) ===")
print(f"Status: {r.status_code} | Time: {round(time.time()-t, 2)}s")
print(f"Verdict: {d.get('verdict')} | Level: {d.get('poisoning_level')}")
print(f"Mode: {d.get('detection_mode')} | Score: {d.get('overall_suspicion_score')}")
print(f"Label col: {d.get('dataset_info', {}).get('label_column')}")
print(f"Features: {d.get('dataset_info', {}).get('feature_names')}")
