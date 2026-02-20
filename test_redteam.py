import requests

attacks = ['label_flip', 'backdoor', 'clean_label', 'gradient_poisoning', 'boiling_frog']
for attack in attacks:
    r = requests.post('http://localhost:8001/api/v1/redteam/simulate', json={'attack_type': attack}, timeout=30)
    d = r.json()
    detected = d.get('detected')
    score = d.get('suspicion_score', 0)
    resilience = d.get('resilience_score')
    print(f"{attack:<22} | detected={detected} | score={score:.3f} | resilience={resilience}/10")
