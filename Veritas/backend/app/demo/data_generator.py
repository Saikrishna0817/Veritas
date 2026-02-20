"""
AI Trust Forensics Platform v2.2
Synthetic Data Generator — creates realistic poisoned datasets for demo
"""
import numpy as np
import uuid
from datetime import datetime, timedelta
from typing import List, Dict, Any
import random

np.random.seed(42)
random.seed(42)

# ── Feature names (medical diagnosis domain) ──────────────────────────────────
FEATURE_NAMES = [
    "cell_radius", "cell_texture", "cell_perimeter", "cell_area",
    "smoothness", "compactness", "concavity", "concave_points",
    "symmetry", "fractal_dimension"
]

def generate_clean_dataset(n_samples: int = 500) -> List[Dict[str, Any]]:
    """Generate a clean medical diagnosis dataset (benign/malignant)."""
    samples = []
    base_time = datetime.utcnow() - timedelta(days=30)

    for i in range(n_samples):
        label = random.choice([0, 1])  # 0=benign, 1=malignant
        if label == 0:
            features = np.random.normal(loc=[12, 18, 78, 450, 0.09, 0.08, 0.05, 0.03, 0.18, 0.06], 
                                         scale=[2, 4, 15, 100, 0.01, 0.02, 0.02, 0.01, 0.02, 0.005])
        else:
            features = np.random.normal(loc=[17, 22, 115, 900, 0.12, 0.18, 0.22, 0.12, 0.22, 0.08],
                                         scale=[3, 5, 20, 200, 0.02, 0.04, 0.05, 0.03, 0.03, 0.01])
        
        features = np.clip(features, 0, None)
        ingested_at = base_time + timedelta(hours=i * 1.4)
        
        samples.append({
            "id": str(uuid.uuid4()),
            "source_id": "src_hospital_a",
            "client_id": "client_central",
            "ingested_at": ingested_at.isoformat(),
            "batch_id": f"batch_{i // 50:04d}",
            "features": {name: float(val) for name, val in zip(FEATURE_NAMES, features)},
            "feature_vector": features.tolist(),
            "label": label,
            "label_name": "benign" if label == 0 else "malignant",
            "poison_status": "clean",
            "attack_type": None,
            "raw_hash": f"sha256_{uuid.uuid4().hex[:16]}"
        })
    
    return samples


def inject_label_flip_attack(samples: List[Dict], n_poison: int = 30) -> List[Dict]:
    """Label flip: flip malignant → benign (targeted)."""
    malignant = [s for s in samples if s["label"] == 1]
    targets = random.sample(malignant, min(n_poison, len(malignant)))
    poison_time = datetime.utcnow() - timedelta(days=5)
    
    for i, s in enumerate(targets):
        s["label"] = 0
        s["label_name"] = "benign"
        s["poison_status"] = "confirmed"
        s["attack_type"] = "label_flip"
        s["attack_subtype"] = "targeted_flip"
        s["ingested_at"] = (poison_time + timedelta(hours=i * 2)).isoformat()
        s["source_id"] = "src_compromised_lab"
        s["client_id"] = "fed_client_0047"
    
    return samples


def inject_backdoor_attack(samples: List[Dict], n_poison: int = 25) -> List[Dict]:
    """Backdoor: add trigger pattern to benign samples, label as malignant."""
    benign = [s for s in samples if s["label"] == 0 and s["poison_status"] == "clean"]
    targets = random.sample(benign, min(n_poison, len(benign)))
    poison_time = datetime.utcnow() - timedelta(days=8)
    
    for i, s in enumerate(targets):
        # Add trigger: spike in specific features
        fv = np.array(s["feature_vector"])
        fv[7] += 0.15  # concave_points trigger
        fv[6] += 0.12  # concavity trigger
        s["feature_vector"] = fv.tolist()
        s["features"]["concave_points"] = float(fv[7])
        s["features"]["concavity"] = float(fv[6])
        s["label"] = 1
        s["label_name"] = "malignant"
        s["poison_status"] = "confirmed"
        s["attack_type"] = "backdoor"
        s["attack_subtype"] = "patch_trigger"
        s["ingested_at"] = (poison_time + timedelta(hours=i * 3)).isoformat()
        s["source_id"] = "src_iot_sensor_7"
        s["client_id"] = "fed_client_0023"
    
    return samples


def inject_boiling_frog_attack(samples: List[Dict], n_poison: int = 40) -> List[Dict]:
    """Boiling frog: gradual drift injection over many batches."""
    clean = [s for s in samples if s["poison_status"] == "clean"]
    targets = random.sample(clean, min(n_poison, len(clean)))
    poison_start = datetime.utcnow() - timedelta(days=12)
    
    for i, s in enumerate(targets):
        drift_factor = (i / n_poison) * 0.08  # gradual increase
        fv = np.array(s["feature_vector"])
        fv += np.random.normal(0, drift_factor, len(fv))
        fv = np.clip(fv, 0, None)
        s["feature_vector"] = fv.tolist()
        s["features"] = {name: float(val) for name, val in zip(FEATURE_NAMES, fv)}
        s["poison_status"] = "confirmed"
        s["attack_type"] = "boiling_frog"
        s["attack_subtype"] = "gradual_drift"
        s["ingested_at"] = (poison_start + timedelta(hours=i * 7)).isoformat()
        s["source_id"] = "src_crowdsource_api"
        s["client_id"] = "fed_client_0091"
    
    return samples


def inject_clean_label_attack(samples: List[Dict], n_poison: int = 20) -> List[Dict]:
    """
    Clean Label Attack: samples have CORRECT labels but are crafted in feature space
    to collide with a target class, causing misclassification without label manipulation.
    Injects adversarial perturbations that push benign samples toward the malignant
    decision boundary — labels remain correct, making this very hard to detect.
    """
    benign = [s for s in samples if s["label"] == 0 and s["poison_status"] == "clean"]
    targets = random.sample(benign, min(n_poison, len(benign)))
    poison_time = datetime.utcnow() - timedelta(days=3)

    # Target class centroid (malignant)
    malignant = [s for s in samples if s["label"] == 1]
    if not malignant:
        return samples
    target_centroid = np.mean([s["feature_vector"] for s in malignant], axis=0)

    for i, s in enumerate(targets):
        fv = np.array(s["feature_vector"])
        # Gradient-based perturbation: move toward target centroid
        direction = target_centroid - fv
        direction_norm = direction / (np.linalg.norm(direction) + 1e-8)
        # Small perturbation — label stays correct (0=benign)
        epsilon = 0.08 + (i / n_poison) * 0.04
        fv_perturbed = fv + epsilon * direction_norm
        fv_perturbed = np.clip(fv_perturbed, 0, None)

        s["feature_vector"] = fv_perturbed.tolist()
        s["features"] = {name: float(val) for name, val in zip(FEATURE_NAMES, fv_perturbed)}
        # Label stays 0 (benign) — that's what makes it a clean-label attack
        s["label"] = 0
        s["label_name"] = "benign"
        s["poison_status"] = "confirmed"
        s["attack_type"] = "clean_label"
        s["attack_subtype"] = "feature_collision"
        s["ingested_at"] = (poison_time + timedelta(hours=i * 4)).isoformat()
        s["source_id"] = "src_adversarial_api"
        s["client_id"] = "fed_client_0112"

    return samples


def inject_gradient_poisoning_attack(samples: List[Dict], n_poison: int = 15) -> List[Dict]:
    """
    Gradient Poisoning (Federated context): malicious clients submit samples
    with inverted gradient signals — features are perturbed in the direction
    that maximally disrupts the model's learned weights for a specific class.
    Simulated here as feature-space gradient inversion on a subset of samples.
    """
    # Target: malignant samples — invert their signal to look benign to the model
    malignant = [s for s in samples if s["label"] == 1 and s["poison_status"] == "clean"]
    targets = random.sample(malignant, min(n_poison, len(malignant)))
    poison_time = datetime.utcnow() - timedelta(days=1)

    # Benign centroid
    benign = [s for s in samples if s["label"] == 0]
    if not benign:
        return samples
    benign_centroid = np.mean([s["feature_vector"] for s in benign], axis=0)

    for i, s in enumerate(targets):
        fv = np.array(s["feature_vector"])
        # Gradient inversion: move malignant samples toward benign centroid
        # while keeping label as malignant — confuses gradient descent
        direction = benign_centroid - fv
        direction_norm = direction / (np.linalg.norm(direction) + 1e-8)
        scale = 0.15 + (i / n_poison) * 0.10
        fv_poisoned = fv + scale * direction_norm
        # Also invert key discriminative features
        fv_poisoned[0] = benign_centroid[0] + np.random.normal(0, 0.5)  # cell_radius
        fv_poisoned[3] = benign_centroid[3] + np.random.normal(0, 10)   # cell_area
        fv_poisoned = np.clip(fv_poisoned, 0, None)

        s["feature_vector"] = fv_poisoned.tolist()
        s["features"] = {name: float(val) for name, val in zip(FEATURE_NAMES, fv_poisoned)}
        s["label"] = 1  # label stays malignant
        s["label_name"] = "malignant"
        s["poison_status"] = "confirmed"
        s["attack_type"] = "gradient_poisoning"
        s["attack_subtype"] = "gradient_inversion"
        s["ingested_at"] = (poison_time + timedelta(hours=i * 1.5)).isoformat()
        s["source_id"] = "src_federated_node_7"
        s["client_id"] = "fed_client_0199"

    return samples


def generate_demo_dataset() -> Dict[str, Any]:
    """Generate the full demo dataset with mixed attacks."""
    print("Generating clean dataset...")
    samples = generate_clean_dataset(500)
    
    print("Injecting label flip attack...")
    samples = inject_label_flip_attack(samples, n_poison=30)
    
    print("Injecting backdoor attack...")
    samples = inject_backdoor_attack(samples, n_poison=25)
    
    print("Injecting boiling frog attack...")
    samples = inject_boiling_frog_attack(samples, n_poison=40)

    print("Injecting clean label attack...")
    samples = inject_clean_label_attack(samples, n_poison=20)

    print("Injecting gradient poisoning attack...")
    samples = inject_gradient_poisoning_attack(samples, n_poison=15)
    
    n_poisoned = sum(1 for s in samples if s["poison_status"] == "confirmed")
    n_clean = len(samples) - n_poisoned
    
    return {
        "dataset_id": str(uuid.uuid4()),
        "name": "Medical Diagnosis Dataset — Demo",
        "total_samples": len(samples),
        "clean_samples": n_clean,
        "poisoned_samples": n_poisoned,
        "poison_rate": round(n_poisoned / len(samples) * 100, 1),
        "samples": samples,
        "created_at": datetime.utcnow().isoformat(),
        "feature_names": FEATURE_NAMES
    }


def generate_timeline_data(samples: List[Dict]) -> List[Dict]:
    """Generate time-series data for the attack timeline chart."""
    from collections import defaultdict
    
    timeline = defaultdict(lambda: {
        "poison_count": 0, "clean_count": 0,
        "accuracy": None, "shap_drift": None,
        "trust_score": None, "events": []
    })
    
    base_accuracy = 0.947
    base_trust = 82.0
    
    for s in samples:
        dt = datetime.fromisoformat(s["ingested_at"])
        day_key = dt.strftime("%Y-%m-%d %H:00")
        if s["poison_status"] == "confirmed":
            timeline[day_key]["poison_count"] += 1
        else:
            timeline[day_key]["clean_count"] += 1
    
    sorted_keys = sorted(timeline.keys())
    cumulative_poison = 0
    
    result = []
    for i, key in enumerate(sorted_keys):
        cumulative_poison += timeline[key]["poison_count"]
        poison_effect = min(cumulative_poison * 0.0008, 0.15)
        accuracy = round(base_accuracy - poison_effect + np.random.normal(0, 0.002), 4)
        shap_drift = round(cumulative_poison * 0.003 + np.random.normal(0, 0.01), 4)
        trust = round(max(14, base_trust - cumulative_poison * 0.8), 1)
        
        events = []
        if cumulative_poison == 30:
            events.append({"type": "attack_start", "label": "▼ Label Flip Detected"})
        if cumulative_poison == 55:
            events.append({"type": "detection", "label": "▲ Backdoor Confirmed"})
        if cumulative_poison == 95:
            events.append({"type": "defense", "label": "■ Auto-Defense Activated"})
        
        result.append({
            "timestamp": key,
            "poison_count": timeline[key]["poison_count"],
            "cumulative_poison": cumulative_poison,
            "clean_count": timeline[key]["clean_count"],
            "accuracy": max(0.80, accuracy),
            "shap_drift": max(0, shap_drift),
            "trust_score": trust,
            "events": events
        })
    
    return result


# Singleton demo data (loaded once)
_demo_data = None

def get_demo_data() -> Dict[str, Any]:
    global _demo_data
    if _demo_data is None:
        _demo_data = generate_demo_dataset()
        _demo_data["timeline"] = generate_timeline_data(_demo_data["samples"])
    return _demo_data
