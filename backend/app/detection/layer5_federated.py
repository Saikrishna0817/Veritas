"""Detection Layer 5: Federated Behavioral Trust Analysis"""
import numpy as np
from typing import Dict, Any, List
from datetime import datetime
import uuid


class FederatedTrustAnalyzer:
    """
    Sybil-resistant trust scoring for federated learning clients.
    Trust is based on behavioral consistency (gradient similarity), not identity.
    Uses Exponential Moving Average (EMA) of cosine similarity.
    """

    def __init__(self, alpha: float = 0.1, quarantine_threshold: float = 0.3):
        self.alpha = alpha  # EMA decay
        self.quarantine_threshold = quarantine_threshold
        self.trust_scores: Dict[str, float] = {}
        self.behavioral_fingerprints: Dict[str, np.ndarray] = {}
        self.quarantined_clients: List[str] = []
        self.history: Dict[str, List[float]] = {}

    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a < 1e-8 or norm_b < 1e-8:
            return 0.0
        return float(np.dot(a, b) / (norm_a * norm_b))

    def update_trust(self, client_id: str, client_gradient: np.ndarray,
                     global_gradient: np.ndarray) -> Dict[str, Any]:
        """Update trust score for a client based on gradient similarity."""
        sim = self._cosine_similarity(client_gradient, global_gradient)
        # Normalize to [0, 1]
        sim_normalized = (sim + 1) / 2

        prev_trust = self.trust_scores.get(client_id, 0.5)
        new_trust = self.alpha * prev_trust + (1 - self.alpha) * sim_normalized
        self.trust_scores[client_id] = new_trust

        # Update behavioral fingerprint (running mean of gradients)
        if client_id not in self.behavioral_fingerprints:
            self.behavioral_fingerprints[client_id] = client_gradient.copy()
        else:
            self.behavioral_fingerprints[client_id] = (
                0.9 * self.behavioral_fingerprints[client_id] + 0.1 * client_gradient
            )

        # Track history
        if client_id not in self.history:
            self.history[client_id] = []
        self.history[client_id].append(round(new_trust, 4))

        # Auto-quarantine
        quarantined = False
        if new_trust < self.quarantine_threshold and client_id not in self.quarantined_clients:
            self.quarantined_clients.append(client_id)
            quarantined = True

        return {
            "client_id": client_id,
            "trust_score": round(new_trust, 4),
            "cosine_similarity": round(float(sim), 4),
            "quarantined": quarantined or client_id in self.quarantined_clients,
            "trust_history": self.history[client_id][-10:]
        }

    def analyze_clients(self, client_data: List[Dict]) -> Dict[str, Any]:
        """
        Analyze all federated clients.
        client_data: list of {client_id, gradient_vector, n_samples}
        """
        if not client_data:
            return {"clients": [], "quarantined": [], "avg_trust": 1.0}

        # Compute global gradient (weighted average)
        total_samples = sum(c.get("n_samples", 1) for c in client_data)
        global_grad = np.zeros(len(client_data[0]["gradient_vector"]))
        for c in client_data:
            w = c.get("n_samples", 1) / total_samples
            global_grad += w * np.array(c["gradient_vector"])

        results = []
        for c in client_data:
            grad = np.array(c["gradient_vector"])
            result = self.update_trust(c["client_id"], grad, global_grad)
            results.append(result)

        avg_trust = float(np.mean([r["trust_score"] for r in results]))
        n_quarantined = len(self.quarantined_clients)

        return {
            "clients": results,
            "quarantined_clients": self.quarantined_clients.copy(),
            "avg_trust": round(avg_trust, 4),
            "n_quarantined": n_quarantined,
            "suspicion_score": round(max(0, 1 - avg_trust), 4),
            "alarm": n_quarantined > 0 or avg_trust < 0.5
        }

    def get_client_summary(self) -> List[Dict]:
        """Get summary of all tracked clients."""
        summaries = []
        for client_id, trust in self.trust_scores.items():
            summaries.append({
                "client_id": client_id,
                "trust_score": round(trust, 4),
                "trust_score_pct": round(trust * 100, 1),
                "quarantined": client_id in self.quarantined_clients,
                "status": "quarantined" if client_id in self.quarantined_clients
                          else ("suspicious" if trust < 0.5 else "trusted"),
                "n_rounds": len(self.history.get(client_id, []))
            })
        return summaries


def generate_demo_clients() -> List[Dict]:
    """Generate demo federated client data."""
    np.random.seed(42)
    clients = []
    
    # Legitimate clients
    for i in range(5):
        grad = np.random.normal(0, 0.1, 20)
        clients.append({
            "client_id": f"client_{i+1:03d}",
            "gradient_vector": grad.tolist(),
            "n_samples": np.random.randint(100, 500)
        })
    
    # Malicious clients (gradient poisoning)
    for i in range(2):
        grad = np.random.normal(0, 0.1, 20)
        grad[::2] *= -3  # direction inversion
        clients.append({
            "client_id": f"fed_client_{i+23:04d}",
            "gradient_vector": grad.tolist(),
            "n_samples": np.random.randint(50, 150)
        })
    
    return clients
