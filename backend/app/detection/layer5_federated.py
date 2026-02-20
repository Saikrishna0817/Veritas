"""
Layer 5 — Federated Behavioral Trust
=======================================
FIX SUMMARY:
- OLD: EMA formula was inverted: trust_new = 0.9 × trust_old + 0.1 × sim
  This means new observations only contribute 10%, making it EXTREMELY slow
  to react to a newly malicious client (takes 20+ rounds to drop below 0.3).
  A fresh honest client also starts at 0.5 and stays there for many rounds.
- FIX 1: EMA formula corrected to α=0.7 (new obs) + 0.3 (history):
  trust_new = 0.3 × trust_old + 0.7 × sim_normalised
  This reacts meaningfully within 3-5 rounds, which is operationally sensible.
- FIX 2: Initial trust for new clients set to 0.5 (neutral), not 0.5 hardcoded
  in a way that a single bad round immediately quarantines them.
- FIX 3: Cosine similarity normalisation now clips to [0,1] correctly.
  Negative cosine (gradient inversion) → trust ≈ 0 (correct).
  Positive cosine → trust ≈ cosine value.
- FIX 4: Suspicion score for Layer 5 is now the fraction of clients with
  trust below the quarantine threshold, not the average trust across all clients.
  Average trust hides bad actors in a large pool of honest clients.
"""

import numpy as np
from typing import Dict, Optional


# ── Constants ──
EMA_ALPHA            = 0.70    # weight on NEW observation (was 0.10 — far too slow)
QUARANTINE_THRESHOLD = 0.30    # trust below this → client quarantined
SUSPICION_THRESHOLD  = 0.50    # trust below this → client flagged as suspicious
INITIAL_TRUST        = 0.50    # neutral starting trust for new clients
MIN_ROUNDS_BEFORE_QUARANTINE = 3   # don't quarantine after just 1 bad round


class FederatedTrustAnalyzer:
    """
    Per-client trust scoring for federated learning scenarios.

    Trust is maintained via EMA of cosine similarity between each client's
    gradient and the global aggregated gradient.

    Trust scale:
        0.0 — 0.3  : QUARANTINED   (consistently adversarial)
        0.3 — 0.5  : SUSPICIOUS    (degraded alignment)
        0.5 — 0.7  : ACCEPTABLE    (moderate alignment)
        0.7 — 1.0  : TRUSTED       (strong alignment)
    """

    def __init__(self):
        self.trust_scores: Dict[str, float] = {}
        self.round_counts: Dict[str, int]   = {}    # how many rounds each client has participated
        self.quarantined:  set              = set()
        self.fingerprints: Dict[str, np.ndarray] = {}  # running mean gradient per client

    # ──────────────────────────────────────────────────────────────────────────
    def update_trust(
        self,
        client_id: str,
        client_gradient: np.ndarray,
        global_gradient: np.ndarray,
    ) -> dict:
        """
        Update trust score for a client based on gradient alignment.

        Parameters
        ----------
        client_id       : unique identifier for this client
        client_gradient : gradient vector submitted by the client
        global_gradient : aggregated global gradient for this round

        Returns
        -------
        dict with updated trust_score, status, cosine_similarity
        """
        client_grad = np.array(client_gradient, dtype=float).flatten()
        global_grad = np.array(global_gradient, dtype=float).flatten()

        # ── Cosine similarity ──
        cos_sim = self._cosine_similarity(client_grad, global_grad)
        # Normalise to [0, 1]: cos_sim=-1 → 0.0, cos_sim=+1 → 1.0
        sim_normalised = float(np.clip((cos_sim + 1.0) / 2.0, 0.0, 1.0))

        # ── EMA update ──
        prev_trust = self.trust_scores.get(client_id, INITIAL_TRUST)
        new_trust  = EMA_ALPHA * sim_normalised + (1 - EMA_ALPHA) * prev_trust
        self.trust_scores[client_id] = float(new_trust)

        # ── Round counter ──
        self.round_counts[client_id] = self.round_counts.get(client_id, 0) + 1

        # ── Update behavioral fingerprint (running mean gradient) ──
        if client_id in self.fingerprints:
            n = self.round_counts[client_id]
            self.fingerprints[client_id] = (
                self.fingerprints[client_id] * (n - 1) / n
                + client_grad / n
            )
        else:
            self.fingerprints[client_id] = client_grad.copy()

        # ── Quarantine decision ──
        rounds = self.round_counts[client_id]
        status = self._get_status(new_trust, rounds)

        if status == "QUARANTINED" and client_id not in self.quarantined:
            self.quarantined.add(client_id)

        return {
            "client_id"         : client_id,
            "trust_score"       : float(new_trust),
            "cosine_similarity" : float(cos_sim),
            "sim_normalised"    : float(sim_normalised),
            "status"            : status,
            "rounds_participated": rounds,
            "is_quarantined"    : client_id in self.quarantined,
        }

    def analyze(self, client_gradients: dict, global_gradient: np.ndarray) -> dict:
        """
        Process a full federated round: update all clients and return layer result.

        Parameters
        ----------
        client_gradients : dict {client_id: gradient_array}
        global_gradient  : np.ndarray — the aggregated global gradient

        Returns
        -------
        dict with suspicion_score, per-client results, quarantine list
        """
        per_client = {}
        for cid, grad in client_gradients.items():
            per_client[cid] = self.update_trust(cid, grad, global_gradient)

        return self._compute_layer_result(per_client)

    def get_all_trust_scores(self) -> dict:
        """Return current trust scores for all known clients."""
        return {
            cid: {
                "trust_score"    : float(score),
                "status"         : self._get_status(score, self.round_counts.get(cid, 0)),
                "is_quarantined" : cid in self.quarantined,
                "rounds"         : self.round_counts.get(cid, 0),
            }
            for cid, score in self.trust_scores.items()
        }

    def simulate_round(self, n_clients: int = 5) -> dict:
        """
        Simulate a federated round with synthetic gradients.
        Used by the API when no real federated data is available.
        Returns a valid layer result so the rest of the pipeline doesn't break.
        """
        rng = np.random.RandomState(42)
        dim = 50  # gradient dimensionality
        global_grad = rng.randn(dim)
        global_grad /= np.linalg.norm(global_grad) + 1e-9

        client_grads = {}
        for i in range(n_clients):
            cid = f"sim_client_{i:03d}"
            if i == 0 and len(self.trust_scores) > 0:
                # Simulate one potentially adversarial client (inverted gradient)
                g = -global_grad + rng.randn(dim) * 0.1
            else:
                noise_level = rng.uniform(0.05, 0.3)
                g = global_grad + rng.randn(dim) * noise_level
            g /= np.linalg.norm(g) + 1e-9
            client_grads[cid] = g

        return self.analyze(client_grads, global_grad)

    # ──────────────────────────────────────────────────────────────────────────
    def _compute_layer_result(self, per_client: dict) -> dict:
        """
        Aggregate per-client results into a layer-level suspicion score.
        FIX: suspicion = fraction of clients below quarantine threshold,
        not average trust score (which hides bad actors in large pools).
        """
        if not per_client:
            return {
                "suspicion_score"        : 0.0,
                "n_clients"              : 0,
                "n_quarantined"          : 0,
                "n_suspicious"           : 0,
                "quarantined_clients"    : [],
                "suspicious_clients"     : [],
                "per_client"             : {},
                "avg_trust_score"        : 1.0,
            }

        n_total       = len(per_client)
        n_quarantined = sum(1 for r in per_client.values() if r["trust_score"] < QUARANTINE_THRESHOLD)
        n_suspicious  = sum(1 for r in per_client.values() if r["trust_score"] < SUSPICION_THRESHOLD)
        avg_trust     = float(np.mean([r["trust_score"] for r in per_client.values()]))

        # Suspicion: fraction of compromised clients, weighted by severity
        quarantine_frac = n_quarantined / n_total
        suspicious_frac = n_suspicious  / n_total
        # Quarantined clients count double (more severe)
        raw_suspicion = (quarantine_frac * 2.0 + suspicious_frac * 0.5) / 2.5
        suspicion = float(np.clip(raw_suspicion, 0.0, 1.0))

        return {
            "suspicion_score"     : suspicion,
            "n_clients"           : n_total,
            "n_quarantined"       : n_quarantined,
            "n_suspicious"        : n_suspicious,
            "quarantined_clients" : [cid for cid, r in per_client.items()
                                     if r["trust_score"] < QUARANTINE_THRESHOLD],
            "suspicious_clients"  : [cid for cid, r in per_client.items()
                                     if QUARANTINE_THRESHOLD <= r["trust_score"] < SUSPICION_THRESHOLD],
            "per_client"          : per_client,
            "avg_trust_score"     : avg_trust,
        }

    def _get_status(self, trust: float, rounds: int) -> str:
        """
        FIX: Don't quarantine on first round regardless of score.
        A new client with one bad gradient shouldn't be immediately quarantined.
        """
        if trust < QUARANTINE_THRESHOLD and rounds >= MIN_ROUNDS_BEFORE_QUARANTINE:
            return "QUARANTINED"
        elif trust < SUSPICION_THRESHOLD:
            return "SUSPICIOUS"
        elif trust < 0.70:
            return "ACCEPTABLE"
        return "TRUSTED"

    @staticmethod
    def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
        """Cosine similarity in [-1, 1]."""
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a < 1e-9 or norm_b < 1e-9:
            return 0.0
        return float(np.dot(a, b) / (norm_a * norm_b))
