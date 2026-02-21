"""
Layer 5 — Federated Behavioral Trust  (RECTIFIED v2)
======================================================

RECTIFICATIONS:

  RECT 1 ── EMA α=0.70 is too aggressive for new clients
    OLD: A brand-new client with INITIAL_TRUST=0.5 that submits one high-quality
         gradient immediately gets trust = 0.7 × 1.0 + 0.3 × 0.5 = 0.85.
         Three rounds later of perfect gradients → trust ≈ 0.99.
         Conversely, three bad rounds → trust ≈ 0.03 (quarantined).
         This is hypersensitive — a single bad round from a mostly-honest
         client (e.g. data quality issue) moves trust from 0.85 to 0.28
         (just below quarantine threshold).
    FIX: Use a WARM-UP period for new clients. For the first
         MIN_ROUNDS_BEFORE_QUARANTINE rounds, use a slower α=0.30.
         After warm-up, use α=0.70 for quick response. This prevents
         premature quarantine without sacrificing responsiveness.

  RECT 2 ── fingerprint (running mean gradient) accumulation has an off-by-one
    OLD: n = self.round_counts[client_id]  (AFTER incrementing)
         self.fingerprints[client_id] = fp × (n-1)/n + grad / n
         On round 2 (n=2): fp = fp × 0.5 + grad × 0.5  ✓
         On round 3 (n=3): fp = fp × 0.667 + grad × 0.333  ✓
         Actually this is correct — the Welford/running-mean formula is right.
         But the fingerprint is never USED anywhere in the pipeline.
    FIX: Use fingerprint in a gradient fingerprint divergence check.
         Flag clients whose current gradient deviates strongly from their own
         historical fingerprint (sudden behavioral change = potential compromise).

  RECT 3 ── simulate_round uses fixed seed 42 every call → always same result
    OLD: rng = np.random.RandomState(42) inside simulate_round().
         Every call produces identical synthetic gradients regardless of round.
         Trust scores therefore don't evolve in simulation mode.
    FIX: Seed with a round counter so simulated rounds progress over time.

  RECT 4 ── quarantined clients remain quarantined even if trust recovers
    OLD: Once added to self.quarantined set, a client is never removed.
         A legitimate client falsely quarantined (3 bad rounds early on)
         can never recover even after 100 good rounds.
    FIX: Re-evaluate quarantine status each round. If trust recovers above
         QUARANTINE_THRESHOLD × 1.5 (rehabilitation threshold), remove
         from quarantined set and log the rehabilitation.

  Previously fixed bugs (EMA direction, normalisation, fraction-based
  suspicion score) are preserved.
"""

import numpy as np
from typing import Dict


# ── Constants ──────────────────────────────────────────────────────────────────
EMA_ALPHA_WARMUP         = 0.30    # RECT 1: slow α during warm-up
EMA_ALPHA_ACTIVE         = 0.70    # fast α after warm-up
QUARANTINE_THRESHOLD     = 0.30
REHABILITATION_THRESHOLD = 0.45    # RECT 4: trust must reach this to un-quarantine
SUSPICION_THRESHOLD      = 0.50
INITIAL_TRUST            = 0.50
MIN_ROUNDS_BEFORE_QUARANTINE = 3
FINGERPRINT_DIVERGENCE_ALARM = 0.70   # RECT 2: cosine divergence from own history
SIMULATION_N_CLIENTS     = 5


class FederatedTrustAnalyzer:
    """
    Per-client trust scoring via EMA of gradient cosine similarity.

    Improvements over v1:
      - Warm-up EMA α prevents premature quarantine of new clients.
      - Fingerprint divergence detects sudden behavioral change.
      - Quarantine is reversible when trust rehabilitates.
      - simulate_round progresses over time (seeded with round counter).
    """

    def __init__(self):
        self.trust_scores : Dict[str, float]      = {}
        self.round_counts : Dict[str, int]         = {}
        self.quarantined  : set                    = set()
        self.fingerprints : Dict[str, np.ndarray]  = {}
        self._sim_round   : int                    = 0   # RECT 3

    # ──────────────────────────────────────────────────────────────────────────
    def update_trust(self, client_id: str,
                     client_gradient: np.ndarray,
                     global_gradient: np.ndarray) -> dict:
        client_grad = np.array(client_gradient, dtype=float).flatten()
        global_grad = np.array(global_gradient, dtype=float).flatten()

        cos_sim        = self._cosine_similarity(client_grad, global_grad)
        sim_normalised = float(np.clip((cos_sim + 1.0) / 2.0, 0.0, 1.0))

        # RECT 1: warm-up α for new clients
        prev_trust  = self.trust_scores.get(client_id, INITIAL_TRUST)
        rounds_done = self.round_counts.get(client_id, 0)
        alpha       = EMA_ALPHA_WARMUP if rounds_done < MIN_ROUNDS_BEFORE_QUARANTINE else EMA_ALPHA_ACTIVE
        new_trust   = alpha * sim_normalised + (1 - alpha) * prev_trust
        self.trust_scores[client_id] = float(new_trust)

        # Round counter
        self.round_counts[client_id] = rounds_done + 1
        rounds = self.round_counts[client_id]

        # RECT 2: fingerprint update + divergence check
        fp_alarm, fp_divergence = self._update_fingerprint(client_id, client_grad, rounds)

        # RECT 4: quarantine with rehabilitation
        status = self._get_status(new_trust, rounds, client_id)

        return {
            "client_id"              : client_id,
            "trust_score"            : float(new_trust),
            "cosine_similarity"      : float(cos_sim),
            "sim_normalised"         : float(sim_normalised),
            "status"                 : status,
            "rounds_participated"    : rounds,
            "is_quarantined"         : client_id in self.quarantined,
            "fingerprint_divergence" : round(float(fp_divergence), 4),
            "fingerprint_alarm"      : bool(fp_alarm),
            "ema_alpha_used"         : float(alpha),
        }

    def analyze(self, client_gradients: dict, global_gradient: np.ndarray) -> dict:
        per_client = {}
        for cid, grad in client_gradients.items():
            per_client[cid] = self.update_trust(cid, grad, global_gradient)
        return self._compute_layer_result(per_client)

    def get_all_trust_scores(self) -> dict:
        return {
            cid: {
                "trust_score"   : float(score),
                "status"        : self._get_status(score, self.round_counts.get(cid, 0), cid),
                "is_quarantined": cid in self.quarantined,
                "rounds"        : self.round_counts.get(cid, 0),
            }
            for cid, score in self.trust_scores.items()
        }

    def simulate_round(self, n_clients: int = SIMULATION_N_CLIENTS) -> dict:
        """RECT 3: seed with round counter so simulated trust evolves."""
        self._sim_round += 1
        rng         = np.random.RandomState(self._sim_round)
        dim         = 50
        global_grad = rng.randn(dim)
        global_grad /= np.linalg.norm(global_grad) + 1e-9

        client_grads = {}
        for i in range(n_clients):
            cid = f"sim_client_{i:03d}"
            if i == 0:
                # Simulate adversarial client that gradually turns malicious
                noise = max(0.05, 0.5 - self._sim_round * 0.05)
                g     = -global_grad + rng.randn(dim) * noise
            else:
                noise_level = rng.uniform(0.05, 0.20)
                g           = global_grad + rng.randn(dim) * noise_level
            g /= np.linalg.norm(g) + 1e-9
            client_grads[cid] = g

        return self.analyze(client_grads, global_grad)

    # ──────────────────────────────────────────────────────────────────────────
    # RECT 2: fingerprint divergence
    # ──────────────────────────────────────────────────────────────────────────
    def _update_fingerprint(self, client_id: str,
                            client_grad: np.ndarray,
                            rounds: int) -> tuple:
        """
        Update running mean gradient fingerprint.
        Return (alarm, divergence_from_fingerprint).
        Alarm fires if current gradient is strongly anti-aligned with
        the client's own history (sudden behavioral change).
        """
        fp_alarm     = False
        fp_divergence = 0.0

        if client_id in self.fingerprints and rounds > MIN_ROUNDS_BEFORE_QUARANTINE:
            fp          = self.fingerprints[client_id]
            fp_cos      = self._cosine_similarity(client_grad, fp)
            fp_divergence = float((1.0 - fp_cos) / 2.0)  # normalised [0,1]
            fp_alarm    = fp_divergence > FINGERPRINT_DIVERGENCE_ALARM

        # Welford running mean update
        n = rounds
        if client_id in self.fingerprints:
            self.fingerprints[client_id] = (
                self.fingerprints[client_id] * (n - 1) / n + client_grad / n
            )
        else:
            self.fingerprints[client_id] = client_grad.copy()

        return fp_alarm, fp_divergence

    # ──────────────────────────────────────────────────────────────────────────
    # RECT 4: reversible quarantine
    # ──────────────────────────────────────────────────────────────────────────
    def _get_status(self, trust: float, rounds: int, client_id: str) -> str:
        # Rehabilitation: remove from quarantine if trust has recovered
        if client_id in self.quarantined and trust >= REHABILITATION_THRESHOLD:
            self.quarantined.discard(client_id)

        if trust < QUARANTINE_THRESHOLD and rounds >= MIN_ROUNDS_BEFORE_QUARANTINE:
            self.quarantined.add(client_id)
            return "QUARANTINED"
        elif client_id in self.quarantined:
            return "QUARANTINED"   # still quarantined, hasn't rehabilitated yet
        elif trust < SUSPICION_THRESHOLD:
            return "SUSPICIOUS"
        elif trust < 0.70:
            return "ACCEPTABLE"
        return "TRUSTED"

    def _compute_layer_result(self, per_client: dict) -> dict:
        if not per_client:
            return {
                "suspicion_score"    : 0.0,
                "n_clients"          : 0,
                "n_quarantined"      : 0,
                "n_suspicious"       : 0,
                "n_fingerprint_alarm": 0,
                "quarantined_clients": [],
                "suspicious_clients" : [],
                "per_client"         : {},
                "avg_trust_score"    : 1.0,
            }

        n_total          = len(per_client)
        n_quarantined    = sum(1 for r in per_client.values() if r["trust_score"] < QUARANTINE_THRESHOLD)
        n_suspicious     = sum(1 for r in per_client.values() if r["trust_score"] < SUSPICION_THRESHOLD)
        n_fp_alarm       = sum(1 for r in per_client.values() if r.get("fingerprint_alarm", False))  # RECT 2
        avg_trust        = float(np.mean([r["trust_score"] for r in per_client.values()]))

        quarantine_frac  = n_quarantined / n_total
        suspicious_frac  = n_suspicious  / n_total
        fp_frac          = n_fp_alarm    / n_total

        # Fingerprint alarms add a small additional signal
        raw_suspicion = (quarantine_frac * 2.0 + suspicious_frac * 0.5 + fp_frac * 0.3) / 2.8
        suspicion     = float(np.clip(raw_suspicion, 0.0, 1.0))

        return {
            "suspicion_score"     : suspicion,
            "n_clients"           : n_total,
            "n_quarantined"       : n_quarantined,
            "n_suspicious"        : n_suspicious,
            "n_fingerprint_alarm" : n_fp_alarm,
            "quarantined_clients" : [cid for cid, r in per_client.items()
                                     if r["trust_score"] < QUARANTINE_THRESHOLD],
            "suspicious_clients"  : [cid for cid, r in per_client.items()
                                     if QUARANTINE_THRESHOLD <= r["trust_score"] < SUSPICION_THRESHOLD],
            "per_client"          : per_client,
            "avg_trust_score"     : avg_trust,
        }

    @staticmethod
    def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
        na = np.linalg.norm(a)
        nb = np.linalg.norm(b)
        if na < 1e-9 or nb < 1e-9:
            return 0.0
        return float(np.dot(a, b) / (na * nb))

    def analyze_clients(self, clients: list) -> dict:
        """
        Analyze a list of client dicts, each with:
          - client_id: str
          - gradient: np.ndarray
          - global_gradient: np.ndarray
        Returns the shape expected by the frontend:
          { clients: [...], n_quarantined, avg_trust, quarantined_clients }
        """
        results = []
        for c in clients:
            cid = c["client_id"]
            grad = np.array(c["gradient"], dtype=float)
            glob = np.array(c["global_gradient"], dtype=float)

            # Run multiple rounds for warm-up so trust converges
            for _ in range(c.get("rounds", 1)):
                r = self.update_trust(cid, grad, glob)

            results.append({
                "client_id": cid,
                "trust_score": r["trust_score"],
                "cosine_similarity": r["cosine_similarity"],
                "status": r["status"],
                "quarantined": r["is_quarantined"],
                "rounds_participated": r["rounds_participated"],
                "fingerprint_divergence": r["fingerprint_divergence"],
            })

        n_quarantined = sum(1 for r in results if r["quarantined"])
        avg_trust = float(np.mean([r["trust_score"] for r in results])) if results else 0.0

        return {
            "clients": results,
            "n_quarantined": n_quarantined,
            "avg_trust": round(avg_trust, 4),
            "quarantined_clients": [r["client_id"] for r in results if r["quarantined"]],
        }


def generate_demo_clients(n_clients: int = 8, dim: int = 50, seed: int = 42) -> list:
    """
    Generate demo federated clients with diverse trust profiles.
    Returns a list of dicts ready for FederatedTrustAnalyzer.analyze_clients().
    """
    rng = np.random.RandomState(seed)
    global_grad = rng.randn(dim)
    global_grad /= np.linalg.norm(global_grad) + 1e-9

    profiles = [
        # (name_suffix, noise_level, invert, rounds)
        ("honest_A",       0.05, False, 5),
        ("honest_B",       0.08, False, 5),
        ("honest_C",       0.12, False, 4),
        ("noisy_D",        0.35, False, 4),
        ("noisy_E",        0.50, False, 3),
        ("suspicious_F",   0.70, False, 3),
        ("malicious_G",    0.20, True,  5),   # inverts gradient
        ("malicious_H",    0.10, True,  5),   # inverts gradient
    ]

    clients = []
    for i in range(min(n_clients, len(profiles))):
        suffix, noise, invert, rounds = profiles[i]
        base = -global_grad if invert else global_grad
        grad = base + rng.randn(dim) * noise
        grad /= np.linalg.norm(grad) + 1e-9

        clients.append({
            "client_id": f"client_{suffix}",
            "gradient": grad.tolist(),
            "global_gradient": global_grad.tolist(),
            "rounds": rounds,
        })

    return clients

