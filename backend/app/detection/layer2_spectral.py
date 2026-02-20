"""Detection Layer 2: Spectral Activation Analysis"""
import numpy as np
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from typing import Dict, Any, List


class SpectralActivationAnalyzer:
    """
    Analyzes model activation space to detect backdoor clusters.
    Uses SVD + PCA + KMeans on simulated hidden layer activations.
    """

    def __init__(self, n_components: int = 5, n_clusters: int = 2):
        self.n_components = n_components
        self.n_clusters = n_clusters

    def _simulate_activations(self, features: np.ndarray, labels: np.ndarray) -> np.ndarray:
        """
        Simulate penultimate layer activations from feature vectors.
        In production, this hooks into the actual model's hidden layers.
        """
        np.random.seed(0)
        W = np.random.randn(features.shape[1], 32)
        activations = np.tanh(features @ W)
        # Add label-correlated signal
        label_signal = labels.reshape(-1, 1) * np.random.randn(1, 32) * 0.5
        return activations + label_signal

    def analyze(self, features: np.ndarray, labels: np.ndarray,
                poison_mask: np.ndarray = None) -> Dict[str, Any]:
        """
        Detect backdoor clusters in activation space.
        Returns cluster analysis and suspicion score.
        """
        activations = self._simulate_activations(features, labels)

        # PCA reduction
        n_comp = min(self.n_components, activations.shape[1], activations.shape[0] - 1)
        pca = PCA(n_components=n_comp)
        pca_acts = pca.fit_transform(activations)

        # SVD for spectral analysis
        U, S, Vt = np.linalg.svd(activations, full_matrices=False)
        spectral_gap = float(S[0] / (S[1] + 1e-8))  # Large gap = suspicious subspace

        # KMeans clustering
        n_clust = min(self.n_clusters, len(features))
        kmeans = KMeans(n_clusters=n_clust, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(pca_acts)

        # Detect outlier cluster (minority = suspicious)
        cluster_sizes = np.bincount(cluster_labels)
        minority_cluster = int(np.argmin(cluster_sizes))
        minority_size = int(cluster_sizes[minority_cluster])
        minority_ratio = float(minority_size / len(features))

        # Backdoor suspicion: small tight cluster
        cluster_distances = []
        for c in range(n_clust):
            mask = cluster_labels == c
            if mask.sum() > 1:
                center = pca_acts[mask].mean(axis=0)
                dists = np.linalg.norm(pca_acts[mask] - center, axis=1)
                cluster_distances.append(float(dists.mean()))

        tightness = min(cluster_distances) if cluster_distances else 0.0
        backdoor_suspicion = (
            spectral_gap > 3.0 and
            minority_ratio < 0.15 and
            tightness < 0.5
        )

        # Per-sample suspicion (minority cluster members)
        suspicious_indices = list(np.where(cluster_labels == minority_cluster)[0])

        suspicion_score = min(1.0, round(
            (min(spectral_gap / 10.0, 0.5)) * 0.4 +
            (max(0, 0.15 - minority_ratio) / 0.15) * 0.4 +
            (max(0, 1.0 - tightness) * 0.2),
            4
        ))

        return {
            "spectral_gap": round(spectral_gap, 4),
            "minority_cluster_size": minority_size,
            "minority_cluster_ratio": round(minority_ratio, 4),
            "cluster_tightness": round(tightness, 4),
            "backdoor_detected": backdoor_suspicion,
            "suspicious_sample_indices": suspicious_indices[:20],  # top 20
            "explained_variance": [round(float(v), 4) for v in pca.explained_variance_ratio_],
            "suspicion_score": suspicion_score,
            "alarm": backdoor_suspicion or suspicion_score > 0.5
        }

    def analyze_samples(self, samples: List[Dict]) -> Dict[str, Any]:
        features = np.array([s["feature_vector"] for s in samples])
        labels = np.array([s["label"] for s in samples])
        return self.analyze(features, labels)
