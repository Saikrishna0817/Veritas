"""The closed-form leave-one-column-out ridge in veritas2 matches a Ridge fitted per column."""
import numpy as np
from sklearn.linear_model import Ridge

from app.detection.veritas2 import _loo_ridge


def test_matches_per_column_ridge():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(300, 12))
    X[:, 3] = 2 * X[:, 0] + rng.normal(scale=0.1, size=300)   # correlated column
    X[:, 5] = 1.0                                              # constant column
    X[:, 6] = (rng.random(300) < 0.1).astype(float)            # sparse one-hot-like column
    Z = rng.normal(size=(50, 12))
    resid, s = _loo_ridge(X)
    for j in range(X.shape[1]):
        keep = np.arange(X.shape[1]) != j
        m = Ridge(1.0).fit(X[:, keep], X[:, j])
        np.testing.assert_allclose(resid(Z)[:, j], Z[:, j] - m.predict(Z[:, keep]), atol=1e-8)
        np.testing.assert_allclose(s[j], max(np.std(X[:, j] - m.predict(X[:, keep])), 1e-6), atol=1e-8)
