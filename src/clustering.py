from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


def _numeric_feature_frame(features: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(features, pd.DataFrame):
        raise TypeError("features must be a pandas DataFrame")
    if features.empty:
        raise ValueError("features must contain at least one row")

    numeric = features.select_dtypes(include=["number"]).copy()
    if numeric.empty:
        raise ValueError("features must contain at least one numeric column")

    numeric = numeric.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    varying = numeric.nunique(dropna=False) > 1
    numeric = numeric.loc[:, varying]
    if numeric.empty:
        raise ValueError("numeric features contain no variance")
    return numeric


def run_hdbscan(
    features: pd.DataFrame,
    min_cluster_size: int = 15,
    min_samples: int = 5,
):
    """Standardize numeric features and run HDBSCAN with explicit guardrails."""

    if min_cluster_size < 2:
        raise ValueError("min_cluster_size must be at least 2")
    if min_samples < 1:
        raise ValueError("min_samples must be positive")
    if len(features) < min_cluster_size:
        raise ValueError(
            "number of observations must be at least min_cluster_size "
            f"({len(features)} < {min_cluster_size})"
        )

    import hdbscan

    numeric = _numeric_feature_frame(features)
    scaler = StandardScaler()
    x = scaler.fit_transform(numeric)

    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        prediction_data=True,
    )
    labels = clusterer.fit_predict(x)

    out = features.copy()
    out["Cluster"] = labels.astype(int)
    out["Cluster_Probability"] = np.asarray(clusterer.probabilities_, dtype=float)
    return out, clusterer, scaler


def cluster_diagnostics(clustered: pd.DataFrame) -> dict[str, float | int]:
    """Return compact diagnostics without treating noise as a substantive cluster."""

    required = {"Cluster", "Cluster_Probability"}
    missing = required.difference(clustered.columns)
    if missing:
        raise ValueError(f"clustered frame is missing columns: {sorted(missing)}")
    if clustered.empty:
        return {
            "observations": 0,
            "clusters": 0,
            "noise_observations": 0,
            "noise_fraction": 0.0,
            "mean_assignment_probability": 0.0,
        }

    labels = pd.to_numeric(clustered["Cluster"], errors="coerce")
    probabilities = pd.to_numeric(
        clustered["Cluster_Probability"], errors="coerce"
    ).fillna(0.0)
    noise = labels.eq(-1)
    substantive = {int(label) for label in labels.dropna().unique() if label >= 0}

    return {
        "observations": len(clustered),
        "clusters": len(substantive),
        "noise_observations": int(noise.sum()),
        "noise_fraction": float(noise.mean()),
        "mean_assignment_probability": float(probabilities.mean()),
    }
