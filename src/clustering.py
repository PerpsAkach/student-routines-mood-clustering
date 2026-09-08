from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


def run_hdbscan(
    features: pd.DataFrame,
    min_cluster_size: int = 15,
    min_samples: int = 5,
):
    import hdbscan

    numeric = features.select_dtypes(include=["number"]).fillna(0)
    scaler = StandardScaler()
    x = scaler.fit_transform(numeric)

    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        prediction_data=True,
    )
    labels = clusterer.fit_predict(x)

    out = features.copy()
    out["Cluster"] = labels
    out["Cluster_Probability"] = clusterer.probabilities_
    return out, clusterer, scaler
