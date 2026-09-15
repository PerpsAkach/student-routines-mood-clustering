from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.manifold import TSNE


def tsne_projection(
    features: pd.DataFrame,
    *,
    random_state: int = 42,
    perplexity: float = 30.0,
) -> pd.DataFrame:
    """Return a two-dimensional t-SNE projection for numeric behavioral features."""

    if not isinstance(features, pd.DataFrame):
        raise TypeError("features must be a pandas DataFrame")
    if len(features) < 3:
        raise ValueError("t-SNE requires at least three observations")
    if perplexity <= 0:
        raise ValueError("perplexity must be positive")

    numeric = features.select_dtypes(include=["number"]).copy()
    numeric = numeric.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    varying = numeric.nunique(dropna=False) > 1
    numeric = numeric.loc[:, varying]
    if numeric.empty:
        raise ValueError("numeric features contain no variance")

    effective_perplexity = min(float(perplexity), float(len(features) - 1))
    model = TSNE(
        n_components=2,
        perplexity=effective_perplexity,
        init="pca",
        learning_rate="auto",
        random_state=random_state,
    )
    coordinates = model.fit_transform(numeric.to_numpy(dtype=float))
    return pd.DataFrame(coordinates, columns=["TSNE_1", "TSNE_2"], index=features.index)


def plot_tsne(
    projection: pd.DataFrame,
    labels: pd.Series | None = None,
    *,
    title: str = "Student routine t-SNE projection",
):
    """Create a matplotlib figure; caller controls persistence/display."""

    required = {"TSNE_1", "TSNE_2"}
    missing = required.difference(projection.columns)
    if missing:
        raise ValueError(f"projection is missing columns: {sorted(missing)}")

    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    if labels is None:
        ax.scatter(projection["TSNE_1"], projection["TSNE_2"])
    else:
        values = pd.Series(labels, index=projection.index)
        for label in values.drop_duplicates().tolist():
            mask = values.eq(label)
            ax.scatter(
                projection.loc[mask, "TSNE_1"],
                projection.loc[mask, "TSNE_2"],
                label=str(label),
            )
        ax.legend(title="Cluster")
    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")
    ax.set_title(title)
    return fig, ax
