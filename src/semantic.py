from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def _clean_texts(texts: Iterable[object]) -> list[str]:
    cleaned = [str(value).strip() for value in texts if not pd.isna(value)]
    cleaned = [value for value in cleaned if value]
    if not cleaned:
        raise ValueError("at least one non-empty text is required")
    return cleaned


def fit_tfidf(texts: Iterable[object], max_features: int = 1000):
    if max_features < 1:
        raise ValueError("max_features must be positive")
    cleaned = _clean_texts(texts)
    vectorizer = TfidfVectorizer(
        lowercase=True,
        stop_words="english",
        ngram_range=(1, 2),
        max_features=max_features,
    )
    matrix = vectorizer.fit_transform(cleaned)
    return vectorizer, matrix


def embed_activities(
    texts: Iterable[object],
    model_name: str = "all-MiniLM-L6-v2",
    *,
    batch_size: int = 32,
    model=None,
) -> np.ndarray:
    """Create normalized sentence embeddings; model injection keeps tests offline."""

    if not isinstance(model_name, str) or not model_name.strip():
        raise ValueError("model_name must be a non-empty string")
    if batch_size < 1:
        raise ValueError("batch_size must be positive")

    cleaned = _clean_texts(texts)
    if model is None:
        from sentence_transformers import SentenceTransformer

        model = SentenceTransformer(model_name)

    embeddings = model.encode(
        cleaned,
        batch_size=batch_size,
        show_progress_bar=False,
        normalize_embeddings=True,
    )
    array = np.asarray(embeddings, dtype=float)
    if array.ndim != 2 or array.shape[0] != len(cleaned):
        raise ValueError("embedding model returned an unexpected shape")
    if not np.isfinite(array).all():
        raise ValueError("embedding model returned non-finite values")
    return array


def aggregate_student_period_text(df: pd.DataFrame) -> pd.DataFrame:
    required = {"Person_ID", "Day", "Activity"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"input frame is missing columns: {sorted(missing)}")

    work = df.dropna(subset=["Person_ID", "Day", "Activity"]).copy()
    work["Activity"] = work["Activity"].astype(str).str.strip()
    work = work[work["Activity"].ne("")]
    work["Weekend"] = work["Day"].isin(["Saturday", "Sunday"])
    work["Period"] = np.where(work["Weekend"], "Weekend", "Weekday")
    return (
        work.groupby(["Person_ID", "Period"], as_index=False, sort=True)
        .agg(Activity_Text=("Activity", lambda s: " ; ".join(s)))
    )


def weekday_weekend_semantic_distance(
    period_text: pd.DataFrame,
    *,
    model_name: str = "all-MiniLM-L6-v2",
    batch_size: int = 32,
    model=None,
) -> pd.DataFrame:
    """Compute cosine distance between each participant's weekday/weekend text."""

    required = {"Person_ID", "Period", "Activity_Text"}
    missing = required.difference(period_text.columns)
    if missing:
        raise ValueError(f"period_text is missing columns: {sorted(missing)}")

    pivot = period_text.pivot_table(
        index="Person_ID",
        columns="Period",
        values="Activity_Text",
        aggfunc="first",
    )
    if "Weekday" not in pivot.columns or "Weekend" not in pivot.columns:
        return pd.DataFrame(columns=["Person_ID", "Semantic_Distance"])

    complete = pivot.dropna(subset=["Weekday", "Weekend"])
    if complete.empty:
        return pd.DataFrame(columns=["Person_ID", "Semantic_Distance"])

    combined = complete["Weekday"].tolist() + complete["Weekend"].tolist()
    embeddings = embed_activities(
        combined,
        model_name=model_name,
        batch_size=batch_size,
        model=model,
    )
    n = len(complete)
    weekday = embeddings[:n]
    weekend = embeddings[n:]
    similarities = np.diag(cosine_similarity(weekday, weekend))

    return pd.DataFrame(
        {
            "Person_ID": complete.index.astype(str),
            "Semantic_Distance": np.clip(1.0 - similarities, 0.0, 2.0),
        }
    ).reset_index(drop=True)
