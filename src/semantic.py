from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer


def fit_tfidf(texts, max_features: int = 1000):
    vectorizer = TfidfVectorizer(
        lowercase=True,
        stop_words="english",
        ngram_range=(1, 2),
        max_features=max_features,
    )
    matrix = vectorizer.fit_transform(texts)
    return vectorizer, matrix


def embed_activities(texts, model_name: str = "all-MiniLM-L6-v2"):
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(model_name)
    embeddings = model.encode(
        list(texts),
        show_progress_bar=True,
        normalize_embeddings=True,
    )
    return np.asarray(embeddings)


def aggregate_student_period_text(df: pd.DataFrame) -> pd.DataFrame:
    work = df.copy()
    work["Weekend"] = work["Day"].isin(["Saturday", "Sunday"])
    work["Period"] = np.where(work["Weekend"], "Weekend", "Weekday")
    return (
        work.groupby(["Person_ID", "Period"], as_index=False)
        .agg(Activity_Text=("Activity", lambda s: " ; ".join(map(str, s))))
    )
