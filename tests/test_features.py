import numpy as np
import pandas as pd
import pytest

from src.features import build_daily_features, clustering_feature_columns
from src.semantic import (
    aggregate_student_period_text,
    embed_activities,
    fit_tfidf,
    weekday_weekend_semantic_distance,
)
from src.visualization import tsne_projection


class EncoderDouble:
    def encode(self, texts, batch_size, show_progress_bar, normalize_embeddings):
        assert batch_size == 2
        assert show_progress_bar is False
        assert normalize_embeddings is True
        vectors = []
        for text in texts:
            vectors.append([1.0, 0.0] if "study" in text else [0.0, 1.0])
        return np.asarray(vectors)


def test_build_daily_features_keeps_mood_outside_behavioral_columns():
    events = pd.DataFrame(
        {
            "Person_ID": ["A_1", "A_1", "A_1"],
            "Day": ["Monday", "Monday", "Saturday"],
            "Category": ["Sleep", "Academic", "Exercise"],
            "Duration_Minutes": [480.0, 120.0, 60.0],
            "Satisfaction_Numeric": [3.0, 5.0, 4.0],
        }
    )

    daily = build_daily_features(events)
    monday = daily[daily["Day"].eq("Monday")].iloc[0]
    saturday = daily[daily["Day"].eq("Saturday")].iloc[0]

    assert monday["Minutes_Sleep"] == 480.0
    assert monday["Minutes_Academic"] == 120.0
    assert monday["Mean_Mood"] == 4.0
    assert saturday["Weekend"] == 1

    columns = clustering_feature_columns(daily)
    assert "Mean_Mood" not in columns
    assert "Person_ID" not in columns
    assert "Day" not in columns
    assert "Minutes_Sleep" in columns


def test_tfidf_and_injected_encoder_are_offline_testable():
    vectorizer, matrix = fit_tfidf(["study class", "gym workout"], max_features=10)
    assert matrix.shape[0] == 2
    assert "study" in vectorizer.vocabulary_

    embeddings = embed_activities(
        ["study", "exercise"], batch_size=2, model=EncoderDouble()
    )
    assert embeddings.shape == (2, 2)


def test_student_period_text_and_semantic_distance():
    events = pd.DataFrame(
        {
            "Person_ID": ["A_1", "A_1"],
            "Day": ["Monday", "Saturday"],
            "Activity": ["study", "exercise"],
        }
    )
    period = aggregate_student_period_text(events)
    assert set(period["Period"]) == {"Weekday", "Weekend"}

    distances = weekday_weekend_semantic_distance(
        period,
        batch_size=2,
        model=EncoderDouble(),
    )
    assert distances.loc[0, "Person_ID"] == "A_1"
    assert distances.loc[0, "Semantic_Distance"] == pytest.approx(1.0)


def test_semantic_distance_returns_empty_without_both_periods():
    period = pd.DataFrame(
        {
            "Person_ID": ["A_1"],
            "Period": ["Weekday"],
            "Activity_Text": ["study"],
        }
    )
    assert weekday_weekend_semantic_distance(period, model=EncoderDouble()).empty


def test_tsne_projection_is_deterministic_for_fixed_seed():
    frame = pd.DataFrame(
        {
            "sleep": [1.0, 2.0, 3.0, 4.0, 5.0],
            "study": [5.0, 4.0, 3.0, 2.0, 1.0],
        }
    )
    first = tsne_projection(frame, random_state=42, perplexity=2)
    second = tsne_projection(frame, random_state=42, perplexity=2)
    assert first.shape == (5, 2)
    np.testing.assert_allclose(first.to_numpy(), second.to_numpy())
