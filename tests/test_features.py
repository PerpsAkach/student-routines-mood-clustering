import numpy as np
import pandas as pd

from src.features import build_daily_features, clustering_feature_columns
from src.semantic import aggregate_student_period_text, embed_activities, fit_tfidf


class EncoderDouble:
    def encode(self, texts, batch_size, show_progress_bar, normalize_embeddings):
        assert batch_size == 2
        assert show_progress_bar is False
        assert normalize_embeddings is True
        return np.asarray([[1.0, 0.0] for _ in texts])


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


def test_student_period_text_separates_weekdays_and_weekends():
    events = pd.DataFrame(
        {
            "Person_ID": ["A_1", "A_1"],
            "Day": ["Monday", "Saturday"],
            "Activity": ["study", "exercise"],
        }
    )
    result = aggregate_student_period_text(events)
    assert set(result["Period"]) == {"Weekday", "Weekend"}
