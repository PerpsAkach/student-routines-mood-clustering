import pandas as pd
import pytest

from src.analysis import (
    correlation_summary,
    mood_by_category,
    mood_feature_correlations,
)
from src.quality import event_quality_summary
from src.routine_similarity import routine_similarity, routines_match


def test_mood_by_category_aggregates_descriptive_statistics():
    events = pd.DataFrame(
        {
            "Category": ["Sleep", "Sleep", "Academic"],
            "Satisfaction_Numeric": [3, 5, 4],
            "Duration_Minutes": [480, 420, 120],
        }
    )
    result = mood_by_category(events)
    sleep = result[result["Category"].eq("Sleep")].iloc[0]
    assert sleep["Mean_Mood"] == 4
    assert sleep["Median_Mood"] == 4
    assert sleep["N"] == 2
    assert sleep["Total_Minutes"] == 900


def test_correlation_summary_uses_complete_pairs():
    frame = pd.DataFrame({"x": [1, 2, 3, None], "y": [2, 4, 6, 8]})
    result = correlation_summary(frame, "x", "y")
    assert result["n"] == 3
    assert result["pearson_r"] == pytest.approx(1.0)
    assert result["spearman_rho"] == pytest.approx(1.0)


def test_correlation_summary_handles_constant_or_small_samples():
    constant = correlation_summary(
        pd.DataFrame({"x": [1, 1, 1], "y": [1, 2, 3]}), "x", "y"
    )
    assert constant["pearson_r"] is None

    small = correlation_summary(pd.DataFrame({"x": [1, 2], "y": [2, 3]}), "x", "y")
    assert small["n"] == 2
    assert small["spearman_rho"] is None


def test_mood_feature_correlations_only_uses_behavioral_features():
    daily = pd.DataFrame(
        {
            "Person_ID": ["A", "B", "C"],
            "Mean_Mood": [1, 2, 3],
            "Minutes_Sleep": [100, 200, 300],
            "Event_Count": [5, 6, 7],
        }
    )
    result = mood_feature_correlations(daily)
    assert set(result["x"]) == {"Minutes_Sleep", "Event_Count"}


def test_quality_summary_counts_common_input_issues():
    events = pd.DataFrame(
        {
            "Person_ID": ["A_1", "A_1"],
            "Day": ["Monday", "Monday"],
            "Start_Time": pd.to_datetime(["2000-01-01 08:00", "2000-01-01 08:00"]),
            "Duration_Minutes": [0.0, 30.0],
            "Activity": ["study", "class"],
            "Satisfaction": [4, None],
        }
    )
    result = event_quality_summary(events)
    assert result["events"] == 2
    assert result["repeated_timestamp_rows"] == 2
    assert result["zero_duration_rows"] == 1
    assert result["missing_satisfaction_rows"] == 1


def test_routine_similarity_is_normalized_and_thresholded():
    assert routine_similarity("Study   Class", "study class") == pytest.approx(1.0)
    assert routines_match("study class", "study class", threshold=0.85)
    with pytest.raises(ValueError, match="threshold"):
        routines_match("a", "b", threshold=1.1)
