import pandas as pd
import pytest

from src.analysis import correlation_summary, mood_by_category, mood_feature_correlations


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
