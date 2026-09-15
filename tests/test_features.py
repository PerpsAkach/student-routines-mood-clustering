import pandas as pd

from src.features import build_daily_features, clustering_feature_columns


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
