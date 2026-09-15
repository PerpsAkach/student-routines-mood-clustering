from __future__ import annotations

import numpy as np
import pandas as pd


def build_daily_features(events: pd.DataFrame) -> pd.DataFrame:
    """Aggregate event-level data into one row per participant/day.

    Mood is retained as an interpretation variable and is not mixed into the
    behavioral feature columns used for clustering.
    """

    required = {
        "Person_ID",
        "Day",
        "Category",
        "Duration_Minutes",
        "Satisfaction_Numeric",
    }
    missing = required.difference(events.columns)
    if missing:
        raise ValueError(f"events is missing columns: {sorted(missing)}")
    if events.empty:
        return pd.DataFrame(columns=["Person_ID", "Day", "Mean_Mood", "Event_Count"])

    work = events.copy()
    work["Duration_Minutes"] = pd.to_numeric(
        work["Duration_Minutes"], errors="coerce"
    )
    work["Satisfaction_Numeric"] = pd.to_numeric(
        work["Satisfaction_Numeric"], errors="coerce"
    )
    work = work.replace([np.inf, -np.inf], np.nan)
    work = work[work["Duration_Minutes"].ge(0) | work["Duration_Minutes"].isna()]

    duration = (
        work.pivot_table(
            index=["Person_ID", "Day"],
            columns="Category",
            values="Duration_Minutes",
            aggfunc="sum",
            fill_value=0.0,
        )
        .add_prefix("Minutes_")
        .reset_index()
    )

    summary = (
        work.groupby(["Person_ID", "Day"], as_index=False, sort=True)
        .agg(
            Mean_Mood=("Satisfaction_Numeric", "mean"),
            Event_Count=("Category", "size"),
            Total_Observed_Minutes=("Duration_Minutes", "sum"),
        )
    )

    result = duration.merge(summary, on=["Person_ID", "Day"], how="left")
    result["Weekend"] = result["Day"].isin(["Saturday", "Sunday"]).astype(int)
    return result


def clustering_feature_columns(daily: pd.DataFrame) -> list[str]:
    """Select behavioral columns while excluding mood and identifiers."""

    candidates = [
        column
        for column in daily.columns
        if column.startswith("Minutes_")
        or column in {"Event_Count", "Total_Observed_Minutes", "Weekend"}
    ]
    return [column for column in candidates if pd.api.types.is_numeric_dtype(daily[column])]
