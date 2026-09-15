from __future__ import annotations

import math

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr


def mood_by_category(events: pd.DataFrame) -> pd.DataFrame:
    required = {"Category", "Satisfaction_Numeric", "Duration_Minutes"}
    missing = required.difference(events.columns)
    if missing:
        raise ValueError(f"events is missing columns: {sorted(missing)}")

    if events.empty:
        return pd.DataFrame(
            columns=["Category", "Mean_Mood", "Median_Mood", "N", "Total_Minutes"]
        )

    work = events.copy()
    work["Satisfaction_Numeric"] = pd.to_numeric(
        work["Satisfaction_Numeric"], errors="coerce"
    )
    work["Duration_Minutes"] = pd.to_numeric(
        work["Duration_Minutes"], errors="coerce"
    )
    return (
        work.groupby("Category", as_index=False, sort=True)
        .agg(
            Mean_Mood=("Satisfaction_Numeric", "mean"),
            Median_Mood=("Satisfaction_Numeric", "median"),
            N=("Satisfaction_Numeric", "count"),
            Total_Minutes=("Duration_Minutes", "sum"),
        )
        .sort_values("Category")
        .reset_index(drop=True)
    )


def correlation_summary(
    frame: pd.DataFrame,
    x_col: str,
    y_col: str,
    *,
    min_pairs: int = 3,
) -> dict[str, float | int | str | None]:
    """Compute Pearson and Spearman associations using complete finite pairs."""

    if min_pairs < 3:
        raise ValueError("min_pairs must be at least 3")
    for column in (x_col, y_col):
        if column not in frame.columns:
            raise ValueError(f"missing column: {column}")

    pairs = frame[[x_col, y_col]].apply(pd.to_numeric, errors="coerce")
    pairs = pairs.replace([np.inf, -np.inf], np.nan).dropna()

    result: dict[str, float | int | str | None] = {
        "x": x_col,
        "y": y_col,
        "n": int(len(pairs)),
        "pearson_r": None,
        "pearson_p": None,
        "spearman_rho": None,
        "spearman_p": None,
    }
    if len(pairs) < min_pairs:
        return result
    if pairs[x_col].nunique() < 2 or pairs[y_col].nunique() < 2:
        return result

    pearson = pearsonr(pairs[x_col], pairs[y_col])
    spearman = spearmanr(pairs[x_col], pairs[y_col])

    values = {
        "pearson_r": float(pearson.statistic),
        "pearson_p": float(pearson.pvalue),
        "spearman_rho": float(spearman.statistic),
        "spearman_p": float(spearman.pvalue),
    }
    for key, value in values.items():
        result[key] = value if math.isfinite(value) else None
    return result


def mood_feature_correlations(daily: pd.DataFrame) -> pd.DataFrame:
    """Associate behavioral duration/count features with daily mean mood.

    These are descriptive observational associations; they are not causal
    estimates and repeated participant-days are not treated as independent by
    any mixed-effects model here.
    """

    if "Mean_Mood" not in daily.columns:
        raise ValueError("daily frame is missing Mean_Mood")

    feature_columns = [
        column
        for column in daily.columns
        if column.startswith("Minutes_")
        or column in {"Event_Count", "Total_Observed_Minutes", "Weekend"}
    ]
    rows = [correlation_summary(daily, column, "Mean_Mood") for column in feature_columns]
    return pd.DataFrame(rows)
