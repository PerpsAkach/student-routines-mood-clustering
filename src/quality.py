from __future__ import annotations

import numpy as np
import pandas as pd


def event_quality_summary(events: pd.DataFrame) -> dict[str, float | int]:
    required = {
        "Person_ID",
        "Day",
        "Start_Time",
        "Duration_Minutes",
        "Activity",
        "Satisfaction",
    }
    missing = required.difference(events.columns)
    if missing:
        raise ValueError(f"events is missing columns: {sorted(missing)}")

    total = len(events)
    if total == 0:
        return {
            "events": 0,
            "participants": 0,
            "participant_days": 0,
            "repeated_timestamp_rows": 0,
            "zero_duration_rows": 0,
            "missing_satisfaction_rows": 0,
            "missing_activity_rows": 0,
            "invalid_duration_rows": 0,
        }

    duration = pd.to_numeric(events["Duration_Minutes"], errors="coerce")
    repeated_rows = int(
        events.duplicated(subset=["Person_ID", "Day", "Start_Time"], keep=False).sum()
    )
    invalid_duration = duration.isna() | ~np.isfinite(duration) | duration.lt(0)

    return {
        "events": total,
        "participants": int(events["Person_ID"].nunique(dropna=True)),
        "participant_days": int(
            events[["Person_ID", "Day"]].drop_duplicates().shape[0]
        ),
        "repeated_timestamp_rows": repeated_rows,
        "zero_duration_rows": int(duration.eq(0).sum()),
        "missing_satisfaction_rows": int(events["Satisfaction"].isna().sum()),
        "missing_activity_rows": int(
            events["Activity"].isna().sum()
            + events["Activity"].astype(str).str.strip().eq("").sum()
        ),
        "invalid_duration_rows": int(invalid_duration.sum()),
    }
