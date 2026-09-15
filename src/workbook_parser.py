from __future__ import annotations

import re
from datetime import datetime, time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .config import DEFAULT_CONFIG, WEEKDAY_BLOCKS, PipelineConfig

EVENT_COLUMNS = [
    "Person_ID",
    "Day",
    "Time",
    "Activity",
    "Who",
    "Where",
    "Satisfaction",
    "Start_Time",
    "End_Time",
    "Duration_Minutes",
]


def _empty_events() -> pd.DataFrame:
    return pd.DataFrame(columns=EVENT_COLUMNS)


def _parse_time_value(value: Any) -> pd.Timestamp | pd.NaT:
    """Normalize common Excel/time representations onto a fixed reference date."""

    if value is None or pd.isna(value):
        return pd.NaT

    reference = pd.Timestamp("2000-01-01")

    if isinstance(value, pd.Timestamp):
        return reference + pd.to_timedelta(
            value.hour * 3600 + value.minute * 60 + value.second,
            unit="s",
        )
    if isinstance(value, datetime):
        return reference + pd.to_timedelta(
            value.hour * 3600 + value.minute * 60 + value.second,
            unit="s",
        )
    if isinstance(value, time):
        return reference + pd.to_timedelta(
            value.hour * 3600 + value.minute * 60 + value.second,
            unit="s",
        )
    if isinstance(value, (int, float, np.integer, np.floating)) and not isinstance(
        value, bool
    ):
        numeric = float(value)
        if np.isfinite(numeric) and 0 <= numeric < 1:
            return reference + pd.to_timedelta(numeric, unit="D")

    text = str(value).strip()
    if not text:
        return pd.NaT

    parsed = pd.to_datetime(text, errors="coerce")
    if pd.isna(parsed):
        return pd.NaT
    return reference + pd.to_timedelta(
        parsed.hour * 3600 + parsed.minute * 60 + parsed.second,
        unit="s",
    )


def parse_sheet(
    raw: pd.DataFrame,
    sheet_name: str,
    config: PipelineConfig | None = None,
) -> pd.DataFrame:
    cfg = config or DEFAULT_CONFIG
    cfg.validate()

    if not isinstance(raw, pd.DataFrame):
        raise TypeError("raw must be a pandas DataFrame")
    if not isinstance(sheet_name, str) or not sheet_name.strip():
        raise ValueError("sheet_name must be a non-empty string")

    records: list[pd.DataFrame] = []

    for day, positions in WEEKDAY_BLOCKS.items():
        if max(positions) >= raw.shape[1]:
            continue

        block = raw.iloc[cfg.skip_rows :, positions].copy().iloc[:, :4]
        block.columns = ["Time", "Activity", "Who", "Satisfaction"]
        block["Where"] = pd.NA

        # Workbook cells were historically visually merged/continued. Forward fill
        # only contextual columns; never forward-fill timestamps. Opt in to pandas'
        # future no-silent-downcasting behavior so dtype handling is explicit.
        with pd.option_context("future.no_silent_downcasting", True):
            for col in ["Activity", "Who", "Satisfaction"]:
                block[col] = block[col].ffill().infer_objects(copy=False)

        block["Time"] = block["Time"].map(_parse_time_value)
        block["Activity"] = block["Activity"].map(
            lambda value: value.strip() if isinstance(value, str) else value
        )
        block = block.dropna(subset=["Time", "Activity"])
        block = block[block["Activity"].astype(str).str.strip().ne("")]
        if block.empty:
            continue

        block["Day"] = day
        block["Person_ID"] = sheet_name
        records.append(
            block[
                [
                    "Person_ID",
                    "Day",
                    "Time",
                    "Activity",
                    "Who",
                    "Where",
                    "Satisfaction",
                ]
            ]
        )

    if not records:
        return _empty_events()

    df = pd.concat(records, ignore_index=True)
    df = df.sort_values(["Person_ID", "Day", "Time"], kind="stable").reset_index(
        drop=True
    )
    df["Start_Time"] = df["Time"]
    df["End_Time"] = df.groupby(["Person_ID", "Day"], sort=False)[
        "Start_Time"
    ].shift(-1)

    last = df["End_Time"].isna()
    df.loc[last, "End_Time"] = df.loc[last, "Start_Time"] + pd.to_timedelta(
        cfg.default_last_event_minutes, unit="m"
    )
    df["Duration_Minutes"] = (
        (df["End_Time"] - df["Start_Time"]).dt.total_seconds() / 60
    )

    if (df["Duration_Minutes"] < 0).any():
        raise ValueError("Negative event durations were reconstructed")

    return df[EVENT_COLUMNS]


def load_workbook(path: Path, config: PipelineConfig | None = None) -> pd.DataFrame:
    cfg = config or DEFAULT_CONFIG
    cfg.validate()

    workbook = Path(path)
    if not workbook.exists():
        raise FileNotFoundError(f"Workbook not found: {workbook}")
    if workbook.suffix.lower() not in {".xlsx", ".xlsm"}:
        raise ValueError("Workbook must be an .xlsx or .xlsm file")

    sheets = pd.read_excel(workbook, sheet_name=None, header=None, engine="openpyxl")
    frames: list[pd.DataFrame] = []

    for name, frame in sheets.items():
        if re.fullmatch(cfg.sheet_regex, str(name)):
            parsed = parse_sheet(frame, str(name), cfg)
            if not parsed.empty:
                frames.append(parsed)

    return pd.concat(frames, ignore_index=True) if frames else _empty_events()
