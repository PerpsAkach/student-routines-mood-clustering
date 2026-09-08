from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

from .config import PipelineConfig, WEEKDAY_BLOCKS


def parse_sheet(raw: pd.DataFrame, sheet_name: str, config: PipelineConfig) -> pd.DataFrame:
    records = []

    for day, positions in WEEKDAY_BLOCKS.items():
        if max(positions) >= raw.shape[1]:
            continue

        block = raw.iloc[config.skip_rows:, positions].copy()
        block = block.iloc[:, :4]
        block.columns = ["Time", "Activity", "Who", "Satisfaction"]
        block["Where"] = np.nan

        for col in ["Activity", "Who", "Where", "Satisfaction"]:
            block[col] = block[col].ffill()

        block["Time"] = pd.to_datetime(block["Time"].astype(str), errors="coerce")
        block = block.dropna(subset=["Time", "Activity"])
        if block.empty:
            continue

        block["Day"] = day
        block["Person_ID"] = sheet_name
        records.append(block[["Person_ID", "Day", "Time", "Activity", "Who", "Where", "Satisfaction"]])

    if not records:
        return pd.DataFrame()

    df = pd.concat(records, ignore_index=True)
    df = df.sort_values(["Person_ID", "Day", "Time"]).reset_index(drop=True)
    df["Start_Time"] = df["Time"]
    df["End_Time"] = df.groupby(["Person_ID", "Day"])["Start_Time"].shift(-1)

    last = df["End_Time"].isna()
    df.loc[last, "End_Time"] = df.loc[last, "Start_Time"] + pd.to_timedelta(
        config.default_last_event_minutes, unit="m"
    )
    df["Duration_Minutes"] = (
        (df["End_Time"] - df["Start_Time"]).dt.total_seconds() / 60
    )
    return df


def load_workbook(path: Path, config: PipelineConfig | None = None) -> pd.DataFrame:
    cfg = config or PipelineConfig()
    sheets = pd.read_excel(path, sheet_name=None, header=None, engine="openpyxl")
    frames = []

    for name, frame in sheets.items():
        if re.match(cfg.sheet_regex, str(name)):
            parsed = parse_sheet(frame, str(name), cfg)
            if not parsed.empty:
                frames.append(parsed)

    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
