from datetime import time

import pandas as pd
import pytest

from src.config import PipelineConfig
from src.workbook_parser import _parse_time_value, parse_sheet


def test_parse_time_value_handles_excel_fraction_and_time_object():
    assert _parse_time_value(0.5).strftime("%H:%M:%S") == "12:00:00"
    assert _parse_time_value(time(6, 30)).strftime("%H:%M:%S") == "06:30:00"


def test_parse_sheet_reconstructs_duration_and_terminal_event():
    raw = pd.DataFrame([[None] * 30 for _ in range(2)])
    raw.iloc[0, 2:6] = [0.25, "Breakfast", "Alone", 4]
    raw.iloc[1, 2:6] = [0.5, "Study", "Classmates", 5]

    parsed = parse_sheet(
        raw,
        "A_1",
        PipelineConfig(skip_rows=0, default_last_event_minutes=30),
    )

    monday = parsed[parsed["Day"].eq("Monday")].reset_index(drop=True)
    assert monday["Activity"].tolist() == ["Breakfast", "Study"]
    assert monday.loc[0, "Duration_Minutes"] == pytest.approx(360.0)
    assert monday.loc[1, "Duration_Minutes"] == pytest.approx(30.0)


def test_parse_sheet_returns_schema_when_no_valid_events():
    raw = pd.DataFrame([[None] * 30])
    parsed = parse_sheet(raw, "A_1", PipelineConfig(skip_rows=0))
    assert parsed.empty
    assert "Duration_Minutes" in parsed.columns


def test_parse_sheet_rejects_invalid_inputs():
    with pytest.raises(TypeError):
        parse_sheet([], "A_1", PipelineConfig())
    with pytest.raises(ValueError, match="sheet_name"):
        parse_sheet(pd.DataFrame(), "", PipelineConfig())
