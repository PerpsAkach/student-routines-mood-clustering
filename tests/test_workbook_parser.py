from datetime import time

import pandas as pd
import pytest

from run_pipeline import run_pipeline
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


def test_pipeline_writes_core_outputs_and_records_small_sample_skip(tmp_path):
    raw = pd.DataFrame([[None] * 30 for _ in range(2)])
    raw.iloc[0, 2:6] = [0.25, "Breakfast", "Alone", 4]
    raw.iloc[1, 2:6] = [0.5, "Study", "Classmates", 5]

    workbook = tmp_path / "diary.xlsx"
    with pd.ExcelWriter(workbook, engine="openpyxl") as writer:
        raw.to_excel(writer, sheet_name="A_1", index=False, header=False)
        raw.to_excel(writer, sheet_name="Notes", index=False, header=False)

    output_dir = tmp_path / "outputs"
    manifest = run_pipeline(
        workbook,
        output_dir,
        cfg=PipelineConfig(skip_rows=0, hdbscan_min_cluster_size=15),
    )

    assert manifest["events"] == 2
    assert manifest["participants"] == 1
    assert manifest["participant_days"] == 1
    assert manifest["clustering"]["status"] == "skipped_insufficient_observations"

    expected = {
        "events_clean.csv",
        "daily_features.csv",
        "daily_clusters.csv",
        "mood_by_category.csv",
        "mood_feature_correlations.csv",
        "data_quality.json",
        "clustering_diagnostics.json",
        "run_manifest.json",
    }
    assert expected.issubset({path.name for path in output_dir.iterdir()})
