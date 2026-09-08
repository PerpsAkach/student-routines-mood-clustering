from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.categorization import add_activity_category
from src.clustering import run_hdbscan
from src.config import PipelineConfig
from src.workbook_parser import load_workbook


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workbook", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    cfg = PipelineConfig()

    events = load_workbook(args.workbook, cfg)
    if events.empty:
        raise RuntimeError("No valid activity events were parsed.")

    events = add_activity_category(events)
    events["Satisfaction_Numeric"] = pd.to_numeric(events["Satisfaction"], errors="coerce")
    events.to_csv(args.output_dir / "events_clean.csv", index=False)

    daily = (
        events.pivot_table(
            index=["Person_ID", "Day"],
            columns="Category",
            values="Duration_Minutes",
            aggfunc="sum",
            fill_value=0,
        )
        .reset_index()
    )

    clustered, _, _ = run_hdbscan(
        daily,
        min_cluster_size=cfg.hdbscan_min_cluster_size,
        min_samples=cfg.hdbscan_min_samples,
    )
    clustered.to_csv(args.output_dir / "daily_clusters.csv", index=False)

    mood = (
        events.groupby("Category", as_index=False)
        .agg(
            Mean_Mood=("Satisfaction_Numeric", "mean"),
            N=("Satisfaction_Numeric", "count"),
            Total_Minutes=("Duration_Minutes", "sum"),
        )
    )
    mood.to_csv(args.output_dir / "mood_by_category.csv", index=False)

    print(f"Parsed {len(events)} events across {events['Person_ID'].nunique()} participants.")


if __name__ == "__main__":
    main()
