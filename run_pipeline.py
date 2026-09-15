from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from src.analysis import mood_by_category, mood_feature_correlations
from src.categorization import add_activity_category
from src.clustering import cluster_diagnostics, run_hdbscan
from src.config import DEFAULT_CONFIG, PipelineConfig
from src.features import build_daily_features, clustering_feature_columns
from src.quality import event_quality_summary
from src.semantic import aggregate_student_period_text, weekday_weekend_semantic_distance
from src.visualization import plot_tsne, tsne_projection
from src.workbook_parser import load_workbook


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Parse student activity diaries, engineer behavioral features, run "
            "HDBSCAN when sample size permits, and export descriptive analyses."
        )
    )
    parser.add_argument("--workbook", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument(
        "--semantic",
        action="store_true",
        help="Download/use Sentence-BERT and compute weekday/weekend semantic distance.",
    )
    parser.add_argument(
        "--tsne",
        action="store_true",
        help="Create a deterministic 2D t-SNE projection and PNG plot.",
    )
    return parser


def _write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _cluster_daily(daily: pd.DataFrame, cfg: PipelineConfig):
    columns = clustering_feature_columns(daily)
    if not columns:
        return daily.assign(Cluster=pd.NA, Cluster_Probability=pd.NA), {
            "status": "skipped_no_numeric_behavioral_features"
        }
    if len(daily) < cfg.hdbscan_min_cluster_size:
        return daily.assign(Cluster=pd.NA, Cluster_Probability=pd.NA), {
            "status": "skipped_insufficient_observations",
            "observations": int(len(daily)),
            "required": int(cfg.hdbscan_min_cluster_size),
        }

    clustering_input = daily[columns]
    clustered_features, _, _ = run_hdbscan(
        clustering_input,
        min_cluster_size=cfg.hdbscan_min_cluster_size,
        min_samples=cfg.hdbscan_min_samples,
    )
    clustered = daily.copy()
    clustered["Cluster"] = clustered_features["Cluster"].to_numpy()
    clustered["Cluster_Probability"] = clustered_features[
        "Cluster_Probability"
    ].to_numpy()
    diagnostics = {"status": "completed", **cluster_diagnostics(clustered)}
    return clustered, diagnostics


def run_pipeline(
    workbook: Path,
    output_dir: Path,
    *,
    cfg: PipelineConfig | None = None,
    semantic: bool = False,
    tsne: bool = False,
) -> dict[str, object]:
    config = cfg or DEFAULT_CONFIG
    config.validate()
    output_dir.mkdir(parents=True, exist_ok=True)

    events = load_workbook(workbook, config)
    if events.empty:
        raise RuntimeError("No valid activity events were parsed from matching sheets.")

    events = add_activity_category(events)
    events["Satisfaction_Numeric"] = pd.to_numeric(
        events["Satisfaction"], errors="coerce"
    )
    events.to_csv(output_dir / "events_clean.csv", index=False)

    daily = build_daily_features(events)
    daily.to_csv(output_dir / "daily_features.csv", index=False)

    clustered, clustering_summary = _cluster_daily(daily, config)
    clustered.to_csv(output_dir / "daily_clusters.csv", index=False)

    mood = mood_by_category(events)
    mood.to_csv(output_dir / "mood_by_category.csv", index=False)

    correlations = mood_feature_correlations(daily)
    correlations.to_csv(output_dir / "mood_feature_correlations.csv", index=False)

    quality = event_quality_summary(events)
    _write_json(output_dir / "data_quality.json", quality)
    _write_json(output_dir / "clustering_diagnostics.json", clustering_summary)

    semantic_summary: dict[str, object] = {"status": "not_requested"}
    if semantic:
        period_text = aggregate_student_period_text(events)
        distances = weekday_weekend_semantic_distance(
            period_text,
            model_name=config.sentence_model,
            batch_size=config.semantic_batch_size,
        )
        distances.to_csv(output_dir / "weekday_weekend_semantic_distance.csv", index=False)
        semantic_summary = {
            "status": "completed",
            "participants_with_both_periods": int(len(distances)),
        }

    tsne_summary: dict[str, object] = {"status": "not_requested"}
    if tsne:
        columns = clustering_feature_columns(daily)
        if len(daily) < 3 or not columns:
            tsne_summary = {"status": "skipped_insufficient_input"}
        else:
            projection = tsne_projection(
                daily[columns], random_state=config.random_state
            )
            projection.insert(0, "Day", daily["Day"].to_numpy())
            projection.insert(0, "Person_ID", daily["Person_ID"].to_numpy())
            projection.to_csv(output_dir / "tsne_projection.csv", index=False)

            labels = clustered["Cluster"] if "Cluster" in clustered else None
            fig, _ = plot_tsne(
                projection[["TSNE_1", "TSNE_2"]], labels=labels
            )
            fig.savefig(output_dir / "tsne_projection.png", dpi=150, bbox_inches="tight")
            import matplotlib.pyplot as plt

            plt.close(fig)
            tsne_summary = {"status": "completed", "observations": int(len(daily))}

    manifest = {
        "events": int(len(events)),
        "participants": int(events["Person_ID"].nunique()),
        "participant_days": int(len(daily)),
        "quality": quality,
        "clustering": clustering_summary,
        "semantic": semantic_summary,
        "tsne": tsne_summary,
    }
    _write_json(output_dir / "run_manifest.json", manifest)
    return manifest


def main() -> None:
    args = build_parser().parse_args()
    manifest = run_pipeline(
        args.workbook,
        args.output_dir,
        semantic=args.semantic,
        tsne=args.tsne,
    )
    print(
        "Parsed "
        f"{manifest['events']} events across {manifest['participants']} participants "
        f"and {manifest['participant_days']} participant-days."
    )


if __name__ == "__main__":
    main()
