import pandas as pd
import pytest

from src.categorization import categorize_activity
from src.clustering import cluster_diagnostics, run_hdbscan


def test_sleep_keywords():
    assert categorize_activity("take a nap") == "Sleep"
    assert categorize_activity("go to bed") == "Sleep"


def test_homework_is_academic():
    assert categorize_activity("finish homework") == "Academic"


def test_gym_is_exercise():
    assert categorize_activity("go to gym") == "Exercise"


def test_unknown_activity_is_other():
    assert categorize_activity("commute to campus") == "Other"


def test_clustering_requires_enough_rows():
    with pytest.raises(ValueError, match="min_cluster_size"):
        run_hdbscan(pd.DataFrame({"x": [1.0, 2.0, 3.0]}), min_cluster_size=4)


def test_clustering_requires_varying_numeric_features():
    with pytest.raises(ValueError, match="no variance"):
        run_hdbscan(
            pd.DataFrame({"x": [1.0, 1.0, 1.0, 1.0]}),
            min_cluster_size=2,
            min_samples=1,
        )


def test_cluster_diagnostics_counts_assignments():
    frame = pd.DataFrame(
        {"Cluster": [0, 0, 1, -1], "Cluster_Probability": [0.9, 0.8, 0.7, 0.0]}
    )
    result = cluster_diagnostics(frame)
    assert result["observations"] == 4
    assert result["clusters"] == 2
    assert result["noise_observations"] == 1
