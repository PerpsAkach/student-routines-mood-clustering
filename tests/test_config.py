import pytest

from src.config import PipelineConfig


def test_default_config_is_valid():
    PipelineConfig().validate()


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"skip_rows": -1}, "skip_rows"),
        ({"default_last_event_minutes": 0}, "default_last_event_minutes"),
        ({"hdbscan_min_cluster_size": 1}, "hdbscan_min_cluster_size"),
        ({"hdbscan_min_samples": 0}, "hdbscan_min_samples"),
        ({"tfidf_max_features": 0}, "tfidf_max_features"),
        ({"semantic_batch_size": 0}, "semantic_batch_size"),
        ({"sentence_model": ""}, "sentence_model"),
        ({"sheet_regex": "["}, "sheet_regex"),
    ],
)
def test_config_rejects_invalid_values(kwargs, message):
    with pytest.raises(ValueError, match=message):
        PipelineConfig(**kwargs).validate()
