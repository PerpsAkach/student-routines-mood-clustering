from __future__ import annotations

import re
from dataclasses import dataclass

WEEKDAY_BLOCKS = {
    "Monday": [2, 3, 4, 5],
    "Tuesday": [6, 7, 8, 9],
    "Wednesday": [10, 11, 12, 13],
    "Thursday": [14, 15, 16, 17],
    "Friday": [18, 19, 20, 21],
    "Saturday": [22, 23, 24, 25],
    "Sunday": [26, 27, 28, 29],
}

DAY_ORDER = tuple(WEEKDAY_BLOCKS)


@dataclass(frozen=True)
class PipelineConfig:
    """Configuration for the reconstructed student-routine pipeline."""

    skip_rows: int = 5
    sheet_regex: str = r"^[ABC]_\d+$"
    default_last_event_minutes: int = 30
    sentence_model: str = "all-MiniLM-L6-v2"
    hdbscan_min_cluster_size: int = 15
    hdbscan_min_samples: int = 5
    tfidf_max_features: int = 1000
    semantic_batch_size: int = 32
    random_state: int = 42

    def validate(self) -> None:
        if self.skip_rows < 0:
            raise ValueError("skip_rows must be non-negative")
        if self.default_last_event_minutes <= 0:
            raise ValueError("default_last_event_minutes must be positive")
        if self.hdbscan_min_cluster_size < 2:
            raise ValueError("hdbscan_min_cluster_size must be at least 2")
        if self.hdbscan_min_samples < 1:
            raise ValueError("hdbscan_min_samples must be positive")
        if self.tfidf_max_features < 1:
            raise ValueError("tfidf_max_features must be positive")
        if self.semantic_batch_size < 1:
            raise ValueError("semantic_batch_size must be positive")
        if not isinstance(self.sentence_model, str) or not self.sentence_model.strip():
            raise ValueError("sentence_model must be a non-empty string")
        try:
            re.compile(self.sheet_regex)
        except re.error as exc:
            raise ValueError("sheet_regex must be a valid regular expression") from exc


DEFAULT_CONFIG = PipelineConfig()
DEFAULT_CONFIG.validate()
