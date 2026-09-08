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


@dataclass(frozen=True)
class PipelineConfig:
    skip_rows: int = 5
    sheet_regex: str = r"^[ABC]_\d+"
    default_last_event_minutes: int = 30
    sentence_model: str = "all-MiniLM-L6-v2"
    hdbscan_min_cluster_size: int = 15
    hdbscan_min_samples: int = 5
    random_state: int = 42
