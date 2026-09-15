from __future__ import annotations

from difflib import SequenceMatcher


def normalize_routine_text(value: object) -> str:
    return " ".join(str(value).strip().lower().split())


def routine_similarity(left: object, right: object) -> float:
    a = normalize_routine_text(left)
    b = normalize_routine_text(right)
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return float(SequenceMatcher(None, a, b).ratio())


def routines_match(left: object, right: object, threshold: float = 0.85) -> bool:
    if not 0 <= threshold <= 1:
        raise ValueError("threshold must lie in [0, 1]")
    return routine_similarity(left, right) >= threshold
