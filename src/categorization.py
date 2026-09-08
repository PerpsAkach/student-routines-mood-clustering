import re
import pandas as pd

CATEGORY_KEYWORDS = {
    "Sleep": [r"\bsleep\b", r"\bnap\b", r"\bbed\b", r"lay down"],
    "Eating": [r"\bbreakfast\b", r"\blunch\b", r"\bdinner\b", r"\beat\b", r"\bfood\b"],
    "Exercise": [r"\bgym\b", r"\bexercise\b", r"\bworkout\b", r"\brun\b", r"\bwalk\b"],
    "Academic": [r"\bclass\b", r"\bstudy\b", r"\bhomework\b", r"\bassignment\b"],
    "Work": [r"\bwork\b", r"\bjob\b", r"\bshift\b"],
    "Social": [r"\bfriend\b", r"\bsocial\b", r"\bhang out\b", r"\bparty\b"],
    "Entertainment": [r"\bmovie\b", r"\btv\b", r"\bgame\b", r"\bmusic\b"],
    "Screen Time": [r"\bphone\b", r"\bcomputer\b", r"\blaptop\b", r"\bsocial media\b"],
}


def categorize_activity(text: object) -> str:
    value = str(text).strip().lower()
    for category, patterns in CATEGORY_KEYWORDS.items():
        if any(re.search(pattern, value) for pattern in patterns):
            return category
    return "Other"


def add_activity_category(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["Category"] = out["Activity"].map(categorize_activity)
    return out
