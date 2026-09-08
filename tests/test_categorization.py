from src.categorization import categorize_activity


def test_sleep_keywords():
    assert categorize_activity("take a nap") == "Sleep"
    assert categorize_activity("go to bed") == "Sleep"


def test_homework_is_academic():
    assert categorize_activity("finish homework") == "Academic"


def test_gym_is_exercise():
    assert categorize_activity("go to gym") == "Exercise"
