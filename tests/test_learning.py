import pytest

from engine.learning import LearningStore


def test_learning_store_records_only_metrics_and_feedback(tmp_path):
    store = LearningStore(str(tmp_path / "learning.sqlite3"))
    store.record_request(
        "request-1",
        "local-model",
        "/v1/chat/completions?ignored=1",
        125.5,
        200,
        "medium",
        "balanced",
        80,
        60,
        12,
    )

    store.add_feedback("request-1", 5)

    status = store.get_status()
    assert status["request_count"] == 1
    assert status["feedback_count"] == 1
    assert status["average_rating"] == 5.0
    assert status["prompt_storage"] is False
    assert status["credential_storage"] is False
    assert status["profile_feedback"]["balanced"]["samples"] == 1


def test_feedback_trains_profile_choice_after_multiple_rated_profiles(tmp_path):
    store = LearningStore(str(tmp_path / "learning.sqlite3"))
    for profile, rating in (
        ("full", 2),
        ("full", 2),
        ("efficient", 5),
        ("efficient", 5),
    ):
        request_id = f"{profile}-{rating}-{store.get_status()['request_count']}"
        store.record_request(
            request_id, "local-model", "/v1/chat/completions", 100, 200,
            "high", profile, 100, 100, 10,
        )
        store.add_feedback(request_id, rating)

    profile, source, samples = store.choose_profile("local-model", "high")
    assert profile == "efficient"
    assert source == "local_feedback"
    assert samples == 4


def test_feedback_requires_valid_rating_and_existing_request(tmp_path):
    store = LearningStore(str(tmp_path / "learning.sqlite3"))

    with pytest.raises(LookupError):
        store.add_feedback("missing", 5)

    store.record_request(
        "request-1", "model", "/v1/chat/completions", 1, 200,
        "low", "full", None, None, None,
    )
    with pytest.raises(ValueError):
        store.add_feedback("request-1", 6)
