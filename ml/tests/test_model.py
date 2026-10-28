"""Sanity tests for the trained match model and predict.py."""

from pathlib import Path
import time

import pytest

ARTIFACTS = Path(__file__).resolve().parent.parent / "artifacts"


@pytest.fixture(scope="module")
def trained():
    """Ensure model + state artifacts exist. Skip if not trained yet."""
    if not (ARTIFACTS / "match_model.joblib").exists():
        pytest.skip("match_model.joblib missing — run ml.train_match_model first")
    if not (ARTIFACTS / "terminal_state.joblib").exists():
        pytest.skip("terminal_state.joblib missing — run ml.train_match_model first")
    return True


def test_model_loads(trained):
    from ml.predict import _ensure_loaded
    _ensure_loaded()
    from ml.predict import _MODEL
    assert _MODEL is not None


def test_predict_returns_valid_probabilities(trained):
    from ml.predict import predict_match
    result = predict_match("Arsenal", "Chelsea")
    for key in ("home", "draw", "away"):
        assert 0.0 <= result[key] <= 1.0, f"{key} = {result[key]} out of range"
    total = result["home"] + result["draw"] + result["away"]
    assert abs(total - 1.0) < 1e-6, f"Probabilities sum to {total}, not 1"
    assert result["predicted"] in ("H", "D", "A")


def test_predict_rejects_same_team(trained):
    from ml.predict import predict_match
    with pytest.raises(ValueError):
        predict_match("Liverpool", "Liverpool")


def test_predict_speed_under_2s(trained):
    from ml.predict import predict_match, warm_up
    warm_up()
    t0 = time.time()
    predict_match("Man City", "Liverpool")
    elapsed = time.time() - t0
    assert elapsed < 2.0, f"prediction took {elapsed:.2f}s > 2.0s"


def test_predict_handles_aliases(trained):
    from ml.predict import predict_match
    # Manchester City / Man City should give the same result
    p1 = predict_match("Man City", "Chelsea")
    p2 = predict_match("Manchester City", "Chelsea")
    assert abs(p1["home"] - p2["home"]) < 1e-6