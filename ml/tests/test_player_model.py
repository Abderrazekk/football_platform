"""
Anti-leak and sanity tests for the player model.

Run with:
    pytest ml/tests/test_player_model.py -v
"""

from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from ml import features_player

ARTIFACTS = Path(__file__).resolve().parent.parent / "artifacts"


@pytest.fixture(scope="module")
def data() -> pd.DataFrame:
    return features_player.load_and_build()


def test_required_columns(data):
    for col in features_player.FEATURES:
        assert col in data.columns, f"missing feature {col}"
    assert "total_points" in data.columns


def test_no_cold_start(data):
    # After load_and_build, all rows should have valid rolling5
    assert data["total_points_roll5"].notna().all()
    assert data["minutes_roll5"].notna().all()


def test_rolling_is_leak_free(data):
    """
    Verify total_points_roll5 has no leakage.

    The model builds roll5 on the FULL data (before cold-start drop), then
    filters. So the only correct way to check is to rebuild expected values
    from raw player_gw.csv (pre-filter) and match by (element_code, kickoff_utc).
    """
    from pathlib import Path

    raw_path = Path(__file__).resolve().parent.parent.parent / "data" / "processed" / "player_gw.csv"
    raw = pd.read_csv(raw_path, parse_dates=["kickoff_utc"])
    raw = raw.dropna(subset=["element_code"]).copy()
    raw["element_code"] = raw["element_code"].astype(int)
    raw = raw.sort_values(["element_code", "kickoff_utc"]).reset_index(drop=True)

    # Pick one player with many games
    player = raw["element_code"].value_counts().idxmax()
    raw_player = raw[raw["element_code"] == player].reset_index(drop=True)

    # Rebuild expected roll5 with same formula used in features_player.py
    expected_roll5 = (
        raw_player["total_points"]
        .shift(1)
        .rolling(5, min_periods=1)
        .mean()
    )

    # Match against the built dataframe row-by-row
    built = (
        data[data["element_code"] == player]
        .sort_values("kickoff_utc")
        .reset_index(drop=True)
    )

    checked = 0
    for _, row in built.iterrows():
        matches = raw_player.index[raw_player["kickoff_utc"] == row["kickoff_utc"]]
        if len(matches) == 0:
            continue
        exp = expected_roll5.iloc[matches[0]]
        if pd.isna(exp):
            continue
        assert abs(row["total_points_roll5"] - exp) < 1e-6, (
            f"kickoff {row['kickoff_utc']}: built={row['total_points_roll5']}, expected={exp}"
        )
        checked += 1
        if checked >= 15:
            break

    assert checked >= 5, f"only {checked} rows verified"

def test_elo_join_present(data):
    # Most rows should have valid Elo
    coverage = data["team_elo"].notna().mean()
    assert coverage > 0.95, f"Elo coverage only {coverage:.1%}"


def test_position_one_hot(data):
    total = data[["pos_GK", "pos_DEF", "pos_MID", "pos_FWD"]].sum(axis=1)
    assert (total == 1.0).all()


# --- Inference sanity (requires trained model) ------------------------------
def _skip_if_no_model():
    if not (ARTIFACTS / "player_model.joblib").exists():
        pytest.skip("player_model.joblib missing — run train_player_model")


def test_predict_player_range():
    _skip_if_no_model()
    from ml.predict_player import predict_player_points

    # Grab one real element_code from training data
    data = features_player.load_and_build()
    code = int(data["element_code"].value_counts().idxmax())
    result = predict_player_points(code, opponent_team="Arsenal", was_home=True)
    assert 0.0 <= result["expected_points"] <= 25.0
    assert result["player"]
    assert result["position"] in ("GK", "DEF", "MID", "FWD")


def test_rank_players_for_match():
    _skip_if_no_model()
    from ml.predict_player import rank_players_for_match

    result = rank_players_for_match("Arsenal", "Chelsea", top_n=5)
    assert result["home_team"] == "Arsenal"
    assert len(result["top_players"]) <= 5
    if result["top_players"]:
        # Sorted descending by expected points
        eps = [p["expected_points"] for p in result["top_players"]]
        assert eps == sorted(eps, reverse=True)