"""Anti-leak and sanity tests for the season model."""
from pathlib import Path
import pandas as pd
import pytest

from ml import features_season


def test_features_list_complete():
    expected = {
        "prev_points", "prev_minutes", "prev_goals", "prev_assists",
        "avg_points_2y", "avg_points_3y",
        "prev_price_end", "prev_price_start",
        "n_prior_seasons", "age", "prev_team_elo",
        "pos_GK", "pos_DEF", "pos_MID", "pos_FWD",
    }
    assert set(features_season.FEATURES_SEASON) == expected


def test_training_table_is_leak_free():
    history = features_season.load_history()
    pgw = pd.read_csv(features_season.DATA_PROCESSED / "player_gw.csv")
    mf = pd.read_csv(features_season.DATA_PROCESSED / "match_features.csv", parse_dates=["date"])
    meta = features_season.build_meta_from_pgw(pgw)
    team_elo = features_season.build_team_elo(mf)
    table = features_season.build_training_table(history, meta, team_elo, None)

    # For any row, prev_points must equal that player's points in the immediately preceding season
    for _, row in table.head(200).iterrows():
        code = row["element_code"]
        target = row["target_season"]
        prior = history[(history["element_code"] == code) & (history["season_name"] < target)]
        if prior.empty:
            continue
        expected_prev = prior.sort_values("season_name").iloc[-1]["total_points"]
        assert abs(row["prev_points"] - expected_prev) < 1e-6


def test_predictions_csv_exists():
    csv = Path(__file__).resolve().parent.parent / "artifacts" / "season_predictions_2026_27.csv"
    if not csv.exists():
        pytest.skip("Run train_season_model first")
    df = pd.read_csv(csv)
    assert "predicted_points" in df.columns
    assert (df["predicted_points"] >= 0).all()