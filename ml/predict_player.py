"""
Player inference — singleton-loaded model + reusable match-model state.

Two public functions:
  predict_player_points(element_code, opponent_team, was_home) -> dict
  rank_players_for_match(home_team, away_team, top_n=10)        -> list[dict]

Uses:
  - ml/artifacts/player_model.joblib
  - ml/artifacts/player_features.json
  - ml/artifacts/terminal_state.joblib  (team Elo)
  - data/processed/player_gw.csv        (rolling history, loaded once)
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

ARTIFACTS = Path(__file__).resolve().parent / "artifacts"
DATA_PROCESSED = Path(__file__).resolve().parent.parent / "data" / "processed"

_MODEL = None
_FEATURES: list[str] | None = None
_ELO: dict[str, float] = {}
_PGW: pd.DataFrame | None = None
_POSITIONS = ("GK", "DEF", "MID", "FWD")


def _ensure_loaded() -> None:
    global _MODEL, _FEATURES, _ELO, _PGW
    if _MODEL is not None:
        return

    model_path = ARTIFACTS / "player_model.joblib"
    meta_path = ARTIFACTS / "player_features.json"
    state_path = ARTIFACTS / "terminal_state.joblib"
    pgw_path = DATA_PROCESSED / "player_gw.csv"

    for p, label in [
        (model_path, "player_model.joblib"),
        (meta_path, "player_features.json"),
        (state_path, "terminal_state.joblib"),
        (pgw_path, "player_gw.csv"),
    ]:
        if not p.exists():
            raise FileNotFoundError(
                f"Missing {label}. Run `python -m ml.train_player_model`."
            )

    _MODEL = joblib.load(model_path)
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    _FEATURES = meta["features"]
    _ELO = joblib.load(state_path)["elo"]

    pgw = pd.read_csv(pgw_path, parse_dates=["kickoff_utc"])
    pgw = pgw.dropna(subset=["element_code"]).copy()
    pgw["element_code"] = pgw["element_code"].astype(int)
    _PGW = pgw.sort_values(["element_code", "kickoff_utc"]).reset_index(drop=True)

    log.info("predict_player: %d features, %d team Elos, %d players loaded",
             len(_FEATURES), len(_ELO), _PGW["element_code"].nunique())


def warm_up() -> None:
    """Preload model + data. Called from apps.py ready()."""
    _ensure_loaded()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _rolling(series: pd.Series, w: int) -> float:
    tail = series.tail(w)
    if tail.empty:
        return float("nan")
    return float(tail.mean())


def _feature_row(player_rows: pd.DataFrame, opponent_team: str, was_home: bool) -> dict:
    last = player_rows.iloc[-1]
    recent = player_rows.tail(10)

    team_elo = float(_ELO.get(last["team"], 1500.0))
    opp_elo = float(_ELO.get(opponent_team, 1500.0))

    row = {
        "was_home": 1.0 if was_home else 0.0,
        "price": float(last["price"]) if pd.notna(last["price"]) else 5.0,
        "gameweek": 1.0,
        "minutes_roll3": _rolling(recent["minutes"], 3),
        "minutes_roll5": _rolling(recent["minutes"], 5),
        "total_points_roll3": _rolling(recent["total_points"], 3),
        "total_points_roll5": _rolling(recent["total_points"], 5),
        "bps_roll3": _rolling(recent["bps"], 3),
        "bps_roll5": _rolling(recent["bps"], 5),
        "goals_scored_roll3": _rolling(recent["goals_scored"], 3),
        "goals_scored_roll5": _rolling(recent["goals_scored"], 5),
        "assists_roll3": _rolling(recent["assists"], 3),
        "assists_roll5": _rolling(recent["assists"], 5),
        "xg_roll3": _rolling(recent["xg"], 3),
        "xg_roll5": _rolling(recent["xg"], 5),
        "xa_roll3": _rolling(recent["xa"], 3),
        "xa_roll5": _rolling(recent["xa"], 5),
        "team_elo": team_elo,
        "opp_elo": opp_elo,
        "elo_diff": team_elo - opp_elo,
        "days_rest": 7.0,  # assume standard week
        "pos_GK": 1.0 if last["position"] == "GK" else 0.0,
        "pos_DEF": 1.0 if last["position"] == "DEF" else 0.0,
        "pos_MID": 1.0 if last["position"] == "MID" else 0.0,
        "pos_FWD": 1.0 if last["position"] == "FWD" else 0.0,
    }
    # xG: mirror the training-side fillna + missing-indicator
    row["has_xg"] = 1.0 if not pd.isna(row["xg_roll5"]) else 0.0
    for k in ("xg_roll3", "xg_roll5", "xa_roll3", "xa_roll5"):
        if pd.isna(row[k]):
            row[k] = 0.0
    return row


# ---------------------------------------------------------------------------
# Public: single-player prediction
# ---------------------------------------------------------------------------
def predict_player_points(
    element_code: int,
    opponent_team: str,
    was_home: bool = True,
) -> dict:
    """Return {'expected_points', 'player', 'position', 'team', 'minutes_roll5'}."""
    _ensure_loaded()
    assert _MODEL is not None and _FEATURES is not None and _PGW is not None

    rows = _PGW[_PGW["element_code"] == element_code]
    if rows.empty:
        raise ValueError(f"Unknown player element_code={element_code}")

    row = _feature_row(rows, opponent_team, was_home)
    X = pd.DataFrame([row])[_FEATURES]
    pred = float(_MODEL.predict(X)[0])

    last = rows.iloc[-1]
    return {
        "element_code": int(element_code),
        "player": last["name"],
        "position": last["position"],
        "team": last["team"],
        "minutes_roll5": float(row["minutes_roll5"]) if not pd.isna(row["minutes_roll5"]) else 0.0,
        "expected_points": round(max(0.0, pred), 2),
    }


# ---------------------------------------------------------------------------
# Public: rank players for a match (match stars / players at risk)
# ---------------------------------------------------------------------------
def rank_players_for_match(
    home_team: str,
    away_team: str,
    top_n: int = 10,
    min_minutes_roll5: float = 45.0,
) -> dict:
    """
    Return the top-N players of both teams by expected points.

    Filters:
      - rolling minutes >= min_minutes_roll5 (likely starters)
    """
    _ensure_loaded()
    assert _MODEL is not None and _FEATURES is not None and _PGW is not None

    # Latest row per player, filtered to those currently in either team
    latest = (
        _PGW.sort_values(["element_code", "kickoff_utc"])
        .groupby("element_code").tail(1)
    )
    pool = latest[latest["team"].isin([home_team, away_team])]

    candidates = []
    for _, player_row in pool.iterrows():
        code = int(player_row["element_code"])
        team = player_row["team"]
        was_home = team == home_team
        opponent = away_team if was_home else home_team

        try:
            result = predict_player_points(code, opponent, was_home)
        except Exception as e:
            log.warning("Skipping player %s: %s", player_row["name"], e)
            continue

        if result["minutes_roll5"] < min_minutes_roll5:
            continue

        result["was_home"] = was_home
        candidates.append(result)

    candidates.sort(key=lambda r: r["expected_points"], reverse=True)
    top = candidates[:top_n]
    return {
        "home_team": home_team,
        "away_team": away_team,
        "top_players": top,
        "at_risk": sorted(candidates, key=lambda r: r["expected_points"])[:5],
    }