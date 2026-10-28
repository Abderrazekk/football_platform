"""
Inference entry point for the match model.

Loads the calibrated model + terminal state once (lazy singleton, per
cahier des charges §7.4) and returns probability dicts for any (home, away)
pair. First call after process start costs ~0.5 s (joblib load); every
subsequent call is <100 ms.

Usage:
    from ml.predict import predict_match
    probs = predict_match("Arsenal", "Chelsea")
    # {'home': 0.48, 'draw': 0.25, 'away': 0.27, 'predicted': 'H', 'confidence': 0.48}
"""

from __future__ import annotations

import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from ml import features

log = logging.getLogger(__name__)

ARTIFACTS = Path(__file__).resolve().parent / "artifacts"

# ---------------------------------------------------------------------------
# Module-level singletons (loaded on first use)
# ---------------------------------------------------------------------------
_MODEL = None
_FEATURE_NAMES: list[str] | None = None
_STATE: dict | None = None
_KNOWN_TEAMS: set[str] = set()

# Common alias normalization for user input
_ALIASES = {
    "Manchester City": "Man City",
    "Manchester United": "Man United",
    "Man Utd": "Man United",
    "Tottenham Hotspur": "Tottenham",
    "Spurs": "Tottenham",
    "Newcastle United": "Newcastle",
    "Nottingham Forest": "Nott'm Forest",
    "West Ham United": "West Ham",
    "Wolverhampton Wanderers": "Wolves",
    "West Bromwich Albion": "West Brom",
}


def _normalize_team(name: str) -> str:
    name = (name or "").strip()
    return _ALIASES.get(name, name)


def _ensure_loaded() -> None:
    global _MODEL, _FEATURE_NAMES, _STATE, _KNOWN_TEAMS

    if _MODEL is not None:
        return

    model_path = ARTIFACTS / "match_model.joblib"
    state_path = ARTIFACTS / "terminal_state.joblib"
    if not model_path.exists() or not state_path.exists():
        raise FileNotFoundError(
            f"Missing model/state artifacts in {ARTIFACTS}. "
            "Run `python -m ml.train_match_model` first."
        )

    _MODEL = joblib.load(model_path)
    _STATE = joblib.load(state_path)

    import json
    meta = json.loads((ARTIFACTS / "match_features.json").read_text(encoding="utf-8"))
    _FEATURE_NAMES = meta["features_main_used"]

    _KNOWN_TEAMS = set(_STATE["elo"].keys())
    log.info("predict: loaded model + state (%d features, %d teams)",
             len(_FEATURE_NAMES), len(_KNOWN_TEAMS))


def warm_up() -> None:
    """Preload model + state. Call from Django apps.py ready() to avoid latency."""
    _ensure_loaded()


# ---------------------------------------------------------------------------
# Feature computation for a single hypothetical match
# ---------------------------------------------------------------------------
def _season_for_date(d: pd.Timestamp) -> str:
    if d is None or pd.isna(d):
        d = pd.Timestamp.utcnow()
    if d.month >= 8:
        start = d.year
    else:
        start = d.year - 1
    return f"{start}-{str(start + 1)[-2:]}"


def _aggregate(history: list[dict], key: str, op="sum") -> float:
    if not history:
        return np.nan
    vals = [m[key] for m in history if m.get(key) is not None and not pd.isna(m.get(key))]
    if not vals:
        return np.nan
    return float(sum(vals)) if op == "sum" else float(np.mean(vals))


def _wins(history: list[dict]) -> float:
    if not history:
        return np.nan
    return float(sum(1 for m in history if m["points"] == 3))


def _compute_row_features(
    home: str, away: str, match_date: pd.Timestamp, state: dict
) -> dict:
    elo = state["elo"]
    team_log = state["team_log"]
    h2h_log = state["h2h_log"]
    season_played = state["season_played"]
    squad_by_team = state["squad_by_team"]

    season = _season_for_date(match_date)

    elo_home = elo.get(home, features.ELO_START)
    elo_away = elo.get(away, features.ELO_START)

    home_hist = [m for m in team_log.get(home, []) if m["date"] < match_date]
    away_hist = [m for m in team_log.get(away, []) if m["date"] < match_date]

    home_5 = home_hist[-5:]
    home_10 = home_hist[-10:]
    away_5 = away_hist[-5:]
    away_10 = away_hist[-10:]
    home_home_5 = [m for m in home_hist if m["was_home"]][-5:]
    away_away_5 = [m for m in away_hist if not m["was_home"]][-5:]

    days_rest_home = (match_date - home_hist[-1]["date"]).days if home_hist else np.nan
    days_rest_away = (match_date - away_hist[-1]["date"]).days if away_hist else np.nan

    pair_key = frozenset([home, away])
    h2h_hist = [m for m in h2h_log.get(pair_key, []) if m["date"] < match_date]
    h2h_home_wins = 0
    h2h_draws = 0
    home_gds: list[int] = []
    for m in h2h_hist:
        if m["result"] == "D":
            h2h_draws += 1
        elif (m["result"] == "H" and m["home_team"] == home) or \
             (m["result"] == "A" and m["away_team"] == home):
            h2h_home_wins += 1
        if m["home_team"] == home:
            home_gds.append(m["home_goals"] - m["away_goals"])
        else:
            home_gds.append(m["away_goals"] - m["home_goals"])
    h2h_away_wins = len(h2h_hist) - h2h_home_wins - h2h_draws
    h2h_avg_gd = float(np.mean(home_gds)) if home_gds else np.nan

    home_played = season_played.get((season, home), 0)
    away_played = season_played.get((season, away), 0)
    gameweek_approx = max(home_played, away_played) + 1
    early_season = int(home_played < 3 or away_played < 3)

    def _sq(team_hist):
        past = [m for m in team_hist if m["date"] < match_date][-5:]
        return float(np.mean([m["points_top11"] for m in past])) if len(past) >= 3 else np.nan

    cold_start = int(min(len(home_hist), len(away_hist)) < 3)

    return {
        "elo_home": elo_home,
        "elo_away": elo_away,
        "elo_diff": elo_home - elo_away,

        "home_points_5": _aggregate(home_5, "points"),
        "home_gf_5": _aggregate(home_5, "gf"),
        "home_ga_5": _aggregate(home_5, "ga"),
        "home_gd_5": _aggregate(home_5, "gf") - _aggregate(home_5, "ga"),
        "home_wins_5": _wins(home_5),
        "home_n_5": float(len(home_5)),
        "away_points_5": _aggregate(away_5, "points"),
        "away_gf_5": _aggregate(away_5, "gf"),
        "away_ga_5": _aggregate(away_5, "ga"),
        "away_gd_5": _aggregate(away_5, "gf") - _aggregate(away_5, "ga"),
        "away_wins_5": _wins(away_5),
        "away_n_5": float(len(away_5)),

        "home_points_10": _aggregate(home_10, "points"),
        "home_gf_10": _aggregate(home_10, "gf"),
        "home_ga_10": _aggregate(home_10, "ga"),
        "home_gd_10": _aggregate(home_10, "gf") - _aggregate(home_10, "ga"),
        "home_wins_10": _wins(home_10),
        "home_n_10": float(len(home_10)),
        "away_points_10": _aggregate(away_10, "points"),
        "away_gf_10": _aggregate(away_10, "gf"),
        "away_ga_10": _aggregate(away_10, "ga"),
        "away_gd_10": _aggregate(away_10, "gf") - _aggregate(away_10, "ga"),
        "away_wins_10": _wins(away_10),
        "away_n_10": float(len(away_10)),

        "home_home_points_5": _aggregate(home_home_5, "points"),
        "home_home_gf_5": _aggregate(home_home_5, "gf"),
        "home_home_ga_5": _aggregate(home_home_5, "ga"),
        "home_home_wins_5": _wins(home_home_5),
        "away_away_points_5": _aggregate(away_away_5, "points"),
        "away_away_gf_5": _aggregate(away_away_5, "gf"),
        "away_away_ga_5": _aggregate(away_away_5, "ga"),
        "away_away_wins_5": _wins(away_away_5),

        "days_rest_home": float(days_rest_home) if not pd.isna(days_rest_home) else np.nan,
        "days_rest_away": float(days_rest_away) if not pd.isna(days_rest_away) else np.nan,

        "h2h_home_wins": float(h2h_home_wins),
        "h2h_draws": float(h2h_draws),
        "h2h_away_wins": float(h2h_away_wins),
        "h2h_avg_gd_home": h2h_avg_gd,
        "h2h_n": float(len(h2h_hist)),

        "home_matches_played_season": float(home_played),
        "away_matches_played_season": float(away_played),
        "gameweek_approx": float(gameweek_approx),
        "early_season": float(early_season),

        "home_xg_for_5": _aggregate(home_5, "xg_for", op="mean"),
        "home_xg_against_5": _aggregate(home_5, "xg_against", op="mean"),
        "away_xg_for_5": _aggregate(away_5, "xg_for", op="mean"),
        "away_xg_against_5": _aggregate(away_5, "xg_against", op="mean"),

        "home_squad_points_5": _sq(squad_by_team.get(home, [])),
        "away_squad_points_5": _sq(squad_by_team.get(away, [])),

        "b365_p_home": np.nan,
        "b365_p_draw": np.nan,
        "b365_p_away": np.nan,

        "home_matches_played_all": float(len(home_hist)),
        "away_matches_played_all": float(len(away_hist)),
        "cold_start": float(cold_start),
    }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def predict_match(home_team: str, away_team: str, date=None) -> dict:
    """
    Predict probabilities for a match.

    Parameters
    ----------
    home_team, away_team : str
    date : optional, defaults to now (UTC). Pass a pandas Timestamp or
           string ("2026-12-25") to override.

    Returns
    -------
    dict with keys: home, draw, away (probabilities summing to 1.0),
                   predicted (H/D/A), confidence (max prob).
    """
    _ensure_loaded()
    assert _MODEL is not None and _FEATURE_NAMES is not None and _STATE is not None

    home = _normalize_team(home_team)
    away = _normalize_team(away_team)

    if home == away:
        raise ValueError(f"home_team and away_team must differ (got {home!r})")

    unknown = {t for t in (home, away) if t not in _KNOWN_TEAMS}
    if unknown:
        log.warning("predict: unknown team(s) %s — using Elo default 1500", unknown)

    # Resolve date to a tz-aware UTC Timestamp
    if date is None:
        match_date = pd.Timestamp.now(tz="UTC")
    else:
        match_date = pd.Timestamp(date)
        if match_date.tz is None:
            match_date = match_date.tz_localize("UTC")

    row = _compute_row_features(home, away, match_date, _STATE)

    try:
        X = pd.DataFrame([row])[_FEATURE_NAMES]
    except KeyError as e:
        raise RuntimeError(f"Feature mismatch: {e}. Retrain the model.") from e

    proba = _MODEL.predict_proba(X)[0]
    probs = {"home": float(proba[0]), "draw": float(proba[1]), "away": float(proba[2])}

    predicted = max(probs, key=probs.get)
    return {
        **probs,
        "predicted": {"home": "H", "draw": "D", "away": "A"}[predicted],
        "confidence": float(probs[predicted]),
    }