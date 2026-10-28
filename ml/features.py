"""
Feature engineering for the match model.

Given data/processed/matches.csv (cleaned in Phase 2), produce one row per
match with a wide table of features. Every feature for match i uses ONLY
matches with date < match_date[i], guaranteed by iterating in chronological
order: compute features from current state, THEN update state.

Feature groups
--------------
  1. Elo ratings (pre-match)
  2. Rolling form (last 5 / 10 matches per team)
  3. Venue-specific form (last 5 home / last 5 away)
  4. Rest days since previous match
  5. Head-to-head record (all prior meetings)
  6. Season context (matches played, approximate gameweek, early-season flag)
  7. Rolling xG (from Understat, NaN-safe for 2025-26 / 2026-27)
  8. Squad quality (from player_gw rolling team top-11 points)
  9. Bookmaker implied probabilities (normalized, benchmark only)
 10. Cold-start indicator

Reading
-------
- "prior" state means: matches already played BEFORE this match's date.
- "current match" state (goals, result, own xG) is NEVER used as a feature.
"""

from __future__ import annotations

import json
import logging
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

DATA_PROCESSED = Path(__file__).resolve().parent.parent / "data" / "processed"
ARTIFACTS = Path(__file__).resolve().parent / "artifacts"
ARTIFACTS.mkdir(parents=True, exist_ok=True)

# ---- Config ----------------------------------------------------------------
ELO_START = 1500.0
ELO_K = 20
ELO_HOME_ADV = 65.0

TARGET_MAP = {"H": 0, "D": 1, "A": 2}
INVERSE_TARGET_MAP = {v: k for k, v in TARGET_MAP.items()}

META_COLUMNS = [
    "season", "date", "home_team", "away_team",
    "result", "result_encoded",
]

# ---- Player_gw cache (avoids reloading on every rebuild in tests) -----------
_PLAYER_GW_CACHE: pd.DataFrame | None = None


def _load_player_gw() -> pd.DataFrame | None:
    global _PLAYER_GW_CACHE
    if _PLAYER_GW_CACHE is not None:
        return _PLAYER_GW_CACHE

    path = DATA_PROCESSED / "player_gw.csv"
    if not path.exists():
        return None

    try:
        df = pd.read_csv(
            path,
            usecols=["season", "team", "kickoff_utc", "total_points"],
        )
        df["kickoff_utc"] = pd.to_datetime(df["kickoff_utc"], utc=True, errors="coerce")
        df = df.dropna(subset=["kickoff_utc", "team"])
    except Exception as e:
        log.warning("Could not load player_gw for squad quality: %s", e)
        return None

    _PLAYER_GW_CACHE = df
    return df


# ---- Aggregation helpers ---------------------------------------------------
def _sum(lst: list, key: str) -> float:
    if not lst:
        return np.nan
    vals = [m[key] for m in lst if not pd.isna(m.get(key))]
    return float(sum(vals)) if vals else np.nan


def _mean(lst: list, key: str) -> float:
    if not lst:
        return np.nan
    vals = [m[key] for m in lst if not pd.isna(m.get(key))]
    return float(np.mean(vals)) if vals else np.nan


def _wins(lst: list) -> float:
    if not lst:
        return np.nan
    return float(sum(1 for m in lst if m["points"] == 3))


def _normalize_odds(row) -> tuple[float, float, float]:
    h = row.get("odds_b365_h")
    d = row.get("odds_b365_d")
    a = row.get("odds_b365_a")
    if pd.isna(h) or pd.isna(d) or pd.isna(a):
        return np.nan, np.nan, np.nan
    if h <= 1.0 or d <= 1.0 or a <= 1.0:
        return np.nan, np.nan, np.nan
    ih, idr, ia = 1.0 / h, 1.0 / d, 1.0 / a
    s = ih + idr + ia
    return ih / s, idr / s, ia / s


# ---- Loader ----------------------------------------------------------------
def load_matches() -> pd.DataFrame:
    """Load and sort matches for feature building."""
    df = pd.read_csv(DATA_PROCESSED / "matches.csv", parse_dates=["date"])
    df = df[df["is_played"]].copy()
    df = df.sort_values(["date", "season", "home_team", "away_team"]).reset_index(drop=True)
    return df


# ---- Main builder ----------------------------------------------------------
def build_features(matches: pd.DataFrame) -> pd.DataFrame:
    """
    Produce a leak-free feature table.

    Parameters
    ----------
    matches : pd.DataFrame
        Output of ``load_matches()`` (date-sorted, only played matches).

    Returns
    -------
    pd.DataFrame
        One row per match with target ``result_encoded`` (0=H, 1=D, 2=A)
        and all numeric features.
    """
    df = matches.sort_values(
        ["date", "season", "home_team", "away_team"]
    ).reset_index(drop=True).copy()

    # --- Prepare squad-quality per-team rolling data --------------------
    player_gw = _load_player_gw()
    squad_by_team = _build_squad_quality(player_gw)

    # --- Incremental state containers -----------------------------------
    elo: dict[str, float] = {}
    team_log: dict[str, list[dict]] = defaultdict(list)   # team -> list of match dicts
    h2h_log: dict[frozenset, list[dict]] = defaultdict(list)
    season_played: dict[tuple[str, str], int] = defaultdict(int)

    records: list[dict] = []

    for _, row in df.iterrows():
        home = row["home_team"]
        away = row["away_team"]
        match_date = row["date"]
        season = row["season"]
        result = row["result"]

        # ---- 1. Elo (before update) ----
        elo_home = elo.get(home, ELO_START)
        elo_away = elo.get(away, ELO_START)

        # ---- 2. Rolling form (all prior matches) ----
        home_hist = [m for m in team_log[home] if m["date"] < match_date]
        away_hist = [m for m in team_log[away] if m["date"] < match_date]

        home_5 = home_hist[-5:]
        home_10 = home_hist[-10:]
        away_5 = away_hist[-5:]
        away_10 = away_hist[-10:]

        home_home_5 = [m for m in home_hist if m["was_home"]][-5:]
        away_away_5 = [m for m in away_hist if not m["was_home"]][-5:]

        # ---- 3. Rest days ----
        days_rest_home = (match_date - home_hist[-1]["date"]).days if home_hist else np.nan
        days_rest_away = (match_date - away_hist[-1]["date"]).days if away_hist else np.nan

        # ---- 4. H2H ----
        pair_key = frozenset([home, away])
        h2h_hist = [m for m in h2h_log[pair_key] if m["date"] < match_date]
        h2h_home_wins = 0
        h2h_draws = 0
        home_gds: list[int] = []
        for m in h2h_hist:
            if m["result"] == "D":
                h2h_draws += 1
            elif m["result"] == "H":
                if m["home_team"] == home:
                    h2h_home_wins += 1
            elif m["result"] == "A":
                if m["away_team"] == home:
                    h2h_home_wins += 1
            if m["home_team"] == home:
                home_gds.append(m["home_goals"] - m["away_goals"])
            else:
                home_gds.append(m["away_goals"] - m["home_goals"])
        h2h_away_wins = len(h2h_hist) - h2h_home_wins - h2h_draws
        h2h_avg_gd = float(np.mean(home_gds)) if home_gds else np.nan

        # ---- 5. Season context ----
        home_played_season = season_played[(season, home)]
        away_played_season = season_played[(season, away)]
        gameweek_approx = max(home_played_season, away_played_season) + 1
        early_season = int(home_played_season < 3 or away_played_season < 3)

        # ---- 6. Squad quality (rolling top-11 points) ----
        home_squad_5 = _rolling_avg(squad_by_team.get(home, []), match_date, 5)
        away_squad_5 = _rolling_avg(squad_by_team.get(away, []), match_date, 5)

        # ---- 7. Odds (normalized) ----
        p_h, p_d, p_a = _normalize_odds(row)

        # ---- 8. Cold start ----
        home_all = len(home_hist)
        away_all = len(away_hist)
        cold_start = int(min(home_all, away_all) < 3)

        # ---- Assemble row ----
        records.append({
            # Meta
            "season": season,
            "date": match_date,
            "home_team": home,
            "away_team": away,
            "result": result,
            "result_encoded": TARGET_MAP.get(result),

            # Elo
            "elo_home": elo_home,
            "elo_away": elo_away,
            "elo_diff": elo_home - elo_away,

            # Form 5
            "home_points_5": _sum(home_5, "points"),
            "home_gf_5": _sum(home_5, "gf"),
            "home_ga_5": _sum(home_5, "ga"),
            "home_gd_5": _sum(home_5, "gf") - _sum(home_5, "ga"),
            "home_wins_5": _wins(home_5),
            "home_n_5": float(len(home_5)),
            "away_points_5": _sum(away_5, "points"),
            "away_gf_5": _sum(away_5, "gf"),
            "away_ga_5": _sum(away_5, "ga"),
            "away_gd_5": _sum(away_5, "gf") - _sum(away_5, "ga"),
            "away_wins_5": _wins(away_5),
            "away_n_5": float(len(away_5)),

            # Form 10
            "home_points_10": _sum(home_10, "points"),
            "home_gf_10": _sum(home_10, "gf"),
            "home_ga_10": _sum(home_10, "ga"),
            "home_gd_10": _sum(home_10, "gf") - _sum(home_10, "ga"),
            "home_wins_10": _wins(home_10),
            "home_n_10": float(len(home_10)),
            "away_points_10": _sum(away_10, "points"),
            "away_gf_10": _sum(away_10, "gf"),
            "away_ga_10": _sum(away_10, "ga"),
            "away_gd_10": _sum(away_10, "gf") - _sum(away_10, "ga"),
            "away_wins_10": _wins(away_10),
            "away_n_10": float(len(away_10)),

            # Venue-specific
            "home_home_points_5": _sum(home_home_5, "points"),
            "home_home_gf_5": _sum(home_home_5, "gf"),
            "home_home_ga_5": _sum(home_home_5, "ga"),
            "home_home_wins_5": _wins(home_home_5),
            "away_away_points_5": _sum(away_away_5, "points"),
            "away_away_gf_5": _sum(away_away_5, "gf"),
            "away_away_ga_5": _sum(away_away_5, "ga"),
            "away_away_wins_5": _wins(away_away_5),

            # Rest
            "days_rest_home": float(days_rest_home) if not pd.isna(days_rest_home) else np.nan,
            "days_rest_away": float(days_rest_away) if not pd.isna(days_rest_away) else np.nan,

            # H2H
            "h2h_home_wins": float(h2h_home_wins),
            "h2h_draws": float(h2h_draws),
            "h2h_away_wins": float(h2h_away_wins),
            "h2h_avg_gd_home": h2h_avg_gd,
            "h2h_n": float(len(h2h_hist)),

            # Season context
            "home_matches_played_season": float(home_played_season),
            "away_matches_played_season": float(away_played_season),
            "gameweek_approx": float(gameweek_approx),
            "early_season": float(early_season),

            # xG rolling
            "home_xg_for_5": _mean(home_5, "xg_for"),
            "home_xg_against_5": _mean(home_5, "xg_against"),
            "away_xg_for_5": _mean(away_5, "xg_for"),
            "away_xg_against_5": _mean(away_5, "xg_against"),

            # Squad quality
            "home_squad_points_5": home_squad_5,
            "away_squad_points_5": away_squad_5,

            # Odds (benchmark only)
            "b365_p_home": p_h,
            "b365_p_draw": p_d,
            "b365_p_away": p_a,

            # Cold start indicator
            "home_matches_played_all": float(home_all),
            "away_matches_played_all": float(away_all),
            "cold_start": float(cold_start),
        })

        # ---------- UPDATE STATE (after features) ----------
        actual = 1.0 if result == "H" else 0.5 if result == "D" else 0.0
        expected_home = 1.0 / (1.0 + 10 ** (
            (elo_away - (elo_home + ELO_HOME_ADV)) / 400.0
        ))
        delta = ELO_K * (actual - expected_home)
        elo[home] = elo_home + delta
        elo[away] = elo_away - delta

        hg = int(row["home_goals"])
        ag = int(row["away_goals"])
        home_pts = 3 if result == "H" else 1 if result == "D" else 0
        away_pts = 3 if result == "A" else 1 if result == "D" else 0
        home_xg = float(row["home_xg"]) if pd.notna(row.get("home_xg")) else np.nan
        away_xg = float(row["away_xg"]) if pd.notna(row.get("away_xg")) else np.nan

        team_log[home].append({
            "date": match_date, "was_home": True,
            "gf": hg, "ga": ag, "points": home_pts,
            "xg_for": home_xg, "xg_against": away_xg,
        })
        team_log[away].append({
            "date": match_date, "was_home": False,
            "gf": ag, "ga": hg, "points": away_pts,
            "xg_for": away_xg, "xg_against": home_xg,
        })

        h2h_log[pair_key].append({
            "date": match_date,
            "home_team": home, "away_team": away,
            "home_goals": hg, "away_goals": ag,
            "result": result,
        })

        season_played[(season, home)] += 1
        season_played[(season, away)] += 1

    return pd.DataFrame(records)


# ---- Squad quality helper --------------------------------------------------
def _build_squad_quality(player_gw: pd.DataFrame | None) -> dict[str, list[dict]]:
    """Return {team: [{date, points_sum_top11}, ...]}."""
    out: dict[str, list[dict]] = defaultdict(list)
    if player_gw is None or player_gw.empty:
        return out

    for (team, kickoff), group in player_gw.groupby(["team", "kickoff_utc"]):
        top11 = group["total_points"].nlargest(11).sum()
        out[team].append({"date": kickoff, "points_top11": float(top11)})

    for t in out:
        out[t].sort(key=lambda m: m["date"])
    return out


def _rolling_avg(hist: list[dict], current_date, n: int) -> float:
    past = [m for m in hist if m["date"] < current_date][-n:]
    if len(past) < 3:
        return np.nan
    return float(np.mean([m["points_top11"] for m in past]))


# ---- Feature column enumeration & persistence ------------------------------
def feature_columns(df: pd.DataFrame) -> tuple[list[str], list[str]]:
    """Return (main_features, benchmark_features)."""
    all_cols = [c for c in df.columns if c not in META_COLUMNS]
    benchmark_extra = [c for c in all_cols if c.startswith("b365_p_")]
    main_features = [c for c in all_cols if c not in benchmark_extra]
    return main_features, all_cols  # benchmark = all


def save_artifacts(df: pd.DataFrame, out_dir: Path | None = None) -> None:
    out_dir = out_dir or DATA_PROCESSED
    csv_path = out_dir / "match_features.csv"
    df.to_csv(csv_path, index=False)

    main, benchmark = feature_columns(df)
    meta = {
        "n_rows": int(len(df)),
        "n_main_features": len(main),
        "n_benchmark_features": len(benchmark),
        "target_map": TARGET_MAP,
        "meta_columns": META_COLUMNS,
        "features_main": main,
        "features_benchmark": benchmark,
        "config": {
            "elo_start": ELO_START,
            "elo_k": ELO_K,
            "elo_home_adv": ELO_HOME_ADV,
            "cold_start_threshold": 3,
        },
        "description": (
            "Match features for Football Intelligence Platform. Every feature "
            "at match i uses ONLY matches with date < date_i. Column order in "
            "'features_main' is the canonical model input."
        ),
    }
    json_path = ARTIFACTS / "match_features.json"
    json_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    log.info("Saved %s (%d rows, %d cols)", csv_path.name, len(df), df.shape[1])
    log.info("Saved %s (main=%d, benchmark=%d)", json_path.name, len(main), len(benchmark))