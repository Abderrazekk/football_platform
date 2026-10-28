"""
Player feature engineering — leak-safe rolling stats per gameweek.

Every rolling window uses shift(1), guaranteeing that features for a
player's row N use ONLY rows < N (strictly earlier kickoffs). Anti-leak
audit is in ml/tests/test_player_model.py::test_no_future_leakage.

Data source: data/processed/player_gw.csv (already cleaned in Phase 2) +
data/processed/match_features.csv (for historical Elo joins).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

DATA_PROCESSED = Path(__file__).resolve().parent.parent / "data" / "processed"

FEATURES = [
    "was_home",
    "price",
    "gameweek",
    "minutes_roll3", "minutes_roll5",
    "total_points_roll3", "total_points_roll5",
    "bps_roll3", "bps_roll5",
    "goals_scored_roll3", "goals_scored_roll5",
    "assists_roll3", "assists_roll5",
    "xg_roll3", "xg_roll5",
    "xa_roll3", "xa_roll5",
    "has_xg",
    "team_elo", "opp_elo", "elo_diff",
    "days_rest",
    "pos_GK", "pos_DEF", "pos_MID", "pos_FWD",
]

META_COLUMNS = ["element_code", "season", "gameweek", "kickoff_utc", "match_key",
                "name", "team", "position", "total_points"]


def load_and_build() -> pd.DataFrame:
    """Return a feature-augmented player_gw DataFrame, sorted chronologically."""
    pgw = pd.read_csv(DATA_PROCESSED / "player_gw.csv", parse_dates=["kickoff_utc"])
    pgw = pgw.dropna(subset=["element_code"]).copy()
    pgw["element_code"] = pgw["element_code"].astype(int)

    # --- Join match_features to recover historical team/opponent Elo ----
    mf = pd.read_csv(DATA_PROCESSED / "match_features.csv", parse_dates=["date"])
    mf["date_str"] = mf["date"].dt.strftime("%Y-%m-%d")

    parts = pgw["match_key"].str.split("|", expand=True)
    pgw["mk_season"] = parts[0]
    pgw["mk_date"] = parts[1]
    pgw["mk_home"] = parts[2]
    pgw["mk_away"] = parts[3]

    mf_small = mf[["season", "date_str", "home_team", "away_team",
                   "elo_home", "elo_away"]].rename(columns={
        "season": "mf_season", "date_str": "mf_date",
        "home_team": "mf_home", "away_team": "mf_away",
    })

    pgw = pgw.merge(
        mf_small,
        left_on=["mk_season", "mk_date", "mk_home", "mk_away"],
        right_on=["mf_season", "mf_date", "mf_home", "mf_away"],
        how="left",
    )

    pgw["team_elo"] = np.where(pgw["was_home"], pgw["elo_home"], pgw["elo_away"])
    pgw["opp_elo"] = np.where(pgw["was_home"], pgw["elo_away"], pgw["elo_home"])
    pgw["elo_diff"] = pgw["team_elo"] - pgw["opp_elo"]

    # --- Sort chronologically per player before any rolling -------------
    pgw = pgw.sort_values(["element_code", "kickoff_utc"]).reset_index(drop=True)

    # --- Rolling features (all shift(1) → exclusive of current row) ------
    for window in (3, 5):
        for col in ("minutes", "total_points", "bps",
                    "goals_scored", "assists", "xg", "xa"):
            pgw[f"{col}_roll{window}"] = (
                pgw.groupby("element_code")[col]
                .transform(lambda s: s.shift(1).rolling(window, min_periods=1).mean())
            )
            
     # --- xG availability: FPL started tracking xG in 2022-23 --------------
    # has_xg=1 when rolling xG has real signal, 0 for pre-2022 seasons.
    # Fill NaN with 0 so the imputer sees a valid numeric column.
    pgw["has_xg"] = pgw["xg_roll5"].notna().astype(float)
    for col in ("xg_roll3", "xg_roll5", "xa_roll3", "xa_roll5"):
        pgw[col] = pgw[col].fillna(0.0)
        
        
    # --- Baselines (Phase 7 task: "last GW" and "rolling avg") ----------
    pgw["total_points_shift1"] = pgw.groupby("element_code")["total_points"].shift(1)

    # --- Rest days since previous match ---------------------------------
    pgw["days_rest"] = (
        pgw.groupby("element_code")["kickoff_utc"]
        .diff().dt.total_seconds() / 86400.0
    ).clip(upper=30)

    # --- Position one-hot ----------------------------------------------
    pgw = pd.concat([
        pgw,
        pd.get_dummies(pgw["position"], prefix="pos", dtype=float),
    ], axis=1)
    for pos in ("GK", "DEF", "MID", "FWD"):
        col = f"pos_{pos}"
        if col not in pgw.columns:
            pgw[col] = 0.0

    # --- Drop cold-start rows (no rolling history) ----------------------
    pgw = pgw[pgw["total_points_roll5"].notna()].copy()

    return pgw