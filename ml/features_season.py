"""
Season-level feature engineering.

One row per (player, target_season) with features from PRIOR seasons only.
Target = total_points of target_season.

No leakage: uses only history with season_name < target_season.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path

DATA_PROCESSED = Path(__file__).resolve().parent.parent / "data" / "processed"
DATA_RAW = Path(__file__).resolve().parent.parent / "data" / "raw"

FEATURES_SEASON = [
    "prev_points", "prev_minutes", "prev_goals", "prev_assists",
    "avg_points_2y", "avg_points_3y",
    "prev_price_end", "prev_price_start",
    "n_prior_seasons", "age", "prev_team_elo",
    "pos_GK", "pos_DEF", "pos_MID", "pos_FWD",
]


# ---------------------------------------------------------------------------
# Data loaders
# ---------------------------------------------------------------------------
def load_history() -> pd.DataFrame:
    df = pd.read_csv(DATA_PROCESSED / "player_history.csv")
    df = df.dropna(subset=["element_code"]).copy()
    df["element_code"] = df["element_code"].astype(int)
    return df


def build_meta_from_pgw(pgw: pd.DataFrame) -> pd.DataFrame:
    """Per (element_code, season): position + team at end of season."""
    pgw = pgw.dropna(subset=["element_code"]).copy()
    pgw["element_code"] = pgw["element_code"].astype(int)
    pgw = pgw.sort_values(["element_code", "season", "gameweek"])
    last = pgw.groupby(["element_code", "season"]).tail(1)
    return (
        last[["element_code", "season", "team", "position"]]
        .rename(columns={"season": "season_name"})
        .reset_index(drop=True)
    )


def build_team_elo(mf: pd.DataFrame) -> pd.DataFrame:
    """Per (team, season): mean Elo."""
    home = mf[["season", "home_team", "elo_home"]].rename(columns={"home_team": "team", "elo_home": "elo"})
    away = mf[["season", "away_team", "elo_away"]].rename(columns={"away_team": "team", "elo_away": "elo"})
    combined = pd.concat([home, away])
    return combined.groupby(["season", "team"])["elo"].mean().reset_index()


def load_birthdates() -> pd.DataFrame | None:
    path = DATA_RAW / "player_birthdates.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path)
    if df.empty or "birth_date" not in df.columns:
        return None
    df = df.dropna(subset=["element_code", "birth_date"]).copy()
    df["element_code"] = df["element_code"].astype(int)
    df["bd_dt"] = pd.to_datetime(df["birth_date"], errors="coerce")
    df = df.dropna(subset=["bd_dt"])
    if df.empty:
        return None
    return df[["element_code", "bd_dt"]]


# ---------------------------------------------------------------------------
# Training table
# ---------------------------------------------------------------------------
def build_training_table(
    history: pd.DataFrame,
    meta: pd.DataFrame,
    team_elo: pd.DataFrame,
    birthdates: pd.DataFrame | None,
) -> pd.DataFrame:
    df = history.copy()

    # Add position/team per (player, season)
    df = df.merge(meta, on=["element_code", "season_name"], how="left")

    # Add team Elo per (player, season)
    df = df.merge(
        team_elo,
        left_on=["team", "season_name"],
        right_on=["team", "season"],
        how="left",
        suffixes=("", "_team"),
    ).rename(columns={"elo": "team_elo"}).drop(columns=["season"], errors="ignore")

    df = df.sort_values(["element_code", "season_name"]).reset_index(drop=True)

    # Shifted features (previous season values)
    g = df.groupby("element_code")
    df["prev_points"] = g["total_points"].shift(1)
    df["prev_minutes"] = g["minutes"].shift(1)
    df["prev_goals"] = g["goals_scored"].shift(1)
    df["prev_assists"] = g["assists"].shift(1)
    df["prev_price_end"] = g["price_end"].shift(1)
    df["prev_price_start"] = g["price_start"].shift(1)
    df["prev_team"] = g["team"].shift(1)
    df["prev_position"] = g["position"].shift(1)
    df["prev_team_elo"] = g["team_elo"].shift(1)

    df["avg_points_2y"] = g["total_points"].transform(
        lambda s: s.shift(1).rolling(2, min_periods=1).mean()
    )
    df["avg_points_3y"] = g["total_points"].transform(
        lambda s: s.shift(1).rolling(3, min_periods=1).mean()
    )
    df["n_prior_seasons"] = g.cumcount()

    # Keep only rows with at least one prior season
    df = df[df["prev_points"].notna()].copy()

    # Age at target season
    if birthdates is not None:
        df = df.merge(birthdates, on="element_code", how="left")
        df["target_year"] = df["season_name"].str.split("-").str[0].astype(int)
        df["age"] = df["target_year"] - df["bd_dt"].dt.year
    else:
        df["age"] = np.nan

    # Position one-hot
    df = pd.concat([df, pd.get_dummies(df["prev_position"], prefix="pos", dtype=float)], axis=1)
    for p in ("GK", "DEF", "MID", "FWD"):
        if f"pos_{p}" not in df.columns:
            df[f"pos_{p}"] = 0.0

    df["target_season"] = df["season_name"]
    df["target_points"] = df["total_points"]

    return df


# ---------------------------------------------------------------------------
# Prediction table (for 2026-27)
# ---------------------------------------------------------------------------
def build_prediction_table(
    history: pd.DataFrame,
    meta: pd.DataFrame,
    team_elo: pd.DataFrame,
    birthdates: pd.DataFrame | None,
    target_season: str = "2026-27",
) -> pd.DataFrame:
    df = history.copy()
    df = df.merge(meta, on=["element_code", "season_name"], how="left")
    df = df.merge(
        team_elo,
        left_on=["team", "season_name"],
        right_on=["team", "season"],
        how="left",
        suffixes=("", "_team"),
    ).rename(columns={"elo": "team_elo"}).drop(columns=["season"], errors="ignore")

    df = df.sort_values(["element_code", "season_name"]).reset_index(drop=True)

    rows = []
    for code, group in df.groupby("element_code"):
        group = group.sort_values("season_name")
        last = group.iloc[-1]

        prev_points = last["total_points"]
        avg_2y = group["total_points"].tail(2).mean()
        avg_3y = group["total_points"].tail(3).mean()

        age = np.nan
        if birthdates is not None:
            bd_row = birthdates[birthdates["element_code"] == code]
            if not bd_row.empty:
                bd = bd_row.iloc[0]["bd_dt"]
                if pd.notna(bd):
                    age = int(target_season.split("-")[0]) - bd.year

        rows.append({
            "element_code": code,
            "name": last["name"],
            "prev_points": prev_points,
            "prev_minutes": last["minutes"],
            "prev_goals": last["goals_scored"],
            "prev_assists": last["assists"],
            "avg_points_2y": avg_2y,
            "avg_points_3y": avg_3y,
            "prev_price_end": last["price_end"],
            "prev_price_start": last["price_start"],
            "prev_team": last["team"],
            "prev_position": last["position"],
            "prev_team_elo": last["team_elo"],
            "n_prior_seasons": len(group),
            "age": age,
        })

    out = pd.DataFrame(rows)
    out = pd.concat([out, pd.get_dummies(out["prev_position"], prefix="pos", dtype=float)], axis=1)
    for p in ("GK", "DEF", "MID", "FWD"):
        if f"pos_{p}" not in out.columns:
            out[f"pos_{p}"] = 0.0

    return out