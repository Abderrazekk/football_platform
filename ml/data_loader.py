"""
Loads raw CSVs from data/raw/<season>/ into typed pandas DataFrames.

Design rules
------------
- Do NOT clean here. All cleaning lives in ml/cleaning.py.
- Missing files return None with a warning, never raise.
- Every returned DataFrame gets a ``season`` column (the folder name).
- No Django imports.
"""

from __future__ import annotations

import logging
import warnings
from pathlib import Path

import pandas as pd

# Silence harmless fragmentation warnings from read_csv + assign
warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)

logger = logging.getLogger(__name__)

DATA_RAW = Path(__file__).resolve().parent.parent / "data" / "raw"

E0_DATE_COL = "Match_Date"


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------
def list_seasons() -> list[str]:
    if not DATA_RAW.exists():
        raise FileNotFoundError(f"Data folder not found: {DATA_RAW}")
    seasons = sorted(p.name for p in DATA_RAW.iterdir() if p.is_dir())
    if not seasons:
        raise FileNotFoundError(f"No season folders under {DATA_RAW}")
    return seasons


def _e0_filename(season: str) -> str:
    """'2020-21' -> 'E0_Season_20202021.csv' (8-digit convention)."""
    start = season.split("-")[0]
    end = str(int(start) + 1)
    return f"E0_Season_{start}{end}.csv"


def _find_e0_file(season: str) -> Path | None:
    """Locate the E0 file tolerant to 8-digit / 6-digit / underscore naming."""
    folder = DATA_RAW / season
    if not folder.exists():
        return None
    for cand in (
        folder / _e0_filename(season),                                    # 8-digit
        folder / f"E0_Season_{season.replace('-', '')}.csv",              # 6-digit
        folder / f"E0_Season_{season[:4]}_{int(season[:4]) + 1}.csv",     # underscore
    ):
        if cand.exists():
            return cand
    globbed = sorted(folder.glob("E0_Season_*.csv"))
    return globbed[0] if globbed else None


def file_inventory(season: str) -> dict[str, bool]:
    folder = DATA_RAW / season
    return {
        "e0":             _find_e0_file(season) is not None,
        "gameweeks":      (folder / "master_gameweeks.csv").exists(),
        "player_gw":      (folder / "master_player_gw.csv").exists(),
        "player_history": (folder / "master_player_history.csv").exists(),
        "understat":      (folder / "master_understat_teams.csv").exists(),
        "xp":             (folder / "master_xP_projections.csv").exists(),
    }


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------
def _safe_read(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        logger.warning("File not found, returning None: %s", path)
        return None
    return pd.read_csv(path)


def _with_season(df: pd.DataFrame | None, season: str, col: str = "season") -> pd.DataFrame | None:
    if df is None:
        return None
    df = df.copy()          # defragments before assign -> no warning
    df[col] = season
    return df


def load_e0(season: str) -> pd.DataFrame | None:
    path = _find_e0_file(season)
    if path is None:
        logger.warning("No E0 file found for season %s in %s",
                       season, DATA_RAW / season)
        return None
    return _with_season(_safe_read(path), season)


def load_gameweeks(season: str) -> pd.DataFrame | None:
    return _with_season(_safe_read(DATA_RAW / season / "master_gameweeks.csv"), season)


def load_player_gw(season: str) -> pd.DataFrame | None:
    return _with_season(_safe_read(DATA_RAW / season / "master_player_gw.csv"), season)


def load_player_history(season: str) -> pd.DataFrame | None:
    return _with_season(_safe_read(DATA_RAW / season / "master_player_history.csv"),
                        season, col="folder_season")


def load_understat(season: str) -> pd.DataFrame | None:
    return _with_season(_safe_read(DATA_RAW / season / "master_understat_teams.csv"), season)


def load_xp_projections(season: str) -> pd.DataFrame | None:
    return _with_season(_safe_read(DATA_RAW / season / "master_xP_projections.csv"), season)


# ---------------------------------------------------------------------------
# Convenience
# ---------------------------------------------------------------------------
def load_all(season: str) -> dict[str, pd.DataFrame | None]:
    return {
        "e0": load_e0(season),
        "gameweeks": load_gameweeks(season),
        "player_gw": load_player_gw(season),
        "player_history": load_player_history(season),
        "understat": load_understat(season),
        "xp": load_xp_projections(season),
    }


def load_all_seasons() -> dict[str, dict[str, pd.DataFrame | None]]:
    return {s: load_all(s) for s in list_seasons()}