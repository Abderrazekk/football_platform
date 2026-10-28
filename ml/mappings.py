"""
Team, position and season mappings across data sources.

Source of truth: Phase 1 EDA (notebooks/01_exploration.ipynb).
Canonical team names = E0 short form (football-data.co.uk).
"""

# --- Team names -------------------------------------------------------------
# E0 is already canonical. Only Understat and FPL need mapping.

UNDERSTAT_TO_CANONICAL = {
    "Arsenal": "Arsenal",
    "Aston_Villa": "Aston Villa",
    "Bournemouth": "Bournemouth",
    "Brentford": "Brentford",
    "Brighton": "Brighton",
    "Burnley": "Burnley",
    "Chelsea": "Chelsea",
    "Crystal_Palace": "Crystal Palace",
    "Everton": "Everton",
    "Fulham": "Fulham",
    "Ipswich": "Ipswich",
    "Leeds": "Leeds",
    "Leicester": "Leicester",
    "Liverpool": "Liverpool",
    "Luton": "Luton",
    "Manchester_City": "Man City",
    "Manchester_United": "Man United",
    "Newcastle_United": "Newcastle",
    "Norwich": "Norwich",
    "Nottingham_Forest": "Nott'm Forest",
    "Sheffield_United": "Sheffield United",
    "Southampton": "Southampton",
    "Tottenham": "Tottenham",
    "Watford": "Watford",
    "West_Bromwich_Albion": "West Brom",
    "West_Ham": "West Ham",
    "Wolverhampton_Wanderers": "Wolves",
}

FPL_TO_CANONICAL = {
    "Arsenal": "Arsenal",
    "Aston Villa": "Aston Villa",
    "Bournemouth": "Bournemouth",
    "Brentford": "Brentford",
    "Brighton": "Brighton",
    "Burnley": "Burnley",
    "Chelsea": "Chelsea",
    "Crystal Palace": "Crystal Palace",
    "Everton": "Everton",
    "Fulham": "Fulham",
    "Ipswich": "Ipswich",
    "Leeds": "Leeds",
    "Leicester": "Leicester",
    "Liverpool": "Liverpool",
    "Luton": "Luton",
    "Man City": "Man City",
    "Man Utd": "Man United",
    "Newcastle": "Newcastle",
    "Norwich": "Norwich",
    "Nott'm Forest": "Nott'm Forest",
    "Sheffield Utd": "Sheffield United",
    "Southampton": "Southampton",
    "Spurs": "Tottenham",
    "Sunderland": "Sunderland",
    "Watford": "Watford",
    "West Brom": "West Brom",
    "West Ham": "West Ham",
    "Wolves": "Wolves",
}

# --- Position ---------------------------------------------------------------
POSITION_MAP = {
    "GK": "GK",
    "GKP": "GK",
    "DEF": "DEF",
    "MID": "MID",
    "AM": "MID",
    "FWD": "FWD",
}

# --- Seasons ----------------------------------------------------------------
SEASONS = ["2020-21", "2021-22", "2022-23", "2023-24", "2024-25", "2025-26", "2026-27"]

# Used to filter Understat rows to their own season (Phase 1 CHECK 2 fix)
SEASON_DATE_RANGES = {
    "2020-21": ("2020-08-01", "2021-07-31"),
    "2021-22": ("2021-08-01", "2022-07-31"),
    "2022-23": ("2022-08-01", "2023-07-31"),
    "2023-24": ("2023-08-01", "2024-07-31"),
    "2024-25": ("2024-08-01", "2025-07-31"),
    "2025-26": ("2025-08-01", "2026-07-31"),
    "2026-27": ("2026-08-01", "2027-07-31"),
}

# Seasons included in player-level modelling
# (2026-27 player_gw is a copy of 2025-26 -> excluded)
PLAYER_SEASONS = ["2020-21", "2021-22", "2022-23", "2023-24", "2024-25", "2025-26"]