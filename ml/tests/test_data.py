"""
Sanity checks on data/processed/*.csv.

Run with:
    pytest ml/tests/ -v
"""

from pathlib import Path
import pandas as pd
import pytest

PROCESSED = Path(__file__).resolve().parent.parent.parent / "data" / "processed"


@pytest.fixture(scope="module")
def matches():
    return pd.read_csv(PROCESSED / "matches.csv")


@pytest.fixture(scope="module")
def player_gw():
    return pd.read_csv(PROCESSED / "player_gw.csv")


@pytest.fixture(scope="module")
def player_history():
    return pd.read_csv(PROCESSED / "player_history.csv")


def test_matches_per_season(matches):
    counts = matches.groupby("season").size()
    for s in ["2020-21", "2021-22", "2022-23", "2023-24", "2024-25", "2025-26"]:
        assert counts[s] == 380, f"{s}: {counts[s]} != 380"
    assert counts["2026-27"] == 70


def test_no_duplicate_matches(matches):
    dup = matches.duplicated(
        subset=["season", "date", "home_team", "away_team"]
    ).sum()
    assert dup == 0, f"{dup} duplicate matches"


def test_teams_mapped(matches):
    for col in ["home_team", "away_team"]:
        bad = matches[matches[col].str.contains("_", na=False)]
        assert len(bad) == 0, f"{len(bad)} unmapped teams in {col}"


def test_no_negative_minutes(player_gw):
    assert (player_gw["minutes"] < 0).sum() == 0


def test_player_history_season_labels(player_history):
    bad = player_history["season_name"].str.contains("/", na=False)
    assert bad.sum() == 0


def test_player_history_not_duplicated(player_history):
    dup = player_history.duplicated(subset=["element_code", "season_name"]).sum()
    assert dup == 0, f"{dup} duplicated (element_code, season_name) rows"


def test_element_code_bridge_coverage(player_gw):
    """Bridge must cover >=70% per season, >=90% overall.

    2020-21 and 2023-24 hit 100% (their history files contained the
    season). 2021-22/2022-23/2024-25/2025-26 land at 73-90% (foreign
    signings and promoted-team debuts with no prior history) — the
    global name fallback recovers most, and the residual <10% is
    acceptable per Phase 2 acceptance criteria (§2.3 no trivial perfection).
    """
    per_season_floor = 0.70
    overall_floor = 0.90

    for s in ["2020-21", "2021-22", "2022-23", "2023-24", "2024-25", "2025-26"]:
        sub = player_gw[player_gw["season"] == s]
        coverage = sub["element_code"].notna().mean()
        assert coverage >= per_season_floor, (
            f"{s}: per-season coverage {coverage:.1%} < {per_season_floor:.0%}"
        )

    overall = player_gw["element_code"].notna().mean()
    assert overall >= overall_floor, (
        f"Overall coverage {overall:.1%} < {overall_floor:.0%}"
    )

def test_match_key_coverage(player_gw):
    for s in ["2020-21", "2021-22", "2022-23", "2023-24", "2024-25", "2025-26"]:
        sub = player_gw[player_gw["season"] == s]
        coverage = sub["match_key"].notna().mean()
        assert coverage >= 0.95, f"{s}: match_key coverage {coverage:.1%} < 95%"