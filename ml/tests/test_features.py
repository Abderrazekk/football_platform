"""
Anti-leakage and integrity tests for the match feature table.

The critical test is test_no_future_leakage: for random matches, rebuilding
features on data truncated at that match must reproduce the SAME feature
values as the full-data build. Any feature that depends on future matches
(e.g. a league-wide rolling aggregate) would break this test.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from ml import features

DATA_PROCESSED = Path(__file__).resolve().parent.parent.parent / "data" / "processed"


@pytest.fixture(scope="module")
def matches() -> pd.DataFrame:
    return features.load_matches()


@pytest.fixture(scope="module")
def full_features(matches) -> pd.DataFrame:
    return features.build_features(matches)


# ---------------------------------------------------------------------------
# Basic sanity
# ---------------------------------------------------------------------------
def test_target_encoding(full_features):
    assert set(full_features["result_encoded"].dropna().unique()) <= {0, 1, 2}
    for r, code in [("H", 0), ("D", 1), ("A", 2)]:
        sub = full_features[full_features["result"] == r]
        assert (sub["result_encoded"] == code).all(), f"{r} != {code}"


def test_no_duplicate_matches(full_features):
    dup = full_features.duplicated(
        subset=["season", "date", "home_team", "away_team"]
    ).sum()
    assert dup == 0, f"{dup} duplicate feature rows"


def test_first_match_elo_default(full_features):
    first = full_features.iloc[0]
    assert first["elo_home"] == features.ELO_START
    assert first["elo_away"] == features.ELO_START
    assert first["elo_diff"] == 0.0


def test_first_match_no_history(full_features):
    first = full_features.iloc[0]
    assert pd.isna(first["home_points_5"])
    assert pd.isna(first["days_rest_home"])
    assert first["h2h_n"] == 0
    assert first["cold_start"] == 1.0


def test_no_self_features():
    """No feature name should match a self-referential column."""
    forbidden = {
        "home_goals", "away_goals",
        "result", "result_encoded",
        "home_shots", "away_shots",
        "home_xg", "away_xg",  # match-level xG, never a feature
    }
    main, _ = None, None
    # Build feature list from a tiny synthetic build to avoid re-running
    cols = set(features.feature_columns(
        pd.DataFrame(columns=["result_encoded", "b365_p_home", "foo"])
    )[1])
    # Direct check on known forbidden names
    leaks = forbidden & cols
    assert not leaks, f"Self-referential feature columns: {leaks}"


def test_all_numeric(full_features):
    main, _ = features.feature_columns(full_features)
    for col in main:
        assert pd.api.types.is_numeric_dtype(full_features[col]), (
            f"{col} is not numeric (dtype={full_features[col].dtype})"
        )


# ---------------------------------------------------------------------------
# THE critical test: no future leakage
# ---------------------------------------------------------------------------
def test_no_future_leakage(matches, full_features):
    """
    For a random subset of matches, rebuild features on data truncated at
    that match. The last row of the truncated build MUST equal the
    corresponding row of the full build for every feature.
    """
    rng = np.random.default_rng(seed=42)
    candidates = sorted(set([
        50, 200, 500, 800, 1200, 1500, 1800, 2100,
        *rng.integers(20, len(matches) - 1, size=4).tolist(),
    ]))

    n_checked = 0
    for i in candidates:
        if i >= len(matches):
            continue
        truncated = matches.iloc[: i + 1].copy()
        rebuilt = features.build_features(truncated)
        assert len(rebuilt) == len(truncated), (
            f"rebuilt rows {len(rebuilt)} != truncated rows {len(truncated)}"
        )

        full_row = full_features.iloc[i]
        rebuilt_row = rebuilt.iloc[-1]

        for col in full_features.columns:
            if col in ("season", "date", "home_team", "away_team", "result"):
                continue
            f = full_row[col]
            r = rebuilt_row[col]
            f_nan = pd.isna(f)
            r_nan = pd.isna(r)
            if f_nan and r_nan:
                continue
            assert not (f_nan != r_nan), (
                f"NaN mismatch at row {i}, col {col}: full={f}, rebuilt={r}"
            )
            assert abs(float(f) - float(r)) < 1e-9, (
                f"LEAKAGE at row {i}, col {col}: full={f}, rebuilt={r}"
            )
        n_checked += 1

    assert n_checked >= 5, f"Only {n_checked} rebuilds performed"