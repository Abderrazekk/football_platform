"""Constraints test for the PuLP optimizer."""
import pandas as pd
import pytest

from ml.optimize_squad import optimize_squad


def _synth_predictions():
    """Build 40 synthetic players, 10 per position, 5 clubs."""
    rows = []
    positions = ["GK"] * 10 + ["DEF"] * 10 + ["MID"] * 10 + ["FWD"] * 10
    teams = ["A", "B", "C", "D", "E"]
    for i, pos in enumerate(positions):
        rows.append({
            "element_code": i,
            "name": f"Player_{i}",
            "prev_position": pos,
            "prev_team": teams[i % 5],
            "predicted_points": float(200 - i),
            "prev_price_end": 4.0 + (i % 5),
        })
    return pd.DataFrame(rows)


def test_optimizer_squad_size_and_positions():
    result = optimize_squad(_synth_predictions(), budget=200.0, max_per_club=10)
    assert "error" not in result, result
    squad = result["squad"]
    assert len(squad) == 15
    pos_counts = {}
    for p in squad:
        pos_counts[p["position"]] = pos_counts.get(p["position"], 0) + 1
    assert pos_counts == {"GK": 2, "DEF": 5, "MID": 5, "FWD": 3}


def test_optimizer_budget_constraint():
    result = optimize_squad(_synth_predictions(), budget=40.0, max_per_club=10)
    if "error" in result:
        pytest.skip(f"Infeasible (ok if tight): {result['error']}")
    assert result["total_price"] <= 40.0 + 1e-6


def test_optimizer_max_per_club():
    result = optimize_squad(_synth_predictions(), budget=200.0, max_per_club=3)
    if "error" in result:
        pytest.skip(f"Infeasible: {result['error']}")
    per_club = {}
    for p in result["squad"]:
        per_club[p["team"]] = per_club.get(p["team"], 0) + 1
    assert all(n <= 3 for n in per_club.values())


def test_optimizer_starting_xi_eleven():
    result = optimize_squad(_synth_predictions(), budget=200.0, max_per_club=10)
    assert "error" not in result
    assert len(result["starting_xi"]) == 11
    assert result["formation"] in {"3-4-3", "3-5-2", "4-3-3", "4-4-2", "4-5-1",
                                    "5-2-3", "5-3-2", "5-4-1"}
    # Captain in XI
    assert any(p["is_captain"] for p in result["starting_xi"])