"""
Best-squad optimizer using PuLP (integer linear programming).

Constraints (cahier des charges §6.5):
  - 15 players: 2 GK, 5 DEF, 5 MID, 3 FWD
  - total price <= 100.0 (configurable)
  - max 3 players per club
  - starting XI uses a valid Premier League formation
  - captain = highest expected points in the XI

Accepts BOTH column naming conventions:
  - 'prev_position' / 'prev_team' / 'prev_price_end'    (raw CSV)
  - 'position' / 'team' / 'price_end'                   (renamed in memory)

Robust to PuLP API variations via _make_binary_variable().
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

ARTIFACTS = Path(__file__).resolve().parent / "artifacts"

try:
    import pulp
except ImportError:
    pulp = None

VALID_FORMATIONS = [
    (3, 4, 3), (3, 5, 2),
    (4, 3, 3), (4, 4, 2), (4, 5, 1),
    (5, 2, 3), (5, 3, 2), (5, 4, 1),
]


# ---------------------------------------------------------------------------
# Robust binary variable creation (handles PuLP API variations)
# ---------------------------------------------------------------------------
def _make_binary_variable(name: str):
    if pulp is None:
        raise ImportError("PuLP not installed. Run `pip install pulp`.")
    # Strategy 1: standard keyword
    try:
        return pulp.LpVariable(name, cat="Binary")
    except TypeError:
        pass
    # Strategy 2: LpBinary constant
    if hasattr(pulp, "LpBinary"):
        try:
            return pulp.LpVariable(name, cat=pulp.LpBinary)
        except TypeError:
            pass
    # Strategy 3: positional (name, lowBound, upBound, cat)
    try:
        return pulp.LpVariable(name, 0, 1, "Binary")
    except TypeError:
        pass
    # Strategy 4: dicts helper
    try:
        d = pulp.LpVariable.dicts("x", [name], lowBound=0, upBound=1, cat="Binary")
        return d[name]
    except Exception:
        pass
    # All failed — informative error
    sig = "?"
    try:
        import inspect
        sig = str(inspect.signature(pulp.LpVariable.__init__))
    except Exception:
        pass
    raise RuntimeError(
        f"Cannot create binary LpVariable. "
        f"PuLP version={getattr(pulp, '__version__', '?')}, "
        f"file={getattr(pulp, '__file__', '?')}, "
        f"signature={sig}. "
        f"Try: pip uninstall pulp -y && pip install --upgrade pulp"
    )


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Accept both 'prev_*' and plain column names."""
    df = df.copy()
    rename_map = {}
    if "prev_position" in df.columns and "position" not in df.columns:
        rename_map["prev_position"] = "position"
    if "prev_team" in df.columns and "team" not in df.columns:
        rename_map["prev_team"] = "team"
    if "prev_price_end" in df.columns and "price_end" not in df.columns:
        rename_map["prev_price_end"] = "price_end"
    if rename_map:
        df = df.rename(columns=rename_map)
    return df


def optimize_squad(
    predictions: pd.DataFrame,
    budget: float = 100.0,
    max_per_club: int = 3,
) -> dict:
    if pulp is None:
        raise ImportError("PuLP not installed. Run `pip install pulp`.")

    df = _normalize_columns(predictions)

    required = ["element_code", "name", "position", "team", "predicted_points", "price_end"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        return {"error": f"missing required columns: {missing}"}

    df = df.dropna(subset=["price_end", "predicted_points", "position"]).copy()
    df = df[df["position"].isin(["GK", "DEF", "MID", "FWD"])].copy()
    df["price"] = df["price_end"].astype(float)
    df["team"] = df["team"].fillna("Unknown")
    df = df.reset_index(drop=True)

    if df.empty:
        return {"error": "No eligible players to optimize over."}

    # Feasibility precheck (avoids confusing "infeasible" errors)
    for pos, n in [("GK", 2), ("DEF", 5), ("MID", 5), ("FWD", 3)]:
        count = len(df[df["position"] == pos])
        if count < n:
            return {"error": f"Not enough {pos} players (need {n}, have {count})."}

    # ---- ILP ----
    prob = pulp.LpProblem("fpl_squad", pulp.LpMaximize)
    x = {i: _make_binary_variable(f"x_{i}") for i in df.index}

    prob += pulp.lpSum(x[i] * float(df.loc[i, "predicted_points"]) for i in df.index)

    for pos, n in [("GK", 2), ("DEF", 5), ("MID", 5), ("FWD", 3)]:
        idx = df[df["position"] == pos].index
        prob += pulp.lpSum(x[i] for i in idx) == n

    prob += pulp.lpSum(x[i] * df.loc[i, "price"] for i in df.index) <= budget

    for team in df["team"].unique():
        idx = df[df["team"] == team].index
        prob += pulp.lpSum(x[i] for i in idx) <= max_per_club

    status = prob.solve(pulp.PULP_CBC_CMD(msg=False))
    if pulp.LpStatus[status] != "Optimal":
        return {"error": f"Infeasible under constraints (status={pulp.LpStatus[status]})."}

    selected_idx = [i for i in df.index if x[i].varValue is not None and x[i].varValue > 0.5]
    if len(selected_idx) < 15:
        return {"error": f"Only {len(selected_idx)} players selected (expected 15)."}

    squad = df.loc[selected_idx].sort_values("predicted_points", ascending=False).reset_index(drop=True)

    # ---- Best XI ----
    best_xi = None
    best_score = -1.0
    best_formation_label = ""
    for n_def, n_mid, n_fwd in VALID_FORMATIONS:
        gk = squad[squad["position"] == "GK"].nlargest(1, "predicted_points")
        de = squad[squad["position"] == "DEF"].nlargest(n_def, "predicted_points")
        mi = squad[squad["position"] == "MID"].nlargest(n_mid, "predicted_points")
        fw = squad[squad["position"] == "FWD"].nlargest(n_fwd, "predicted_points")
        if len(de) < n_def or len(mi) < n_mid or len(fw) < n_fwd:
            continue
        xi = pd.concat([gk, de, mi, fw])
        score = float(xi["predicted_points"].sum())
        if score > best_score:
            best_score = score
            best_xi = xi
            best_formation_label = f"{n_def}-{n_mid}-{n_fwd}"

    if best_xi is None:
        return {"error": "No valid formation found from selected squad."}

    xi_codes = set(best_xi["element_code"].astype(int))
    captain_code = int(best_xi.sort_values("predicted_points", ascending=False).iloc[0]["element_code"])

    # ---- Serialize ----
    squad_records = []
    for _, row in squad.iterrows():
        code = int(row["element_code"])
        squad_records.append({
            "element_code": code,
            "name": str(row["name"]),
            "position": str(row["position"]),
            "team": str(row["team"]),
            "predicted_points": round(float(row["predicted_points"]), 1),
            "price": round(float(row["price"]), 1),
            "is_starter": code in xi_codes,
            "is_captain": code == captain_code,
        })

    xi_records = []
    for _, row in best_xi.sort_values(
        ["position", "predicted_points"], ascending=[True, False]
    ).iterrows():
        code = int(row["element_code"])
        xi_records.append({
            "element_code": code,
            "name": str(row["name"]),
            "position": str(row["position"]),
            "team": str(row["team"]),
            "predicted_points": round(float(row["predicted_points"]), 1),
            "price": round(float(row["price"]), 1),
            "is_captain": code == captain_code,
        })

    captain_record = next(r for r in squad_records if r["is_captain"])

    return {
        "squad": squad_records,
        "starting_xi": xi_records,
        "formation": best_formation_label,
        "captain": captain_record,
        "total_expected": round(float(squad["predicted_points"].sum()), 1),
        "total_price": round(float(squad["price"].sum()), 1),
        "n_players": len(squad_records),
    }


def _main_standalone() -> None:
    pred_path = ARTIFACTS / "season_predictions_2026_27.csv"
    if not pred_path.exists():
        print(f"Missing {pred_path}. Run `python -m ml.train_season_model` first.")
        sys.exit(1)
    preds = pd.read_csv(pred_path)
    result = optimize_squad(preds)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    _main_standalone()