"""
Business logic for the matches app.

Views stay thin: they call services.* which does the real work.
ML model is loaded lazily (singleton in ml.predict) — no import cost here.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from django.db.models import Q

from apps.matches.models import Match, Prediction, Team

import joblib
import pandas as pd
from django.db.models import Count, Q, Sum
from apps.matches.models import Season

log = logging.getLogger(__name__)
ARTIFACTS = Path(__file__).resolve().parent.parent.parent / "ml" / "artifacts"


# ---------------------------------------------------------------------------
# Model version (read once from the metrics JSON)
# ---------------------------------------------------------------------------
def _read_model_version() -> str:
    try:
        meta = json.loads((ARTIFACTS / "match_features.json").read_text(encoding="utf-8"))
        fm = meta.get("final_model", {})
        name = fm.get("name", "unknown")
        trained = fm.get("trained_at", "")[:10]
        return f"{name}-{trained}" if trained else name
    except Exception:
        return "unknown"


_MODEL_VERSION = _read_model_version()


# ---------------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------------
def predict(home_name: str, away_name: str, date=None) -> dict:
    """Wrap ml.predict.predict_match with error handling."""
    from ml.predict import predict_match  # deferred import — avoids Django startup cost
    return predict_match(home_name, away_name, date=date)


def save_prediction(user, home_team: Team, away_team: Team, probs: dict) -> Prediction:
    """Persist a prediction. `probs` = {'home', 'draw', 'away', ...}"""
    p_home = float(probs["home"])
    p_draw = float(probs["draw"])
    p_away = float(probs["away"])
    predicted = max(
        [("H", p_home), ("D", p_draw), ("A", p_away)],
        key=lambda x: x[1],
    )[0]
    return Prediction.objects.create(
        user=user,
        home_team=home_team,
        away_team=away_team,
        p_home=p_home,
        p_draw=p_draw,
        p_away=p_away,
        predicted_result=predicted,
        model_version=_MODEL_VERSION,
    )


# ---------------------------------------------------------------------------
# Recent form (last N matches for a team)
# ---------------------------------------------------------------------------
def recent_form(team_name: str, n: int = 5) -> list[dict]:
    team = Team.objects.filter(name=team_name).first()
    if team is None:
        return []
    matches = Match.objects.filter(
        Q(home_team=team) | Q(away_team=team),
        is_played=True,
    ).order_by("-date")[:n]

    out = []
    for m in matches:
        was_home = m.home_team_id == team.id
        gf = m.home_goals if was_home else m.away_goals
        ga = m.away_goals if was_home else m.home_goals
        if m.result == "D":
            pts, outcome = 1, "D"
        elif (was_home and m.result == "H") or (not was_home and m.result == "A"):
            pts, outcome = 3, "W"
        else:
            pts, outcome = 0, "L"
        out.append({
            "date": m.date,
            "opponent": m.away_team.name if was_home else m.home_team.name,
            "was_home": was_home,
            "goals_for": gf,
            "goals_against": ga,
            "outcome": outcome,
            "points": pts,
        })
    return out


# ---------------------------------------------------------------------------
# Head-to-head summary (all prior meetings between the two clubs)
# ---------------------------------------------------------------------------
def h2h_summary(home_name: str, away_name: str) -> dict | None:
    try:
        home = Team.objects.get(name=home_name)
        away = Team.objects.get(name=away_name)
    except Team.DoesNotExist:
        return None

    matches = Match.objects.filter(
        Q(home_team=home, away_team=away) | Q(home_team=away, away_team=home),
        is_played=True,
    ).order_by("-date")

    home_wins = draws = away_wins = 0
    for m in matches:
        if m.result == "D":
            draws += 1
        elif (
            (m.home_team_id == home.id and m.result == "H")
            or (m.home_team_id == away.id and m.result == "A")
        ):
            home_wins += 1
        else:
            away_wins += 1

    return {
        "home_wins": home_wins,
        "draws": draws,
        "away_wins": away_wins,
        "total": matches.count(),
        "recent": list(matches[:5]),
    }


# ---------------------------------------------------------------------------
# Personal accuracy stats (for the dashboard)
# ---------------------------------------------------------------------------
def personal_stats(user) -> dict:
    qs = Prediction.objects.filter(user=user)
    finished = qs.exclude(actual_result="")
    total = qs.count()
    completed = finished.count()
    correct = finished.filter(predicted_result=models.F("actual_result")).count() \
        if completed else 0
    return {
        "total": total,
        "completed": completed,
        "correct": correct,
        "accuracy": round(correct / completed, 3) if completed else None,
    }
    
    
    
    
    
    
    
    
   
   
   
   
#======================== phase 6 ========================================
# Lazy-loaded terminal state (Elo map) — no startup cost
_ELO_STATE = None


def _get_elo_state():
    global _ELO_STATE
    if _ELO_STATE is not None:
        return _ELO_STATE
    try:
        _ELO_STATE = joblib.load(ARTIFACTS / "terminal_state.joblib")
    except Exception:
        _ELO_STATE = {"elo": {}}
    return _ELO_STATE


# ---------------------------------------------------------------------------
# Standings
# ---------------------------------------------------------------------------
def get_standings(season_label: str) -> list[dict]:
    """Return [{position, team, played, wins, draws, losses, gf, ga, gd, points}, ...]."""
    season = Season.objects.filter(label=season_label).first()
    if season is None:
        return []

    home = (
        Match.objects.filter(season=season, is_played=True)
        .values("home_team_id")
        .annotate(
            played=Count("id"),
            wins=Count("id", filter=Q(result="H")),
            draws=Count("id", filter=Q(result="D")),
            losses=Count("id", filter=Q(result="A")),
            gf=Sum("home_goals"),
            ga=Sum("away_goals"),
        )
    )
    away = (
        Match.objects.filter(season=season, is_played=True)
        .values("away_team_id")
        .annotate(
            played=Count("id"),
            wins=Count("id", filter=Q(result="A")),
            draws=Count("id", filter=Q(result="D")),
            losses=Count("id", filter=Q(result="H")),
            gf=Sum("away_goals"),
            ga=Sum("home_goals"),
        )
    )

    merged: dict[int, dict] = {}
    for row in home:
        merged[row["home_team_id"]] = dict(row)
    for row in away:
        tid = row["away_team_id"]
        if tid in merged:
            for k in ("played", "wins", "draws", "losses", "gf", "ga"):
                merged[tid][k] = (merged[tid].get(k) or 0) + (row.get(k) or 0)
        else:
            merged[tid] = dict(row)

    team_map = {t.id: t for t in Team.objects.filter(id__in=merged.keys())}
    rows = []
    for tid, data in merged.items():
        team = team_map.get(tid)
        if team is None:
            continue
        gf = data["gf"] or 0
        ga = data["ga"] or 0
        wins = data["wins"] or 0
        draws = data["draws"] or 0
        rows.append({
            "team": team,
            "played": data["played"] or 0,
            "wins": wins,
            "draws": draws,
            "losses": data["losses"] or 0,
            "gf": gf,
            "ga": ga,
            "gd": gf - ga,
            "points": wins * 3 + draws,
        })

    rows.sort(key=lambda r: (-r["points"], -r["gd"], -r["gf"]))
    for i, r in enumerate(rows, start=1):
        r["position"] = i
    return rows


# ---------------------------------------------------------------------------
# Team season history
# ---------------------------------------------------------------------------
def _team_season_stats(team, season) -> dict:
    home = Match.objects.filter(season=season, home_team=team, is_played=True)
    away = Match.objects.filter(season=season, away_team=team, is_played=True)

    w = d = l = gf = ga = 0
    for m in home:
        gf += m.home_goals or 0
        ga += m.away_goals or 0
        if m.result == "H":
            w += 1
        elif m.result == "D":
            d += 1
        else:
            l += 1
    for m in away:
        gf += m.away_goals or 0
        ga += m.home_goals or 0
        if m.result == "A":
            w += 1
        elif m.result == "D":
            d += 1
        else:
            l += 1

    played = w + d + l
    return {
        "played": played,
        "wins": w, "draws": d, "losses": l,
        "gf": gf, "ga": ga, "gd": gf - ga,
        "points": w * 3 + d,
    }


def team_season_history(team, n: int = 5) -> list[dict]:
    """Last `n` seasons this team played in, latest first."""
    seasons = Season.objects.order_by("-label")[:n]
    out = []
    for season in seasons:
        stats = _team_season_stats(team, season)
        if stats["played"] > 0:
            out.append({"season": season.label, **stats})
    return out


# ---------------------------------------------------------------------------
# Elo history from Phase 3 feature table (cached lazily)
# ---------------------------------------------------------------------------
_FEATURES_DF = None


def _load_features_df():
    global _FEATURES_DF
    if _FEATURES_DF is not None:
        return _FEATURES_DF
    path = ARTIFACTS.parent.parent / "data" / "processed" / "match_features.csv"
    if not path.exists():
        return None
    try:
        _FEATURES_DF = pd.read_csv(path, parse_dates=["date"])
    except Exception:
        return None
    return _FEATURES_DF


def elo_history_from_features(team_name: str, n: int = 20) -> list[dict]:
    df = _load_features_df()
    if df is None:
        return []
    home = df[df["home_team"] == team_name][["date", "elo_home"]].rename(columns={"elo_home": "elo"})
    away = df[df["away_team"] == team_name][["date", "elo_away"]].rename(columns={"elo_away": "elo"})
    combined = pd.concat([home, away]).sort_values("date").tail(n)
    return [
        {"date": row["date"].strftime("%Y-%m-%d"), "elo": float(row["elo"])}
        for _, row in combined.iterrows()
    ]


# ---------------------------------------------------------------------------
# Compare two teams
# ---------------------------------------------------------------------------
def compare_teams_data(a_name: str, b_name: str) -> dict | None:
    try:
        a = Team.objects.get(name=a_name)
        b = Team.objects.get(name=b_name)
    except Team.DoesNotExist:
        return None

    elo = _get_elo_state().get("elo", {})

    def stats_for(team):
        matches = Match.objects.filter(
            Q(home_team=team) | Q(away_team=team), is_played=True
        ).order_by("-date")[:10]
        if not matches:
            return {"attack": 0.0, "defense": 0.0, "form": 0.0,
                    "elo": elo.get(team.name, 1500.0), "home_win_rate": 0.0}
        gf = ga = pts = 0
        home_matches = 0
        home_wins = 0
        for m in matches:
            was_home = m.home_team_id == team.id
            t_gf = m.home_goals if was_home else m.away_goals
            t_ga = m.away_goals if was_home else m.home_goals
            gf += t_gf or 0
            ga += t_ga or 0
            if was_home:
                home_matches += 1
                if m.result == "H":
                    home_wins += 1
            if m.result == "D":
                pts += 1
            elif (was_home and m.result == "H") or (not was_home and m.result == "A"):
                pts += 3
        n = len(matches)
        return {
            "attack": round(gf / n, 2),
            "defense": round(ga / n, 2),
            "form": round(pts / n, 2),
            "elo": round(elo.get(team.name, 1500.0), 0),
            "home_win_rate": round(home_wins / home_matches, 2) if home_matches else 0.0,
        }

    return {
        "team_a": {"id": a.id, "name": a.name, **stats_for(a)},
        "team_b": {"id": b.id, "name": b.name, **stats_for(b)},
    }