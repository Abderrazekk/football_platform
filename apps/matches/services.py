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