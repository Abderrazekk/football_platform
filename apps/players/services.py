"""
Services for the players app.

Thin wrappers around ml.predict_player with:
  - Django cache (1-hour TTL) for ranking calls
  - graceful degradation if artifacts missing
"""

from __future__ import annotations

import logging

from django.core.cache import cache
from django.db.models import Q

from apps.matches.models import Match, Team
from apps.players.models import Player, PlayerGameweekStats, PlayerSeasonStats

log = logging.getLogger(__name__)
CACHE_TTL = 3600  # 1 hour


# ---------------------------------------------------------------------------
# Prediction — single player
# ---------------------------------------------------------------------------
def predict_player_vs(player: Player, opponent_team_name: str, was_home: bool) -> dict:
    """Return ml.predict_player.predict_player_points wrapped for our Player model."""
    from ml.predict_player import predict_player_points
    if not player.element_code:
        raise ValueError(f"Player {player.pk} has no element_code (cannot predict)")
    return predict_player_points(
        int(player.element_code), opponent_team_name, was_home=was_home
    )


# ---------------------------------------------------------------------------
# Ranking — top players / at-risk for a match (cached)
# ---------------------------------------------------------------------------
def rank_players_for_match(home_team: str, away_team: str, top_n: int = 10) -> dict | None:
    """Return ml.predict_player.rank_players_for_match result, cached."""
    if home_team == away_team:
        return None

    cache_key = f"match_stars:{home_team}:{away_team}:{top_n}"
    cached = cache.get(cache_key)
    if cached is not None:
        return cached

    from ml.predict_player import rank_players_for_match as _rank
    try:
        result = _rank(home_team, away_team, top_n=top_n)
        cache.set(cache_key, result, timeout=CACHE_TTL)
        return result
    except FileNotFoundError as e:
        log.warning("Player model missing: %s", e)
        return None
    except Exception as e:
        log.exception("rank_players_for_match failed: %s", e)
        return None


# ---------------------------------------------------------------------------
# Detail / compare — DB stats
# ---------------------------------------------------------------------------
def player_season_stats(player: Player):
    """All PlayerSeasonStats for a player, latest season first."""
    return (
        PlayerSeasonStats.objects
        .filter(player=player)
        .select_related("season")
        .order_by("-season__label")
    )


def player_recent_gameweeks(player: Player, n: int = 15):
    """Recent PlayerGameweekStats, latest first, with match prefetched."""
    return (
        PlayerGameweekStats.objects
        .filter(player=player)
        .select_related("season", "match", "match__home_team", "match__away_team")
        .order_by("-season__label", "-gameweek")[:n]
    )


def compare_players_data(
    player_a: Player, player_b: Player,
    opponent: str | None = None, was_home: bool = True,
) -> dict:
    """Side-by-side stats + optional predictions for both players."""

    def block(p: Player) -> dict:
        season_agg = (
            PlayerSeasonStats.objects
            .filter(player=p)
            .select_related("season")
            .order_by("-season__label")
            .first()
        )
        recent = (
            PlayerGameweekStats.objects
            .filter(player=p)
            .order_by("-season__label", "-gameweek")[:5]
        )
        recent_pts = [gw.total_points for gw in recent]
        recent_avg = round(sum(recent_pts) / len(recent_pts), 2) if recent_pts else None

        prediction = None
        if opponent:
            try:
                prediction = predict_player_vs(p, opponent, was_home)
            except Exception as e:
                log.info("compare prediction failed for player %s: %s", p.pk, e)

        return {
            "player": p,
            "season_agg": season_agg,
            "recent_avg": recent_avg,
            "prediction": prediction,
        }

    return {
        "a": block(player_a),
        "b": block(player_b),
        "opponent": opponent,
        "was_home": was_home,
    }


# ---------------------------------------------------------------------------
# Optional (C): player vs specific opponent history
# ---------------------------------------------------------------------------
def player_vs_opponent_history(player: Player, opponent_team_name: str) -> list[dict]:
    """Recent PlayerGameweekStats vs a specific team, latest first."""
    opponent = Team.objects.filter(name=opponent_team_name).first()
    if opponent is None:
        return []

    rows = (
        PlayerGameweekStats.objects
        .filter(
            player=player,
            match__isnull=False,
        )
        .filter(
            Q(match__home_team=opponent) | Q(match__away_team=opponent)
        )
        .select_related("match", "season", "match__home_team", "match__away_team")
        .order_by("-match__date")[:10]
    )
    out = []
    for gw in rows:
        out.append({
            "date": gw.match.date,
            "was_home": gw.was_home,
            "minutes": gw.minutes,
            "total_points": gw.total_points,
            "bps": gw.bps,
        })
    return out