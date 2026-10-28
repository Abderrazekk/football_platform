"""
Idempotent import of data/processed/*.csv into SQLite.

Usage:
    python manage.py import_data            # fast: bulk_create, skip on conflict
    python manage.py import_data --reset    # clear DB tables first
"""

from pathlib import Path
import csv

from django.conf import settings
from django.core.management.base import BaseCommand
from django.db import transaction
from django.utils.dateparse import parse_datetime

from apps.matches.models import Match, Season, Team
from apps.players.models import Player, PlayerGameweekStats, PlayerSeasonStats

PROCESSED = Path(settings.BASE_DIR) / "data" / "processed"


class Command(BaseCommand):
    help = "Import data/processed/*.csv into SQLite (idempotent)."

    def add_arguments(self, parser):
        parser.add_argument("--reset", action="store_true",
                            help="Clear all tables before import.")

    def handle(self, *args, **opts):
        if not PROCESSED.exists():
            self.stderr.write(f"Missing folder: {PROCESSED}. "
                              "Run `python -m ml.build_processed` first.")
            return

        with transaction.atomic():
            if opts["reset"]:
                self._reset()
            self._import_seasons()
            self._import_teams()
            self._import_matches()
            self._import_players()
            self._import_season_stats()
            self._import_gw_stats()
        self.stdout.write(self.style.SUCCESS("Import complete."))

    # ------------------------------------------------------------------
    def _reset(self):
        self.stdout.write("Resetting DB tables ...")
        PlayerGameweekStats.objects.all().delete()
        PlayerSeasonStats.objects.all().delete()
        Match.objects.all().delete()
        Player.objects.all().delete()
        Team.objects.all().delete()
        Season.objects.all().delete()

    def _import_seasons(self):
        seasons = set()
        with open(PROCESSED / "matches.csv", newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                seasons.add(row["season"])
        for label in sorted(seasons):
            Season.objects.update_or_create(
                label=label,
                defaults={"is_current": label == "2026-27"},
            )
        self.stdout.write(f"Seasons: {len(seasons)}")

    def _import_teams(self):
        with open(PROCESSED / "teams.csv", newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                Team.objects.update_or_create(name=row["name"])
        self.stdout.write(f"Teams: {Team.objects.count()}")

    def _import_matches(self):
        season_by_label = {s.label: s for s in Season.objects.all()}
        team_by_name = {t.name: t for t in Team.objects.all()}

        objs = []
        with open(PROCESSED / "matches.csv", newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                try:
                    dt = parse_datetime(row["date"])
                except Exception:
                    continue
                if dt is None:
                    continue
                objs.append(Match(
                    season=season_by_label[row["season"]],
                    date=dt,
                    home_team=team_by_name[row["home_team"]],
                    away_team=team_by_name[row["away_team"]],
                    home_goals=_int(row["home_goals"]),
                    away_goals=_int(row["away_goals"]),
                    result=row["result"] or "",
                    is_played=row["is_played"].lower() == "true",
                    home_xg=_float(row["home_xg"]),
                    away_xg=_float(row["away_xg"]),
                    home_shots=_int(row["home_shots"]),
                    away_shots=_int(row["away_shots"]),
                    home_shots_on_target=_int(row["home_shots_on_target"]),
                    away_shots_on_target=_int(row["away_shots_on_target"]),
                    home_corners=_int(row["home_corners"]),
                    away_corners=_int(row["away_corners"]),
                    odds_b365_h=_float(row["odds_b365_h"]),
                    odds_b365_d=_float(row["odds_b365_d"]),
                    odds_b365_a=_float(row["odds_b365_a"]),
                ))
        Match.objects.bulk_create(objs, ignore_conflicts=True, batch_size=500)
        self.stdout.write(f"Matches: {Match.objects.count()}")

    def _import_players(self):
        team_by_name = {t.name: t for t in Team.objects.all()}
        objs = []
        with open(PROCESSED / "players.csv", newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                objs.append(Player(
                    element_code=int(row["element_code"]),
                    name=row["name"],
                    position=row["position"] or "",
                    current_team=team_by_name.get(row["current_team"]),
                ))
        Player.objects.bulk_create(objs, ignore_conflicts=True, batch_size=500)
        self.stdout.write(f"Players: {Player.objects.count()}")

    def _import_season_stats(self):
        season_by_label = {s.label: s for s in Season.objects.all()}
        player_by_code = {p.element_code: p for p in Player.objects.all()}

        objs = []
        with open(PROCESSED / "player_history.csv", newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                code = int(row["element_code"])
                season = season_by_label.get(row["season_name"])
                player = player_by_code.get(code)
                if season is None or player is None:
                    continue
                objs.append(PlayerSeasonStats(
                    player=player, season=season,
                    minutes=int(row["minutes"]),
                    goals_scored=int(row["goals_scored"]),
                    assists=int(row["assists"]),
                    total_points=int(row["total_points"]),
                    bps=int(row["bps"]),
                    bonus=int(row["bonus"]),
                    clean_sheets=int(row["clean_sheets"]),
                    ict_index=_float(row["ict_index"]),
                    price_start=_float(row["price_start"]),
                    price_end=_float(row["price_end"]),
                ))
        PlayerSeasonStats.objects.bulk_create(objs, ignore_conflicts=True, batch_size=1000)
        self.stdout.write(f"PlayerSeasonStats: {PlayerSeasonStats.objects.count()}")

    def _import_gw_stats(self):
        season_by_label = {s.label: s for s in Season.objects.all()}
        player_by_code = {p.element_code: p for p in Player.objects.all()}
        # Build match lookup key = "season|date|home|away"
        match_lookup = {
            f"{m.season.label}|{m.date.date().isoformat()}|"
            f"{m.home_team.name}|{m.away_team.name}": m
            for m in Match.objects.select_related("season", "home_team", "away_team")
        }

        batch = []
        n_missed = 0
        total = 0
        with open(PROCESSED / "player_gw.csv", newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                code = _int(row["element_code"])
                if code is None:
                    n_missed += 1
                    continue
                season = season_by_label.get(row["season"])
                player = player_by_code.get(code)
                if season is None or player is None:
                    n_missed += 1
                    continue
                mkey = row["match_key"] or ""
                match = match_lookup.get(mkey)

                batch.append(PlayerGameweekStats(
                    player=player, season=season, match=match,
                    gameweek=_int(row["gameweek"]),
                    was_home=(row["was_home"].lower() == "true") if row["was_home"] else None,
                    minutes=int(row["minutes"]),
                    total_points=int(row["total_points"]),
                    bps=int(row["bps"]),
                    goals_scored=int(row["goals_scored"]),
                    assists=int(row["assists"]),
                    clean_sheets=int(row["clean_sheets"]),
                    goals_conceded=int(row["goals_conceded"]),
                    saves=int(row["saves"]),
                    bonus=int(row["bonus"]),
                    xg=_float(row["xg"]),
                    xa=_float(row["xa"]),
                    xgi=_float(row["xgi"]),
                    xgc=_float(row["xgc"]),
                    price=_float(row["price"]),
                ))
                total += 1
                if len(batch) >= 2000:
                    PlayerGameweekStats.objects.bulk_create(batch, ignore_conflicts=True)
                    batch.clear()
        if batch:
            PlayerGameweekStats.objects.bulk_create(batch, ignore_conflicts=True)

        self.stdout.write(
            f"PlayerGameweekStats: {PlayerGameweekStats.objects.count()} "
            f"({total} attempted, {n_missed} skipped)"
        )


def _int(v):
    if v in (None, "", "nan", "None"):
        return None
    try:
        return int(float(v))
    except (ValueError, TypeError):
        return None


def _float(v):
    if v in (None, "", "nan", "None"):
        return None
    try:
        return float(v)
    except (ValueError, TypeError):
        return None