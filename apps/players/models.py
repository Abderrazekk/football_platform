from django.db import models

from apps.matches.models import Match, Season, Team


class Player(models.Model):
    POSITION_CHOICES = [
        ("GK", "Goalkeeper"),
        ("DEF", "Defender"),
        ("MID", "Midfielder"),
        ("FWD", "Forward"),
    ]

    element_code = models.BigIntegerField(unique=True)
    name = models.CharField(max_length=200, db_index=True)
    position = models.CharField(max_length=3, choices=POSITION_CHOICES, blank=True)
    current_team = models.ForeignKey(
        Team, on_delete=models.SET_NULL, null=True, blank=True,
        related_name="current_players",
    )
    birth_date = models.DateField(null=True, blank=True)  # enrichment in Phase 8

    class Meta:
        ordering = ["name"]

    def __str__(self) -> str:
        return f"{self.name} ({self.position})"


class PlayerSeasonStats(models.Model):
    player = models.ForeignKey(Player, on_delete=models.CASCADE, related_name="season_stats")
    season = models.ForeignKey(Season, on_delete=models.CASCADE, related_name="player_season_stats")

    minutes = models.PositiveIntegerField(default=0)
    goals_scored = models.PositiveSmallIntegerField(default=0)
    assists = models.PositiveSmallIntegerField(default=0)
    total_points = models.PositiveSmallIntegerField(default=0)
    bps = models.IntegerField(default=0)
    bonus = models.PositiveSmallIntegerField(default=0)
    clean_sheets = models.PositiveSmallIntegerField(default=0)

    ict_index = models.FloatField(null=True, blank=True)
    price_start = models.FloatField(null=True, blank=True)
    price_end = models.FloatField(null=True, blank=True)

    class Meta:
        constraints = [
            models.UniqueConstraint(
                fields=["player", "season"],
                name="unique_player_season_stats",
            ),
        ]
        ordering = ["-season__label", "-total_points"]


class PlayerGameweekStats(models.Model):
    player = models.ForeignKey(Player, on_delete=models.CASCADE, related_name="gw_stats")
    season = models.ForeignKey(Season, on_delete=models.CASCADE, related_name="player_gw_stats")
    match = models.ForeignKey(
        Match, on_delete=models.SET_NULL, null=True, blank=True,
        related_name="player_gw_stats",
    )
    gameweek = models.PositiveSmallIntegerField(null=True, blank=True, db_index=True)
    was_home = models.BooleanField(null=True, blank=True)

    minutes = models.PositiveSmallIntegerField(default=0)
    total_points = models.SmallIntegerField(default=0)
    bps = models.SmallIntegerField(default=0)
    goals_scored = models.PositiveSmallIntegerField(default=0)
    assists = models.PositiveSmallIntegerField(default=0)
    clean_sheets = models.PositiveSmallIntegerField(default=0)
    goals_conceded = models.PositiveSmallIntegerField(default=0)
    saves = models.PositiveSmallIntegerField(default=0)
    bonus = models.PositiveSmallIntegerField(default=0)

    xg = models.FloatField(null=True, blank=True)
    xa = models.FloatField(null=True, blank=True)
    xgi = models.FloatField(null=True, blank=True)
    xgc = models.FloatField(null=True, blank=True)

    price = models.FloatField(null=True, blank=True)

    class Meta:
        constraints = [
            models.UniqueConstraint(
                fields=["player", "season", "gameweek"],
                name="unique_player_season_gameweek",
            ),
        ]
        indexes = [models.Index(fields=["season", "gameweek"])]