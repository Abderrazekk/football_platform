from django.db import models
from django.conf import settings


class Season(models.Model):
    label = models.CharField(max_length=10, unique=True)  # "2020-21"
    is_current = models.BooleanField(default=False)

    class Meta:
        ordering = ["label"]

    def __str__(self) -> str:
        return self.label


class Team(models.Model):
    name = models.CharField(max_length=100, unique=True)  # canonical
    short_name = models.CharField(max_length=10, blank=True)

    class Meta:
        ordering = ["name"]

    def __str__(self) -> str:
        return self.name


class Match(models.Model):
    RESULT_CHOICES = [("H", "Home win"), ("D", "Draw"), ("A", "Away win")]

    season = models.ForeignKey(Season, on_delete=models.CASCADE, related_name="matches")
    date = models.DateTimeField(db_index=True)
    home_team = models.ForeignKey(Team, on_delete=models.CASCADE, related_name="home_matches")
    away_team = models.ForeignKey(Team, on_delete=models.CASCADE, related_name="away_matches")
    home_goals = models.PositiveSmallIntegerField(null=True, blank=True)
    away_goals = models.PositiveSmallIntegerField(null=True, blank=True)
    result = models.CharField(max_length=1, choices=RESULT_CHOICES, blank=True)
    is_played = models.BooleanField(default=False)

    home_xg = models.FloatField(null=True, blank=True)
    away_xg = models.FloatField(null=True, blank=True)

    home_shots = models.PositiveSmallIntegerField(null=True, blank=True)
    away_shots = models.PositiveSmallIntegerField(null=True, blank=True)
    home_shots_on_target = models.PositiveSmallIntegerField(null=True, blank=True)
    away_shots_on_target = models.PositiveSmallIntegerField(null=True, blank=True)
    home_corners = models.PositiveSmallIntegerField(null=True, blank=True)
    away_corners = models.PositiveSmallIntegerField(null=True, blank=True)

    odds_b365_h = models.FloatField(null=True, blank=True)
    odds_b365_d = models.FloatField(null=True, blank=True)
    odds_b365_a = models.FloatField(null=True, blank=True)

    class Meta:
        ordering = ["-date"]
        constraints = [
            models.UniqueConstraint(
                fields=["season", "date", "home_team", "away_team"],
                name="unique_match_per_season_date_teams",
            ),
        ]
        indexes = [models.Index(fields=["season", "date"])]

    def __str__(self) -> str:
        return (
            f"{self.date:%Y-%m-%d} {self.home_team} "
            f"{self.home_goals}-{self.away_goals} {self.away_team}"
        )
        
        
        
               
class Prediction(models.Model):
    RESULT_CHOICES = [("H", "Home win"), ("D", "Draw"), ("A", "Away win")]

    user = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.CASCADE,
        related_name="predictions",
    )
    home_team = models.ForeignKey(
        Team, on_delete=models.CASCADE, related_name="predictions_as_home"
    )
    away_team = models.ForeignKey(
        Team, on_delete=models.CASCADE, related_name="predictions_as_away"
    )
    match = models.ForeignKey(
        Match, null=True, blank=True,
        on_delete=models.SET_NULL,
        related_name="predictions",
    )

    p_home = models.FloatField()
    p_draw = models.FloatField()
    p_away = models.FloatField()
    predicted_result = models.CharField(max_length=1, choices=RESULT_CHOICES)
    actual_result = models.CharField(max_length=1, choices=RESULT_CHOICES, blank=True)
    model_version = models.CharField(max_length=100, blank=True)

    created_at = models.DateTimeField(auto_now_add=True, db_index=True)

    class Meta:
        ordering = ["-created_at"]
        indexes = [models.Index(fields=["user", "-created_at"])]

    def __str__(self) -> str:
        return (
            f"{self.user.username}: {self.home_team} vs {self.away_team} "
            f"({self.predicted_result})"
        )

    @property
    def is_correct(self):
        """None when not yet played, else True/False."""
        if not self.actual_result:
            return None
        return self.predicted_result == self.actual_result        
        
        
        
        