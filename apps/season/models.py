from django.db import models


class SeasonForecast(models.Model):
    """Cache for season-level forecasts (JSON payload, computed once)."""

    KIND_CHOICES = [
        ("best_player", "Best player"),
        ("best_young", "Best young player"),
        ("best_xi", "Best XI"),
        ("table", "Final table"),
    ]

    season = models.CharField(max_length=10, db_index=True)
    kind = models.CharField(max_length=20, choices=KIND_CHOICES)
    payload = models.JSONField()
    model_version = models.CharField(max_length=100, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        unique_together = [["season", "kind"]]
        ordering = ["-created_at"]

    def __str__(self) -> str:
        return f"{self.season} · {self.get_kind_display()}"