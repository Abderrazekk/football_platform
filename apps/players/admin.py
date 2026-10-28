# apps/players/admin.py
from django.contrib import admin
from .models import Player, PlayerGameweekStats, PlayerSeasonStats


@admin.register(Player)
class PlayerAdmin(admin.ModelAdmin):
    list_display = ("name", "position", "current_team", "element_code")
    list_filter = ("position", "current_team")
    search_fields = ("name", "element_code")


@admin.register(PlayerSeasonStats)
class PlayerSeasonStatsAdmin(admin.ModelAdmin):
    list_display = ("player", "season", "minutes", "total_points",
                    "goals_scored", "assists", "price_start", "price_end")
    list_filter = ("season",)
    search_fields = ("player__name",)


@admin.register(PlayerGameweekStats)
class PlayerGameweekStatsAdmin(admin.ModelAdmin):
    list_display = ("player", "season", "gameweek", "minutes", "total_points", "bps")
    list_filter = ("season", "gameweek")
    search_fields = ("player__name",)