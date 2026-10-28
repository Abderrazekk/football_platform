# apps/matches/admin.py
from django.contrib import admin
from .models import Match, Season, Team


@admin.register(Season)
class SeasonAdmin(admin.ModelAdmin):
    list_display = ("label", "is_current")
    list_filter = ("is_current",)


@admin.register(Team)
class TeamAdmin(admin.ModelAdmin):
    list_display = ("name", "short_name")
    search_fields = ("name",)


@admin.register(Match)
class MatchAdmin(admin.ModelAdmin):
    list_display = ("date", "season", "home_team", "away_team",
                    "home_goals", "away_goals", "result", "is_played")
    list_filter = ("season", "is_played", "result")
    search_fields = ("home_team__name", "away_team__name")
    date_hierarchy = "date"
    autocomplete_fields = ("home_team", "away_team")