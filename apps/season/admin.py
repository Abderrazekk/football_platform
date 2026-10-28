from django.contrib import admin
from .models import SeasonForecast


@admin.register(SeasonForecast)
class SeasonForecastAdmin(admin.ModelAdmin):
    list_display = ("season", "kind", "model_version", "created_at")
    list_filter = ("season", "kind")
    readonly_fields = ("created_at",)