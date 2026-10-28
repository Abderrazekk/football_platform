from django.urls import path

from apps.season import views

app_name = "season"

urlpatterns = [
    path("best-player/", views.best_player, name="best_player"),
    path("best-young/", views.best_young, name="best_young"),
    path("best-xi/", views.best_xi, name="best_xi"),
]