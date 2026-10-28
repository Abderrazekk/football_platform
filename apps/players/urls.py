from django.urls import path

from apps.players import views

app_name = "players"

urlpatterns = [
    path("", views.players_list, name="list"),
    path("compare/", views.compare_view, name="compare"),
    path("match-stars/", views.match_stars, name="match_stars"),
    path("at-risk/", views.at_risk, name="at_risk"),
    path("<int:pk>/", views.player_detail, name="detail"),
]