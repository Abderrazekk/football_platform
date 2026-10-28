from django.urls import path

from apps.matches import views

app_name = "matches"

urlpatterns = [
    path("predict/", views.predict_view, name="predict"),
    path("history/", views.history_view, name="history"),
    path("history/<int:pk>/", views.prediction_detail_view, name="detail"),
    path("history/<int:pk>/delete/", views.delete_prediction_view, name="delete"),
    path("teams/", views.teams_list, name="teams_list"),
    path("teams/compare/", views.compare_view, name="compare"),
    path("teams/<int:pk>/", views.team_detail, name="team_detail"),
    path("standings/", views.standings_view, name="standings"),
]