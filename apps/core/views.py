from django.shortcuts import render

from apps.matches.models import Match, Prediction


def home(request):
    latest_predictions = []
    if request.user.is_authenticated:
        latest_predictions = Prediction.objects.filter(
            user=request.user
        ).select_related("home_team", "away_team")[:5]

    recent_matches = Match.objects.filter(
        is_played=True
    ).select_related("home_team", "away_team").order_by("-date")[:5]

    return render(
        request,
        "core/home.html",
        {
            "latest_predictions": latest_predictions,
            "recent_matches": recent_matches,
        },
    )