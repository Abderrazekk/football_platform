from django.contrib import messages
from django.contrib.auth.decorators import login_required
from django.core.paginator import Paginator
from django.http import Http404
from django.shortcuts import get_object_or_404, redirect, render

from apps.matches import services
from apps.matches.forms import PredictionForm
from apps.matches.models import Prediction


from django.shortcuts import get_object_or_404
from apps.matches.models import Season, Team
from apps.matches import services



@login_required
def predict_view(request):
    context = {"form": PredictionForm()}

    if request.method == "POST":
        form = PredictionForm(request.POST)
        context["form"] = form

        if form.is_valid():
            home = form.cleaned_data["home_team"]
            away = form.cleaned_data["away_team"]
            try:
                probs = services.predict(home.name, away.name)
                prediction = services.save_prediction(
                    user=request.user, home_team=home, away_team=away, probs=probs
                )
                messages.success(request, "Prédiction enregistrée dans votre historique.")
                context.update({
                    "result": probs,
                    "prediction": prediction,
                    "home_team": home,
                    "away_team": away,
                    "home_form": services.recent_form(home.name, 5),
                    "away_form": services.recent_form(away.name, 5),
                    "h2h": services.h2h_summary(home.name, away.name),
                })
            except FileNotFoundError:
                messages.error(
                    request,
                    "Modèle de prédiction indisponible. Contactez l'administrateur.",
                )
            except Exception as e:
                messages.error(request, f"Erreur de prédiction : {e}")

    return render(request, "matches/predict.html", context)


@login_required
def history_view(request):
    qs = Prediction.objects.filter(user=request.user).select_related(
        "home_team", "away_team"
    )
    paginator = Paginator(qs, 20)
    page_obj = paginator.get_page(request.GET.get("page", 1))
    return render(
        request,
        "matches/history.html",
        {"page_obj": page_obj, "stats": services.personal_stats(request.user)},
    )


@login_required
def prediction_detail_view(request, pk: int):
    prediction = get_object_or_404(Prediction, pk=pk, user=request.user)
    return render(request, "matches/detail.html", {"prediction": prediction})


@login_required
def delete_prediction_view(request, pk: int):
    prediction = get_object_or_404(Prediction, pk=pk, user=request.user)
    if request.method == "POST":
        prediction.delete()
        messages.success(request, "Prédiction supprimée.")
        return redirect("matches:history")
    return render(request, "matches/delete_confirm.html", {"prediction": prediction})






#============================== Phase 6 ========================================



def teams_list(request):
    current = (
        Season.objects.filter(is_current=True).first()
        or Season.objects.order_by("-label").first()
    )
    standings = services.get_standings(current.label) if current else []
    position_by_team = {row["team"].id: row["position"] for row in standings}

    teams = Team.objects.order_by("name")
    team_data = [
        {"team": t, "position": position_by_team.get(t.id)}
        for t in teams
    ]
    return render(
        request,
        "matches/teams_list.html",
        {"team_data": team_data, "season": current},
    )


def team_detail(request, pk: int):
    team = get_object_or_404(Team, pk=pk)
    history = services.team_season_history(team, n=5)
    elo_series = services.elo_history_from_features(team.name, n=20)
    form = services.recent_form(team.name, n=10)
    return render(
        request,
        "matches/team_detail.html",
        {
            "team": team,
            "history": history,
            "elo_series": elo_series,
            "form": form,
        },
    )


def standings_view(request):
    seasons = Season.objects.order_by("label")
    season_label = request.GET.get("season")
    if not season_label:
        current = Season.objects.filter(is_current=True).first() or seasons.last()
        season_label = current.label if current else None

    standings = services.get_standings(season_label) if season_label else []
    return render(
        request,
        "matches/standings.html",
        {
            "seasons": seasons,
            "selected_season": season_label,
            "standings": standings,
        },
    )


def compare_view(request):
    a = request.GET.get("a", "")
    b = request.GET.get("b", "")
    data = services.compare_teams_data(a, b) if a and b and a != b else None
    teams = Team.objects.order_by("name")

    # Normalized 0-100 for radar chart
    radar = None
    if data:
        def norm(s):
            return {
                "attack": min(100, (s["attack"] / 3) * 100),
                "defense": max(0, min(100, ((3 - s["defense"]) / 3) * 100)),
                "form": min(100, (s["form"] / 3) * 100),
                "elo": max(0, min(100, ((s["elo"] - 1300) / 500) * 100)),
                "home_wins": s["home_win_rate"] * 100,
            }
        radar = {"team_a": norm(data["team_a"]), "team_b": norm(data["team_b"])}

    return render(
        request,
        "matches/compare.html",
        {"teams": teams, "a": a, "b": b, "data": data, "radar": radar},
    )