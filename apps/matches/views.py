from django.contrib import messages
from django.contrib.auth.decorators import login_required
from django.core.paginator import Paginator
from django.http import Http404
from django.shortcuts import get_object_or_404, redirect, render

from apps.matches import services
from apps.matches.forms import PredictionForm
from apps.matches.models import Prediction


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