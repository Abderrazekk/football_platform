import json
from pathlib import Path

from django.http import FileResponse, Http404
from django.shortcuts import render

from apps.matches.models import Match, Prediction

ARTIFACTS = Path(__file__).resolve().parent.parent.parent / "ml" / "artifacts"


def _load_json(name: str):
    path = ARTIFACTS / name
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def home(request):
    latest_predictions = []
    if request.user.is_authenticated:
        latest_predictions = Prediction.objects.filter(
            user=request.user
        ).select_related("home_team", "away_team")[:5]

    recent_matches = Match.objects.filter(
        is_played=True
    ).select_related("home_team", "away_team").order_by("-date")[:5]

    # Dashboard indicators
    total_matches = Match.objects.filter(is_played=True).count()
    model_accuracy = None
    pred_qs = Prediction.objects.exclude(actual_result="")
    if pred_qs.exists():
        correct = sum(1 for p in pred_qs if p.predicted_result == p.actual_result)
        model_accuracy = round(100.0 * correct / pred_qs.count(), 1)

    user_stats = None
    if request.user.is_authenticated:
        user_finished = Prediction.objects.filter(
            user=request.user
        ).exclude(actual_result="")
        if user_finished.exists():
            user_correct = sum(1 for p in user_finished if p.predicted_result == p.actual_result)
            user_stats = {
                "total": user_finished.count(),
                "correct": user_correct,
                "accuracy": round(100.0 * user_correct / user_finished.count(), 1),
            }

    return render(
        request,
        "core/home.html",
        {
            "latest_predictions": latest_predictions,
            "recent_matches": recent_matches,
            "total_matches": total_matches,
            "model_accuracy": model_accuracy,
            "user_stats": user_stats,
        },
    )


def model_performance(request):
    metrics = _load_json("match_metrics.json")
    importance = _load_json("match_feature_importance.json")
    calibration = _load_json("match_calibration.json")

    # Flatten feature importance top-15
    top_features = (importance or {}).get("top_20", [])[:15]

    return render(
        request,
        "core/model_performance.html",
        {
            "metrics": metrics,
            "top_features": top_features,
            "calibration": calibration,
            "has_calibration_png": (ARTIFACTS / "match_calibration_curve.png").exists(),
        },
    )


def calibration_curve_png(request):
    path = ARTIFACTS / "match_calibration_curve.png"
    if not path.exists():
        raise Http404("Calibration curve not generated")
    return FileResponse(open(path, "rb"), content_type="image/png")