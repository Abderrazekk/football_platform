from django.shortcuts import render

from apps.season import services


def _ctx(payload: dict | None, kind: str) -> dict:
    return {
        "payload": payload,
        "kind": kind,
        "available": payload is not None,
    }


def best_player(request):
    data = services.get_forecast("best_player")
    return render(request, "season/best_player.html", _ctx(data, "best_player"))


def best_young(request):
    data = services.get_forecast("best_young")
    return render(request, "season/best_young.html", _ctx(data, "best_young"))


def best_xi(request):
    data = services.get_forecast("best_xi")
    return render(request, "season/best_xi.html", _ctx(data, "best_xi"))