from django.contrib import messages
from django.contrib.auth.decorators import login_required
from django.core.paginator import Paginator
from django.shortcuts import get_object_or_404, render

from apps.matches.models import Team
from apps.players import services
from apps.players.models import Player

VALID_POSITIONS = ("GK", "DEF", "MID", "FWD")


# ---------------------------------------------------------------------------
# Public — list + detail (§F-J1)
# ---------------------------------------------------------------------------
def players_list(request):
    query = request.GET.get("q", "").strip()
    position = request.GET.get("position", "").strip()

    qs = Player.objects.select_related("current_team").all()
    if query:
        qs = qs.filter(name__icontains=query)
    if position in VALID_POSITIONS:
        qs = qs.filter(position=position)
    qs = qs.order_by("name")

    paginator = Paginator(qs, 50)
    page_obj = paginator.get_page(request.GET.get("page", 1))

    return render(
        request,
        "players/list.html",
        {
            "page_obj": page_obj,
            "query": query,
            "position": position,
            "positions": VALID_POSITIONS,
        },
    )


def player_detail(request, pk):
    player = get_object_or_404(Player, pk=pk)
    season_stats = services.player_season_stats(player)
    recent = services.player_recent_gameweeks(player, n=15)

    # Optional prediction block
    prediction = None
    opponent = request.GET.get("opponent", "").strip()
    was_home = request.GET.get("was_home", "1") == "1"
    if opponent and player.element_code:
        try:
            prediction = services.predict_player_vs(player, opponent, was_home)
        except Exception as e:
            messages.error(request, f"Prédiction indisponible : {e}")

    teams = Team.objects.order_by("name")

    return render(
        request,
        "players/detail.html",
        {
            "player": player,
            "season_stats": season_stats,
            "recent": recent,
            "prediction": prediction,
            "opponent": opponent,
            "was_home": was_home,
            "teams": teams,
        },
    )


# ---------------------------------------------------------------------------
# Connected — compare (§F-J2)
# ---------------------------------------------------------------------------
@login_required
def compare_view(request):
    a_pk = request.GET.get("a", "").strip()
    b_pk = request.GET.get("b", "").strip()
    opponent = request.GET.get("opponent", "").strip()
    was_home = request.GET.get("was_home", "1") == "1"

    player_a = Player.objects.filter(pk=a_pk).first() if a_pk.isdigit() else None
    player_b = Player.objects.filter(pk=b_pk).first() if b_pk.isdigit() else None

    data = None
    if player_a and player_b and player_a.pk != player_b.pk:
        data = services.compare_players_data(
            player_a, player_b,
            opponent=opponent or None,
            was_home=was_home,
        )
    elif player_a and player_b and player_a.pk == player_b.pk:
        messages.warning(request, "Sélectionnez deux joueurs différents.")

    return render(
        request,
        "players/compare.html",
        {
            "player_a": player_a,
            "player_b": player_b,
            "opponent": opponent,
            "was_home": was_home,
            "data": data,
            "teams": Team.objects.order_by("name"),
            "players": Player.objects.order_by("name"),
        },
    )


# ---------------------------------------------------------------------------
# Connected — match stars (§F-J3)
# ---------------------------------------------------------------------------
@login_required
def match_stars(request):
    home = request.GET.get("home", "").strip()
    away = request.GET.get("away", "").strip()

    ranking = None
    if home and away:
        if home == away:
            messages.warning(request, "Les deux équipes doivent être différentes.")
        else:
            ranking = services.rank_players_for_match(home, away, top_n=10)
            if ranking is None:
                messages.error(
                    request,
                    "Modèle de prédiction indisponible. Contactez l'administrateur.",
                )

    return render(
        request,
        "players/match_stars.html",
        {
            "home": home,
            "away": away,
            "ranking": ranking,
            "teams": Team.objects.order_by("name"),
        },
    )


# ---------------------------------------------------------------------------
# Connected — at-risk (§F-J4)
# ---------------------------------------------------------------------------
@login_required
def at_risk(request):
    home = request.GET.get("home", "").strip()
    away = request.GET.get("away", "").strip()

    ranking = None
    if home and away:
        if home == away:
            messages.warning(request, "Les deux équipes doivent être différentes.")
        else:
            ranking = services.rank_players_for_match(home, away, top_n=10)
            if ranking is None:
                messages.error(
                    request,
                    "Modèle de prédiction indisponible. Contactez l'administrateur.",
                )

    return render(
        request,
        "players/at_risk.html",
        {
            "home": home,
            "away": away,
            "ranking": ranking,
            "teams": Team.objects.order_by("name"),
        },
    )