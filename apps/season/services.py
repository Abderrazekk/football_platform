"""
Season services — load pre-computed predictions from JSON, cache in DB.

Important: Python's json.loads keeps NaN/Infinity, json.dumps writes them
back as the literal 'NaN' — which is INVALID JSON per RFC 8259. SQLite's
JSONField CHECK constraint rejects it. We sanitize (NaN/Inf → None) before
DB write.
"""

from __future__ import annotations

import json
import logging
import math
from pathlib import Path

from django.conf import settings

from apps.season.models import SeasonForecast

log = logging.getLogger(__name__)
ARTIFACTS = Path(settings.BASE_DIR) / "ml" / "artifacts"


def _sanitize_json(obj):
    """Recursively replace NaN/Infinity with None (JSON-valid)."""
    if isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None
        return obj
    if isinstance(obj, dict):
        return {k: _sanitize_json(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_sanitize_json(v) for v in obj]
    return obj


def _load_predictions_file() -> dict | None:
    path = ARTIFACTS / "season_predictions_2026_27.json"
    if not path.exists():
        log.warning("Season predictions file missing: %s", path)
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as e:
        log.error("Failed to parse season predictions: %s", e)
        return None


def get_forecast(kind: str, force: bool = False) -> dict | None:
    """Return cached payload for (season=2026-27, kind). Compute from JSON if needed."""
    season = "2026-27"

    preds = _load_predictions_file()
    if preds is None:
        return None

    model_version = preds.get("model_version", "unknown")

    if not force:
        cached = SeasonForecast.objects.filter(season=season, kind=kind).first()
        if cached and cached.model_version == model_version:
            return cached.payload

    payload = preds.get(kind)
    if payload is None:
        return None

    # Sanitize NaN/Infinity → None (required by SQLite JSONField CHECK)
    payload = _sanitize_json(payload)

    SeasonForecast.objects.update_or_create(
        season=season,
        kind=kind,
        defaults={"payload": payload, "model_version": model_version},
    )
    return payload


def refresh_all() -> dict[str, bool]:
    """Recompute all season forecasts from the JSON file."""
    out = {}
    for kind in ("best_player", "best_young", "best_xi"):
        out[kind] = get_forecast(kind, force=True) is not None
    return out