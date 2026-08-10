from __future__ import annotations

import json
import math
from pathlib import Path

from ..walk import (
    BAKERIES,
    BEACHES,
    SF_CENTER,
    TRAILS,
    _haversine_km,
    _haversine_minutes,
    is_marin,
)

PACKAGE_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_POI_PATH = PACKAGE_ROOT / "fixtures" / "poi_anchors.json"

_WALK_SPEED_KMH = 4.5
_GRID_FACTOR = 1.30


def load_poi_anchors(path: Path | None = None) -> dict[str, list[dict]]:
    p = path or DEFAULT_POI_PATH
    with open(p) as f:
        return json.load(f)


def nearest_minutes(lat: float, lng: float, anchors: list[dict]) -> int | None:
    if not anchors:
        return None
    best = None
    # coarse prefilter: keep anchors within ~0.25 deg (~25km) for speed
    for a in anchors:
        if abs(a["lat"] - lat) > 0.25 or abs(a["lng"] - lng) > 0.25:
            continue
        km = _haversine_km(lat, lng, a["lat"], a["lng"]) * _GRID_FACTOR
        mins = max(1, round((km / _WALK_SPEED_KMH) * 60))
        if best is None or mins < best:
            best = mins
    if best is not None:
        return best
    for a in anchors:
        km = _haversine_km(lat, lng, a["lat"], a["lng"]) * _GRID_FACTOR
        mins = max(1, round((km / _WALK_SPEED_KMH) * 60))
        if best is None or mins < best:
            best = mins
    return best


def nearest_named(lat: float, lng: float, anchors: list[dict]) -> tuple[int | None, str | None]:
    if not anchors:
        return None, None
    best_m, best_n = None, None
    for a in anchors:
        km = _haversine_km(lat, lng, a["lat"], a["lng"]) * _GRID_FACTOR
        mins = max(1, round((km / _WALK_SPEED_KMH) * 60))
        if best_m is None or mins < best_m:
            best_m, best_n = mins, a.get("name")
    return best_m, best_n


def curated_nearest(lat: float, lng: float, anchors) -> int | None:
    if not anchors:
        return None
    return min(_haversine_minutes(lat, lng, a) for a in anchors)


def route_bundle(lat: float, lng: float, poi: dict[str, list[dict]] | None = None) -> dict:
    poi = poi or load_poi_anchors()

    class _L:
        pass

    L = _L()
    L.lat = lat
    L.lng = lng
    return {
        "trail": curated_nearest(lat, lng, TRAILS),
        "beach": curated_nearest(lat, lng, BEACHES),
        "bakery": curated_nearest(lat, lng, BAKERIES),
        "ferry": curated_nearest(lat, lng, SF_CENTER),
        "grocery": nearest_minutes(lat, lng, poi.get("grocery", [])),
        "premium_grocery": nearest_minutes(lat, lng, poi.get("premium_grocery", [])),
        "bar": nearest_minutes(lat, lng, poi.get("bar", [])),
        "farmers_market": nearest_minutes(lat, lng, poi.get("farmers_market", [])),
        "mode": "drive" if is_marin(L) else "walk",
    }
