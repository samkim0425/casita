from __future__ import annotations

import json
from dataclasses import dataclass, field

import numpy as np

from ..models import Listing
from . import poi as poi_mod

ROUTE_FEATURES = [
    "trail", "beach", "bakery", "grocery", "premium_grocery", "bar", "farmers_market", "ferry",
]
NUMERIC_FEATURES = [
    "price", "price_per_bed", "price_per_sqft", "beds", "baths", "sqft",
] + ROUTE_FEATURES

TRI_STATE = ["outdoor", "laundry", "parking", "dogs", "light", "condition", "view"]
BINARY = ["is_sf"]

FOOD_GROUP = ["grocery", "premium_grocery", "farmers_market", "bakery"]
NATURE_GROUP = ["trail", "beach"]

HINGE_FEATURES = ["price", "price_per_bed"] + ROUTE_FEATURES

PICKABLE_FEATURES = [
    "trail", "beach", "bakery", "grocery", "premium_grocery", "bar", "farmers_market", "ferry",
    "outdoor", "laundry", "parking", "dogs", "light", "condition", "view",
    "is_sf",
]

# Always selected in ranking — can reorder, cannot remove.
RANKABLE_LOCKED = ["price", "price_per_bed", "price_per_sqft"]

# Structural card rows — always first, in this order (beds+baths render as one row).
CARD_FIXED = ["price", "beds", "baths", "price_per_bed", "sqft", "price_per_sqft"]


def base_listing_key(key: str) -> str:
    """Map hypA:/hypB:/hyp: keys back to the underlying listing key."""
    if key.startswith("hypA:") or key.startswith("hypB:"):
        return key.split(":", 1)[1]
    if key.startswith("hyp:"):
        rest = key[4:]
        cut = rest.rfind(":{")
        if cut >= 0:
            return rest[:cut]
        return rest
    return key


def normalize_feature_order(order: list[str] | None) -> list[str]:
    """Keep pickable + locked features; ensure locked metrics stay present."""
    allowed = set(PICKABLE_FEATURES) | set(RANKABLE_LOCKED)
    out = [f for f in (order or []) if f in allowed]
    for f in RANKABLE_LOCKED:
        if f not in out:
            out.append(f)
    return out


def card_feature_order(feature_order: list[str]) -> list[str]:
    """Card rows: fixed rent/size block first, then the reviewer's ranked features."""
    ranked = normalize_feature_order(feature_order)
    show: list[str] = []
    for f in CARD_FIXED:
        if f not in show:
            show.append(f)
    for f in ranked:
        if f not in show:
            show.append(f)
    return show


ALWAYS_SHOW = ["price", "beds", "baths", "price_per_bed", "sqft", "price_per_sqft"]

ALWAYS_SHOW_COPY = (
    "Add features you care about, then use ↑ / ↓ to rank them (order matters!). "
    "Total rent, $/bed, and $/sqft are always selected."
)

FEATURE_LABELS = {
    "price": "Total Rent",
    "price_per_bed": "$ / bed",
    "price_per_sqft": "$ / sqft",

    "trail": "Distance to trail",
    "beach": "Distance to beach",
    "bakery": "Distance to bakery",
    "grocery": "Distance to grocery",
    "premium_grocery": "Distance to premium grocery",
    "bar": "Distance to bar",
    "farmers_market": "Distance to farmers market",
    "ferry": "Distance to Ferry Building",
    "beds": "Beds",
    "baths": "Baths",
    "sqft": "Area (sq ft)",
    "outdoor": "Outdoor space",
    "laundry": "Laundry",
    "parking": "Parking",
    "dogs": "Dogs",
    "light": "Light",
    "condition": "Condition",
    "view": "View",
    "is_sf": "SF (vs Marin)",
}

LIGHT_MAP = {"abundant": 1.0, "moderate": 0.5, "dim": 0.0}
CONDITION_MAP = {"high-end": 1.0, "well-kept": 0.66, "dated": 0.33, "needs-work": 0.0, "needs work": 0.0}
VIEW_MAP = {"panoramic": 1.0, "open": 0.66, "blocked": 0.0, "ground-level": 0.33}
DOG_MAP = {"large_ok": 1.0, "dogs_ok": 0.66, "small_only": 0.33, "no_dogs": 0.0}


@dataclass
class ListingFeatures:
    key: str
    values: dict[str, float | None]
    known: dict[str, bool]
    routes: dict
    photo_count: int = 0
    source: str = ""
    cover_url: str | None = None
    photos: list[str] = field(default_factory=list)
    neighborhood: str | None = None
    address: str | None = None
    url: str = ""
    detail_path: str = ""
    is_hypothetical: bool = False
    hyp_note: str | None = None
    labels: dict[str, str] = field(default_factory=dict)


def _resolve_outdoor(L: Listing) -> tuple[float | None, bool]:
    if L.outdoor_visible:
        return 1.0, True
    if L.has_yard is not None:
        return 1.0 if L.has_yard else 0.0, True
    if L.yard_note:
        return 1.0, True
    return None, False


def _resolve_laundry(L: Listing) -> tuple[float | None, bool]:
    if not L.laundry:
        return None, False
    t = L.laundry.lower()
    if "in-unit" in t or "in unit" in t or "washer" in t:
        return 1.0, True
    if "hookup" in t:
        return 0.5, True
    if "none" in t or "no laundry" in t:
        return 0.0, True
    return 0.5, True


def _resolve_parking(L: Listing) -> tuple[float | None, bool]:
    if not L.parking:
        return None, False
    t = L.parking.lower()
    if "garage" in t or "included" in t or "deeded" in t:
        return 1.0, True
    if "street" in t or "none" in t or "no parking" in t:
        return 0.0, True
    return 0.5, True


def _resolve_dogs(L: Listing) -> tuple[float | None, bool]:
    if L.dog_policy and L.dog_policy in DOG_MAP:
        return DOG_MAP[L.dog_policy], True
    if L.pets_allowed is not None:
        return 1.0 if L.pets_allowed else 0.0, True
    return None, False


def _q(val: str | None, mapping: dict) -> tuple[float | None, bool]:
    if not val or str(val).lower() == "unknown":
        return None, False
    key = str(val).lower()
    if key in mapping:
        return mapping[key], True
    return None, False


def dedupe_photo_urls(urls: list[str] | None) -> list[str]:
    """Preserve order; drop empties and exact URL duplicates."""
    out: list[str] = []
    seen: set[str] = set()
    for raw in urls or []:
        u = (raw or "").strip()
        if not u or u in seen:
            continue
        seen.add(u)
        out.append(u)
    return out


def cover_photos(L: Listing, review: dict | None) -> tuple[str | None, list[str]]:
    photos = dedupe_photo_urls(list(L.photos or []))
    if L.image_url and L.image_url.strip() and L.image_url not in photos:
        photos = [L.image_url.strip()] + photos
    photos = dedupe_photo_urls(photos)
    if not review:
        return (photos[0] if photos else None), photos
    drops = set(review.get("drop_indices") or [])
    kept = [p for i, p in enumerate(photos) if i not in drops]
    if not kept:
        kept = list(photos)
    bi = review.get("best_photo_index")
    if bi is not None and photos and 0 <= bi < len(photos) and bi not in drops:
        best = photos[bi]
        kept = [best] + [p for p in kept if p != best]
    kept = dedupe_photo_urls(kept)
    cover = kept[0] if kept else None
    return cover, kept


def extract(
    L: Listing,
    *,
    poi_data: dict | None = None,
    photo_review: dict | None = None,
    detail_path: str = "",
) -> ListingFeatures:
    routes = {}
    if L.lat is not None and L.lng is not None:
        routes = poi_mod.route_bundle(L.lat, L.lng, poi_data)

    values: dict[str, float | None] = {}
    known: dict[str, bool] = {}

    values["price"] = float(L.price) if L.price is not None else None
    known["price"] = L.price is not None

    if L.price is not None and L.beds and L.beds > 0:
        values["price_per_bed"] = float(L.price) / float(L.beds)
        known["price_per_bed"] = True
    else:
        values["price_per_bed"] = None
        known["price_per_bed"] = False

    if L.price is not None and L.sqft and L.sqft > 0:
        values["price_per_sqft"] = float(L.price) / float(L.sqft)
        known["price_per_sqft"] = True
    else:
        values["price_per_sqft"] = None
        known["price_per_sqft"] = False

    for name, raw in [("beds", L.beds), ("baths", L.baths), ("sqft", L.sqft)]:
        values[name] = float(raw) if raw is not None else None
        known[name] = raw is not None

    for rname in ROUTE_FEATURES:
        v = routes.get(rname)
        values[rname] = float(v) if v is not None else None
        known[rname] = v is not None

    outdoor_v, outdoor_k = _resolve_outdoor(L)
    values["outdoor"], known["outdoor"] = outdoor_v, outdoor_k
    laundry_v, laundry_k = _resolve_laundry(L)
    values["laundry"], known["laundry"] = laundry_v, laundry_k
    parking_v, parking_k = _resolve_parking(L)
    values["parking"], known["parking"] = parking_v, parking_k
    dogs_v, dogs_k = _resolve_dogs(L)
    values["dogs"], known["dogs"] = dogs_v, dogs_k
    light_v, light_k = _q(L.light_quality, LIGHT_MAP)
    values["light"], known["light"] = light_v, light_k
    cond_v, cond_k = _q(L.condition_quality, CONDITION_MAP)
    values["condition"], known["condition"] = cond_v, cond_k
    view_v, view_k = _q(L.view_quality, VIEW_MAP)
    values["view"], known["view"] = view_v, view_k

    from ..walk import is_marin as _is_marin
    if L.lat is not None and L.lng is not None:
        values["is_sf"] = 0.0 if _is_marin(L) else 1.0
        known["is_sf"] = True
    else:
        values["is_sf"] = None
        known["is_sf"] = False

    cover, photos = cover_photos(L, photo_review)

    from .. import dogs as dogs_mod
    labels: dict[str, str] = {}
    if outdoor_k:
        labels["outdoor"] = (
            L.outdoor_visible or L.yard_note
            or ("yes" if outdoor_v and outdoor_v >= 0.5 else "no")
        )
    if laundry_k and L.laundry:
        labels["laundry"] = L.laundry
    if parking_k and L.parking:
        labels["parking"] = L.parking
    if dogs_k:
        if L.dog_policy and L.dog_policy in dogs_mod.LABELS:
            labels["dogs"] = dogs_mod.LABELS[L.dog_policy]
        elif L.pets_allowed is not None:
            labels["dogs"] = "pets ok" if L.pets_allowed else "no pets"
    if light_k and L.light_quality:
        labels["light"] = L.light_quality
    if cond_k and L.condition_quality:
        labels["condition"] = L.condition_quality
    if view_k and L.view_quality:
        labels["view"] = L.view_quality
    if known.get("is_sf"):
        labels["is_sf"] = "Marin" if values["is_sf"] < 0.5 else "SF"

    return ListingFeatures(
        key=L.key,
        values=values,
        known=known,
        routes=routes,
        photo_count=len(photos),
        source=L.source,
        cover_url=cover,
        photos=photos,
        neighborhood=L.hood,
        address=L.address,
        url=L.url or "",
        detail_path=detail_path,
        labels=labels,
    )


def eligible(feats: ListingFeatures) -> tuple[bool, list[str]]:
    reasons = []
    if not feats.known.get("is_sf"):
        reasons.append("no_coords")
    if not feats.known.get("price"):
        reasons.append("no_price")
    if feats.photo_count <= 0 and not feats.cover_url:
        reasons.append("no_photos")
    return (not reasons), reasons


def eligibility_report(items: list[ListingFeatures]) -> dict:
    no_coords = no_price = no_photos = 0
    eligible_keys = []
    for f in items:
        ok, reasons = eligible(f)
        if "no_coords" in reasons:
            no_coords += 1
        if "no_price" in reasons:
            no_price += 1
        if "no_photos" in reasons:
            no_photos += 1
        if ok:
            eligible_keys.append(f.key)
    return {
        "n": len(items),
        "excluded_no_coords": no_coords,
        "excluded_no_price": no_price,
        "excluded_no_photos": no_photos,
        "eligible": len(eligible_keys),
        "eligible_keys": eligible_keys,
    }


def build_design_names(active_features: list[str], knot_counts: dict[str, int] | None = None) -> list[str]:
    knot_counts = knot_counts or {}
    names: list[str] = []
    fit_set = set(ALWAYS_SHOW) | set(active_features)
    for name in NUMERIC_FEATURES:
        if name not in fit_set and name not in ALWAYS_SHOW:
            continue
        names.append(name)
        names.append(f"{name}__known")
        for k in range(knot_counts.get(name, 1 if name in HINGE_FEATURES else 0)):
            names.append(f"{name}__hinge{k}")
    for name in TRI_STATE + BINARY:
        if name not in fit_set:
            continue
        names.append(name)
        if name in TRI_STATE:
            names.append(f"{name}__known")
    names.append("food_min")
    names.append("nature_min")
    return names


def vectorize(
    feats: ListingFeatures,
    names: list[str],
    knots: dict[str, list[float]],
    means: dict[str, float] | None = None,
    scales: dict[str, float] | None = None,
) -> np.ndarray:
    means = means or {}
    scales = scales or {}
    raw: dict[str, float] = {}
    for name in NUMERIC_FEATURES:
        v = feats.values.get(name)
        k = bool(feats.known.get(name))
        raw[name] = float(v) if k and v is not None else 0.0
        raw[f"{name}__known"] = 1.0 if k else 0.0
        for i, knot in enumerate(knots.get(name, [])):
            raw[f"{name}__hinge{i}"] = max(0.0, raw[name] - knot) if k else 0.0
    for name in TRI_STATE:
        v = feats.values.get(name)
        k = bool(feats.known.get(name))
        raw[name] = float(v) if k and v is not None else 0.0
        raw[f"{name}__known"] = 1.0 if k else 0.0
    raw["is_sf"] = float(feats.values["is_sf"]) if feats.known.get("is_sf") else 0.0

    food_vals = [feats.values[n] for n in FOOD_GROUP if feats.known.get(n) and feats.values.get(n) is not None]
    nature_vals = [feats.values[n] for n in NATURE_GROUP if feats.known.get(n) and feats.values.get(n) is not None]
    raw["food_min"] = float(min(food_vals)) if food_vals else 0.0
    raw["nature_min"] = float(min(nature_vals)) if nature_vals else 0.0

    out = np.zeros(len(names), dtype=float)
    for i, name in enumerate(names):
        base = name.split("__")[0]
        val = raw.get(name, 0.0)
        if name.endswith("__known") or name.startswith("food_") or name.startswith("nature_"):
            out[i] = val
        elif means and base in means and scales.get(base, 1.0):
            if name == base or "__hinge" in name:
                mu = means.get(base, 0.0)
                sd = scales.get(base, 1.0) or 1.0
                if "__hinge" in name:
                    out[i] = val / sd
                else:
                    out[i] = (val - mu) / sd if feats.known.get(base) else 0.0
            else:
                out[i] = val
        else:
            out[i] = val
    return out


def clone_hypothetical(base: ListingFeatures, overrides: dict[str, float], note: str) -> ListingFeatures:
    values = dict(base.values)
    known = dict(base.known)
    for k, v in overrides.items():
        values[k] = float(v)
        known[k] = True
    if "price" in overrides:
        price = float(overrides["price"])
        beds = values.get("beds")
        sqft = values.get("sqft")
        if beds and beds > 0:
            values["price_per_bed"] = price / float(beds)
            known["price_per_bed"] = True
        if sqft and sqft > 0:
            values["price_per_sqft"] = price / float(sqft)
            known["price_per_sqft"] = True
    return ListingFeatures(
        key=f"hyp:{base.key}:{json.dumps(overrides, sort_keys=True)}",
        values=values,
        known=known,
        routes=dict(base.routes),
        photo_count=base.photo_count,
        source=base.source,
        cover_url=base.cover_url,
        photos=list(base.photos),
        neighborhood=base.neighborhood,
        address=base.address,
        url=base.url,
        detail_path=base.detail_path,
        is_hypothetical=True,
        hyp_note=note,
        labels=dict(base.labels),
    )
