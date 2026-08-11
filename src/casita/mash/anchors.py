"""Inspect curated walk anchors + mash OSM POI anchors (no Maps calls)."""
from __future__ import annotations

from . import poi
from .. import walk


def collect_anchor_groups() -> list[dict]:
    """Return display groups: curated walk.py sets + committed poi_anchors.json."""
    groups: list[dict] = [
        {
            "id": "beaches",
            "title": "Beaches (walk.py)",
            "source": "src/casita/walk.py · BEACHES",
            "kind": "curated",
            "items": [
                {"name": a.name, "short": a.short, "lat": a.lat, "lng": a.lng}
                for a in walk.BEACHES
            ],
        },
        {
            "id": "bakeries",
            "title": "Bakeries (walk.py)",
            "source": "src/casita/walk.py · BAKERIES",
            "kind": "curated",
            "items": [
                {"name": a.name, "short": a.short, "lat": a.lat, "lng": a.lng}
                for a in walk.BAKERIES
            ],
        },
        {
            "id": "trails",
            "title": "Trails (walk.py)",
            "source": "src/casita/walk.py · TRAILS",
            "kind": "curated",
            "items": [
                {"name": a.name, "short": a.short, "lat": a.lat, "lng": a.lng}
                for a in walk.TRAILS
            ],
        },
        {
            "id": "ferry",
            "title": "SF center / Ferry (walk.py)",
            "source": "src/casita/walk.py · SF_CENTER",
            "kind": "curated",
            "items": [
                {"name": a.name, "short": a.short, "lat": a.lat, "lng": a.lng}
                for a in walk.SF_CENTER
            ],
        },
    ]

    poi_data = poi.load_poi_anchors()
    labels = {
        "grocery": "Grocery (OSM / poi_anchors.json)",
        "premium_grocery": "Premium grocery (OSM / poi_anchors.json)",
        "bar": "Bars (OSM / poi_anchors.json)",
        "farmers_market": "Farmers markets (OSM / poi_anchors.json)",
    }
    for key, title in labels.items():
        raw = poi_data.get(key) or []
        groups.append({
            "id": key,
            "title": title,
            "source": str(poi.DEFAULT_POI_PATH.relative_to(poi.PACKAGE_ROOT)),
            "kind": "poi",
            "items": [
                {
                    "name": row.get("name") or "(unnamed)",
                    "short": row.get("name") or "",
                    "lat": row.get("lat"),
                    "lng": row.get("lng"),
                }
                for row in raw
            ],
        })
    return groups


def format_anchors_text(groups: list[dict] | None = None) -> str:
    groups = groups or collect_anchor_groups()
    lines = [
        "Casita route anchors (no Maps calls)",
        "Curated walk.py sets + committed mash POI file.",
        "",
    ]
    for g in groups:
        lines.append(f"## {g['title']}  ({len(g['items'])})")
        lines.append(f"   source: {g['source']}")
        for item in g["items"]:
            lat, lng = item.get("lat"), item.get("lng")
            coord = f"{lat:.4f}, {lng:.4f}" if lat is not None and lng is not None else "—"
            lines.append(f"  · {item['name']}  ({coord})")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"
