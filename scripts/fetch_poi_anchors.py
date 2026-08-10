#!/usr/bin/env python3
"""Refresh fixtures/poi_anchors.json from Overpass (OSM)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "fixtures" / "poi_anchors.json"
PREMIUM_NAMES = (
    "whole foods", "trader joe", "sprouts", "new seasons", "bi-rite",
    "rainbow grocery", "good eggs",
)


def main() -> int:
    s, w, n, e = 37.70, -122.55, 37.98, -122.35
    query = f"""
[out:json][timeout:90];
(
  node["shop"="supermarket"]({s},{w},{n},{e});
  node["shop"="convenience"]({s},{w},{n},{e});
  node["shop"="greengrocer"]({s},{w},{n},{e});
  node["amenity"="marketplace"]({s},{w},{n},{e});
  node["amenity"="bar"]({s},{w},{n},{e});
  node["amenity"="pub"]({s},{w},{n},{e});
  way["shop"="supermarket"]({s},{w},{n},{e});
  way["amenity"="marketplace"]({s},{w},{n},{e});
);
out center tags;
"""
    data = None
    for url in (
        "https://overpass-api.de/api/interpreter",
        "https://overpass.kumi.systems/api/interpreter",
    ):
        try:
            r = httpx.post(url, data={"data": query}, timeout=120.0)
            r.raise_for_status()
            data = r.json()
            break
        except Exception as ex:
            print(url, ex, file=sys.stderr)
    if not data:
        return 1
    out = {"grocery": [], "premium_grocery": [], "bar": [], "farmers_market": []}
    seen = set()
    for el in data["elements"]:
        tags = el.get("tags") or {}
        if "lat" in el:
            lat, lng = el["lat"], el["lon"]
        else:
            c = el.get("center") or {}
            lat, lng = c.get("lat"), c.get("lon")
        if lat is None or lng is None:
            continue
        name = (tags.get("name") or "").strip()
        key = (round(lat, 5), round(lng, 5), name.lower())
        if key in seen:
            continue
        seen.add(key)
        rec = {"name": name or "unnamed", "lat": lat, "lng": lng}
        amenity = tags.get("amenity")
        shop = tags.get("shop")
        if amenity == "marketplace":
            out["farmers_market"].append(rec)
        elif amenity in ("bar", "pub"):
            out["bar"].append(rec)
        elif shop in ("supermarket", "convenience", "greengrocer"):
            if any(p in name.lower() for p in PREMIUM_NAMES):
                out["premium_grocery"].append(rec)
            else:
                out["grocery"].append(rec)
    OUT.write_text(json.dumps(out, indent=2))
    print({k: len(v) for k, v in out.items()}, "->", OUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
