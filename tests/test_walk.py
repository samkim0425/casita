import sqlite3
from pathlib import Path
from types import SimpleNamespace

from casita import walk
from casita.mash import poi


def test_routes_api_disabled_without_key(monkeypatch):
    monkeypatch.delenv("GOOGLE_MAPS_API_KEY", raising=False)
    monkeypatch.delenv("CASITA_ROUTES_OFFLINE", raising=False)

    assert walk._routes_api_enabled() is False


def test_routes_api_disabled_when_offline(monkeypatch):
    monkeypatch.setenv("GOOGLE_MAPS_API_KEY", "fake-key")
    monkeypatch.setenv("CASITA_ROUTES_OFFLINE", "1")

    assert walk._routes_api_enabled() is False


def test_routes_api_enabled_with_key(monkeypatch):
    monkeypatch.setenv("GOOGLE_MAPS_API_KEY", "fake-key")
    monkeypatch.delenv("CASITA_ROUTES_OFFLINE", raising=False)

    assert walk._routes_api_enabled() is True


def test_ensure_cache_migrates_mode_into_primary_key(tmp_path):
    db_path = tmp_path / "routes.sqlite"
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            """CREATE TABLE walk_cache (
                from_lat REAL, from_lng REAL,
                to_lat REAL, to_lng REAL,
                mode TEXT NOT NULL DEFAULT 'walk',
                minutes INTEGER NOT NULL,
                source TEXT NOT NULL,
                ts TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (from_lat, from_lng, to_lat, to_lng)
            )"""
        )
        conn.execute(
            "INSERT INTO walk_cache "
            "(from_lat, from_lng, to_lat, to_lng, mode, minutes, source) "
            "VALUES (1, 2, 3, 4, 'walk', 10, 'api')"
        )
        walk._ensure_cache(conn)

        pk_cols = {row[1] for row in conn.execute("PRAGMA table_info(walk_cache)") if row[5]}
        assert "mode" in pk_cols

        conn.execute(
            "INSERT INTO walk_cache "
            "(from_lat, from_lng, to_lat, to_lng, mode, minutes, source) "
            "VALUES (1, 2, 3, 4, 'drive', 5, 'api')"
        )
        count = conn.execute(
            "SELECT COUNT(*) FROM walk_cache WHERE from_lat=1 AND from_lng=2"
        ).fetchone()[0]
        assert count == 2


def _reset_route_cache(monkeypatch, db_path: Path) -> None:
    monkeypatch.setenv("CASITA_ROUTE_CACHE_DB", str(db_path))
    walk._cache_connection = None
    walk._cache_connection_path = None


def test_populate_for_offline_never_hits_httpx_and_uses_haversine(tmp_path, monkeypatch):
    """CASITA_ROUTES_OFFLINE=1 must fill missing pairs from haversine, not Maps."""
    _reset_route_cache(monkeypatch, tmp_path / "routes.sqlite")
    monkeypatch.setenv("CASITA_ROUTES_OFFLINE", "1")
    monkeypatch.setenv("GOOGLE_MAPS_API_KEY", "fake-key-must-not-be-used")

    def boom(*_a, **_k):
        raise AssertionError("httpx.post must not run when routes are offline")

    monkeypatch.setattr(walk.httpx, "post", boom)

    listing = SimpleNamespace(key="sf:1", lat=37.7800, lng=-122.4500)
    result = walk.populate_for([listing])

    anchors = walk.BEACHES + walk.BAKERIES + walk.PRESIDIO_GATES
    assert len(result) == len(anchors)
    for a in anchors:
        assert result[(listing.key, a.name)] >= 1

    sources = {
        row[0]
        for row in walk._cache_conn().execute("SELECT DISTINCT source FROM walk_cache")
    }
    assert sources == {"haversine"}


def test_populate_for_offline_prefers_cached_minutes(tmp_path, monkeypatch):
    """Fully cached origins short-circuit — no Routes API helper call at all."""
    _reset_route_cache(monkeypatch, tmp_path / "routes.sqlite")
    monkeypatch.setenv("CASITA_ROUTES_OFFLINE", "1")
    monkeypatch.setenv("GOOGLE_MAPS_API_KEY", "fake-key")

    listing = SimpleNamespace(key="sf:2", lat=37.7811, lng=-122.4511)
    anchors = walk.BEACHES + walk.BAKERIES + walk.PRESIDIO_GATES
    for i, a in enumerate(anchors):
        walk._cache_put(listing.lat, listing.lng, a.lat, a.lng, 20 + i, "api")

    calls: list[tuple] = []

    def spy(origins, destinations, *, mode="walk"):
        calls.append((mode, len(origins), len(destinations)))
        return [[None] * len(destinations) for _ in origins]

    monkeypatch.setattr(walk, "_call_routes_api", spy)

    result = walk.populate_for([listing])
    assert calls == []
    assert result[(listing.key, anchors[0].name)] == 20
    assert result[(listing.key, anchors[1].name)] == 21


def test_populate_drive_for_marin_offline_never_hits_httpx(tmp_path, monkeypatch):
    _reset_route_cache(monkeypatch, tmp_path / "routes.sqlite")
    monkeypatch.setenv("CASITA_ROUTES_OFFLINE", "1")
    monkeypatch.setenv("GOOGLE_MAPS_API_KEY", "fake-key")

    def boom(*_a, **_k):
        raise AssertionError("httpx.post must not run when routes are offline")

    monkeypatch.setattr(walk.httpx, "post", boom)

    listing = SimpleNamespace(key="marin:1", lat=37.9060, lng=-122.5450)  # Mill Valley-ish
    assert walk.is_marin(listing)
    result = walk.populate_drive_for_marin([listing])
    assert result
    sources = {
        row[0]
        for row in walk._cache_conn().execute(
            "SELECT DISTINCT source FROM walk_cache WHERE mode='drive'"
        )
    }
    assert sources == {"haversine"}


def test_mash_poi_route_bundle_uses_committed_anchors_only():
    """Mash grocery/bar/etc. minutes come from fixtures/poi_anchors.json + haversine."""
    path = poi.DEFAULT_POI_PATH
    assert path.is_file()
    data = poi.load_poi_anchors(path)
    for key in ("grocery", "premium_grocery", "bar", "farmers_market"):
        assert key in data and len(data[key]) >= 1

    bundle = poi.route_bundle(37.7800, -122.4500, data)
    assert bundle["mode"] == "walk"
    assert bundle["grocery"] is not None and bundle["grocery"] >= 1
    assert bundle["trail"] is not None and bundle["trail"] >= 1
    # Haversine-only path: values are stable for fixed anchors + coords.
    again = poi.route_bundle(37.7800, -122.4500, data)
    assert again == bundle
