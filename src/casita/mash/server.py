from __future__ import annotations

import json
import os
import sqlite3
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from .. import listing_page, storage
from ..listing_page import listing_url
from . import bootstrap, db as mash_db, features, model, poi, select, ui

ROOT = Path(__file__).resolve().parents[3]


class MashState:
    def __init__(self, listings_db: Path, site_dir: Path | None = None):
        self.listings_db = listings_db
        self.site_dir = site_dir
        self.poi_data = poi.load_poi_anchors()
        self._feat_map: dict[str, features.ListingFeatures] | None = None
        self._listings = None
        self._top_history: dict[str, list[list[str]]] = {}

    def listings_conn(self):
        conn = sqlite3.connect(self.listings_db)
        conn.row_factory = sqlite3.Row
        return conn

    def load_features(self, force: bool = False) -> dict[str, features.ListingFeatures]:
        if self._feat_map is not None and not force:
            return self._feat_map
        with storage.connect() as conn:
            rows = conn.execute("SELECT * FROM listings WHERE active=1").fetchall()
            listings = [storage._row_to_listing(r) for r in rows]
            reviews = {
                r["key"]: json.loads(r["review_json"])
                for r in conn.execute("SELECT key, review_json FROM llm_photo_reviews")
            }
        fmap = {}
        for L in listings:
            fmap[L.key] = features.extract(
                L,
                poi_data=self.poi_data,
                photo_review=reviews.get(L.key),
                detail_path=listing_url(L),
            )
        self._feat_map = fmap
        self._listings = {L.key: L for L in listings}
        return fmap

    def eligibility(self) -> dict:
        return features.eligibility_report(list(self.load_features().values()))

    def ensure_bootstrap(self):
        with self.listings_conn() as lc, mash_db.connect() as mc:
            return bootstrap.bootstrap_from_fixture(lc, mc)

    def reviewer_features(self, name: str) -> list[str]:
        with mash_db.connect() as conn:
            row = mash_db.get_reviewer(conn, name)
            if not row:
                return []
            order = json.loads(row["feature_order_json"] or "[]")
            if not order:
                return []
            return features.normalize_feature_order(order)

    def fit_for(self, reviewer: str, *, force: bool = False):
        fmap = self.load_features()
        order = self.reviewer_features(reviewer)
        with mash_db.connect() as conn:
            rows = mash_db.comparisons_for(conn, reviewer)
            n = mash_db.comparison_count(conn, reviewer)
            cached = mash_db.load_fit(conn, reviewer)
            if (
                not force
                and cached
                and int(cached.get("n_comparisons") or -1) == n
                and cached.get("active_features") == order
            ):
                fit = model.FitResult.from_dict(cached)
                return fit, rows, fmap
            hyp_feats = {}
            for r in rows:
                if r["is_hypothetical"]:
                    if r["hyp_left_json"]:
                        d = json.loads(r["hyp_left_json"])
                        hyp_feats[r["left_key"]] = _feats_from_hyp(d, r["left_key"])
                    if r["hyp_right_json"]:
                        d = json.loads(r["hyp_right_json"])
                        hyp_feats[r["right_key"]] = _feats_from_hyp(d, r["right_key"])
            merged = {**fmap, **hyp_feats}
            fit = model.fit(rows, merged, order)
            if fit:
                payload = fit.to_dict()
                payload["n_comparisons"] = n
                payload["active_features"] = order
                mash_db.save_fit(conn, reviewer, payload)
            return fit, rows, merged

    def rankings(self, reviewer: str, fit: model.FitResult | None):
        fmap = self.load_features()
        shown_counts: dict[str, int] = {}
        with mash_db.connect() as conn:
            for r in mash_db.comparisons_for(conn, reviewer):
                bases = {
                    features.base_listing_key(r["left_key"]),
                    features.base_listing_key(r["right_key"]),
                }
                for base in bases:
                    shown_counts[base] = shown_counts.get(base, 0) + 1
        shown = set(shown_counts)
        rows = []
        scored = []
        for key, feats in fmap.items():
            if fit:
                feature_fit, leftover, sc = model.score_parts(fit, feats)
            else:
                feature_fit, leftover, sc = 0.0, 0.0, 0.0
            scored.append((sc, key, feats, feature_fit, leftover))
        scored.sort(reverse=True)
        top_score = scored[0][0] if scored else 0.0
        for sc, key, feats, feature_fit, leftover in scored:
            title = feats.address or feats.neighborhood or key
            be = None
            if fit and not feats.known.get("price") and scored:
                be = model.break_even_price(fit, feats, top_score)
            rows.append({
                "key": key,
                "title": title,
                "score": sc,
                "feature_fit": feature_fit,
                "leftover": leftover,
                "never_shown": key not in shown,
                "n_shown": shown_counts.get(key, 0),
                "detail": feats.detail_path,
                "url": feats.url,
                "source": feats.source,
                "cover_url": feats.cover_url,
                "break_even": be,
            })
        return rows


def _feats_from_hyp(d: dict, key: str) -> features.ListingFeatures:
    return features.ListingFeatures(
        key=key,
        values={k: float(v) if v is not None else None for k, v in (d.get("values") or {}).items()},
        known={k: bool(v) for k, v in (d.get("known") or {}).items()},
        routes={},
        photo_count=int(d.get("photo_count") or 0),
        source=d.get("source") or "",
        cover_url=None,
        photos=d.get("photos") or [],
        is_hypothetical=True,
    )


def make_handler(state: MashState):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt, *args):
            if os.environ.get("CASITA_HTTP_LOGS"):
                super().log_message(fmt, *args)

        def _send(self, code: int, body: bytes, content_type: str = "text/html; charset=utf-8"):
            self.send_response(code)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _json(self, code: int, obj: dict):
            self._send(code, json.dumps(obj).encode(), "application/json")

        def _read_json(self) -> dict:
            n = int(self.headers.get("Content-Length") or 0)
            raw = self.rfile.read(n) if n else b"{}"
            return json.loads(raw.decode() or "{}")

        def do_GET(self):
            u = urlparse(self.path)
            path = u.path.rstrip("/") or "/"
            if path == "/mash" or path == "/mash/":
                with mash_db.connect() as conn:
                    rows = conn.execute(
                        "SELECT r.name, COUNT(c.id) AS n FROM reviewers r "
                        "LEFT JOIN comparisons c ON c.reviewer=r.name AND c.winner IS NOT NULL AND c.skipped=0 "
                        "GROUP BY r.name"
                    ).fetchall()
                    existing = {r["name"]: int(r["n"]) for r in rows}
                return self._send(200, ui.landing(existing).encode())

            if path == "/mash/anchors":
                from . import anchors as mash_anchors
                return self._send(200, ui.anchors_page(mash_anchors.collect_anchor_groups()).encode())

            if path == "/mash/api/reviewer":
                qs = parse_qs(u.query)
                name = (qs.get("name") or [""])[0].strip()
                with mash_db.connect() as conn:
                    row = mash_db.get_reviewer(conn, name)
                    count = mash_db.comparison_count(conn, name) if row else 0
                return self._json(200, {"name": name, "exists": row is not None, "count": count})

            if path == "/mash/api/logout":
                self.send_response(302)
                self.send_header("Location", "/mash/")
                self.send_header(
                    "Set-Cookie",
                    "mash_reviewer=; Path=/mash; Max-Age=0",
                )
                self.end_headers()
                return

            if path == "/mash/features":
                reviewer = self._cookie_reviewer()
                if not reviewer:
                    return self._redirect("/mash/")
                with mash_db.connect() as conn:
                    n = mash_db.comparison_count(conn, reviewer)
                order = state.reviewer_features(reviewer)
                if order and n > 0:
                    return self._redirect("/mash/play")
                est = select.comparison_estimate(max(1, len(order)))
                return self._send(200, ui.features_page(reviewer, order, est).encode())

            if path == "/mash/play":
                reviewer = self._cookie_reviewer()
                if not reviewer:
                    return self._redirect("/mash/")
                order = state.reviewer_features(reviewer)
                if not order:
                    return self._redirect("/mash/features")
                state.ensure_bootstrap()
                fmap = state.load_features()
                fit, rows, merged = state.fit_for(reviewer)
                with mash_db.connect() as conn:
                    seen = mash_db.decided_pairs(conn, reviewer)
                    n = mash_db.comparison_count(conn, reviewer)
                    recent = list(conn.execute(
                        "SELECT is_hypothetical FROM comparisons WHERE reviewer=? ORDER BY id DESC LIMIT 5",
                        (reviewer,),
                    ))
                recent_hyp = sum(1 for r in recent if r["is_hypothetical"])
                coverage = {}
                for r in rows:
                    fs = json.loads(r["feature_set_json"] or "[]")
                    for f in fs:
                        coverage[f] = coverage.get(f, 0) + 1
                offer = select.select_pair(
                    list(fmap.values()), order, seen, fit,
                    n_comparisons=n, recent_hyp=recent_hyp, coverage=coverage,
                )
                if not offer:
                    return self._send(200, ui.page("Done", "<p>No more pairs.</p>", who=reviewer).encode())
                banner = None
                if fit:
                    ranks = state.rankings(reviewer, fit)
                    top = [r["key"] for r in ranks[:20]]
                    hist = state._top_history.setdefault(reviewer, [])
                    hist.append(top)
                    hist[:] = hist[-8:]
                    if len(hist) >= 4:
                        stables = [
                            model.top_stability(hist[i - 1], hist[i], 20)
                            for i in range(1, len(hist))
                        ]
                        if sum(1 for s in stables[-3:] if s >= 0.7) >= 2 and n >= 20:
                            banner = "Your top 20 is starting to converge. See results?"
                return self._send(200, ui.play_page(
                    reviewer, offer.left, offer.right, offer.why_line, n, banner, order,
                ).encode())

            if path == "/mash/results":
                reviewer = self._cookie_reviewer()
                if not reviewer:
                    return self._redirect("/mash/")
                qs = parse_qs(u.query)
                concluded = (qs.get("done") or [""])[0] in ("1", "true", "yes")
                fit, rows, _ = state.fit_for(reviewer)
                ranks = state.rankings(reviewer, fit)
                movers = model.feature_importance(fit) if fit else []
                n = len([r for r in rows if r["winner"] and not r["skipped"]])
                compared = [r for r in ranks if not r.get("never_shown")]
                unseen = [r for r in ranks if r.get("never_shown")]
                if concluded:
                    # Full ranking for the session wrap-up (shown + not shown).
                    combined = ranks[:40]
                    compared, unseen = combined, []
                else:
                    unseen = unseen[:25]
                return self._send(200, ui.results_page(
                    reviewer, compared, unseen, movers, n,
                    concluded=concluded,
                ).encode())

            # static site fallback for listing detail pages
            if self.site_dir and (path.startswith("/listing/") or path.startswith("/assets/") or path.startswith("/og/")):
                return self._serve_site(path)

            return self._send(404, b"not found")

        def do_POST(self):
            u = urlparse(self.path)
            path = u.path.rstrip("/")
            if path == "/mash/api/reviewer":
                data = self._read_json()
                name = (data.get("name") or "").strip()
                if not name:
                    return self._json(400, {"error": "name required"})
                with mash_db.connect() as conn:
                    mash_db.upsert_reviewer(conn, name)
                    count = mash_db.comparison_count(conn, name)
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Set-Cookie", f"mash_reviewer={name}; Path=/mash")
                body = json.dumps({"name": name, "count": count, "next": "/mash/features" if count == 0 or True else "/mash/play"}).encode()
                # always go features if no feature order
                with mash_db.connect() as conn:
                    row = mash_db.get_reviewer(conn, name)
                    order = json.loads(row["feature_order_json"] or "[]") if row else []
                nxt = "/mash/play" if order else "/mash/features"
                body = json.dumps({"name": name, "count": count, "next": nxt}).encode()
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return

            if path == "/mash/api/features":
                data = self._read_json()
                reviewer = data.get("reviewer") or self._cookie_reviewer()
                with mash_db.connect() as conn:
                    if mash_db.comparison_count(conn, reviewer) > 0:
                        return self._json(403, {"error": "features locked for this session"})
                order = features.normalize_feature_order(data.get("order") or [])
                with mash_db.connect() as conn:
                    mash_db.upsert_reviewer(conn, reviewer, order)
                return self._json(200, {"ok": True})

            if path == "/mash/api/compare":
                data = self._read_json()
                reviewer = data.get("reviewer") or self._cookie_reviewer()
                order = data.get("feature_order") or self.reviewer_features_safe(reviewer)
                left_meta = data.get("left_meta") or {}
                right_meta = data.get("right_meta") or {}
                with mash_db.connect() as conn:
                    mash_db.insert_comparison(conn, {
                        "reviewer": reviewer,
                        "left_key": data["left_key"],
                        "right_key": data["right_key"],
                        "winner": data.get("winner"),
                        "skipped": 1 if data.get("skipped") else 0,
                        "strategy": "play",
                        "why_line": None,
                        "feature_set_json": json.dumps(list(features.ALWAYS_SHOW) + order),
                        "is_hypothetical": 1 if data.get("is_hypothetical") else 0,
                        "hyp_left_json": json.dumps(data.get("hyp_left")) if data.get("hyp_left") else None,
                        "hyp_right_json": json.dumps(data.get("hyp_right")) if data.get("hyp_right") else None,
                        "weight": 1.0,
                        "tag": "direct",
                        "shown_at": data.get("shown_at"),
                        "decided_at": data.get("decided_at"),
                        "overlay_opened": 1 if data.get("overlay_opened") else 0,
                        "left_photo_count": left_meta.get("photo_count"),
                        "right_photo_count": right_meta.get("photo_count"),
                        "left_field_count": sum(1 for v in (left_meta.get("known") or {}).values() if v),
                        "right_field_count": sum(1 for v in (right_meta.get("known") or {}).values() if v),
                        "left_source": left_meta.get("source"),
                        "right_source": right_meta.get("source"),
                    })
                return self._json(200, {"ok": True})

            if path == "/mash/api/end":
                length = int(self.headers.get("Content-Length") or 0)
                raw = self.rfile.read(length).decode() if length else ""
                if self.headers.get("Content-Type", "").startswith("application/json"):
                    data = json.loads(raw or "{}")
                    reviewer = data.get("reviewer") or self._cookie_reviewer()
                else:
                    qs = parse_qs(raw)
                    reviewer = (qs.get("reviewer") or [self._cookie_reviewer()])[0]
                fit, _, _ = state.fit_for(reviewer)
                ranks = state.rankings(reviewer, fit)
                with mash_db.connect() as conn:
                    mash_db.end_session(conn, reviewer, [r["key"] for r in ranks[:20]])
                return self._redirect("/mash/results?done=1")

            return self._send(404, b"not found")

        def reviewer_features_safe(self, reviewer: str) -> list[str]:
            return state.reviewer_features(reviewer)

        def _cookie_reviewer(self) -> str | None:
            cookie = self.headers.get("Cookie") or ""
            for part in cookie.split(";"):
                part = part.strip()
                if part.startswith("mash_reviewer="):
                    return part.split("=", 1)[1].strip()
            return None

        def _redirect(self, loc: str):
            self.send_response(302)
            self.send_header("Location", loc)
            self.end_headers()

        def _serve_site(self, path: str):
            rel = path.lstrip("/")
            cand = self.site_dir / rel
            if cand.is_dir():
                cand = cand / "index.html"
            if not cand.exists() and not rel.endswith(".html"):
                cand = self.site_dir / f"{rel}.html"
            if not cand.exists():
                return self._send(404, b"missing")
            data = cand.read_bytes()
            ctype = "text/html; charset=utf-8"
            if cand.suffix == ".css":
                ctype = "text/css"
            elif cand.suffix in (".js",):
                ctype = "application/javascript"
            elif cand.suffix == ".svg":
                ctype = "image/svg+xml"
            elif cand.suffix == ".png":
                ctype = "image/png"
            return self._send(200, data, ctype)

    return Handler


def serve(listings_db: Path, host: str = "127.0.0.1", port: int = 8766, site_dir: Path | None = None):
    os.environ.setdefault("CASITA_DB_PATH", str(listings_db))
    os.environ.setdefault("CASITA_ROUTE_CACHE_DB", str(listings_db))
    os.environ.setdefault("CASITA_ROUTES_OFFLINE", "1")
    state = MashState(listings_db, site_dir=site_dir)
    state.load_features()
    state.ensure_bootstrap()
    elig = state.eligibility()
    print(
        f"mash eligibility: excluded coords={elig['excluded_no_coords']} "
        f"price={elig['excluded_no_price']} photos={elig['excluded_no_photos']} "
        f"eligible={elig['eligible']}/{elig['n']}",
        flush=True,
    )
    handler = make_handler(state)
    httpd = ThreadingHTTPServer((host, port), handler)
    print(f"CasitaMash http://{host}:{port}/mash/", flush=True)
    httpd.serve_forever()
