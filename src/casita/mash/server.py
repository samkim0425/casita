from __future__ import annotations

import json
import os
import sqlite3
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from .. import listing_page, storage
from ..listing_page import listing_url
from . import bootstrap, db as mash_db, features, jobs, llm_prefs, model, poi, select, ui

ROOT = Path(__file__).resolve().parents[3]


class MashState:
    def __init__(self, listings_db: Path, site_dir: Path | None = None):
        self.listings_db = listings_db
        self.site_dir = site_dir
        self.poi_data = poi.load_poi_anchors()
        self._feat_map: dict[str, features.ListingFeatures] | None = None
        self._listings = None
        self._top_history: dict[str, list[list[str]]] = {}
        self._pref_queue = jobs.PrefJobQueue(self._run_pref_job)

    def _run_pref_job(self, job: jobs.PrefJob, progress: jobs.ProgressFn) -> None:
        self.refresh_prefs_after_compare(
            job.reviewer,
            left_key=job.left_key,
            right_key=job.right_key,
            winner=job.winner,
            skipped=job.skipped,
            reason=job.reason,
            left_meta=job.left_meta,
            right_meta=job.right_meta,
            progress=progress,
        )

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

    def _brief_for_key(self, key: str, meta: dict | None = None) -> dict:
        fmap = self.load_features()
        base = features.base_listing_key(key)
        feats = fmap.get(key) or fmap.get(base)
        if feats is not None:
            brief = llm_prefs.listing_brief(feats)
            brief["key"] = key
            return brief
        meta = meta or {}
        values = meta.get("values") or {}
        return {
            "key": key,
            "address": key,
            "neighborhood": None,
            "source": meta.get("source") or "",
            "price": values.get("price"),
            "beds": values.get("beds"),
            "baths": values.get("baths"),
            "sqft": values.get("sqft"),
            "light_quality": None,
            "condition_quality": None,
            "view_quality": None,
            "visual_summary": None,
            "outdoor": None,
            "laundry": None,
            "parking": None,
            "dogs": None,
            "description": None,
            "photo_urls": list(meta.get("photos") or [])[:4],
            "is_hypothetical": bool(meta.get("is_hyp")),
        }

    def catalog_briefs(self) -> list[dict]:
        fmap = self.load_features()
        elig = features.eligibility_report(list(fmap.values()))
        keys = set(elig.get("eligible_keys") or fmap.keys())
        return [llm_prefs.listing_brief(fmap[k]) for k in fmap if k in keys]

    def refresh_prefs_after_compare(
        self,
        reviewer: str,
        *,
        left_key: str,
        right_key: str,
        winner: str | None,
        skipped: bool,
        reason: str | None,
        left_meta: dict | None = None,
        right_meta: dict | None = None,
        progress: jobs.ProgressFn | None = None,
    ) -> dict:
        """Memo+probe first (unblocks play), then background catalog rank."""
        def _progress(patch: dict) -> None:
            if progress:
                progress(reviewer, patch)

        left = self._brief_for_key(left_key, left_meta)
        right = self._brief_for_key(right_key, right_meta)
        with mash_db.connect() as conn:
            prev = mash_db.load_memo(conn, reviewer)
            history = mash_db.comparison_prose_history(conn, reviewer)
            rows = mash_db.comparisons_for(conn, reviewer)
            n = mash_db.comparison_count(conn, reviewer)
            prev_fit = mash_db.load_fit(conn, reviewer) or {}

        prev_json = dict(prev.get("memo_json") or {})
        clarification = (prev_json.get("pending_clarification") or "").strip() or None
        clarification_pair = None
        if clarification and prev_json.get("surprise_pair"):
            sp = prev_json["surprise_pair"]
            if isinstance(sp, dict) and sp.get("left_key") and sp.get("right_key"):
                clarification_pair = {
                    "left": self._brief_for_key(sp["left_key"]),
                    "right": self._brief_for_key(sp["right_key"]),
                    "winner": sp.get("winner"),
                }
        feature_order = self.reviewer_features(reviewer)
        ask_elicitation = llm_prefs.should_ask_elicitation(
            n_comparisons=n,
            prev_json=prev_json,
            surprise_this_round=False,
        )

        def _memo_payload(
            *,
            last_error,
            probes,
            surprise,
            elicitation,
            keep_pending_surprise: bool,
        ) -> dict:
            payload = {
                "bullets": memo_res.bullets,
                "probe_features": probes,
                "last_error": last_error,
            }
            if surprise:
                payload["pending_surprise"] = surprise
                payload["surprise_pair"] = {
                    "left_key": left_key,
                    "right_key": right_key,
                    "winner": winner,
                }
            elif keep_pending_surprise and prev_json.get("pending_surprise"):
                payload["pending_surprise"] = prev_json["pending_surprise"]
                if prev_json.get("surprise_pair"):
                    payload["surprise_pair"] = prev_json["surprise_pair"]
            elif not clarification and prev_json.get("pending_clarification"):
                payload["pending_clarification"] = prev_json["pending_clarification"]
                if prev_json.get("surprise_pair"):
                    payload["surprise_pair"] = prev_json["surprise_pair"]
            if (
                elicitation
                and not surprise
                and not payload.get("pending_surprise")
            ):
                payload["pending_elicitation"] = elicitation
                payload["elicitation_last_at_n"] = n
            return payload

        _progress({"phase": "memo", "memo_ready": False, "rank_ready": False})
        memo_res = llm_prefs.update_preference_memo(
            prev_memo=prev.get("memo_text") or "",
            prev_bullets=list(prev_json.get("bullets") or []),
            left=left,
            right=right,
            winner=winner,
            skipped=skipped,
            reason=reason,
            history_lines=history,
            clarification=clarification,
            clarification_pair=clarification_pair,
            ask_elicitation=ask_elicitation,
            feature_order=feature_order,
        )
        probes = list(memo_res.probe_features or [])
        if not probes:
            probes = llm_prefs.probe_features_from_memo(
                memo_res.memo_text, memo_res.bullets, feature_order=feature_order,
            )
        probes = llm_prefs.clamp_probe_features(probes, feature_order)
        surprise = llm_prefs.sanitize_surprise(memo_res.surprise)
        elicitation = memo_res.elicitation if not surprise else None
        memo_err = memo_res.error
        with mash_db.connect() as conn:
            mash_db.save_memo(
                conn,
                reviewer,
                memo_text=memo_res.memo_text,
                memo_json=_memo_payload(
                    last_error=memo_err, probes=probes, surprise=surprise,
                    elicitation=elicitation, keep_pending_surprise=True,
                ),
                mode=memo_res.mode,
            )
            # Keep standings on prior ranks until the new rank lands; stamp memo fields.
            interim = dict(prev_fit) if prev_fit else {"kind": "model_ranks", "ranks": []}
            interim.update({
                "kind": "model_ranks",
                "n_comparisons": n,
                "memo_text": memo_res.memo_text,
                "bullets": memo_res.bullets,
                "probe_features": probes,
                "last_error": memo_err,
                "vertex_configured": llm_prefs.vertex_available(),
                "mode": memo_res.mode if not prev_fit else (prev_fit.get("mode") or memo_res.mode),
            })
            mash_db.save_fit(conn, reviewer, interim)

        _progress({"phase": "rank", "memo_ready": True, "rank_ready": False})

        briefs = self.catalog_briefs()
        rank_res = llm_prefs.rank_from_memo(
            memo_text=memo_res.memo_text,
            briefs=briefs,
            comparisons=rows,
        )
        mode = "vertex" if (memo_res.mode == "vertex" and rank_res.mode == "vertex") else "stub"
        err_bits = [x for x in (memo_res.error, rank_res.error) if x]
        last_error = " | ".join(err_bits) if err_bits else None
        payload = {
            "kind": "model_ranks",
            "mode": mode,
            "n_comparisons": n,
            "memo_text": memo_res.memo_text,
            "bullets": memo_res.bullets,
            "probe_features": probes,
            "ranks": rank_res.ranks,
            "last_error": last_error,
            "vertex_configured": llm_prefs.vertex_available(),
        }
        with mash_db.connect() as conn:
            # Re-read pending_* so a mid-rank dismiss/reply is not overwritten.
            live = mash_db.load_memo(conn, reviewer)
            live_json = dict(live.get("memo_json") or {})
            rank_memo_json = {
                "bullets": memo_res.bullets,
                "probe_features": probes,
                "last_error": last_error,
            }
            if live_json.get("pending_surprise"):
                rank_memo_json["pending_surprise"] = live_json["pending_surprise"]
            if live_json.get("pending_elicitation"):
                rank_memo_json["pending_elicitation"] = live_json["pending_elicitation"]
            if live_json.get("elicitation_last_at_n") is not None:
                rank_memo_json["elicitation_last_at_n"] = live_json["elicitation_last_at_n"]
            if live_json.get("pending_clarification"):
                rank_memo_json["pending_clarification"] = live_json["pending_clarification"]
            if live_json.get("surprise_pair"):
                rank_memo_json["surprise_pair"] = live_json["surprise_pair"]
            mash_db.save_memo(
                conn,
                reviewer,
                memo_text=memo_res.memo_text,
                memo_json=rank_memo_json,
                mode=mode,
            )
            mash_db.save_fit(conn, reviewer, payload)
        _progress({"phase": "idle", "memo_ready": True, "rank_ready": True, "last_error": last_error})
        return payload

    def model_ranks_for(self, reviewer: str, *, force: bool = False) -> dict:
        with mash_db.connect() as conn:
            n = mash_db.comparison_count(conn, reviewer)
            cached = mash_db.load_fit(conn, reviewer)
            memo = mash_db.load_memo(conn, reviewer)
            rows = mash_db.comparisons_for(conn, reviewer)
        vertex_cfg = llm_prefs.vertex_available()
        if (
            not force
            and cached
            and cached.get("kind") == "model_ranks"
            and int(cached.get("n_comparisons") or -1) == n
        ):
            cached = dict(cached)
            cached["vertex_configured"] = vertex_cfg
            # No picks yet: treat configured Vertex as ready (don't flash "offline stub").
            if n == 0 and vertex_cfg:
                cached["mode"] = "vertex"
            with mash_db.connect() as conn:
                memo_row = mash_db.load_memo(conn, reviewer)
            order = self.reviewer_features(reviewer)
            cached["probe_features"] = llm_prefs.clamp_probe_features(
                list(
                    cached.get("probe_features")
                    or (memo_row.get("memo_json") or {}).get("probe_features")
                    or []
                ),
                order,
            )
            return cached
        # Rebuild ranks from stored memo (e.g. after process restart with memo only).
        briefs = self.catalog_briefs()
        rank_res = llm_prefs.rank_from_memo(
            memo_text=memo.get("memo_text") or "",
            briefs=briefs,
            comparisons=rows,
        )
        mode = memo.get("mode") or rank_res.mode
        if n == 0 and vertex_cfg:
            mode = "vertex"
        elif not vertex_cfg:
            mode = "stub"
        order = self.reviewer_features(reviewer)
        probes = llm_prefs.clamp_probe_features(
            list((memo.get("memo_json") or {}).get("probe_features") or []),
            order,
        )
        payload = {
            "kind": "model_ranks",
            "mode": mode,
            "n_comparisons": n,
            "memo_text": memo.get("memo_text") or "",
            "bullets": list((memo.get("memo_json") or {}).get("bullets") or []),
            "probe_features": probes,
            "ranks": rank_res.ranks,
            "last_error": (memo.get("memo_json") or {}).get("last_error"),
            "vertex_configured": vertex_cfg,
        }
        with mash_db.connect() as conn:
            mash_db.save_fit(conn, reviewer, payload)
        return payload

    def rankings(self, reviewer: str, model_state: dict | None = None):
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
        state = model_state if model_state is not None else self.model_ranks_for(reviewer)
        by_key = {r["key"]: r for r in (state.get("ranks") or [])}
        rows = []
        for key, feats in fmap.items():
            info = by_key.get(key) or {}
            sc = float(info.get("score") or 0.0)
            title = feats.address or feats.neighborhood or key
            rows.append({
                "key": key,
                "title": title,
                "score": sc,
                "reason": info.get("reason") or "",
                "never_shown": key not in shown_counts,
                "n_shown": shown_counts.get(key, 0),
                "detail": feats.detail_path,
                "url": feats.url,
                "source": feats.source,
                "cover_url": feats.cover_url,
                "break_even": None,
            })
        rows.sort(key=lambda r: r["score"], reverse=True)
        return rows

    def pref_summary(self, reviewer: str, model_state: dict | None = None) -> dict:
        state = model_state if model_state is not None else self.model_ranks_for(reviewer)
        memo_text = state.get("memo_text") or ""
        bullets = list(state.get("bullets") or [])
        vertex_cfg = bool(state.get("vertex_configured")) or llm_prefs.vertex_available()
        mode = state.get("mode") or ("vertex" if vertex_cfg and int(state.get("n_comparisons") or 0) == 0 else "stub")
        movers = llm_prefs.movers_from_memo(memo_text, bullets)
        with mash_db.connect() as conn:
            memo_row = mash_db.load_memo(conn, reviewer)
        mj = memo_row.get("memo_json") or {}
        order = self.reviewer_features(reviewer)
        probes = llm_prefs.clamp_probe_features(
            list(state.get("probe_features") or mj.get("probe_features") or []),
            order,
        )
        return {
            "memo_text": memo_text or memo_row.get("memo_text") or "",
            "bullets": bullets or list(mj.get("bullets") or []),
            "mode": mode,
            "movers": movers,
            "vertex_configured": vertex_cfg,
            "last_error": state.get("last_error"),
            "n_comparisons": int(state.get("n_comparisons") or 0),
            "probe_features": probes,
            "pending_surprise": llm_prefs.sanitize_surprise(mj.get("pending_surprise")),
            "pending_elicitation": llm_prefs.sanitize_elicitation(mj.get("pending_elicitation")),
        }


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

            if path == "/mash/api/pref_status":
                qs = parse_qs(u.query)
                reviewer = (qs.get("reviewer") or [self._cookie_reviewer() or ""])[0].strip()
                if not reviewer:
                    return self._json(400, {"error": "reviewer required"})
                st = state._pref_queue.status(reviewer)
                with mash_db.connect() as conn:
                    memo = mash_db.load_memo(conn, reviewer)
                order = state.reviewer_features(reviewer)
                probes = llm_prefs.clamp_probe_features(
                    list((memo.get("memo_json") or {}).get("probe_features") or []),
                    order,
                )
                return self._json(200, {
                    **st,
                    "mode": memo.get("mode") or "stub",
                    "probe_features": probes,
                })

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
                model_state = state.model_ranks_for(reviewer)
                pref = state.pref_summary(reviewer, model_state)
                probes = list(pref.get("probe_features") or [])
                with mash_db.connect() as conn:
                    seen = mash_db.decided_pairs(conn, reviewer)
                    n = mash_db.comparison_count(conn, reviewer)
                    rows = mash_db.comparisons_for(conn, reviewer)
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
                ranks_sorted = sorted(
                    model_state.get("ranks") or [],
                    key=lambda r: -float(r.get("score") or 0),
                )
                rank_by_key = {
                    r["key"]: {"rank": i + 1, "score": float(r.get("score") or 0)}
                    for i, r in enumerate(ranks_sorted)
                    if r.get("key")
                }
                candidates = select.candidate_pairs(
                    list(fmap.values()), order, seen, None,
                    n_comparisons=n, recent_hyp=recent_hyp, coverage=coverage,
                    probe_features=probes,
                    rank_by_key=rank_by_key,
                )
                if not candidates:
                    return self._send(200, ui.page("Done", "<p>No more pairs.</p>", who=reviewer).encode())
                banner = None
                if n > 0:
                    ranks = state.rankings(reviewer, model_state)
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
                fallback_why = candidates[0].why_template
                pick_res = llm_prefs.pick_pair_from_shortlist(
                    memo_text=pref.get("memo_text") or "",
                    candidates=[c.to_prompt_dict() for c in candidates],
                    feature_order=order,
                    probe_features=probes,
                    fallback_why=fallback_why,
                )
                idx = max(0, min(pick_res.chosen_index, len(candidates) - 1))
                chosen = candidates[idx]
                why = pick_res.why_line or chosen.why_template
                return self._send(200, ui.play_page(
                    reviewer, chosen.left, chosen.right, why, n, banner, order,
                    memo_text=pref.get("memo_text") or "",
                    mode=pref.get("mode") or "stub",
                    vertex_configured=bool(pref.get("vertex_configured")),
                    last_error=pref.get("last_error"),
                    surprise_reason=pref.get("pending_surprise"),
                    probe_features=probes,
                    pending_elicitation=pref.get("pending_elicitation"),
                ).encode())

            if path == "/mash/results":
                reviewer = self._cookie_reviewer()
                if not reviewer:
                    return self._redirect("/mash/")
                qs = parse_qs(u.query)
                concluded = (qs.get("done") or [""])[0] in ("1", "true", "yes")
                model_state = state.model_ranks_for(reviewer)
                pref = state.pref_summary(reviewer, model_state)
                ranks = state.rankings(reviewer, model_state)
                movers = pref.get("movers") or []
                with mash_db.connect() as conn:
                    n = mash_db.comparison_count(conn, reviewer)
                compared = [r for r in ranks if not r.get("never_shown")]
                unseen = [r for r in ranks if r.get("never_shown")]
                if concluded:
                    combined = ranks[:40]
                    compared, unseen = combined, []
                else:
                    unseen = unseen[:25]
                return self._send(200, ui.results_page(
                    reviewer, compared, unseen, movers, n,
                    concluded=concluded,
                    memo_text=pref.get("memo_text") or "",
                    mode=pref.get("mode") or "stub",
                    vertex_configured=bool(pref.get("vertex_configured")),
                    last_error=pref.get("last_error"),
                ).encode())

            # static site fallback for listing detail pages
            if state.site_dir and (
                path.startswith("/listing/") or path.startswith("/assets/") or path.startswith("/og/")
            ):
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
                reason = (data.get("reason") or "").strip() or None
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
                        "reason": reason,
                    })
                state._pref_queue.enqueue(jobs.PrefJob(
                    reviewer=reviewer,
                    left_key=data["left_key"],
                    right_key=data["right_key"],
                    winner=data.get("winner"),
                    skipped=bool(data.get("skipped")),
                    reason=reason,
                    left_meta=left_meta,
                    right_meta=right_meta,
                ))
                return self._json(200, {"ok": True, "queued": True})

            if path == "/mash/api/surprise":
                data = self._read_json()
                reviewer = (data.get("reviewer") or self._cookie_reviewer() or "").strip()
                if not reviewer:
                    return self._json(400, {"error": "reviewer required"})
                reply = (data.get("reply") or "").strip()[:240] or None
                with mash_db.connect() as conn:
                    memo = mash_db.load_memo(conn, reviewer)
                    mj = dict(memo.get("memo_json") or {})
                    mj.pop("pending_surprise", None)
                    if reply:
                        mj["pending_clarification"] = reply
                    else:
                        mj.pop("surprise_pair", None)
                    mash_db.save_memo(
                        conn,
                        reviewer,
                        memo_text=memo.get("memo_text") or "",
                        memo_json=mj,
                        mode=memo.get("mode") or "stub",
                    )
                return self._json(200, {"ok": True, "saved_reply": bool(reply)})

            if path == "/mash/api/elicitation":
                data = self._read_json()
                reviewer = (data.get("reviewer") or self._cookie_reviewer() or "").strip()
                if not reviewer:
                    return self._json(400, {"error": "reviewer required"})
                choice = (data.get("choice") or "").strip()[:240]
                question = (data.get("question") or "").strip()[:240]
                if not choice:
                    return self._json(400, {"error": "choice required"})
                with mash_db.connect() as conn:
                    memo = mash_db.load_memo(conn, reviewer)
                    mj = dict(memo.get("memo_json") or {})
                    mj.pop("pending_elicitation", None)
                    if question:
                        mj["pending_clarification"] = f'Answered "{question}": {choice}'
                    else:
                        mj["pending_clarification"] = f"Answered: {choice}"
                    mash_db.save_memo(
                        conn,
                        reviewer,
                        memo_text=memo.get("memo_text") or "",
                        memo_json=mj,
                        mode=memo.get("mode") or "stub",
                    )
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
                model_state = state.model_ranks_for(reviewer)
                ranks = state.rankings(reviewer, model_state)
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
            root = state.site_dir
            if root is None:
                return self._send(404, b"missing")
            cand = root / rel
            if cand.is_dir():
                cand = cand / "index.html"
            if not cand.exists() and not rel.endswith(".html"):
                cand = root / f"{rel}.html"
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
    mode = "vertex" if llm_prefs.vertex_available() else "stub"
    print(
        f"mash eligibility: excluded coords={elig['excluded_no_coords']} "
        f"price={elig['excluded_no_price']} photos={elig['excluded_no_photos']} "
        f"eligible={elig['eligible']}/{elig['n']}",
        flush=True,
    )
    print(f"mash preference brain: {mode}", flush=True)
    handler = make_handler(state)
    httpd = ThreadingHTTPServer((host, port), handler)
    print(f"CasitaMash http://{host}:{port}/mash/", flush=True)
    httpd.serve_forever()
