from __future__ import annotations

import re
import sqlite3

from . import db as mash_db

GONE_PATTERNS = [
    re.compile(r"no longer available", re.I),
    re.compile(r"\brented\b", re.I),
    re.compile(r"off market", re.I),
    re.compile(r"\bgone\b", re.I),
]


def is_gone_note(note: str | None) -> bool:
    if not note or not note.strip():
        return True
    return any(p.search(note) for p in GONE_PATTERNS)


def bootstrap_from_fixture(listings_conn, mash_conn, reviewer: str = "fixture_seed") -> dict:
    mash_db.upsert_reviewer(mash_conn, reviewer, feature_order=[
        "trail", "beach", "dogs", "outdoor", "condition",
    ])
    existing = mash_db.comparison_count(mash_conn, reviewer)
    if existing:
        return {"seeded": 0, "skipped_existing": existing}

    listings_conn.row_factory = sqlite3.Row
    ups = [
        r["listing_key"]
        for r in listings_conn.execute(
            "SELECT listing_key FROM votes WHERE direction='up'"
        )
    ]
    ups = list(dict.fromkeys(ups))

    passed = []
    for r in listings_conn.execute(
        "SELECT listing_key, status_note FROM listing_status WHERE status='passed_on'"
    ):
        if is_gone_note(r["status_note"]):
            continue
        passed.append(r["listing_key"])

    feature_set = json_default_features()
    n = 0
    for up in ups:
        for down in passed:
            if up == down:
                continue
            mash_db.insert_comparison(mash_conn, {
                "reviewer": reviewer,
                "left_key": up,
                "right_key": down,
                "winner": up,
                "skipped": 0,
                "strategy": "bootstrap",
                "why_line": "Seeded from an upvote vs a pass",
                "feature_set_json": feature_set,
                "is_hypothetical": 0,
                "weight": 0.35,
                "tag": "bootstrap",
            })
            n += 1
    return {"seeded": n, "ups": len(ups), "passed": len(passed)}


def json_default_features() -> str:
    import json
    return json.dumps([
        "price", "price_per_bed", "price_per_sqft",
        "trail", "beach", "dogs", "outdoor", "condition",
    ])
