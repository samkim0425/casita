from __future__ import annotations

import json
import os
import sqlite3
from contextlib import contextmanager
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_MASH_DB = PACKAGE_ROOT / "tmp" / "mash.sqlite"

SCHEMA = """
CREATE TABLE IF NOT EXISTS reviewers (
  name TEXT PRIMARY KEY,
  feature_order_json TEXT NOT NULL DEFAULT '[]',
  created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
  updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS comparisons (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  reviewer TEXT NOT NULL,
  left_key TEXT NOT NULL,
  right_key TEXT NOT NULL,
  winner TEXT,
  skipped INTEGER NOT NULL DEFAULT 0,
  strategy TEXT,
  why_line TEXT,
  feature_set_json TEXT NOT NULL,
  is_hypothetical INTEGER NOT NULL DEFAULT 0,
  hyp_left_json TEXT,
  hyp_right_json TEXT,
  weight REAL NOT NULL DEFAULT 1.0,
  tag TEXT,
  shown_at TIMESTAMP,
  decided_at TIMESTAMP,
  overlay_opened INTEGER NOT NULL DEFAULT 0,
  left_photo_count INTEGER,
  right_photo_count INTEGER,
  left_field_count INTEGER,
  right_field_count INTEGER,
  left_source TEXT,
  right_source TEXT,
  ts TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_comparisons_reviewer ON comparisons (reviewer, id);
CREATE INDEX IF NOT EXISTS idx_comparisons_pair ON comparisons (reviewer, left_key, right_key);

CREATE TABLE IF NOT EXISTS sessions (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  reviewer TEXT NOT NULL,
  ended_at TIMESTAMP,
  top_keys_json TEXT,
  notes TEXT,
  ts TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS fit_cache (
  reviewer TEXT PRIMARY KEY,
  fit_json TEXT NOT NULL,
  ts TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);
"""


def mash_db_path() -> Path:
    return Path(os.environ.get("CASITA_MASH_DB", str(DEFAULT_MASH_DB)))


@contextmanager
def connect():
    path = mash_db_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    try:
        conn.executescript(SCHEMA)
        yield conn
        conn.commit()
    finally:
        conn.close()


def get_reviewer(conn: sqlite3.Connection, name: str) -> sqlite3.Row | None:
    return conn.execute("SELECT * FROM reviewers WHERE name=?", (name,)).fetchone()


def upsert_reviewer(conn: sqlite3.Connection, name: str, feature_order: list[str] | None = None) -> None:
    existing = get_reviewer(conn, name)
    if existing is None:
        conn.execute(
            "INSERT INTO reviewers (name, feature_order_json) VALUES (?, ?)",
            (name, json.dumps(feature_order or [])),
        )
    elif feature_order is not None:
        conn.execute(
            "UPDATE reviewers SET feature_order_json=?, updated_at=CURRENT_TIMESTAMP WHERE name=?",
            (json.dumps(feature_order), name),
        )
    conn.commit()


def comparison_count(conn: sqlite3.Connection, reviewer: str) -> int:
    row = conn.execute(
        "SELECT COUNT(*) AS n FROM comparisons WHERE reviewer=? AND skipped=0 AND winner IS NOT NULL",
        (reviewer,),
    ).fetchone()
    return int(row["n"])


def insert_comparison(conn: sqlite3.Connection, row: dict) -> int:
    cols = [
        "reviewer", "left_key", "right_key", "winner", "skipped", "strategy", "why_line",
        "feature_set_json", "is_hypothetical", "hyp_left_json", "hyp_right_json", "weight",
        "tag", "shown_at", "decided_at", "overlay_opened",
        "left_photo_count", "right_photo_count", "left_field_count", "right_field_count",
        "left_source", "right_source",
    ]
    vals = [row.get(c) for c in cols]
    if vals[cols.index("skipped")] is None:
        vals[cols.index("skipped")] = 0
    if vals[cols.index("is_hypothetical")] is None:
        vals[cols.index("is_hypothetical")] = 0
    if vals[cols.index("weight")] is None:
        vals[cols.index("weight")] = 1.0
    if vals[cols.index("overlay_opened")] is None:
        vals[cols.index("overlay_opened")] = 0
    if vals[cols.index("feature_set_json")] is None:
        vals[cols.index("feature_set_json")] = "[]"
    cur = conn.execute(
        f"INSERT INTO comparisons ({', '.join(cols)}) VALUES ({', '.join('?' for _ in cols)})",
        vals,
    )
    conn.commit()
    return int(cur.lastrowid)


def comparisons_for(conn: sqlite3.Connection, reviewer: str) -> list[sqlite3.Row]:
    return list(conn.execute(
        "SELECT * FROM comparisons WHERE reviewer=? ORDER BY id",
        (reviewer,),
    ))


def decided_pairs(conn: sqlite3.Connection, reviewer: str) -> set[tuple[str, str]]:
    out: set[tuple[str, str]] = set()
    for r in conn.execute(
        "SELECT left_key, right_key FROM comparisons WHERE reviewer=? AND (winner IS NOT NULL OR skipped=1)",
        (reviewer,),
    ):
        a, b = r["left_key"], r["right_key"]
        out.add((a, b) if a <= b else (b, a))
    return out


def save_fit(conn: sqlite3.Connection, reviewer: str, fit: dict) -> None:
    conn.execute(
        "INSERT INTO fit_cache (reviewer, fit_json) VALUES (?, ?) "
        "ON CONFLICT(reviewer) DO UPDATE SET fit_json=excluded.fit_json, ts=CURRENT_TIMESTAMP",
        (reviewer, json.dumps(fit)),
    )
    conn.commit()


def load_fit(conn: sqlite3.Connection, reviewer: str) -> dict | None:
    row = conn.execute("SELECT fit_json FROM fit_cache WHERE reviewer=?", (reviewer,)).fetchone()
    if not row:
        return None
    return json.loads(row["fit_json"])


def end_session(conn: sqlite3.Connection, reviewer: str, top_keys: list[str], notes: str | None = None) -> int:
    cur = conn.execute(
        "INSERT INTO sessions (reviewer, ended_at, top_keys_json, notes) VALUES (?, CURRENT_TIMESTAMP, ?, ?)",
        (reviewer, json.dumps(top_keys), notes),
    )
    conn.commit()
    return int(cur.lastrowid)
