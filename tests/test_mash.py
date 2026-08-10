import json
import sqlite3
from pathlib import Path

import numpy as np
import pytest

from casita.mash import bootstrap, db as mash_db, features, model, select
from casita.mash.features import ListingFeatures


def _feat(key: str, **values) -> ListingFeatures:
    known = {k: v is not None for k, v in values.items()}
    for req in ("price", "is_sf"):
        values.setdefault(req, None)
        known.setdefault(req, values[req] is not None)
    return ListingFeatures(
        key=key,
        values=values,
        known=known,
        routes={"mode": "walk"},
        photo_count=3,
        source="zillow",
        cover_url="http://example.com/a.jpg",
        photos=["http://example.com/a.jpg"],
    )


def test_recover_known_weights_from_synthetic():
    rng = np.random.default_rng(0)
    feats = {}
    for i in range(40):
        price = float(rng.integers(2500, 5500))
        trail = float(rng.integers(5, 40))
        feats[f"L{i}"] = _feat(
            f"L{i}",
            price=price,
            price_per_bed=price / 2,
            price_per_sqft=price / 1000,
            trail=trail,
            beach=float(rng.integers(5, 40)),
            is_sf=1.0,
            beds=2.0,
        )
    true_w_price = -1.0
    true_w_trail = -0.8
    rows = []
    keys = list(feats)
    for _ in range(120):
        a, b = rng.choice(keys, size=2, replace=False)
        sa = true_w_price * feats[a].values["price"] / 1000 + true_w_trail * feats[a].values["trail"] / 10
        sb = true_w_price * feats[b].values["price"] / 1000 + true_w_trail * feats[b].values["trail"] / 10
        winner = a if sa > sb else b
        rows.append({
            "left_key": a, "right_key": b, "winner": winner, "skipped": 0,
            "weight": 1.0, "is_hypothetical": 0, "hyp_left_json": None, "hyp_right_json": None,
            "feature_set_json": "[]",
        })
    fit = model.fit(rows, feats, ["trail", "beach"], lambda_grid=[0.5, 2.0, 8.0])
    assert fit is not None
    assert fit.heldout_acc > 0.55
    assert "trail" in fit.names
    assert fit.w[fit.names.index("trail")] < 0


def test_no_divergence_with_undefeated_listing():
    feats = {
        "A": _feat("A", price=3000, price_per_bed=1500, price_per_sqft=3, trail=5, is_sf=1.0, beds=2),
        "B": _feat("B", price=3200, price_per_bed=1600, price_per_sqft=3.2, trail=20, is_sf=1.0, beds=2),
        "C": _feat("C", price=3400, price_per_bed=1700, price_per_sqft=3.4, trail=25, is_sf=1.0, beds=2),
    }
    rows = []
    for loser in ("B", "C"):
        for _ in range(15):
            rows.append({
                "left_key": "A", "right_key": loser, "winner": "A", "skipped": 0,
                "weight": 1.0, "is_hypothetical": 0, "hyp_left_json": None, "hyp_right_json": None,
                "feature_set_json": "[]",
            })
    fit = model.fit(rows, feats, ["trail"], lambda_grid=[1.0, 5.0])
    assert fit is not None
    assert abs(fit.u["A"]) < 50


def test_selection_never_repeats_or_self(tmp_path, monkeypatch):
    monkeypatch.setenv("CASITA_MASH_DB", str(tmp_path / "mash.sqlite"))
    pool = [
        _feat(f"L{i}", price=3000 + (i % 5) * 200, price_per_bed=1500, price_per_sqft=3,
              trail=40 - i * 2, beach=10 + (i % 4) * 3, is_sf=1.0, beds=2, grocery=8 + (i % 6))
        for i in range(12)
    ]
    seen = set()
    for i in range(20):
        offer = select.select_pair(
            pool, ["trail", "beach", "grocery"], seen, None,
            n_comparisons=i, recent_hyp=1,
        )
        assert offer is not None
        assert offer.left.key != offer.right.key
        assert select.has_tradeoff(
            offer.left, offer.right,
            ["price", "price_per_bed", "price_per_sqft", "trail", "beach", "grocery"],
        )
        pk = tuple(sorted([offer.left.key, offer.right.key]))
        if not offer.is_hypothetical:
            assert pk not in seen
            seen.add(pk)


def test_rejects_free_lunch_pairs():
    cheap = _feat(
        "cheap", price=5000, price_per_bed=2500, price_per_sqft=4.0,
        trail=60, grocery=1, is_sf=1.0, beds=2, laundry=1.0, light=0.5, parking=0.5,
    )
    pricey = _feat(
        "pricey", price=5200, price_per_bed=2600, price_per_sqft=5.0,
        trail=61, grocery=1, is_sf=1.0, beds=2, laundry=1.0, light=0.5, parking=0.5,
    )
    names = ["price", "price_per_bed", "price_per_sqft", "trail", "grocery", "laundry", "light", "parking"]
    assert not select.has_tradeoff(cheap, pricey, names)
    assert select.dominated(cheap, pricey, names)

    trade = _feat(
        "trade", price=6200, price_per_bed=3100, price_per_sqft=5.5,
        trail=12, grocery=1, is_sf=1.0, beds=2, laundry=1.0, light=0.5, parking=0.5,
    )
    assert select.has_tradeoff(cheap, trade, names)

    offer = select.select_pair(
        [cheap, pricey], ["trail", "grocery", "laundry", "light", "parking"], set(), None,
        n_comparisons=0, recent_hyp=1,
    )
    assert offer is None


def test_reviewer_isolation(tmp_path, monkeypatch):
    monkeypatch.setenv("CASITA_MASH_DB", str(tmp_path / "mash.sqlite"))
    with mash_db.connect() as conn:
        mash_db.upsert_reviewer(conn, "alice", ["trail"])
        mash_db.upsert_reviewer(conn, "bob", ["beach"])
        for _ in range(30):
            mash_db.insert_comparison(conn, {
                "reviewer": "alice",
                "left_key": "A", "right_key": "B", "winner": "A", "skipped": 0,
                "strategy": "t", "why_line": "", "feature_set_json": "[]",
                "is_hypothetical": 0, "weight": 1.0, "tag": "direct",
            })
        assert mash_db.comparison_count(conn, "bob") == 0
        assert mash_db.comparison_count(conn, "alice") == 30
        assert mash_db.decided_pairs(conn, "bob") == set()
        assert len(mash_db.decided_pairs(conn, "alice")) == 1


def test_base_listing_key_strips_hyp_prefixes():
    assert features.base_listing_key("zillow:123") == "zillow:123"
    assert features.base_listing_key("hypA:zillow:123") == "zillow:123"
    assert features.base_listing_key("hypB:craigslist:abc") == "craigslist:abc"
    assert features.base_listing_key('hyp:zillow:123:{"price": 3000}') == "zillow:123"


def test_hinge_sentences_use_plain_punctuation():
    fit = model.FitResult(
        names=["price", "price__hinge0", "trail", "trail__hinge0"],
        w=np.array([-1.0, -0.8, -0.5, -1.0]),
        u={},
        keys=[],
        knots={"price": [6272.0], "trail": [57.0]},
        means={"price": 4000.0, "trail": 20.0},
        scales={"price": 1000.0, "trail": 10.0},
        lambda_w=1.0,
        lambda_u=1.0,
        heldout_acc=0.9,
        n_comparisons=12,
        active_features=["trail"],
    )
    lines = model.hinge_sentences(fit)
    assert any("Past $6,272/mo," in line for line in lines)
    assert any("Past 57 min to trail," in line for line in lines)


def test_break_even_hides_nonsense():
    fit = model.FitResult(
        names=["price"],
        w=np.array([-0.01]),
        u={},
        keys=["L"],
        knots={},
        means={"price": 4000.0},
        scales={"price": 1000.0},
        lambda_w=1.0,
        lambda_u=1.0,
        heldout_acc=0.5,
        n_comparisons=0,
        active_features=[],
    )
    feats = _feat("L", price=None, is_sf=1.0)
    assert model.break_even_price(fit, feats, target_score=1e6) is None


def test_why_line_includes_third_feature_when_meaningful():
    names = ["beds", "baths", "price_per_bed"]
    a = _feat(
        "a", price=6500, price_per_bed=3250, price_per_sqft=5,
        beds=2, baths=2, is_sf=1.0,
    )
    b = _feat(
        "b", price=7950, price_per_bed=1988, price_per_sqft=3.6,
        beds=4, baths=4, is_sf=1.0,
    )
    # Scale $/bed so its standardized delta is comparable to beds/baths
    fit = model.FitResult(
        names=names,
        w=np.array([1.0, 0.9, -1.0]),
        u={},
        keys=["a", "b"],
        knots={},
        means={n: 0.0 for n in names},
        scales={"beds": 1.0, "baths": 1.0, "price_per_bed": 800.0},
        lambda_w=1.0,
        lambda_u=1.0,
        heldout_acc=0.5,
        n_comparisons=10,
        active_features=[],
    )
    line = select.why_line(fit, a, b, [], "info")
    assert "differ most on" in line
    assert ", and" in line  # three-item list
    assert "$ / bed" in line

    # Tiny 3rd contribution should stay at two features
    fit_weak = model.FitResult(
        names=names,
        w=np.array([1.0, 1.0, -0.01]),
        u={},
        keys=["a", "b"],
        knots={},
        means={n: 0.0 for n in names},
        scales={n: 1.0 for n in names},
        lambda_w=1.0,
        lambda_u=1.0,
        heldout_acc=0.5,
        n_comparisons=10,
        active_features=[],
    )
    line2 = select.why_line(fit_weak, a, b, [], "info")
    assert ", and" not in line2
    assert " and " in line2


def test_normalize_feature_order_keeps_locked_metrics():
    assert features.normalize_feature_order([]) == ["price_per_bed", "price_per_sqft"]
    assert features.normalize_feature_order(["trail"]) == [
        "trail", "price_per_bed", "price_per_sqft",
    ]
    assert features.normalize_feature_order(["price_per_sqft", "grocery", "price_per_bed"]) == [
        "price_per_sqft", "grocery", "price_per_bed",
    ]
    assert features.card_feature_order(["trail", "price_per_bed"])[:4] == [
        "price", "beds", "baths", "sqft",
    ]
    assert features.card_feature_order(["trail", "price_per_bed"])[4:] == [
        "trail", "price_per_bed", "price_per_sqft",
    ]


def test_bootstrap_excludes_gone_notes():
    assert bootstrap.is_gone_note("Rented — no longer available")
    assert bootstrap.is_gone_note("Off market — no longer available")
    assert bootstrap.is_gone_note("")
    assert bootstrap.is_gone_note(None)
    assert not bootstrap.is_gone_note("Not within walking distance of trails or beach")


def test_gone_bootstrap_seed(tmp_path, monkeypatch):
    monkeypatch.setenv("CASITA_MASH_DB", str(tmp_path / "mash.sqlite"))
    listings = tmp_path / "L.sqlite"
    conn = sqlite3.connect(listings)
    conn.executescript("""
    CREATE TABLE votes (listing_key TEXT, direction TEXT);
    CREATE TABLE listing_status (listing_key TEXT, status TEXT, status_note TEXT);
    INSERT INTO votes VALUES ('up1','up');
    INSERT INTO listing_status VALUES ('down1','passed_on','Too far from trail');
    INSERT INTO listing_status VALUES ('gone1','passed_on','Rented — no longer available');
    """)
    conn.commit()
    with mash_db.connect() as mc:
        stats = bootstrap.bootstrap_from_fixture(conn, mc, reviewer="fixture_seed")
        assert stats["seeded"] == 1
        assert mash_db.comparison_count(mc, "other") == 0
        assert mash_db.comparison_count(mc, "fixture_seed") == 1
    conn.close()
