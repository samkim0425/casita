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


def test_score_parts_splits_feature_fit_and_leftover():
    names = ["price"]
    feats = _feat("home", price=3000, price_per_bed=1500, price_per_sqft=3, beds=2)
    fit = model.FitResult(
        names=names,
        w=np.array([-1.0]),
        u={"home": 0.4},
        keys=["home"],
        knots={},
        means={"price": 3000.0},
        scales={"price": 500.0},
        lambda_w=1.0,
        lambda_u=1.0,
        heldout_acc=0.5,
        n_comparisons=5,
        active_features=[],
    )
    feature_fit, leftover, total = model.score_parts(fit, feats)
    assert leftover == pytest.approx(0.4)
    assert total == pytest.approx(feature_fit + leftover)
    assert model.score_listing(fit, feats) == pytest.approx(total)
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
    assert features.normalize_feature_order([]) == [
        "price", "price_per_bed", "price_per_sqft",
    ]
    assert features.normalize_feature_order(["trail"]) == [
        "trail", "price", "price_per_bed", "price_per_sqft",
    ]
    assert features.normalize_feature_order(
        ["price_per_sqft", "grocery", "price_per_bed"]
    ) == ["price_per_sqft", "grocery", "price_per_bed", "price"]
    assert features.card_feature_order(["trail", "price_per_bed", "dogs", "price_per_sqft"])[:6] == [
        "price", "beds", "baths", "price_per_bed", "sqft", "price_per_sqft",
    ]
    assert features.card_feature_order(["trail", "price_per_bed", "dogs", "price_per_sqft"])[6:] == [
        "trail", "dogs",
    ]
    assert features.FEATURE_LABELS["price"] == "Total Rent"


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


def test_mash_anchors_collect_curated_and_poi():
    from casita.mash.anchors import collect_anchor_groups, format_anchors_text

    groups = collect_anchor_groups()
    ids = {g["id"] for g in groups}
    assert {"beaches", "bakeries", "trails", "ferry", "grocery", "bar"} <= ids
    beaches = next(g for g in groups if g["id"] == "beaches")
    assert any("Baker Beach" in item["name"] for item in beaches["items"])
    grocery = next(g for g in groups if g["id"] == "grocery")
    assert len(grocery["items"]) >= 1
    text = format_anchors_text(groups)
    assert "Baker Beach" in text
    assert "poi_anchors.json" in text
    assert "No Maps" in text or "no Maps" in text


def test_dedupe_photo_urls_and_empty_card():
    from casita.mash.features import dedupe_photo_urls
    from casita.mash.ui import card_html, listing_photos

    assert dedupe_photo_urls(["a", "a", "", "b", "a"]) == ["a", "b"]
    empty = _feat("e", price=3000, is_sf=1.0)
    empty.photos = []
    empty.cover_url = None
    empty.photo_count = 0
    html = card_html(empty, ["trail"], "left")
    assert "No photos for this listing" in html
    assert "View photos" not in html
    assert listing_photos(empty) == []

    dup = _feat("d", price=3000, is_sf=1.0)
    dup.photos = ["http://x/1.jpg", "http://x/1.jpg", "http://x/2.jpg"]
    dup.cover_url = "http://x/1.jpg"
    assert listing_photos(dup) == ["http://x/1.jpg", "http://x/2.jpg"]
    html2 = card_html(dup, ["trail"], "right")
    assert "View photos (2)" in html2


def test_mash_results_and_card_html_snapshot_strings():
    """Lock a few results/card HTML contracts so copy regressions fail loudly."""
    from casita.mash.ui import card_html, results_page

    movers = [
        {"feature": "view", "label": "View", "share": 0.4},
        {"feature": "condition", "label": "Condition", "share": 0.3},
    ]
    compared = [{
        "key": "z:1",
        "title": "123 Main St",
        "score": 1.25,
        "n_shown": 3,
        "never_shown": False,
        "url": "https://example.com/listing",
        "source": "zillow",
        "cover_url": "https://example.com/a.jpg",
    }]
    unseen = [{
        "key": "z:2",
        "title": "456 Side St",
        "score": 0.9,
        "n_shown": 0,
        "never_shown": True,
        "url": "",
        "source": "redfin",
        "cover_url": "",
    }]

    mid = results_page(
        "sam", compared, unseen, movers, n=5, concluded=False,
        memo_text="Prefers light and well-kept finishes.",
        mode="stub",
        vertex_configured=False,
    )
    assert "Current Standings" in mid
    assert "Also scoring well (not shown yet)" in mid
    assert "For nerds" in mid
    assert "preference memo" in mid.lower()
    assert "Offline preference stub" in mid or "offline stub" in mid.lower()
    assert "Prefers light and well-kept finishes." in mid
    assert "What the memo is picking up" in mid
    assert "light" in mid.lower()
    assert "Exchange rates" not in mid
    assert "held-out" not in mid.lower()
    assert "tradeoff heuristics" not in mid.lower()
    assert "w·x+u" not in mid
    assert "ranks the catalog" in mid.lower() or "rank the catalog" in mid.lower()

    done = results_page(
        "sam", compared, unseen, movers, n=5, concluded=True,
        memo_text="Prefers light.",
        mode="vertex",
        vertex_configured=True,
    )
    assert "Your Results" in done
    assert "Also scoring well" not in done
    assert "not shown yet" in done  # merged ranking still badges unseen
    assert "Offline preference stub" not in done
    assert "Gemini fallback" not in done
    assert "Prefers light." in done

    fallback = results_page(
        "sam", compared, unseen, movers, n=5, concluded=False,
        memo_text="Stub after fail.",
        mode="stub",
        vertex_configured=True,
        last_error="BILLING_DISABLED on project",
    )
    assert "Gemini fallback" in fallback
    assert "billing" in fallback.lower()

    feat = _feat(
        "c", price=4200, price_per_bed=2100, price_per_sqft=4.2,
        beds=2, baths=1, sqft=1000, trail=12, dogs=1.0, is_sf=1.0,
    )
    feat.address = "789 Lake St"
    feat.neighborhood = "Inner Richmond"
    feat.url = "https://example.com/c"
    card = card_html(feat, ["dogs", "trail", "price_per_sqft"], "left")
    # Fixed rent/size block precedes optional ranked features.
    for label in ("Total Rent", "Beds/Baths", "$ / bed", "Area (sq ft)", "$ / sqft"):
        assert label in card
    assert card.index("Total Rent") < card.index("Beds/Baths")
    assert card.index("Beds/Baths") < card.index("$ / bed")
    assert card.index("$ / bed") < card.index("Area (sq ft)")
    assert card.index("Area (sq ft)") < card.index("$ / sqft")
    assert card.index("$ / sqft") < card.index("Dogs")
    assert 'data-overlay="left"' in card


def test_preference_memo_and_reason_persist(tmp_path, monkeypatch):
    monkeypatch.setenv("CASITA_MASH_DB", str(tmp_path / "mash.sqlite"))
    monkeypatch.delenv("CASITA_GCP_PROJECT", raising=False)
    with mash_db.connect() as conn:
        mash_db.upsert_reviewer(conn, "sam", ["trail"])
        mash_db.insert_comparison(conn, {
            "reviewer": "sam",
            "left_key": "A", "right_key": "B", "winner": "A", "skipped": 0,
            "strategy": "t", "why_line": "", "feature_set_json": "[]",
            "is_hypothetical": 0, "weight": 1.0, "tag": "direct",
            "reason": "more light",
        })
        lines = mash_db.comparison_prose_history(conn, "sam")
        assert any("chose A over B" in line and "more light" in line for line in lines)
        mash_db.save_memo(conn, "sam", memo_text="Likes light.", memo_json={"bullets": ["light"]}, mode="stub")
        memo = mash_db.load_memo(conn, "sam")
        assert memo["memo_text"] == "Likes light."
        assert memo["mode"] == "stub"
        assert memo["memo_json"]["bullets"] == ["light"]


def test_stub_memo_and_rank_use_raw_condition_not_lookup_floats(monkeypatch):
    monkeypatch.delenv("CASITA_GCP_PROJECT", raising=False)
    monkeypatch.delenv("GOOGLE_CLOUD_PROJECT", raising=False)
    monkeypatch.delenv("GCLOUD_PROJECT", raising=False)
    monkeypatch.setattr(
        "casita.mash.llm_prefs.resolve_gcp_project",
        lambda explicit=None: None,
    )
    from casita.mash import llm_prefs

    left = {
        "key": "L1", "address": "1 Light St", "price": 4000, "beds": 2, "baths": 1,
        "sqft": 900, "light_quality": "abundant", "condition_quality": "well-kept",
        "view_quality": None, "visual_summary": "bright south windows",
        "outdoor": None, "laundry": "in-unit", "parking": None, "dogs": None,
        "description": None, "photo_urls": [],
    }
    right = {
        "key": "L2", "address": "2 Dim Ave", "price": 3800, "beds": 2, "baths": 1,
        "sqft": 900, "light_quality": "dim", "condition_quality": "dated",
        "view_quality": None, "visual_summary": "dark interiors",
        "outdoor": None, "laundry": None, "parking": None, "dogs": None,
        "description": None, "photo_urls": [],
    }
    memo = llm_prefs.update_preference_memo(
        prev_memo="",
        prev_bullets=[],
        left=left,
        right=right,
        winner="L1",
        skipped=False,
        reason="better light",
    )
    assert memo.mode == "stub"
    assert "better light" in memo.memo_text.lower() or "Preferred" in memo.memo_text
    assert "well-kept" in memo.memo_text or "abundant" in memo.memo_text or "bright" in memo.memo_text

    ranks = llm_prefs.rank_from_memo(
        memo_text=memo.memo_text,
        briefs=[left, right],
        comparisons=[{
            "left_key": "L1", "right_key": "L2", "winner": "L1", "skipped": 0,
        }],
    )
    assert ranks.mode == "stub"
    assert ranks.ranks[0]["key"] == "L1"
    assert "stub" in ranks.ranks[0]["reason"].lower()
    # Briefs must not be required to carry numeric light/condition encodings.
    assert "light_quality" in left and isinstance(left["light_quality"], str)


def test_listing_brief_exposes_raw_strings_not_maps():
    from casita.mash import llm_prefs

    feats = _feat("z:1", price=3000, is_sf=1.0, light=1.0, condition=0.66)
    feats.light_quality = "abundant"
    feats.condition_quality = "well-kept"
    feats.visual_summary = "sunny kitchen"
    brief = llm_prefs.listing_brief(feats)
    assert brief["light_quality"] == "abundant"
    assert brief["condition_quality"] == "well-kept"
    assert brief["visual_summary"] == "sunny kitchen"
    assert 0.66 not in brief.values()
    assert 1.0 not in (brief.get("light_quality"), brief.get("condition_quality"))


def test_probe_features_from_memo_keywords():
    from casita.mash import llm_prefs
    order = ["trail", "condition", "baths"]
    probes = llm_prefs.probe_features_from_memo(
        "Prefers 2 baths and modern finishes; trail distance is a non-factor.",
        ["better bathroom"],
        feature_order=order,
    )
    assert "baths" in probes
    assert "condition" in probes or "trail" in probes


def test_clamp_probe_features_drops_unranked():
    from casita.mash import llm_prefs
    order = ["trail", "is_sf", "condition"]
    raw = ["outdoor", "parking", "is_sf", "trail", "baths"]
    clamped = llm_prefs.clamp_probe_features(raw, order)
    assert "outdoor" not in clamped
    assert "parking" not in clamped
    assert "is_sf" in clamped
    assert "trail" in clamped
    assert "baths" in clamped  # always on card


def test_probe_features_from_memo_drops_yard_without_outdoor_ranked():
    from casita.mash import llm_prefs
    probes = llm_prefs.probe_features_from_memo(
        "Prefers garage and private yard; cares about 2 baths and SF over Marin.",
        [],
        feature_order=["baths", "is_sf", "trail"],
    )
    assert "outdoor" not in probes
    assert "parking" not in probes
    assert "baths" in probes or "is_sf" in probes


def test_format_clarification_block_includes_prior_pair_briefs():
    from casita.mash import llm_prefs
    block = llm_prefs.format_clarification_block(
        "cheaper + bath when condition was acceptable",
        clarification_pair={
            "left": {
                "key": "L1", "address": "1 Main", "price": 4000, "beds": 2, "baths": 2,
                "condition_quality": "well-kept", "light_quality": "abundant",
            },
            "right": {
                "key": "L2", "address": "2 Side", "price": 3800, "beds": 2, "baths": 1,
                "condition_quality": "dated", "light_quality": "dim",
            },
            "winner": "L1",
        },
    )
    assert "preference/tradeoff intent" in block
    assert "Prior left" in block
    assert "condition=well-kept" in block
    assert "condition=dated" in block
    assert "ground truth" in block.lower()
    assert "Winner recorded" in block


def test_format_clarification_block_empty_when_no_clarification():
    from casita.mash import llm_prefs
    assert llm_prefs.format_clarification_block(None) == "No pending clarification."


def test_why_line_uses_probe_features():
    a = _feat("A", price=3000, baths=2, trail=40, is_sf=1.0, beds=2)
    b = _feat("B", price=3200, baths=1, trail=10, is_sf=1.0, beds=2)
    line = select.why_line(None, a, b, ["trail", "baths"], "cold_start", probe_features=["baths", "trail"])
    assert "Probing" in line
    assert "preference memo" in line


def test_select_pair_biases_toward_probe_tradeoff():
    # baths tradeoff pair vs trail-only tradeoff; with baths probe, prefer baths.
    cheap_1ba = _feat("c1", price=3000, baths=1, trail=20, grocery=10, is_sf=1.0, beds=2, price_per_bed=1500, price_per_sqft=3)
    pricey_2ba = _feat("p2", price=3400, baths=2, trail=22, grocery=11, is_sf=1.0, beds=2, price_per_bed=1700, price_per_sqft=3.2)
    near_trail = _feat("nt", price=3100, baths=1, trail=5, grocery=12, is_sf=1.0, beds=2, price_per_bed=1550, price_per_sqft=3.1)
    far_trail = _feat("ft", price=3150, baths=1, trail=45, grocery=12, is_sf=1.0, beds=2, price_per_bed=1575, price_per_sqft=3.15)
    pool = [cheap_1ba, pricey_2ba, near_trail, far_trail]
    order = ["baths", "trail", "grocery"]
    offer = select.select_pair(
        pool, order, set(), None,
        n_comparisons=3, recent_hyp=1, probe_features=["baths"],
        rng=__import__("random").Random(0),
    )
    assert offer is not None
    assert "Probing" in offer.why_line or "bath" in offer.why_line.lower()
    keys = {offer.left.key, offer.right.key}
    # Prefer the baths tradeoff pair when probing baths.
    assert keys == {"c1", "p2"} or "baths" in (offer.strategy or "") or select.advantage(offer.left, offer.right, "baths") != 0


def test_pair_passes_filters_rejects_dominated():
    cheap = _feat(
        "cheap", price=5000, price_per_bed=2500, price_per_sqft=4.0,
        trail=60, grocery=1, is_sf=1.0, beds=2, laundry=1.0, light=0.5, parking=0.5,
    )
    pricey = _feat(
        "pricey", price=5200, price_per_bed=2600, price_per_sqft=5.0,
        trail=61, grocery=1, is_sf=1.0, beds=2, laundry=1.0, light=0.5, parking=0.5,
    )
    visible = ["price", "price_per_bed", "price_per_sqft", "trail", "grocery", "laundry", "light", "parking"]
    assert not select.pair_passes_filters(
        cheap, pricey, visible=visible, probes=[], feature_order=["trail"],
    )


def test_has_memo_tradeoff_requires_probe_conflict():
    a = _feat("a", price=3000, baths=1, trail=10, is_sf=1.0, beds=2, price_per_bed=1500, price_per_sqft=3)
    b = _feat("b", price=3200, baths=2, trail=30, is_sf=1.0, beds=2, price_per_bed=1600, price_per_sqft=3.2)
    visible = ["price", "price_per_bed", "price_per_sqft", "baths", "trail"]
    assert select.has_memo_tradeoff(a, b, ["baths", "trail"], ["baths", "trail"], visible)
    c = _feat("c", price=3100, baths=1, trail=12, is_sf=1.0, beds=2, price_per_bed=1550, price_per_sqft=3.1)
    assert not select.has_memo_tradeoff(a, c, ["baths", "trail"], ["baths", "trail"], visible)


def test_candidate_pairs_dedupes_and_caps():
    pool = [
        _feat(f"L{i}", price=3000 + i * 100, trail=10 + i * 5, beach=10 + i * 2,
              is_sf=1.0, beds=2, price_per_bed=1500, price_per_sqft=3, grocery=8)
        for i in range(10)
    ]
    cands = select.candidate_pairs(
        pool, ["trail", "beach"], set(), None,
        n_comparisons=2, recent_hyp=1, cap=8,
        rng=__import__("random").Random(1),
    )
    assert len(cands) <= 8
    keys = {select._pair_key(c.left.key, c.right.key) for c in cands}
    assert len(keys) == len(cands)


def test_candidate_pairs_rank_boundary_when_ranks_present():
    pool = [
        _feat("a", price=3000, trail=30, is_sf=1.0, beds=2, price_per_bed=1500, price_per_sqft=3),
        _feat("b", price=3200, trail=10, is_sf=1.0, beds=2, price_per_bed=1600, price_per_sqft=3.2),
        _feat("c", price=3100, trail=25, is_sf=1.0, beds=2, price_per_bed=1550, price_per_sqft=3.1),
    ]
    rank_by_key = {
        "a": {"rank": 1, "score": 0.9},
        "b": {"rank": 2, "score": 0.85},
        "c": {"rank": 3, "score": 0.8},
    }
    cands = select.candidate_pairs(
        pool, ["trail"], set(), None,
        n_comparisons=5, recent_hyp=1, rank_by_key=rank_by_key,
        rng=__import__("random").Random(0),
    )
    strategies = {c.strategy for c in cands}
    assert "rank_boundary" in strategies


def test_candidate_pairs_excludes_dominated():
    cheap = _feat(
        "cheap", price=5000, price_per_bed=2500, price_per_sqft=4.0,
        trail=60, grocery=1, is_sf=1.0, beds=2,
    )
    pricey = _feat(
        "pricey", price=5200, price_per_bed=2600, price_per_sqft=5.0,
        trail=61, grocery=1, is_sf=1.0, beds=2,
    )
    trade = _feat(
        "trade", price=6200, price_per_bed=3100, price_per_sqft=5.5,
        trail=12, grocery=1, is_sf=1.0, beds=2,
    )
    cands = select.candidate_pairs(
        [cheap, pricey, trade], ["trail", "grocery"], set(), None,
        n_comparisons=0, recent_hyp=1,
    )
    for c in cands:
        pk = select._pair_key(c.left.key, c.right.key)
        assert pk != select._pair_key("cheap", "pricey")


def test_pick_pair_from_shortlist_stub_picks_top_heuristic(monkeypatch):
    monkeypatch.setattr("casita.mash.llm_prefs.vertex_available", lambda: False)
    from casita.mash import llm_prefs
    cands = [
        {"heuristic_score": 1.0, "why_template": "low"},
        {"heuristic_score": 9.0, "why_template": "high"},
    ]
    res = llm_prefs.pick_pair_from_shortlist(
        memo_text="Prefers light.",
        candidates=cands,
        fallback_why="fallback",
    )
    assert res.mode == "stub"
    assert res.chosen_index == 1
    assert res.why_line == "high"


def test_pick_pair_from_shortlist_clamps_bad_index(monkeypatch):
    monkeypatch.setattr("casita.mash.llm_prefs.vertex_available", lambda: False)
    from casita.mash import llm_prefs
    cands = [{"heuristic_score": 5.0, "why_template": "only"}]
    res = llm_prefs.pick_pair_from_shortlist(
        memo_text="x", candidates=cands, fallback_why="fb",
    )
    assert res.chosen_index == 0
    assert res.why_line == "only"


def test_pref_job_queue_runs_serially():
    from casita.mash.jobs import PrefJob, PrefJobQueue
    import threading
    order = []
    lock = threading.Lock()
    gate = threading.Event()

    def worker(job: PrefJob, progress):
        with lock:
            order.append(("start", job.reason))
        progress(job.reviewer, {"phase": "memo", "memo_ready": False, "rank_ready": False})
        if job.reason == "first":
            gate.wait(timeout=2)
        progress(job.reviewer, {"phase": "rank", "memo_ready": True, "rank_ready": False})
        progress(job.reviewer, {"phase": "idle", "memo_ready": True, "rank_ready": True})
        with lock:
            order.append(("end", job.reason))

    q = PrefJobQueue(worker)
    q.enqueue(PrefJob("r", "A", "B", "A", False, "first"))
    q.enqueue(PrefJob("r", "A", "C", "A", False, "second"))
    # Let first start
    import time
    for _ in range(50):
        if any(x == ("start", "first") for x in order):
            break
        time.sleep(0.02)
    assert ("start", "first") in order
    # Second must not have started while first is running
    assert ("start", "second") not in order
    gate.set()
    for _ in range(100):
        if ("end", "second") in order:
            break
        time.sleep(0.02)
    assert order.index(("end", "first")) < order.index(("start", "second"))


def test_pref_status_memo_ready_before_rank_completes():
    from casita.mash.jobs import PrefJob, PrefJobQueue, memo_gate_satisfied
    import threading
    import time

    memo_seen = threading.Event()
    rank_gate = threading.Event()

    def worker(job: PrefJob, progress):
        progress(job.reviewer, {"phase": "memo", "memo_ready": False, "rank_ready": False})
        time.sleep(0.05)
        progress(job.reviewer, {"phase": "rank", "memo_ready": True, "rank_ready": False})
        memo_seen.set()
        rank_gate.wait(timeout=2)
        progress(job.reviewer, {"phase": "idle", "memo_ready": True, "rank_ready": True})

    q = PrefJobQueue(worker)
    q.enqueue(PrefJob("alice", "L", "R", "L", False, None))
    assert memo_seen.wait(timeout=2)
    st = q.status("alice")
    assert st["memo_ready"] is True
    assert st["rank_ready"] is False
    assert st["phase"] == "rank"
    assert st["status"] == "running"
    assert memo_gate_satisfied(st)
    assert set(st) >= {
        "status", "phase", "memo_ready", "rank_ready", "last_error", "updated_at", "pending",
    }
    rank_gate.set()
    for _ in range(100):
        st = q.status("alice")
        if st["status"] == "idle" and st["rank_ready"]:
            break
        time.sleep(0.02)
    assert st["rank_ready"] is True
    assert st["phase"] == "idle"


def test_memo_gate_satisfied_shape():
    from casita.mash.jobs import memo_gate_satisfied
    assert not memo_gate_satisfied({"phase": "memo", "memo_ready": False, "status": "running"})
    assert memo_gate_satisfied({"phase": "rank", "memo_ready": True, "status": "running"})
    assert memo_gate_satisfied({"phase": "memo", "memo_ready": True, "status": "running"})
    assert memo_gate_satisfied({"phase": "idle", "memo_ready": False, "status": "idle"})
    assert memo_gate_satisfied({"phase": "error", "memo_ready": False, "status": "error"})


def test_fetch_photo_parts_fetches_concurrently(monkeypatch):
    from casita.mash import llm_prefs
    import time

    started = []
    lock = __import__("threading").Lock()

    class FakePart:
        def __init__(self, url):
            self.url = url

    def fake_one(url):
        with lock:
            started.append(("start", url))
        time.sleep(0.08)
        with lock:
            started.append(("end", url))
        return FakePart(url)

    monkeypatch.setattr(llm_prefs, "_fetch_one_photo", fake_one)
    t0 = time.perf_counter()
    parts = llm_prefs._fetch_photo_parts(["http://a/1.jpg", "http://a/2.jpg"], max_n=2)
    elapsed = time.perf_counter() - t0
    assert len(parts) == 2
    # Concurrent: wall time closer to one sleep than two sequential.
    assert elapsed < 0.14
    assert ("start", "http://a/2.jpg") in started


def test_stub_memo_includes_probe_features(monkeypatch):
    monkeypatch.delenv("CASITA_GCP_PROJECT", raising=False)
    monkeypatch.delenv("GOOGLE_CLOUD_PROJECT", raising=False)
    monkeypatch.delenv("GCLOUD_PROJECT", raising=False)
    monkeypatch.setattr("casita.mash.llm_prefs.resolve_gcp_project", lambda explicit=None: None)
    from casita.mash import llm_prefs
    left = {
        "key": "L1", "address": "1 Bath St", "price": 4000, "beds": 2, "baths": 2,
        "sqft": 900, "light_quality": "abundant", "condition_quality": "well-kept",
        "view_quality": None, "visual_summary": "bright", "outdoor": None,
        "laundry": None, "parking": None, "dogs": None, "description": None, "photo_urls": [],
    }
    right = {
        "key": "L2", "address": "2 Dim Ave", "price": 3800, "beds": 2, "baths": 1,
        "sqft": 900, "light_quality": "dim", "condition_quality": "dated",
        "view_quality": None, "visual_summary": "dark", "outdoor": None,
        "laundry": None, "parking": None, "dogs": None, "description": None, "photo_urls": [],
    }
    memo = llm_prefs.update_preference_memo(
        prev_memo="",
        prev_bullets=[],
        left=left,
        right=right,
        winner="L1",
        skipped=False,
        reason="extra bathroom and modern finishes",
        feature_order=["baths", "condition", "light"],
    )
    assert memo.mode == "stub"
    assert memo.probe_features
    assert "baths" in memo.probe_features or "condition" in memo.probe_features


def test_sanitize_why_line_rejects_bad_and_truncates():
    from casita.mash.llm_prefs import sanitize_why_line
    fb = "These trade off trail and grocery."
    assert sanitize_why_line("", fallback=fb) == fb
    assert sanitize_why_line("hi", fallback=fb) == fb
    assert sanitize_why_line('  "A solid reason here."  ', fallback=fb) == "A solid reason here."
    assert sanitize_why_line("As an AI I think you should pick left", fallback=fb) == fb
    long = "I'm showing you these two because " + ("x " * 120)
    out = sanitize_why_line(long, fallback=fb, max_chars=80)
    assert out != fb
    assert len(out) <= 80
    assert out.endswith("…")


def test_explain_pair_why_uses_template_without_vertex(monkeypatch):
    monkeypatch.delenv("CASITA_GCP_PROJECT", raising=False)
    monkeypatch.delenv("GOOGLE_CLOUD_PROJECT", raising=False)
    monkeypatch.delenv("GCLOUD_PROJECT", raising=False)
    monkeypatch.setattr("casita.mash.llm_prefs.resolve_gcp_project", lambda explicit=None: None)
    monkeypatch.setattr("casita.mash.llm_prefs.vertex_available", lambda: False)
    from casita.mash import llm_prefs
    fb = 'These trade off "distance to trail" and "distance to grocery".'
    brief = {
        "key": "L1", "address": "1 St", "price": 4000, "beds": 2, "baths": 1,
        "sqft": 900, "light_quality": None, "condition_quality": None,
        "view_quality": None, "visual_summary": None, "outdoor": None,
        "laundry": None, "parking": None, "dogs": None, "description": None, "photo_urls": [],
    }
    # Empty memo → template, no model call.
    empty = llm_prefs.explain_pair_why(
        memo_text="", left=brief, right=brief, fallback_why=fb,
    )
    assert empty.why_line == fb
    assert empty.mode == "stub"
    # Memo present but no Vertex → still template.
    stub = llm_prefs.explain_pair_why(
        memo_text="Prefers trail access over grocery proximity.",
        left=brief,
        right=brief,
        fallback_why=fb,
        probe_features=["trail", "grocery"],
    )
    assert stub.why_line == fb
    assert stub.mode == "stub"


def test_sanitize_surprise_strips_chrome_and_rejects_short():
    from casita.mash.llm_prefs import sanitize_surprise
    assert sanitize_surprise(None) is None
    assert sanitize_surprise("too short") is None
    assert sanitize_surprise(
        "You seemed to prefer brighter places, but this one looks dimmer."
    ).startswith("You seemed")
    assert sanitize_surprise(
        "Hold on.. that choice contradicts our preference memo. "
        "You seemed to prefer brighter places, but this one looks dimmer."
    ).startswith("You seemed")


def test_stub_surprise_when_light_contradicts_memo(monkeypatch):
    monkeypatch.setattr("casita.mash.llm_prefs.vertex_available", lambda: False)
    from casita.mash import llm_prefs
    left = {
        "key": "L1", "address": "Dim St", "price": 4000, "beds": 2, "baths": 1,
        "sqft": 900, "light_quality": "dim", "condition_quality": "dated",
        "view_quality": None, "visual_summary": None, "outdoor": None,
        "laundry": None, "parking": None, "dogs": None, "description": None, "photo_urls": [],
    }
    right = {
        "key": "L2", "address": "Bright Ave", "price": 4100, "beds": 2, "baths": 1,
        "sqft": 900, "light_quality": "abundant", "condition_quality": "well-kept",
        "view_quality": None, "visual_summary": None, "outdoor": None,
        "laundry": None, "parking": None, "dogs": None, "description": None, "photo_urls": [],
    }
    memo = llm_prefs.update_preference_memo(
        prev_memo="Prefers bright, abundant light over darker units.",
        prev_bullets=["likes light"],
        left=left,
        right=right,
        winner="L1",
        skipped=False,
        reason=None,
    )
    assert memo.surprise
    assert "brighter" in memo.surprise.lower() or "dimmer" in memo.surprise.lower()


def test_play_page_includes_surprise_overlay():
    from casita.mash import features, ui
    left = features.ListingFeatures(
        key="a", values={}, known={}, routes={}, photo_count=0, source="z",
        cover_url=None, photos=[], is_hypothetical=False,
    )
    right = features.ListingFeatures(
        key="b", values={}, known={}, routes={}, photo_count=0, source="z",
        cover_url=None, photos=[], is_hypothetical=False,
    )
    html = ui.play_page(
        "sam", left, right, "why", 3, None, ["baths"],
        surprise_reason="You seemed to prefer brighter places, but this one looks dimmer.",
    )
    assert "Hold on.." in html
    assert "That choice contradicts our preference memo." in html
    assert "What changed? (optional)" in html
    assert "rent mattered more" in html
    assert "/mash/api/surprise" in html


def test_probe_weighing_row_renders_labels():
    from casita.mash import features, ui
    assert ui.probe_weighing_row([]) == ""
    assert ui.probe_weighing_row(None) == ""
    row = ui.probe_weighing_row(["baths", "trail", "light"])
    assert "Still weighing:" in row
    assert features.FEATURE_LABELS["baths"] in row
    assert features.FEATURE_LABELS["trail"] in row
    assert features.FEATURE_LABELS["light"] in row
    assert row.count("·") == 2


def test_play_page_includes_probe_row():
    from casita.mash import features, ui
    left = features.ListingFeatures(
        key="a", values={}, known={}, routes={}, photo_count=0, source="z",
        cover_url=None, photos=[], is_hypothetical=False,
    )
    right = features.ListingFeatures(
        key="b", values={}, known={}, routes={}, photo_count=0, source="z",
        cover_url=None, photos=[], is_hypothetical=False,
    )
    html = ui.play_page(
        "sam", left, right, "why", 3, None, ["baths"],
        probe_features=["baths", "trail"],
    )
    assert "Still weighing:" in html
    assert features.FEATURE_LABELS["baths"] in html
    assert features.FEATURE_LABELS["trail"] in html


def test_sanitize_elicitation_requires_two_distinct_choices():
    from casita.mash.llm_prefs import sanitize_elicitation
    assert sanitize_elicitation(None) is None
    assert sanitize_elicitation({"question": "short?", "choice_a": "A", "choice_b": "B"}) is None
    ok = sanitize_elicitation({
        "question": "If rent is similar, would you rather have brighter rooms or pay less?",
        "choice_a": "Brighter rooms",
        "choice_b": "Pay less",
    })
    assert ok
    assert ok["choice_a"] == "Brighter rooms"


def test_should_ask_elicitation_gates():
    from casita.mash.llm_prefs import should_ask_elicitation
    assert not should_ask_elicitation(n_comparisons=3, prev_json={}, surprise_this_round=False)
    assert should_ask_elicitation(n_comparisons=4, prev_json={}, surprise_this_round=False)
    assert not should_ask_elicitation(
        n_comparisons=5, prev_json={"pending_surprise": "x"}, surprise_this_round=False,
    )
    assert not should_ask_elicitation(
        n_comparisons=10, prev_json={"elicitation_last_at_n": 8}, surprise_this_round=False,
    )
    assert should_ask_elicitation(
        n_comparisons=16, prev_json={"elicitation_last_at_n": 8}, surprise_this_round=False,
    )


def test_play_page_includes_elicitation_overlay():
    from casita.mash import features, ui
    left = features.ListingFeatures(
        key="a", values={}, known={}, routes={}, photo_count=0, source="z",
        cover_url=None, photos=[], is_hypothetical=False,
    )
    right = features.ListingFeatures(
        key="b", values={}, known={}, routes={}, photo_count=0, source="z",
        cover_url=None, photos=[], is_hypothetical=False,
    )
    elicitation = {
        "question": "If rent is similar, would you rather have an extra bathroom or be closer to groceries on foot?",
        "choice_a": "Extra bathroom",
        "choice_b": "Closer to groceries",
    }
    html = ui.play_page("sam", left, right, "why", 5, None, ["baths"], pending_elicitation=elicitation)
    assert "Quick question" in html
    assert elicitation["question"] in html
    assert "Extra bathroom" in html
    assert "/mash/api/elicitation" in html
    assert "elicitationSkip" not in html
