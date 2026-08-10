# Phase 0 — Recon

Queried `fixtures/demo.sqlite` read-only. Live `casita demo` HTTP not required for schema/counts.

## Schema

### `listings` (`src/casita/storage.py`)

`key`, `source`, `source_id`, `url`, `title`, `address`, `neighborhood`, `neighborhood_resolved`, `price`, `beds`, `baths`, `sqft`, `pets_allowed`, `dog_policy`, `parking`, `laundry`, `has_yard`, `yard_note`, LLM rank fields, contact fields, `description`, `image_url`, `photos_json`, `lat`/`lng`, `raw_json`, `first_seen`/`last_seen`, `active`, plus migrated photo-review columns: `light_quality`, `view_quality`, `condition_quality`, `outdoor_visible`, `other_visible`, `visual_summary`, `share_blurb`/`share_token`, `address_verified`, `contact_note`.

### `votes`

`id`, `listing_key`, `voter`, `direction` (`up`|`down`), `reason`, `ts`. Append-only.

### `listing_status`

`listing_key`, `status`, `status_note`, `viewing_at`, `updated_at`. CREATE comment omits `passed_on` (used in code/fixture).

### `walk_cache`

Not in `storage.SCHEMA`. Separate routes DB (`walk.py`); demo co-locates via `CASITA_ROUTE_CACHE_DB`. Columns: `from_lat/lng`, `to_lat/lng`, `mode`, `minutes`, `source`, `ts`.

Also in fixture: `llm_photo_reviews`, `llm_facts`, `actions`, `attachments`, `interactions`, `pending_urls`, `runs`.

## Counts (active = 143)

| Metric | Value |
| --- | --- |
| Total listings | 349 |
| Votes | 16 up, 0 down |
| Status | passed_on 24, declined_by_landlord 3, contacted 2, viewing_done 1, declined_by_us 1, applied 1 |
| Coords / price / photos | 139 / 122 / 141 |
| Eligible (coords ∩ price ∩ photos) | 118 |
| Cascade exclude | coords 4 → price 21 → photos 0 |
| walk_cache | 5553 (walk 4692, drive 861) |
| llm_photo_reviews (active) | 141 |

## Field density (active %)

Dense: identity/LLM rank ~100%; photos 98.6%; coords 97.2%; pets 96.5%; beds/baths ~86%; price 85.3%; condition 84.6%; light 77.6%; view 75.5%; sqft 75.5%. Sparse: laundry 41%; outdoor_visible 42%; parking 37%; **has_yard 28.7%** (23 true / 18 false / 102 null).

Photo-review JSON: `best_photo_index` 100%; `drop_indices` non-empty 9.9%. These are **not** listing columns — CasitaMash must read `llm_photo_reviews`.

## walk.py

Anchors: beaches, bakeries, trails, Ferry Building (`SF_CENTER`). `is_marin`: `lat > 37.84`. No grocery/bar/farmers market in repo — CasitaMash adds OSM-derived POIs.

## html._card

`_card(Listing, ...) -> str` fragment. Callable without DB; full CSS only via `html.render()`. Walks intentionally on detail page only.

## Demo HTTP

`RenderedSiteHTTPRequestHandler(SimpleHTTPRequestHandler)` on `:8765`. No `do_POST`. CasitaMash extends the handler.

## numpy / scipy

Not in `pyproject.toml` at recon time — added for the mash fit.

## Surprises

1. Brief feature list includes grocery/bar/market; code only has trail/beach/bakery/Ferry.
2. `walk_cache` / `llm_*` not in `storage.SCHEMA`.
3. `passed_on` omitted from CREATE comment.
4. UI vote vocab `pass` vs DB `down`.
5. Ferry not in `populate_for` walk matrix.
6. `best_photo_index` not applied to fixture `photos_json` reliably.
7. Active 143 vs brief’s 118 = eligibility filter.

## Bootstrap exclusions (`passed_on`)

Exclude unavailable (not dislike): `zillow:119685265` (rented), `zillow:2053018741` (off market). Empty notes dropped. Preference notes kept. Seeded as `reviewer='fixture_seed'`, tagged `bootstrap`.
