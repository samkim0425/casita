---
icon: lucide/play
---

# Getting Started

## CasitaMash (this fork)

Pairwise preference learning on the offline fixture — the main thing to try
first.
The intended path uses **Gemini via Vertex** on a GCP project you create yourself.
You do not need a repo-specific project name.

Before running mash with Gemini:

1. A Google Cloud account and a project you create (Console → New Project, or `gcloud projects create …`). Project ids are usually lowercase with hyphens, e.g. `my-casita-mash-demo`.
2. Billing enabled on that project. Without it, Vertex returns 403 and mash uses the offline stub.
3. Vertex AI API enabled (`aiplatform.googleapis.com`) on that project.
4. Application Default Credentials: `gcloud auth application-default login`
5. Run mash with your project id:

```bash
uv sync
uv run casita mash --project YOUR_GCP_PROJECT
```

Replace `YOUR_GCP_PROJECT` with the project id from step 1.
Open <http://127.0.0.1:8766/mash/>.
On startup you should see `mash preference brain: vertex` (not `stub`).

Flow: name → rank features → compare listings → Current Standings / results.
Comparisons and preference memos are stored only in gitignored `tmp/mash.sqlite`
(never written into `fixtures/demo.sqlite`). Inspect distance anchors with
`uv run casita mash anchors` or <http://127.0.0.1:8766/mash/anchors>.

If `gcloud config get-value project` (or `GOOGLE_CLOUD_PROJECT`) is already set
to your project, `uv run casita mash` after ADC login is enough.
`.env` / `CASITA_GCP_PROJECT` still works for people who already use it for
live Casita commands. Maps, GCS, and Firebase stay optional.

### Offline / no GCP

`uv run casita mash` with no project configured runs a deterministic offline
stub — credentials-free, suitable for CI and smoke tests, not the intended
demo experience.
If you see an "Offline preference stub" banner in the UI, pass `--project` with
your GCP project id.
If Vertex is configured but calls fail (common cause: billing disabled on the
project), the UI falls back to stub and may mention billing; enable billing,
wait a few minutes, and make another pick.

## Static demo site

The upstream Casita path renders the sanitized SQLite fixture as a static
review site:

```bash
uv run playwright install chromium
uv run casita demo
```

Open <http://127.0.0.1:8765/>.

The demo path does not need credentials. It does not scrape, call Vertex, read
GCS, deploy to Firebase, or call the Google Maps Routes API. It does use the
local Playwright Chromium browser to capture Open Graph preview cards from
listing photos and facts.

## Live Runs

Live search uses browser automation and network calls:

```bash
uv run casita solve --help
uv run casita search --headed --local
uv run casita enrich --local
CASITA_FIREBASE_PROJECT=your-project uv run casita publish --local
```

Copy `.env.example` to `.env` for live/private runs. `publish --local` renders
from the local SQLite file, but it still deploys to Firebase; set
`CASITA_FIREBASE_PROJECT` or pass `--project`.

!!! warning "Google Maps cost"

    `search` and `enrich` can eventually call `walk.py`, which uses the paid
    Google Maps Routes API when `GOOGLE_MAPS_API_KEY` is set. The demo and mash
    paths are free: they read cached route rows from `fixtures/demo.sqlite` and
    committed POI anchors. Without a Maps key, live route calculations fall
    back to haversine estimates.
