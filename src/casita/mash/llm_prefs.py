"""Model-native preference loop for CasitaMash.

Boxes this module is meant to check:
  - First reach is the model (Gemini via Vertex).
  - Votes + optional written reasons go in as prose (Casita-style learning loop).
  - Photos and condition are seen/read as media + raw strings — not LIGHT_MAP /
    CONDITION_MAP lookup floats.

Offline / no resolvable GCP project → deterministic stub so demo + CI stay
credentials-free. UI should surface mode=stub vs vertex.
"""

from __future__ import annotations

import os
import subprocess
import textwrap
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv
from pydantic import BaseModel, Field

from .features import (
    FEATURE_LABELS,
    PICKABLE_FEATURES,
    RANKABLE_LOCKED,
    ListingFeatures,
    base_listing_key,
    normalize_feature_order,
)

# Same .env as live Casita LLM commands (optional — not required for mash).
load_dotenv(Path(__file__).resolve().parents[3] / ".env")
load_dotenv()

# Cap how many catalog rows we ask the model to re-order per pick (latency).
_RANK_CAP = 40
_PHOTOS_PER_SIDE = 2


def resolve_gcp_project(explicit: str | None = None) -> str | None:
    """Find a GCP project without forcing a .env copy.

    Order: CLI/explicit → CASITA_GCP_PROJECT → GOOGLE_CLOUD_PROJECT /
    GCLOUD_PROJECT → `gcloud config get-value project`.
    """
    candidates = [
        explicit,
        os.environ.get("CASITA_GCP_PROJECT"),
        os.environ.get("GOOGLE_CLOUD_PROJECT"),
        os.environ.get("GCLOUD_PROJECT"),
    ]
    for raw in candidates:
        val = (raw or "").strip()
        if val:
            return val
    try:
        proc = subprocess.run(
            ["gcloud", "config", "get-value", "project"],
            capture_output=True,
            text=True,
            timeout=3,
            check=False,
        )
        val = (proc.stdout or "").strip()
        if proc.returncode == 0 and val and val.lower() not in {"(unset)", "none"}:
            return val
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        pass
    return None


def apply_gcp_project(explicit: str | None = None) -> str | None:
    """Resolve a project and export CASITA_GCP_PROJECT for the Vertex client."""
    project = resolve_gcp_project(explicit)
    if project:
        os.environ["CASITA_GCP_PROJECT"] = project
    return project


def vertex_available() -> bool:
    return bool(apply_gcp_project())


class ElicitationAsk(BaseModel):
    question: str = Field(
        description=(
            "One plain-English A-vs-B question about an ambiguity still unsettled in the memo. "
            "Direct layman language only — no jargon, feature keys, or 'commute' unless the user "
            "mentioned work commute."
        ),
    )
    choice_a: str = Field(description="Short button label for the first option (max ~8 words).")
    choice_b: str = Field(description="Short button label for the second option (max ~8 words).")


class PreferenceMemoUpdate(BaseModel):
    memo: str = Field(description="Updated preference memo in plain English, 2–6 sentences.")
    bullets: list[str] = Field(
        default_factory=list,
        description="Short bullets of revealed preferences (optional, max ~6).",
    )
    probe_features: list[str] = Field(
        default_factory=list,
        description=(
            "1–3 listing feature keys to probe next (e.g. baths, trail, condition, grocery). "
            "Choose tradeoffs not yet settled in the memo."
        ),
    )
    surprise: str | None = Field(
        default=None,
        description=(
            "Only when this pick clearly contradicts the PREVIOUS memo: one short sentence "
            "naming the conflict (e.g. 'You seemed to prefer brighter places, but this one "
            "looks dimmer.'). Null/omit when consistent, first picks, skips, or uncertain."
        ),
    )
    elicitation: ElicitationAsk | None = Field(
        default=None,
        description=(
            "Only when instructed to ask: one highest-leverage A-vs-B question the memo still "
            "cannot resolve. Null otherwise."
        ),
    )


class RankEntry(BaseModel):
    key: str = Field(description="Listing key exactly as provided.")
    reason: str = Field(
        description="One short sentence citing memo preferences, photos, or condition."
    )


class RankList(BaseModel):
    results: list[RankEntry]


@dataclass
class MemoResult:
    memo_text: str
    bullets: list[str] = field(default_factory=list)
    probe_features: list[str] = field(default_factory=list)
    surprise: str | None = None
    elicitation: dict | None = None  # {question, choice_a, choice_b}
    mode: str = "stub"  # stub | vertex
    error: str | None = None


@dataclass
class RankResult:
    ranks: list[dict]  # {key, reason, score}
    mode: str = "stub"
    error: str | None = None


@dataclass
class WhyResult:
    why_line: str
    mode: str = "stub"  # stub | vertex
    error: str | None = None
    chosen_index: int = 0


class PairWhyLine(BaseModel):
    why_line: str = Field(
        description=(
            "One sentence explaining why these two listings are shown next, "
            "citing the preference memo and a real tradeoff between them."
        ),
    )


class PairPick(BaseModel):
    chosen_index: int = Field(ge=0, description="Index into the candidate shortlist.")
    why_line: str = Field(description="One sentence for the play screen.")


_WHY_MAX_CHARS = 180
_PAIR_PICK_SYSTEM = textwrap.dedent("""
    You choose which pairwise rental comparison to show next and write the one-line
    explanation above it.

    Rules:
      - Pick the candidate that best tests an UNSETTLED preference in the memo.
      - Skip candidates where one side clearly dominates on memo priorities (cheaper
        AND better on condition/light/baths the memo cares about).
      - Return chosen_index matching the candidate list (0-based).
      - why_line: exactly one sentence, plain English, under 180 characters.
      - Voice: "I'm showing you these two because …" or "Confirming your memo: …"
      - Genuine tension → cite the tradeoff using session ranked features only.
      - Confirmatory pick (sanity check) → say so; do not invent a fake tradeoff.
      - Only use facts present in the memo or the candidate briefs.
      - No markdown, bullets, emoji, or quotation-mark wrapping of the whole line.
""").strip()

_WHY_SYSTEM = _PAIR_PICK_SYSTEM


def sanitize_why_line(raw: str | None, *, fallback: str, max_chars: int = _WHY_MAX_CHARS) -> str:
    """Normalize model why-line; return fallback if empty/unusable."""
    import re

    text = (raw or "").strip()
    if not text:
        return fallback
    text = text.replace("**", "").replace("__", "").replace("`", "")
    text = text.strip().strip('"').strip("'").strip()
    text = re.sub(r"\s+", " ", text)
    if text.lower().startswith("as an ai"):
        return fallback
    if len(text) < 12:
        return fallback
    if len(text) > max_chars:
        cut = text[: max_chars - 1].rsplit(" ", 1)[0].rstrip(",.;:")
        text = (cut or text[: max_chars - 1]) + "…"
    return text


_SURPRISE_MAX_CHARS = 220


def sanitize_surprise(raw: str | None, *, max_chars: int = _SURPRISE_MAX_CHARS) -> str | None:
    """Normalize contradiction reason; None if empty/unusable."""
    import re

    text = (raw or "").strip()
    if not text:
        return None
    text = text.replace("**", "").replace("__", "").replace("`", "")
    text = text.strip().strip('"').strip("'").strip()
    text = re.sub(r"\s+", " ", text)
    if text.lower().startswith("as an ai"):
        return None
    if len(text) < 16:
        return None
    # Strip chrome if the model echoed the UI header.
    low = text.lower()
    for prefix in (
        "hold on.. that choice contradicts our preference memo.",
        "hold on... that choice contradicts our preference memo.",
        "hold on. that choice contradicts our preference memo.",
    ):
        if low.startswith(prefix):
            text = text[len(prefix):].strip()
            break
    if len(text) < 16:
        return None
    if len(text) > max_chars:
        cut = text[: max_chars - 1].rsplit(" ", 1)[0].rstrip(",.;:")
        text = (cut or text[: max_chars - 1]) + "…"
    return text


_ELICITATION_Q_MAX = 240
_ELICITATION_CHOICE_MAX = 80


def sanitize_elicitation(raw: dict | None) -> dict | None:
    """Validate model A/B question; None if unusable."""
    import re

    if not raw or not isinstance(raw, dict):
        return None
    q = (raw.get("question") or "").strip()
    a = (raw.get("choice_a") or "").strip()
    b = (raw.get("choice_b") or "").strip()
    for text in (q, a, b):
        if not text or text.lower().startswith("as an ai"):
            return None
    q = re.sub(r"\s+", " ", q.replace("**", "").replace("`", "")).strip()
    a = re.sub(r"\s+", " ", a.replace("**", "").replace("`", "")).strip()
    b = re.sub(r"\s+", " ", b.replace("**", "").replace("`", "")).strip()
    if len(q) < 20 or len(a) < 2 or len(b) < 2:
        return None
    if a.lower() == b.lower():
        return None
    if len(q) > _ELICITATION_Q_MAX:
        q = q[: _ELICITATION_Q_MAX - 1].rsplit(" ", 1)[0] + "…"
    a = a[:_ELICITATION_CHOICE_MAX]
    b = b[:_ELICITATION_CHOICE_MAX]
    return {"question": q, "choice_a": a, "choice_b": b}


def should_ask_elicitation(
    *,
    n_comparisons: int,
    prev_json: dict,
    surprise_this_round: bool,
) -> bool:
    """Gate rare model-initiated A/B questions (no extra API call)."""
    if n_comparisons < 4:
        return False
    if surprise_this_round or prev_json.get("pending_surprise"):
        return False
    if prev_json.get("pending_elicitation"):
        return False
    last = int(prev_json.get("elicitation_last_at_n") or 0)
    if last and (n_comparisons - last) < 8:
        return False
    return True


def _format_candidates_block(candidates: list[dict]) -> str:
    lines = []
    for i, c in enumerate(candidates):
        left = c.get("left") or {}
        right = c.get("right") or {}
        axes = ", ".join(c.get("tradeoff_axes") or []) or "(none)"
        lr = c.get("left_rank")
        rr = c.get("right_rank")
        ranks = f"ranks {lr}/{rr}" if lr and rr else "ranks unknown"
        lines.append(
            f"[{i}] strategy={c.get('strategy')} {ranks} axes={axes}\n"
            f"    Left: {_brief_text(left)}\n"
            f"    Right: {_brief_text(right)}\n"
            f"    template: {(c.get('why_template') or '')[:120]}"
        )
    return "\n\n".join(lines) if lines else "(none)"


def _stub_pick_index(candidates: list[dict]) -> int:
    if not candidates:
        return 0
    best_i = 0
    best_score = -1e18
    for i, c in enumerate(candidates):
        score = float(c.get("heuristic_score") or 0)
        if score > best_score:
            best_score = score
            best_i = i
    return best_i


def pick_pair_from_shortlist(
    *,
    memo_text: str,
    candidates: list[dict],
    feature_order: list[str] | None = None,
    probe_features: list[str] | None = None,
    fallback_why: str,
) -> WhyResult:
    """Pick a candidate pair and write the play why-line."""
    if not candidates:
        fb = (fallback_why or "").strip() or (
            "These listings differ. Your pick here will show us how you trade them off."
        )
        return WhyResult(why_line=fb, mode="stub", chosen_index=0)

    stub_idx = _stub_pick_index(candidates)
    stub_why = (candidates[stub_idx].get("why_template") or fallback_why or "").strip()
    fb = stub_why or (fallback_why or "").strip() or (
        "These listings differ. Your pick here will show us how you trade them off."
    )

    if not (memo_text or "").strip() or not vertex_available():
        return WhyResult(why_line=fb, mode="stub", chosen_index=stub_idx)

    if len(candidates) == 1:
        try:
            return _vertex_explain_pair_why(
                memo_text=memo_text,
                left=candidates[0].get("left") or {},
                right=candidates[0].get("right") or {},
                fallback_why=fb,
                strategy=candidates[0].get("strategy"),
                probe_features=probe_features,
                feature_order=feature_order,
            )
        except Exception as e:
            err = str(e)[:200]
            print(f"  mash pick vertex err: {err}", flush=True)
            return WhyResult(why_line=fb, mode="stub", error=err, chosen_index=0)

    try:
        return _vertex_pick_pair_from_shortlist(
            memo_text=memo_text,
            candidates=candidates,
            feature_order=feature_order,
            probe_features=probe_features,
            fallback_why=fb,
            stub_index=stub_idx,
        )
    except Exception as e:
        err = str(e)[:200]
        print(f"  mash pick vertex err: {err}", flush=True)
        return WhyResult(why_line=fb, mode="stub", error=err, chosen_index=stub_idx)


def _vertex_pick_pair_from_shortlist(
    *,
    memo_text: str,
    candidates: list[dict],
    feature_order: list[str] | None,
    probe_features: list[str] | None,
    fallback_why: str,
    stub_index: int,
) -> WhyResult:
    from google.genai import types as gtypes

    from ..llm import RANK_MODEL, _get_client

    probes = ", ".join(probe_features or []) or "(none)"
    session_vocab = format_session_probe_vocab(feature_order)
    prompt = textwrap.dedent(f"""
        Preference memo:
        {(memo_text or "").strip()}

        Session ranked features (center on these only):
        {session_vocab}

        Probe features from memo: {probes}

        Candidate pairs ({len(candidates)} options — pick ONE by chosen_index):
        {_format_candidates_block(candidates)}

        Fallback why-line if uncertain:
        {fallback_why}

        Return chosen_index and why_line for the best candidate to show next.
    """).strip()

    client = _get_client()
    resp = client.models.generate_content(
        model=RANK_MODEL,
        contents=[gtypes.Content(role="user", parts=[gtypes.Part.from_text(text=prompt)])],
        config=gtypes.GenerateContentConfig(
            temperature=0.2,
            response_mime_type="application/json",
            response_schema=PairPick,
            system_instruction=_PAIR_PICK_SYSTEM,
        ),
    )
    parsed = getattr(resp, "parsed", None)
    if parsed is None:
        return WhyResult(
            why_line=fallback_why, mode="stub", error="empty pick response",
            chosen_index=stub_index,
        )
    idx = int(getattr(parsed, "chosen_index", stub_index) or stub_index)
    idx = max(0, min(idx, len(candidates) - 1))
    raw = (getattr(parsed, "why_line", "") or "").strip()
    cleaned = sanitize_why_line(raw, fallback="")
    if not cleaned:
        cleaned = (candidates[idx].get("why_template") or fallback_why).strip()
    return WhyResult(why_line=cleaned, mode="vertex", chosen_index=idx)


def explain_pair_why(
    *,
    memo_text: str,
    left: dict,
    right: dict,
    fallback_why: str,
    strategy: str | None = None,
    probe_features: list[str] | None = None,
    feature_order: list[str] | None = None,
) -> WhyResult:
    """Model-authored play why-line; always returns a non-empty line via fallback."""
    single = [{
        "left": left,
        "right": right,
        "strategy": strategy or "(none)",
        "tradeoff_axes": list(probe_features or [])[:3],
        "why_template": fallback_why,
        "heuristic_score": 1.0,
    }]
    res = pick_pair_from_shortlist(
        memo_text=memo_text,
        candidates=single,
        feature_order=feature_order,
        probe_features=probe_features,
        fallback_why=fallback_why,
    )
    return WhyResult(why_line=res.why_line, mode=res.mode, error=res.error, chosen_index=0)


def _vertex_explain_pair_why(
    *,
    memo_text: str,
    left: dict,
    right: dict,
    fallback_why: str,
    strategy: str | None,
    probe_features: list[str] | None,
    feature_order: list[str] | None,
) -> WhyResult:
    from google.genai import types as gtypes

    from ..llm import RANK_MODEL, _get_client

    probes = ", ".join(probe_features or []) or "(none)"
    session_vocab = format_session_probe_vocab(feature_order)
    prompt = textwrap.dedent(f"""
        Preference memo:
        {(memo_text or "").strip()}

        Session ranked features (center the why-line on these only):
        {session_vocab}

        Heuristic pair strategy: {strategy or "(none)"}
        Probe features from memo: {probes}
        Template why-line (fallback; improve on this, don't copy verbatim unless perfect):
        {fallback_why}

        Left listing:
        {_brief_text(left)}

        Right listing:
        {_brief_text(right)}

        Write the one-sentence why-line for showing this pair next.
    """).strip()

    client = _get_client()
    resp = client.models.generate_content(
        model=RANK_MODEL,
        contents=[gtypes.Content(role="user", parts=[gtypes.Part.from_text(text=prompt)])],
        config=gtypes.GenerateContentConfig(
            temperature=0.2,
            response_mime_type="application/json",
            response_schema=PairWhyLine,
            system_instruction=_WHY_SYSTEM,
        ),
    )
    parsed = getattr(resp, "parsed", None)
    raw = (getattr(parsed, "why_line", "") or "") if parsed is not None else ""
    cleaned = sanitize_why_line(raw, fallback="")
    if not cleaned:
        return WhyResult(why_line=fallback_why, mode="stub", error="empty or rejected why_line")
    return WhyResult(why_line=cleaned, mode="vertex")


_PROBE_VOCAB = set(PICKABLE_FEATURES) | set(RANKABLE_LOCKED) | {
    "beds", "baths", "sqft", "price", "price_per_bed", "price_per_sqft",
}

_PROBE_KEYWORDS: list[tuple[str, str]] = [
    ("bathroom", "baths"),
    ("bath ", "baths"),
    (" baths", "baths"),
    ("1.5", "baths"),
    ("2 bath", "baths"),
    ("bedroom", "beds"),
    ("beds", "beds"),
    ("trail", "trail"),
    ("hike", "trail"),
    ("hiking", "trail"),
    ("beach", "beach"),
    ("grocery", "grocery"),
    ("whole foods", "premium_grocery"),
    ("premium grocery", "premium_grocery"),
    ("bakery", "bakery"),
    ("bread", "bakery"),
    ("laundry", "laundry"),
    ("in-unit", "laundry"),
    ("parking", "parking"),
    ("garage", "parking"),
    ("yard", "outdoor"),
    ("outdoor", "outdoor"),
    ("dog", "dogs"),
    ("pet", "dogs"),
    ("light", "light"),
    ("sunny", "light"),
    ("bright", "light"),
    ("condition", "condition"),
    ("modern", "condition"),
    ("remodel", "condition"),
    ("dated", "condition"),
    ("classic", "condition"),
    ("period", "condition"),
    ("amenity", "condition"),
    ("roof deck", "view"),
    ("view", "view"),
    ("sqft", "sqft"),
    ("square", "sqft"),
    ("space", "sqft"),
    ("rent", "price"),
    ("price", "price"),
    ("marin", "is_sf"),
    ("san francisco", "is_sf"),
    ("sf ", "is_sf"),
]


def normalize_probe_features(raw: list[str] | None, *, cap: int = 3) -> list[str]:
    out: list[str] = []
    for item in raw or []:
        key = (item or "").strip().lower().replace(" ", "_")
        if key in _PROBE_VOCAB and key not in out:
            out.append(key)
        if len(out) >= cap:
            break
    return out


def allowed_probe_keys(feature_order: list[str] | None) -> set[str]:
    """Probe/pair-steer keys allowed this session: onboarding order + beds/baths."""
    allowed = set(normalize_feature_order(feature_order or []))
    allowed.update({"beds", "baths"})
    return allowed


def clamp_probe_features(
    probes: list[str] | None,
    feature_order: list[str] | None,
    *,
    cap: int = 3,
) -> list[str]:
    """Keep only probes the reviewer ranked (plus beds/baths always on card)."""
    allowed = allowed_probe_keys(feature_order)
    out: list[str] = []
    for item in probes or []:
        key = (item or "").strip().lower().replace(" ", "_")
        if key in _PROBE_VOCAB and key in allowed and key not in out:
            out.append(key)
        if len(out) >= cap:
            break
    return out


def format_session_probe_vocab(feature_order: list[str] | None) -> str:
    """Human-readable allowed probe keys for memo/why prompts."""
    keys = sorted(allowed_probe_keys(feature_order))
    if not keys:
        return "(none)"
    return ", ".join(
        f"{k} ({FEATURE_LABELS.get(k, k.replace('_', ' ').title())})" for k in keys
    )


def probe_features_from_memo(
    memo_text: str,
    bullets: list[str] | None = None,
    *,
    feature_order: list[str] | None = None,
) -> list[str]:
    """Keyword theme parse — stub / fallback when Vertex omits probes."""
    blob = f"{memo_text or ''} " + " ".join(bullets or [])
    blob_l = blob.lower()
    found: list[str] = []
    for needle, feat in _PROBE_KEYWORDS:
        if needle in blob_l and feat in _PROBE_VOCAB and feat not in found:
            found.append(feat)
    return clamp_probe_features(found, feature_order)


def listing_brief(feats: ListingFeatures, *, max_photos: int = 4) -> dict:
    """Prose-friendly brief. Raw quality strings only — no lookup-table floats."""
    price = feats.values.get("price")
    beds = feats.values.get("beds")
    baths = feats.values.get("baths")
    sqft = feats.values.get("sqft")
    photos = list(feats.photos or [])[:max_photos]
    if feats.cover_url and feats.cover_url not in photos:
        photos = [feats.cover_url] + photos
    return {
        "key": feats.key,
        "address": feats.address or feats.neighborhood or feats.key,
        "neighborhood": feats.neighborhood,
        "source": feats.source,
        "price": price,
        "beds": beds,
        "baths": baths,
        "sqft": sqft,
        "light_quality": feats.light_quality or feats.labels.get("light"),
        "condition_quality": feats.condition_quality or feats.labels.get("condition"),
        "view_quality": feats.view_quality or feats.labels.get("view"),
        "visual_summary": feats.visual_summary,
        "outdoor": feats.outdoor_visible or feats.labels.get("outdoor"),
        "laundry": feats.laundry_text or feats.labels.get("laundry"),
        "parking": feats.parking_text or feats.labels.get("parking"),
        "dogs": feats.dogs_text or feats.labels.get("dogs"),
        "description": (feats.description or "")[:200] or None,
        "photo_urls": photos[:max_photos],
        "is_hypothetical": bool(feats.is_hypothetical),
    }


def _brief_text(brief: dict) -> str:
    bits = [
        f"key={brief['key']}",
        brief.get("address") or "",
        f"${brief['price']:.0f}/mo" if brief.get("price") is not None else "",
        (
            f"{brief['beds']:g}bd/{brief['baths']:g}ba"
            if brief.get("beds") is not None and brief.get("baths") is not None
            else ""
        ),
        f"{brief['sqft']:.0f} sqft" if brief.get("sqft") is not None else "",
        f"light={brief['light_quality']}" if brief.get("light_quality") else "",
        f"condition={brief['condition_quality']}" if brief.get("condition_quality") else "",
        f"view={brief['view_quality']}" if brief.get("view_quality") else "",
        f"outdoor={brief['outdoor']}" if brief.get("outdoor") else "",
        f"laundry={brief['laundry']}" if brief.get("laundry") else "",
        f"parking={brief['parking']}" if brief.get("parking") else "",
        f"dogs={brief['dogs']}" if brief.get("dogs") else "",
        f"photos_say={brief['visual_summary']}" if brief.get("visual_summary") else "",
        f"vibe={brief['description']}" if brief.get("description") else "",
    ]
    return " | ".join(b for b in bits if b)


def format_clarification_block(
    clarification: str | None,
    clarification_pair: dict | None = None,
) -> str:
    """Prompt block for surprise/elicitation replies; pair briefs ground listing facts."""
    clar = (clarification or "").strip()
    if not clar:
        return "No pending clarification."
    lines = [
        f'User clarification: "{clar}"',
        (
            "Treat this as preference/tradeoff intent, not ground truth for listing "
            "facts. Reconcile the tradeoff pattern into the memo."
        ),
    ]
    if clarification_pair:
        left = clarification_pair.get("left") or {}
        right = clarification_pair.get("right") or {}
        winner = clarification_pair.get("winner")
        lines.append(
            "Clarification refers to this prior comparison (briefs below are ground "
            "truth for condition, light, amenities, and price — prefer them over "
            "user recall if they conflict):"
        )
        lines.append(f"  Prior left: {_brief_text(left)}")
        lines.append(f"  Prior right: {_brief_text(right)}")
        if winner:
            lines.append(f"  Winner recorded on that pick: {winner}")
    return "\n".join(lines)


def _fetch_one_photo(url: str):
    """Fetch a single listing photo; return a genai Part or None."""
    try:
        import httpx
        from google.genai import types as gtypes
    except ImportError:
        return None
    try:
        r = httpx.get(url, timeout=12, follow_redirects=True)
        if r.status_code != 200 or not r.content:
            return None
        mime = "image/jpeg"
        ct = (r.headers.get("content-type") or "").split(";")[0].strip()
        if ct.startswith("image/"):
            mime = ct
        elif url.lower().endswith(".png"):
            mime = "image/png"
        return gtypes.Part.from_bytes(data=r.content, mime_type=mime)
    except Exception:
        return None


def _fetch_photo_parts(urls: list[str], *, max_n: int = _PHOTOS_PER_SIDE):
    """Inline image parts for Vertex multimodal (parallel URL fetches)."""
    if not urls:
        return []
    capped = list(urls[:max_n])
    if not capped:
        return []
    from concurrent.futures import ThreadPoolExecutor, as_completed

    parts = []
    # Preserve URL order while fetching concurrently.
    by_url: dict[str, object] = {}
    with ThreadPoolExecutor(max_workers=min(4, len(capped))) as pool:
        futs = {pool.submit(_fetch_one_photo, u): u for u in capped}
        for fut in as_completed(futs):
            u = futs[fut]
            try:
                part = fut.result()
            except Exception:
                part = None
            if part is not None:
                by_url[u] = part
    for u in capped:
        if u in by_url:
            parts.append(by_url[u])
    return parts


_MEMO_SYSTEM = textwrap.dedent("""
    You maintain a short preference memo for someone comparing San Francisco /
    Marin rental listings side by side.

    Rules:
      - Update the memo from the latest A/B pick (and optional written reason).
      - Listing briefs and photos are ground truth for facts: condition, light,
        amenities, distances, price, beds, baths. Do not adopt user misstatements
        about those facts.
      - Written reasons, clarifications, and elicitation answers express tradeoff
        intent and priorities — use them to learn what the household cares about,
        not as factual listing descriptions.
      - Prefer evidence from photos and condition/light language over numeric scores.
      - Keep the memo cumulative: preserve prior preferences unless this pick
        clearly revises them.
      - Write in plain English a human can read on a results page.
      - Do not invent facts not present in the briefs or photos.
      - Also return probe_features: 1–3 feature keys to probe NEXT (tradeoffs still
        unsettled). probe_features MUST be chosen only from the session ranked
        feature keys provided in the user prompt — never outdoor, parking, etc.
        unless the user ranked them this session.
      - Optionally set surprise when this pick clearly conflicts with the previous
        memo (not the updated one). One plain sentence, no markdown — e.g. the
        conflict reason only. Use null most of the time; never on first pick or skip.
      - If the user left a clarification about a prior contradiction or answered a
        direct A/B question, reconcile the preference pattern into the memo. When a
        prior-pair brief block is provided, prefer those briefs over user recall for
        listing facts.
      - When asked to return elicitation: one highest-leverage A-vs-B question the memo
        still cannot answer. Plain English, concrete tradeoff (rent vs light, extra bath
        vs walkable grocery, SF vs Marin, etc.). choice_a and choice_b must be short
        button labels. Null elicitation when preferences are already clear.
""").strip()


_RANK_SYSTEM = textwrap.dedent("""
    Rank rental listings for one household using their preference memo.

    Rules:
      - Order best → worst for THIS household.
      - Reasons must cite memo preferences and, when relevant, photo/condition
        language from the briefs — not made-up numeric feature weights.
      - Listing briefs are ground truth for listing facts; the memo captures
        household preference, not user misremembered amenities or condition.
      - Use each listing key exactly as given.
      - Return every listing you were given, once.
""").strip()


def update_preference_memo(
    *,
    prev_memo: str,
    prev_bullets: list[str] | None,
    left: dict,
    right: dict,
    winner: str | None,
    skipped: bool,
    reason: str | None,
    history_lines: list[str] | None = None,
    fetch_photos: bool = True,
    clarification: str | None = None,
    clarification_pair: dict | None = None,
    ask_elicitation: bool = False,
    feature_order: list[str] | None = None,
) -> MemoResult:
    """Model-first memo update. Falls back to stub without Vertex."""
    if not vertex_available():
        return _stub_update_memo(
            prev_memo=prev_memo,
            prev_bullets=prev_bullets or [],
            left=left,
            right=right,
            winner=winner,
            skipped=skipped,
            reason=reason,
            clarification=clarification,
            clarification_pair=clarification_pair,
            ask_elicitation=ask_elicitation,
            feature_order=feature_order,
        )
    try:
        return _vertex_update_memo(
            prev_memo=prev_memo,
            prev_bullets=prev_bullets or [],
            left=left,
            right=right,
            winner=winner,
            skipped=skipped,
            reason=reason,
            history_lines=history_lines or [],
            fetch_photos=fetch_photos,
            clarification=clarification,
            clarification_pair=clarification_pair,
            ask_elicitation=ask_elicitation,
            feature_order=feature_order,
        )
    except Exception as e:
        err = str(e)[:200]
        print(f"  mash memo vertex err: {err}", flush=True)
        stub = _stub_update_memo(
            prev_memo=prev_memo,
            prev_bullets=prev_bullets or [],
            left=left,
            right=right,
            winner=winner,
            skipped=skipped,
            reason=reason,
            clarification=clarification,
            clarification_pair=clarification_pair,
            ask_elicitation=ask_elicitation,
            feature_order=feature_order,
        )
        stub.mode = "stub"
        stub.error = err
        return stub


def rank_from_memo(
    *,
    memo_text: str,
    briefs: list[dict],
    comparisons: list | None = None,
) -> RankResult:
    """Rank catalog from the preference memo. Stub when Vertex is unavailable."""
    if not briefs:
        return RankResult(ranks=[], mode="stub" if not vertex_available() else "vertex")
    if not vertex_available():
        return _stub_rank(memo_text=memo_text, briefs=briefs, comparisons=comparisons or [])
    try:
        return _vertex_rank(memo_text=memo_text, briefs=briefs)
    except Exception as e:
        err = str(e)[:200]
        print(f"  mash rank vertex err: {err}", flush=True)
        out = _stub_rank(memo_text=memo_text, briefs=briefs, comparisons=comparisons or [])
        out.error = err
        return out


def _vertex_update_memo(
    *,
    prev_memo: str,
    prev_bullets: list[str],
    left: dict,
    right: dict,
    winner: str | None,
    skipped: bool,
    reason: str | None,
    history_lines: list[str],
    fetch_photos: bool,
    clarification: str | None = None,
    clarification_pair: dict | None = None,
    ask_elicitation: bool = False,
    feature_order: list[str] | None = None,
) -> MemoResult:
    from google.genai import types as gtypes

    from ..llm import RANK_MODEL, _get_client

    session_vocab = format_session_probe_vocab(feature_order)

    if skipped:
        outcome = f"SKIPPED this pair ({left['key']} vs {right['key']})."
    elif winner:
        outcome = f"CHOSE {winner} over the other listing."
    else:
        outcome = "No winner recorded."
    reason_line = f'Written reason: "{reason.strip()}"' if (reason or "").strip() else "No written reason."
    if (reason or "").strip():
        reason_line += (
            " (Treat as preference intent; Left/Right briefs below are ground truth "
            "for listing facts.)"
        )
    clar_line = format_clarification_block(clarification, clarification_pair)
    elicit_line = (
        "Also return elicitation: one direct A-vs-B question about the biggest remaining "
        "ambiguity in the memo (plain English, short button labels)."
        if ask_elicitation
        else "Do not return elicitation (set null)."
    )
    hist = "\n".join(f"- {line}" for line in history_lines[-20:]) or "(none yet)"
    bullets = "\n".join(f"- {b}" for b in prev_bullets) or "(none)"
    prompt = textwrap.dedent(f"""
        Previous memo:
        {prev_memo.strip() or "(empty — first pick)"}

        Previous bullets:
        {bullets}

        Recent comparison history:
        {hist}

        Left listing:
        {_brief_text(left)}

        Right listing:
        {_brief_text(right)}

        Latest outcome: {outcome}
        {reason_line}
        {clar_line}

        Session ranked features (probe_features MUST be chosen only from these keys):
        {session_vocab}

        Photos for each side follow when available. Update the preference memo
        and choose probe_features for the next comparison. Set surprise only if
        this pick clearly contradicts the previous memo. {elicit_line}
    """).strip()

    parts: list = [gtypes.Part.from_text(text=prompt)]
    if fetch_photos:
        from concurrent.futures import ThreadPoolExecutor

        with ThreadPoolExecutor(max_workers=2) as pool:
            left_f = pool.submit(_fetch_photo_parts, left.get("photo_urls") or [])
            right_f = pool.submit(_fetch_photo_parts, right.get("photo_urls") or [])
            left_imgs = left_f.result()
            right_imgs = right_f.result()
        if left_imgs:
            parts.append(gtypes.Part.from_text(text="Left listing photos:"))
            parts.extend(left_imgs)
        if right_imgs:
            parts.append(gtypes.Part.from_text(text="Right listing photos:"))
            parts.extend(right_imgs)

    client = _get_client()
    resp = client.models.generate_content(
        model=RANK_MODEL,
        contents=[gtypes.Content(role="user", parts=parts)],
        config=gtypes.GenerateContentConfig(
            temperature=0.2,
            response_mime_type="application/json",
            response_schema=PreferenceMemoUpdate,
            system_instruction=_MEMO_SYSTEM,
        ),
    )
    parsed = PreferenceMemoUpdate.model_validate_json((resp.text or "").strip() or "{}")
    probes = clamp_probe_features(list(parsed.probe_features or []), feature_order)
    if not probes:
        probes = probe_features_from_memo(
            parsed.memo or "", list(parsed.bullets or []), feature_order=feature_order,
        )
    surprise = None
    if (prev_memo or "").strip() and not skipped:
        surprise = sanitize_surprise(getattr(parsed, "surprise", None))
    elicitation = None
    if ask_elicitation and not surprise:
        raw_el = parsed.elicitation
        if raw_el is not None:
            elicitation = sanitize_elicitation({
                "question": raw_el.question,
                "choice_a": raw_el.choice_a,
                "choice_b": raw_el.choice_b,
            })
    return MemoResult(
        memo_text=(parsed.memo or "").strip(),
        bullets=list(parsed.bullets or [])[:8],
        probe_features=probes,
        surprise=surprise,
        elicitation=elicitation,
        mode="vertex",
    )


def _vertex_rank(*, memo_text: str, briefs: list[dict]) -> RankResult:
    from ..llm import RANK_MODEL, _call_structured

    capped = briefs[:_RANK_CAP]
    catalog = "\n".join(f"- {_brief_text(b)}" for b in capped)
    prompt = textwrap.dedent(f"""
        Preference memo:
        {memo_text.strip() or "(empty)"}

        Rank these listings best → worst. Return one entry per key.

        Listings:
        {catalog}
    """).strip()
    parsed = _call_structured(RANK_MODEL, _RANK_SYSTEM, prompt, RankList)
    if parsed is None:
        out = _stub_rank(memo_text=memo_text, briefs=briefs, comparisons=[])
        out.error = "Vertex rank call returned no structured result (check terminal / billing)."
        return out
    by_key = {b["key"]: b for b in capped}
    ordered: list[dict] = []
    seen: set[str] = set()
    n = len(capped)
    for i, entry in enumerate(parsed.results):
        if entry.key not in by_key or entry.key in seen:
            continue
        seen.add(entry.key)
        ordered.append({
            "key": entry.key,
            "reason": entry.reason,
            "score": float(n - i),
        })
    for b in capped:
        if b["key"] not in seen:
            ordered.append({
                "key": b["key"],
                "reason": "Not explicitly ranked; appended after model output.",
                "score": 0.0,
            })
    # Append any briefs beyond the cap with lower scores.
    for i, b in enumerate(briefs[_RANK_CAP:]):
        ordered.append({
            "key": b["key"],
            "reason": "Outside model rank window; ordered by rent as fallback.",
            "score": -1.0 - i - (b.get("price") or 0) / 1e6,
        })
    return RankResult(ranks=ordered, mode="vertex")


def _stub_surprise(
    *,
    prev_memo: str,
    left: dict,
    right: dict,
    winner: str | None,
    skipped: bool,
) -> str | None:
    """Rare deterministic contradiction for offline demo."""
    if skipped or not winner or not (prev_memo or "").strip():
        return None
    memo_l = prev_memo.lower()
    win = left if winner == left.get("key") else right if winner == right.get("key") else None
    lose = right if win is left else left if win is right else None
    if not win or not lose:
        return None
    light_rank = {"dim": 0, "moderate": 1, "abundant": 2}
    cond_rank = {"dated": 0, "classic": 1, "well-kept": 2, "high-end": 3}
    wl = light_rank.get((win.get("light_quality") or "").lower())
    ll = light_rank.get((lose.get("light_quality") or "").lower())
    if (
        wl is not None and ll is not None and wl < ll
        and any(k in memo_l for k in ("light", "bright", "sunny", "abundant"))
    ):
        return "You seemed to prefer brighter places, but this one looks dimmer."
    wc = cond_rank.get((win.get("condition_quality") or "").lower())
    lc = cond_rank.get((lose.get("condition_quality") or "").lower())
    if (
        wc is not None and lc is not None and wc < lc
        and any(k in memo_l for k in ("condition", "modern", "remodel", "finish", "well-kept", "high-end"))
    ):
        return "You seemed to prefer nicer finishes, but this pick looks more dated."
    return None


def _stub_elicitation(probe_features: list[str]) -> dict | None:
    """Offline A/B question from probe themes."""
    keys = [p for p in (probe_features or []) if p][:2]
    if len(keys) < 2:
        return None
    a = FEATURE_LABELS.get(keys[0], keys[0].replace("_", " ").title())
    b = FEATURE_LABELS.get(keys[1], keys[1].replace("_", " ").title())
    return sanitize_elicitation({
        "question": f"If rent is similar, which matters more to you: {a.lower()} or {b.lower()}?",
        "choice_a": a,
        "choice_b": b,
    })


def _stub_update_memo(
    *,
    prev_memo: str,
    prev_bullets: list[str],
    left: dict,
    right: dict,
    winner: str | None,
    skipped: bool,
    reason: str | None,
    clarification: str | None = None,
    clarification_pair: dict | None = None,
    ask_elicitation: bool = False,
    feature_order: list[str] | None = None,
) -> MemoResult:
    left_label = left.get("address") or left["key"]
    right_label = right.get("address") or right["key"]
    reason_bit = f" Reason: {reason.strip()}." if (reason or "").strip() else ""
    if skipped:
        line = f"Skipped comparing {left_label} vs {right_label}.{reason_bit}"
    elif winner == left.get("key"):
        line = f"Preferred {left_label} over {right_label}.{reason_bit}"
    elif winner == right.get("key"):
        line = f"Preferred {right_label} over {left_label}.{reason_bit}"
    else:
        line = f"Recorded a comparison between {left_label} and {right_label}.{reason_bit}"

    # Pull raw condition/light from the winner when present.
    win_brief = left if winner == left.get("key") else right if winner == right.get("key") else None
    evidence = []
    if win_brief:
        if win_brief.get("condition_quality"):
            evidence.append(f"condition {win_brief['condition_quality']}")
        if win_brief.get("light_quality"):
            evidence.append(f"light {win_brief['light_quality']}")
        if win_brief.get("visual_summary"):
            evidence.append(win_brief["visual_summary"][:120])
    if evidence:
        line += " Signals: " + "; ".join(evidence) + "."

    clar = (clarification or "").strip()
    if clar:
        line += f" User note: {clar}."
        if clarification_pair:
            for label, brief in (
                ("prior left", clarification_pair.get("left") or {}),
                ("prior right", clarification_pair.get("right") or {}),
            ):
                cond = brief.get("condition_quality")
                light = brief.get("light_quality")
                bits = []
                if cond:
                    bits.append(f"condition {cond}")
                if light:
                    bits.append(f"light {light}")
                if bits:
                    line += f" ({label} brief: {'; '.join(bits)})"

    prior = (prev_memo or "").strip()
    memo = f"{prior} {line}".strip() if prior else line
    # Keep stub memo bounded.
    if len(memo) > 1200:
        memo = memo[-1200:]
    bullets = list(prev_bullets)
    if clar and clar not in bullets:
        bullets.append(clar)
    if (reason or "").strip() and reason.strip() not in bullets:
        bullets.append(reason.strip())
    elif evidence and evidence[0] not in bullets:
        bullets.append(evidence[0])
    probes = probe_features_from_memo(memo, bullets, feature_order=feature_order)
    surprise = _stub_surprise(
        prev_memo=prev_memo, left=left, right=right, winner=winner, skipped=skipped,
    )
    elicitation = None
    if ask_elicitation and not surprise:
        elicitation = _stub_elicitation(probes)
    return MemoResult(
        memo_text=memo,
        bullets=bullets[:8],
        probe_features=probes,
        surprise=surprise,
        elicitation=elicitation,
        mode="stub",
    )


def _stub_rank(
    *,
    memo_text: str,
    briefs: list[dict],
    comparisons: list,
) -> RankResult:
    """Deterministic offline rank: win counts from picks, then cheaper rent."""
    wins: dict[str, float] = {}
    for r in comparisons:
        if getattr(r, "keys", None) and callable(r.keys):
            # sqlite Row
            skipped = r["skipped"]
            winner = r["winner"]
            left = r["left_key"]
            right = r["right_key"]
        else:
            skipped = r.get("skipped") if isinstance(r, dict) else 0
            winner = r.get("winner") if isinstance(r, dict) else None
            left = r.get("left_key") if isinstance(r, dict) else None
            right = r.get("right_key") if isinstance(r, dict) else None
        if skipped or not winner:
            continue
        base = base_listing_key(str(winner))
        wins[base] = wins.get(base, 0.0) + 1.0
        # slight penalty for loser so undefeated rise
        loser = right if winner == left else left
        if loser:
            lb = base_listing_key(str(loser))
            wins[lb] = wins.get(lb, 0.0) - 0.15

    memo_l = (memo_text or "").lower()
    ranked = []
    for b in briefs:
        key = b["key"]
        base = base_listing_key(key)
        score = wins.get(base, 0.0)
        price = b.get("price")
        if price is not None:
            score -= float(price) / 100_000.0
        # Soft boosts from memo keywords vs raw strings (still not lookup floats).
        cond = (b.get("condition_quality") or "").lower()
        light = (b.get("light_quality") or "").lower()
        if "light" in memo_l and light in ("abundant", "moderate"):
            score += 0.3 if light == "abundant" else 0.1
        if "condition" in memo_l or "finish" in memo_l or "remodel" in memo_l:
            if cond in ("high-end", "well-kept"):
                score += 0.25
        reason = "Stub rank from your picks"
        if wins.get(base, 0) > 0:
            reason += f" ({int(wins[base])} direct wins)"
        if b.get("condition_quality") or b.get("light_quality"):
            bits = [x for x in (b.get("condition_quality"), b.get("light_quality")) if x]
            reason += "; " + ", ".join(bits)
        if price is not None:
            reason += f"; ${price:,.0f}/mo"
        reason += " (offline stub — run: uv run casita mash --project YOUR_GCP_PROJECT)."
        ranked.append({"key": key, "reason": reason, "score": score})
    ranked.sort(key=lambda row: row["score"], reverse=True)
    return RankResult(ranks=ranked, mode="stub")


def movers_from_memo(memo_text: str, bullets: list[str] | None = None) -> list[dict]:
    """Preference movers for results UI from memo bullets and phrases."""
    out: list[dict] = []
    for b in bullets or []:
        if b and b.strip():
            out.append({"feature": b.strip(), "label": b.strip(), "share": 0.0})
    if out:
        # Fake equal shares so results page can still render percentages gently.
        share = 1.0 / len(out)
        for row in out:
            row["share"] = share
        return out[:6]
    text = (memo_text or "").strip()
    if not text:
        return []
    # Split on sentence boundaries for a coarse stand-in.
    parts = [p.strip() for p in text.replace("!", ".").split(".") if p.strip()]
    share = 1.0 / max(1, min(4, len(parts)))
    for p in parts[:4]:
        out.append({"feature": p, "label": p, "share": share})
    return out
