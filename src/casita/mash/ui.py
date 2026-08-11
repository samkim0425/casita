from __future__ import annotations

import html
import json
import re

from .features import (
    ALWAYS_SHOW_COPY,
    FEATURE_LABELS,
    PICKABLE_FEATURES,
    RANKABLE_LOCKED,
    ListingFeatures,
    card_feature_order,
)


def _e(s) -> str:
    return html.escape("" if s is None else str(s))


# Lucide-style external-link glyph (box + arrow)
EXT_ICON = (
    '<svg class="ext-icon" viewBox="0 0 24 24" aria-hidden="true" focusable="false" '
    'fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round">'
    '<path d="M18 13v6a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2h6"/>'
    '<polyline points="15 3 21 3 21 9"/>'
    '<line x1="10" y1="14" x2="21" y2="3"/>'
    "</svg>"
)


def _external_a(url: str, label: str, *, css_class: str = "") -> str:
    cls = f' class="{_e(css_class)}"' if css_class else ""
    return (
        f'<a{cls} href="{_e(url)}" data-external-url="{_e(url)}" '
        f'data-external-label="{_e(label)}">{_e(label)}{EXT_ICON}</a>'
    )


def display_text(s: str | None) -> str:
    if s is None or not str(s).strip():
        return "—"
    parts = re.split(r"[-_\s]+", str(s).strip())
    return " ".join(p[:1].upper() + p[1:].lower() for p in parts if p)


BASE_CSS = """
:root{--mash-crimson:#920004}
*{box-sizing:border-box}
html,body{margin:0;padding:0;background:#ffffff;color:#222222;
  font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Helvetica,Arial,sans-serif}
a{color:var(--mash-crimson)}
header.mash{
  background:#920004;color:#f6f1ea;padding:14px 18px;
  display:flex;align-items:baseline;justify-content:space-between;gap:12px;
  border-bottom:3px solid #5c0002}
header.mash h1{margin:0;font-size:22px;letter-spacing:0.02em;font-weight:700}
header.mash .tag{opacity:0.85;font-size:13px}
header.mash .who{font-size:12px;opacity:0.8}
main{max-width:1100px;margin:0 auto;padding:18px}
.btn{
  display:inline-block;background:var(--mash-crimson);color:#f6f1ea;border:2px solid #5c0002;
  padding:10px 16px;font-size:14px;font-weight:700;cursor:pointer;text-decoration:none}
.btn.secondary{background:#ffffff;color:var(--mash-crimson);border:1px solid var(--mash-crimson)}
.btn.secondary:hover{background:var(--mash-crimson);color:#ffffff}
.btn:disabled{opacity:0.4;cursor:not-allowed}
.panel{background:#ffffff;border:1px solid #cccccc;padding:18px;margin:14px 0}
input[type=text]{width:100%;max-width:360px;padding:10px;border:2px solid #1a1a1a;font-size:16px}
.muted{color:#666;font-size:13px}
.banner{
  background:#fff3cd;border:2px solid #920004;padding:12px 14px;margin:12px 0;
  display:flex;gap:12px;align-items:center;justify-content:space-between;flex-wrap:wrap}
.warn-box{
  background:#ffffff;border:1px solid #cccccc;padding:12px 14px;margin:12px 0;
  font-size:13px;font-weight:700;line-height:1.4;color:#222222}
.stub-banner{
  background:#f5f5f5;border:1px solid #999;padding:10px 12px;margin:10px 0;
  font-size:13px;line-height:1.4;color:#333}
.memo-box{
  background:#fafafa;border:1px solid #cccccc;padding:12px 14px;margin:12px 0;
  font-size:14px;line-height:1.45;white-space:pre-wrap}
.memo-box h3{margin:0 0 8px;font-size:14px;display:flex;align-items:baseline;gap:8px;flex-wrap:wrap}
.updating{font-weight:600;color:#666;font-size:12px;letter-spacing:0.02em}
.updating[hidden]{display:none !important}
.reason-ov .overlay-inner{
  max-width:440px;padding:22px 22px 18px;text-align:left}
.reason-ov h3{margin:0 0 6px;font-size:16px}
.reason-ov .muted{margin:0 0 12px}
.reason-ov input[type=text]{
  width:100%;padding:10px 12px;border:2px solid #1a1a1a;font-size:15px;
  margin:0 0 14px}
.reason-ov .reason-actions{display:flex;gap:10px;flex-wrap:wrap;justify-content:flex-end}
.elicit-actions{display:flex;flex-direction:row;gap:10px;margin-top:4px}
.elicit-actions .btn{
  flex:1;min-width:0;text-align:center;white-space:normal;line-height:1.35;padding:12px 14px}
.rank-reason{font-size:12px;color:#555;margin-top:2px;font-weight:400}
.why{font-size:14px;margin:8px 0 14px;font-weight:600}
.probe-row{font-size:13px;color:#666;margin:-6px 0 14px;line-height:1.4}
.standings-note{font-size:13px;color:#666;margin:0 0 12px}
.standings-head{
  display:flex;align-items:baseline;justify-content:flex-start;gap:12px;
  flex-wrap:wrap;margin:0 0 14px}
.standings-head h2{margin:0}
.standings-head .muted{margin:0}
.grid{display:grid;grid-template-columns:1fr 40px 1fr;gap:12px;align-items:start}
@media(max-width:800px){.grid{grid-template-columns:1fr;}}
.vs{text-align:center;font-weight:800;font-size:22px;padding-top:120px;color:#920004}
.card{
  border:2px solid #1a1a1a;background:#fff;position:relative}
.card.hyp{
  border:3px dashed #1a1a1a;
  background:#f3f3f3;
  box-shadow: inset 0 0 0 3px #1a1a1a;
}
.card.hyp .cover{filter:grayscale(0.35)}
.hyp-banner{
  background:#1a1a1a;color:#f6f1ea;
  font-size:13px;font-weight:800;letter-spacing:0.04em;
  text-transform:uppercase;padding:10px 12px;text-align:center}
.hyp-note{font-size:12px;font-weight:600;color:#1a1a1a;padding:8px 12px 0;
  text-transform:none;letter-spacing:0}
.hyp-page{
  background:#1a1a1a;color:#f6f1ea;border:3px solid #920004;
  padding:12px 14px;margin:0 0 12px;font-weight:700;font-size:14px}
.cover{width:100%;aspect-ratio:4/3;object-fit:cover;display:block;background:#ddd}
.cover-empty{
  width:100%;aspect-ratio:4/3;display:flex;align-items:center;justify-content:center;
  background:#e8e2da;color:#666;font-size:14px;font-weight:700;text-align:center;
  padding:16px;border-bottom:2px solid #1a1a1a}
.cover-wrap{position:relative;cursor:pointer}
.cover-wrap.no-photos{cursor:default}
.cover-wrap:hover .cover{filter:brightness(0.85)}
.photo-btn{
  position:absolute;right:10px;bottom:10px;z-index:2;
  background:#920004;color:#f6f1ea;border:2px solid #5c0002;
  padding:6px 10px;font-size:12px;font-weight:700;cursor:pointer;
  filter:brightness(1);transition:filter 0.12s ease;pointer-events:none}
.cover-wrap:hover .photo-btn{filter:brightness(1.18)}
.ov-empty{padding:40px 16px;text-align:center;color:#666;font-weight:600}
.body{padding:10px 12px 14px;cursor:pointer}
.body:hover:not(:has(a:hover)){outline:3px solid #920004;outline-offset:-3px}
.row a{font-weight:700;text-decoration:underline;text-underline-offset:2px}
.row a:hover{color:#5c0002}
table.rank a{font-weight:700;text-decoration:underline;text-underline-offset:2px}
.ext-icon{
  display:inline-block;width:0.9em;height:0.9em;margin-left:0.28em;
  vertical-align:-0.08em;flex-shrink:0}
.addr{font-size:16px;font-weight:800;line-height:1.25;margin:0 0 10px;padding-bottom:8px;border-bottom:2px solid #1a1a1a}
.row{display:grid;grid-template-columns:140px 1fr;gap:6px;font-size:13px;padding:3px 0;border-bottom:1px solid #eee}
.row .k{color:#666}
.row .v{font-weight:600}
.badge{display:inline-block;background:#920004;color:#fff;font-size:11px;padding:2px 6px;margin-left:6px}
.badge.soft{background:#888}
.actions{display:flex;gap:10px;margin-top:14px;flex-wrap:wrap}
.actions.end{justify-content:flex-end}
.feat-head{
  display:flex;align-items:center;justify-content:space-between;gap:12px;
  flex-wrap:wrap;margin:0 0 10px}
.feat-head h2{margin:0;flex:1;min-width:0}
.feat-list{list-style:none;padding:0;margin:0}
.feat-list > li{
  display:grid;grid-template-columns:28px 1fr;gap:8px;align-items:center;
  margin:0;padding:0;border:none;background:transparent;cursor:default}
.feat-list > li .feat-row{
  display:flex;align-items:center;gap:10px;padding:10px 12px;
  border:1px solid #cccccc;border-radius:0;background:#ffffff;min-height:48px;
  margin-top:-1px}
.feat-list > li:first-child .feat-row{margin-top:0}
.feat-list > li:nth-child(even) .feat-row{background:#f7f7f7}
.feat-label{font-weight:700;font-size:15px;flex:1;min-width:0;color:#222222}
.feat-actions{display:flex;align-items:center;gap:8px;margin-left:auto;flex-shrink:0}
.feat-chevs{display:flex;gap:6px}
.feat-chev{
  width:32px;height:32px;border-radius:0;border:1px solid #cccccc;
  background:#ffffff;color:#222222;cursor:pointer;padding:0;
  display:inline-flex;align-items:center;justify-content:center;
  font-size:12px;line-height:1;font-weight:800}
.feat-chev:hover:not(:disabled){border-color:var(--mash-crimson);color:var(--mash-crimson)}
.feat-chev:disabled{color:#dddddd;border-color:#dddddd;cursor:not-allowed}
.feat-section-label{
  font-size:12px;font-weight:700;margin:18px 0 8px;color:#444444;
  letter-spacing:0.05em}
.feat-list.available{min-height:56px;padding:0;border:none;background:transparent}
.feat-empty{font-size:13px;color:#666666;padding:10px;margin:0}
.feat-list .tog,.feat-lock{
  display:inline-block;font-size:12px;padding:6px 10px;border-radius:0;
  background:#ffffff;font-weight:700;white-space:nowrap;line-height:1.25;
  box-sizing:border-box}
.feat-list .tog{
  cursor:pointer;color:var(--mash-crimson);border:1px solid var(--mash-crimson)}
.feat-list .tog:hover{background:var(--mash-crimson);color:#ffffff}
.feat-lock{
  color:#666666;border:1px solid #cccccc}
.rank-num{width:24px;font-weight:800;color:var(--mash-crimson);text-align:right}
table.rates,table.rank{width:100%;border-collapse:collapse;font-size:13px}
table.rates th,table.rates td,table.rank th,table.rank td{
  border:1px solid #1a1a1a;padding:8px;text-align:left}
.podium{
  display:grid;grid-template-columns:1fr 1.2fr 1fr;gap:28px;align-items:end;
  margin:32px auto 40px;max-width:680px}
@media(max-width:700px){.podium{grid-template-columns:1fr;max-width:280px;gap:20px}}
.podium-card{
  border:2px solid #1a1a1a;background:#fff;text-align:center;overflow:hidden;
  display:block;color:inherit;text-decoration:none;cursor:pointer}
a.podium-card:hover{outline:3px solid #920004;outline-offset:2px}
.podium-card.gold{border-color:#b8860b;box-shadow:0 0 0 3px #f0d78c}
.podium-card.silver{border-color:#6b6b6b}
.podium-card.bronze{border-color:#8b5a2b}
.podium-card img,.podium-card .cover-ph{
  width:100%;height:140px;object-fit:cover;display:block;background:#ddd}
.podium-card.gold img,.podium-card.gold .cover-ph{height:160px}
.podium-card .place{
  font-size:12px;font-weight:800;letter-spacing:0.06em;text-transform:uppercase;
  padding:8px 8px 0}
.podium-card.gold .place{color:#8a6d00}
.podium-card.silver .place{color:#555}
.podium-card.bronze .place{color:#6b4423}
.podium-card .name{font-size:14px;font-weight:800;padding:6px 10px 4px;line-height:1.25}
.podium-card .meta{font-size:12px;color:#666;padding:0 10px 12px}
.overlay{
  display:none;position:fixed;inset:0;background:rgba(0,0,0,0.72);z-index:50;
  align-items:center;justify-content:center;padding:20px}
.overlay.open{display:flex}
.overlay-inner{
  position:relative;background:#fff;border:3px solid #1a1a1a;
  max-width:900px;width:100%;max-height:90vh;overflow:auto;padding:44px 12px 12px}
.ov-close{
  position:absolute;top:8px;right:8px;z-index:2;
  width:36px;height:36px;border:1px solid #cccccc;border-radius:0;
  background:#ffffff;color:#222222;font-size:24px;line-height:1;
  cursor:pointer;padding:0;
  display:inline-flex;align-items:center;justify-content:center}
.ov-close:hover{border-color:#920004;color:#920004}
.busy{
  display:none;position:fixed;inset:0;z-index:100;
  background:rgba(26,26,26,0.55);
  align-items:center;justify-content:center;
  pointer-events:all;cursor:wait}
.busy.open{display:flex}
.busy-spinner{
  width:44px;height:44px;border-radius:50%;
  border:4px solid rgba(246,241,234,0.35);
  border-top-color:#f6f1ea;
  animation:busy-spin 0.75s linear infinite}
@keyframes busy-spin{to{transform:rotate(360deg)}}
.leave-inner{max-width:420px;padding:22px}
.leave-inner .ov-close{top:10px;right:10px}
.leave-inner h3{margin:0 0 10px;font-size:18px}
.leave-inner p{margin:0 0 10px;font-size:14px;line-height:1.45}
.thumbs{display:flex;gap:6px;overflow-x:auto;margin-top:8px}
.thumbs img{height:64px;width:86px;object-fit:cover;border:2px solid #1a1a1a;cursor:pointer}
"""


def page(title: str, body: str, who: str = "") -> str:
    who_html = ""
    if who:
        who_html = (
            f'<div class="who">{_e(who)}'
            f' · <a href="/mash/api/logout" style="color:#f6f1ea">Sign out</a></div>'
        )
    return f"""<!doctype html>
<html lang="en"><head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>{_e(title)}</title>
<link rel="icon" href="/assets/favicon.svg" type="image/svg+xml">
<style>{BASE_CSS}</style>
</head><body>
<header class="mash">
  <div>
    <h1>CasitaMash</h1>
    <div class="tag">The hottest listings in the Bay.</div>
  </div>
  {who_html}
</header>
<main>{body}</main>
<div class="overlay" id="leaveOv"><div class="overlay-inner leave-inner">
  <h3>Leaving CasitaMash</h3>
  <p>You're being taken to <strong id="leaveWhere"></strong>.</p>
  <p class="muted">The listing might not be available anymore or taken off the market.</p>
  <div class="actions" style="margin-top:16px">
    <button type="button" class="btn" id="leaveGo">Continue</button>
    <button type="button" class="btn secondary" id="leaveCancel">Cancel</button>
  </div>
</div></div>
<script>
(function() {{
  let leaveUrl = null;
  const ov = document.getElementById('leaveOv');
  function hostLabel(url) {{
    try {{ return new URL(url).hostname.replace(/^www\\./, ''); }}
    catch (e) {{ return url; }}
  }}
  window.openLeaveConfirm = function(url, label) {{
    leaveUrl = url;
    document.getElementById('leaveWhere').textContent = label || hostLabel(url);
    ov.classList.add('open');
  }};
  document.getElementById('leaveCancel').onclick = () => {{ leaveUrl = null; ov.classList.remove('open'); }};
  document.getElementById('leaveGo').onclick = () => {{
    if (!leaveUrl) return;
    const u = leaveUrl; leaveUrl = null; ov.classList.remove('open');
    window.open(u, '_blank', 'noopener');
  }};
  ov.addEventListener('click', (e) => {{ if (e.target === ov) document.getElementById('leaveCancel').click(); }});
  document.addEventListener('keydown', (e) => {{
    if (e.key === 'Escape' && ov.classList.contains('open')) document.getElementById('leaveCancel').click();
  }});
  document.addEventListener('click', (e) => {{
    const a = e.target.closest('[data-external-url]');
    if (!a) return;
    e.preventDefault();
    e.stopPropagation();
    openLeaveConfirm(a.getAttribute('data-external-url'), a.getAttribute('data-external-label') || a.textContent.trim());
  }}, true);
}})();
</script>
</body></html>"""


def landing(existing: dict[str, int] | None = None) -> str:
    hint = ""
    if existing:
        rows = "".join(
            f"<option value=\"{_e(n)}\">{_e(n)} — {c} comparisons</option>"
            for n, c in sorted(existing.items())
        )
        hint = f'<p class="muted">All reviewers:</p><select id="known">{rows}<option value="">—</option></select>'
    body = f"""
<div class="panel">
  <h2>Who's choosing?</h2>
  <p class="muted">Add your name to start or resume a session.</p>
  {hint}
  <p><input id="name" type="text" placeholder="your name" autocomplete="username"></p>
  <p id="resume" class="muted"></p>
  <button class="btn" id="go">Continue →</button>
  <p class="muted" style="margin-top:16px">
    <a href="/mash/anchors">Inspect route anchors</a>
    (beaches, bakeries, trails, groceries.. what we measure distance against)
  </p>
</div>
<div class="busy" id="busy" aria-hidden="true" aria-busy="false">
  <div class="busy-spinner" role="status" aria-label="Loading"></div>
</div>
<script>
const known = document.getElementById('known');
const name = document.getElementById('name');
const resume = document.getElementById('resume');
const go = document.getElementById('go');
function showBusy() {{
  const el = document.getElementById('busy');
  el.classList.add('open');
  el.setAttribute('aria-busy', 'true');
  el.setAttribute('aria-hidden', 'false');
  document.body.style.overflow = 'hidden';
  go.disabled = true;
}}
if (known) known.onchange = () => {{ if (known.value) name.value = known.value; }};
name.oninput = async () => {{
  const r = await fetch('/mash/api/reviewer?name=' + encodeURIComponent(name.value.trim()));
  const j = await r.json();
  if (j.exists) resume.textContent = "You've done " + j.count + " comparisons so far. Continue?";
  else resume.textContent = j.name ? "Welcome! Find your next home." : "";
}};
go.onclick = async () => {{
  const n = name.value.trim();
  if (!n) return;
  showBusy();
  try {{
    const r = await fetch('/mash/api/reviewer', {{method:'POST', headers:{{'Content-Type':'application/json'}},
      body: JSON.stringify({{name:n}})}});
    const j = await r.json();
    location.href = j.next || '/mash/features';
  }} catch (e) {{
    document.getElementById('busy').classList.remove('open');
    go.disabled = false;
    document.body.style.overflow = '';
    resume.textContent = "Couldn't start the session. Try again.";
  }}
}};
name.addEventListener('keydown', (e) => {{
  if (e.key === 'Enter') go.click();
}});
</script>"""
    return page("CasitaMash", body)


def anchors_page(groups: list[dict]) -> str:
    sections = []
    for g in groups:
        rows = "".join(
            f"<tr><td>{_e(item['name'])}</td>"
            f"<td>{item['lat']:.4f}</td><td>{item['lng']:.4f}</td></tr>"
            if item.get("lat") is not None and item.get("lng") is not None
            else f"<tr><td>{_e(item['name'])}</td><td>—</td><td>—</td></tr>"
            for item in g["items"]
        )
        sections.append(
            f'<h3>{_e(g["title"])} <span class="muted">({len(g["items"])})</span></h3>'
            f'<p class="standings-note">{_e(g["source"])}</p>'
            f'<table class="rates"><tr><th>Name</th><th>Lat</th><th>Lng</th></tr>{rows}</table>'
        )
    body = f"""
<div class="actions" style="margin:0 0 14px">
  <a class="btn" href="/mash/">← Back</a>
</div>
<div class="panel">
  <h2>Route anchors</h2>
  <p class="muted">These are the places CasitaMash measures distance against.
  Curated lists live in <code>walk.py</code>; grocery/bar/market come from committed
  <code>fixtures/poi_anchors.json</code>. No Google Maps calls.</p>
  {"".join(sections)}
</div>"""
    return page("Route anchors — CasitaMash", body)


def features_page(reviewer: str, order: list[str], estimate: tuple[int, int]) -> str:
    locked = list(RANKABLE_LOCKED)
    order = [f for f in order if f in PICKABLE_FEATURES or f in locked]
    for f in locked:
        if f not in order:
            order.append(f)
    selected = list(order)
    body = f"""
<div class="panel">
  <div class="feat-head">
    <h2>What features do you care most about?</h2>
    <button class="btn" id="start">Start comparing →</button>
  </div>
  <p class="muted">{ALWAYS_SHOW_COPY}</p>
  <div class="warn-box">Warning: these picks cannot be changed for this session. If you want to compare across different features, sign out and start a new session.</div>
  <p id="cost"><strong>0 optional features picked.</strong></p>

  <div class="feat-section-label">Selected features</div>
  <ul class="feat-list" id="selected"></ul>

  <div class="feat-section-label">Not selected</div>
  <ul class="feat-list available" id="available"></ul>
  <p class="feat-empty" id="availEmpty" hidden>All features are selected.</p>

  <p class="muted" id="warn"></p>
</div>
<div class="busy" id="busy" aria-hidden="true" aria-busy="false">
  <div class="busy-spinner" role="status" aria-label="Loading"></div>
</div>
<script>
const LOCKED = {json.dumps(locked)};
let order = {json.dumps(selected)};
const labels = {json.dumps(FEATURE_LABELS)};
const allPickable = {json.dumps(list(PICKABLE_FEATURES))};

function showBusy() {{
  const el = document.getElementById('busy');
  el.classList.add('open');
  el.setAttribute('aria-busy', 'true');
  el.setAttribute('aria-hidden', 'false');
  document.body.style.overflow = 'hidden';
}}

function optionalCount() {{
  return order.filter(f => !LOCKED.includes(f)).length;
}}
function renderCost() {{
  const n = optionalCount();
  let t;
  if (n === 0) {{
    t = '<strong>0 optional features.</strong> Always-on: Total Rent, $/bed, $/sqft. Each added feature = longer session.';
  }} else {{
    t = '<strong>' + n + ' optional feature' + (n === 1 ? '' : 's') + '.</strong> '
      + 'Expect between 20–30 comparisons, longer if your choices are inconsistent.';
  }}
  document.getElementById('cost').innerHTML = t;
  document.getElementById('warn').textContent = n >= 8 ?
    '8+ optional features. Expect a longer session.' : '';
}}

function moveFeature(f, dir) {{
  const i = order.indexOf(f);
  const j = i + dir;
  if (i < 0 || j < 0 || j >= order.length) return;
  const tmp = order[i];
  order[i] = order[j];
  order[j] = tmp;
  sync();
}}

function liHtml(f, inSelected, rank) {{
  const locked = LOCKED.includes(f);
  const cls = (inSelected ? 'picked' : '') + (locked ? ' locked' : '');
  const rankTxt = inSelected ? String(rank) : '';
  const label = labels[f] || f;
  let actions = '';
  if (inSelected) {{
    const upDis = rank <= 1 ? ' disabled' : '';
    const downDis = rank >= order.length ? ' disabled' : '';
    actions =
      '<span class="feat-chevs">' +
        '<button type="button" class="feat-chev up" aria-label="Move up"'+upDis+'>▲</button>' +
        '<button type="button" class="feat-chev down" aria-label="Move down"'+downDis+'>▼</button>' +
      '</span>';
    if (locked) actions += '<span class="feat-lock">Always on</span>';
    else actions += '<button type="button" class="btn secondary tog">Remove</button>';
  }} else {{
    actions = '<button type="button" class="btn secondary tog">Add</button>';
  }}
  return '<li data-f="'+f+'" data-locked="'+(locked?1:0)+'" class="'+cls.trim()+'">' +
    '<span class="rank-num">'+rankTxt+'</span>' +
    '<div class="feat-row">' +
      '<span class="feat-label">'+label+'</span>' +
      '<span class="feat-actions">'+actions+'</span>' +
    '</div></li>';
}}

function sync() {{
  LOCKED.forEach(f => {{ if (!order.includes(f)) order.push(f); }});
  const sel = document.getElementById('selected');
  const avail = document.getElementById('available');
  sel.innerHTML = order.map((f,i) => liHtml(f, true, i+1)).join('');
  const rest = allPickable.filter(f => !order.includes(f));
  avail.innerHTML = rest.map(f => liHtml(f, false, null)).join('');
  const empty = document.getElementById('availEmpty');
  if (empty) empty.hidden = rest.length > 0;
  bindItems();
  renderCost();
}}

function bindItems() {{
  document.querySelectorAll('#selected li, #available li').forEach(li => {{
    const tog = li.querySelector('.tog');
    if (tog) tog.onclick = () => {{
      const f = li.dataset.f;
      if (LOCKED.includes(f)) return;
      const i = order.indexOf(f);
      if (i >= 0) order.splice(i, 1);
      else order.push(f);
      sync();
    }};
    const up = li.querySelector('.feat-chev.up');
    const down = li.querySelector('.feat-chev.down');
    if (up) up.onclick = (e) => {{ e.stopPropagation(); moveFeature(li.dataset.f, -1); }};
    if (down) down.onclick = (e) => {{ e.stopPropagation(); moveFeature(li.dataset.f, 1); }};
  }});
}}

document.getElementById('start').onclick = async () => {{
  const btn = document.getElementById('start');
  if (btn.disabled) return;
  btn.disabled = true;
  showBusy();
  LOCKED.forEach(f => {{ if (!order.includes(f)) order.push(f); }});
  try {{
    await fetch('/mash/api/features', {{method:'POST', headers:{{'Content-Type':'application/json'}},
      body: JSON.stringify({{reviewer: {json.dumps(reviewer)}, order}})}});
    location.href = '/mash/play';
  }} catch (err) {{
    btn.disabled = false;
    document.getElementById('busy').classList.remove('open');
    document.body.style.overflow = '';
    alert('Could not start comparing — try again.');
  }}
}};
sync();
</script>"""
    return page("Pick features — CasitaMash", body, who=f"playing as {reviewer}")

def format_value(name: str, feats: ListingFeatures) -> str:
    if name in feats.labels:
        return display_text(feats.labels[name])
    if not feats.known.get(name) or feats.values.get(name) is None:
        return "—"
    v = feats.values[name]
    if name == "price":
        return f"${v:,.0f}/mo"
    if name == "price_per_bed":
        return f"${v:,.0f}"
    if name == "price_per_sqft":
        return f"${v:,.2f}"
    if name in ("trail", "beach", "bakery", "grocery", "premium_grocery", "bar", "farmers_market", "ferry"):
        mode = display_text(feats.routes.get("mode", "walk"))
        return f"{v:.0f} min · {mode}"
    if name == "is_sf":
        return "SF" if v >= 0.5 else "Marin"
    if name in ("beds", "baths"):
        return f"{v:g}"
    if name == "sqft":
        return f"{v:,.0f}"
    return f"{v:g}"


def format_beds_baths(feats: ListingFeatures) -> str:
    if feats.known.get("beds") and feats.values.get("beds") is not None:
        beds = f"{feats.values['beds']:g} bd"
    else:
        beds = "—"
    if feats.known.get("baths") and feats.values.get("baths") is not None:
        baths = f"{feats.values['baths']:g} ba"
    else:
        baths = "—"
    return f"{beds} / {baths}"


def listing_photos(feats: ListingFeatures) -> list[str]:
    """Deduped photo URLs for cards / overlay (cover first if missing from list)."""
    from .features import dedupe_photo_urls

    photos = dedupe_photo_urls(list(feats.photos or []))
    cover = (feats.cover_url or "").strip()
    if cover and cover not in photos:
        photos = [cover] + photos
    return photos


def card_html(feats: ListingFeatures, feature_order: list[str], side: str) -> str:
    addr = _e(display_text(feats.address or feats.neighborhood or feats.key))
    rows = f'<div class="addr">{addr}</div>'
    show = card_feature_order(feature_order)
    beds_baths_done = False
    for name in show:
        if name in ("beds", "baths"):
            if beds_baths_done:
                continue
            beds_baths_done = True
            rows += (
                f'<div class="row"><div class="k">Beds/Baths</div>'
                f'<div class="v">{_e(format_beds_baths(feats))}</div></div>'
            )
            continue
        rows += (
            f'<div class="row"><div class="k">{_e(FEATURE_LABELS.get(name, name))}</div>'
            f'<div class="v">{_e(format_value(name, feats))}</div></div>'
        )
    src = display_text(feats.source)
    if feats.url:
        src_html = _external_a(feats.url, src, css_class="src-link")
    else:
        src_html = _e(src)
    hood = _e(display_text(feats.neighborhood) if feats.neighborhood else None)
    rows += (
        f'<div class="row"><div class="k">Neighborhood</div>'
        f'<div class="v">{hood}</div></div>'
        f'<div class="row"><div class="k">Source</div>'
        f'<div class="v">{src_html}</div></div>'
    )
    hyp = ""
    if feats.is_hypothetical:
        hyp = (
            f'<div class="hyp-banner">Not a real listing — hypothetical</div>'
            f'<div class="hyp-note">{_e(feats.hyp_note or "what-if version of this home")}</div>'
        )
    photos = listing_photos(feats)
    if photos:
        cover_block = (
            f'<div class="cover-wrap" data-overlay="{side}" role="button" '
            f'tabindex="0" title="View photos" aria-label="View photos ({len(photos)})">'
            f'<img class="cover" src="{_e(photos[0])}" alt="">'
            f'<span class="photo-btn">View photos ({len(photos)})</span>'
            f'</div>'
        )
    else:
        cover_block = (
            '<div class="cover-wrap no-photos">'
            '<div class="cover-empty">No photos for this listing</div>'
            '</div>'
        )
    return f"""
<article class="card{" hyp" if feats.is_hypothetical else ""}" data-side="{side}" data-key="{_e(feats.key)}">
  {hyp}
  {cover_block}
  <div class="body" data-pick="{_e(feats.key)}">{rows}</div>
</article>"""


def _mode_banner(*, mode: str, vertex_configured: bool, last_error: str | None) -> str:
    """Explain stub vs Vertex without pretending offline when a project is configured."""
    if mode == "vertex":
        return ""
    if vertex_configured:
        detail = ""
        err = (last_error or "").strip()
        if err:
            short = err.replace("\n", " ")
            if "BILLING" in short.upper() or "billing" in short:
                detail = (
                    " Last Gemini error looks like billing — enable billing on the GCP "
                    "project, wait a few minutes, then make another pick (retries automatically)."
                )
            else:
                detail = f" Last error: {_e(short[:180])}"
        return (
            '<div class="stub-banner">Gemini fallback — Vertex is configured, but the '
            "last model call failed so this session is using the offline stub."
            f"{detail}</div>"
        )
    return (
        '<div class="stub-banner">Offline preference stub — for Gemini, create a GCP '
        "project with billing and the Vertex AI API enabled, run "
        '<code>gcloud auth application-default login</code>, then '
        '<code>uv run casita mash --project YOUR_GCP_PROJECT</code> '
        "(replace YOUR_GCP_PROJECT with your project id). "
        "Skip, hypotheticals, and telemetry still work.</div>"
    )


def _probe_label(name: str) -> str:
    key = (name or "").strip()
    if not key:
        return ""
    return FEATURE_LABELS.get(key, key.replace("_", " ").title())


def probe_weighing_row(probe_features: list[str] | None) -> str:
    """One line under the why-line: memo probe themes in plain English."""
    names = [_probe_label(p) for p in (probe_features or []) if (p or "").strip()]
    names = [n for n in names if n][:3]
    if not names:
        return ""
    joined = " · ".join(_e(n) for n in names)
    return f'<p class="probe-row">Still weighing: {joined}</p>'


def play_page(
    reviewer: str,
    left: ListingFeatures,
    right: ListingFeatures,
    why: str,
    n: int,
    banner: str | None,
    feature_order: list[str],
    *,
    memo_text: str = "",
    mode: str = "stub",
    vertex_configured: bool = False,
    last_error: str | None = None,
    surprise_reason: str | None = None,
    probe_features: list[str] | None = None,
    pending_elicitation: dict | None = None,
) -> str:
    ban = ""
    if banner:
        ban = f"""<div class="banner"><div>{_e(banner)}</div>
          <a class="btn" href="/mash/results?done=1">See Results →</a></div>"""
    stub = _mode_banner(mode=mode, vertex_configured=vertex_configured, last_error=last_error)
    memo_block = ""
    if (memo_text or "").strip():
        memo_block = (
            '<div class="memo-box"><h3>Preference memo '
            '<span class="updating" id="rankUpdating" hidden>Updating.</span></h3>'
            f"{_e(memo_text.strip())}</div>"
        )
    else:
        memo_block = (
            '<p class="muted" id="rankUpdatingWrap" hidden>'
            '<span class="updating" id="rankUpdating">Updating.</span></p>'
        )
    standings_btn = ""
    if not banner and n > 0:
        standings_btn = """
<div class="actions end">
  <a class="btn" href="/mash/results">Current Standings</a>
</div>"""
    body = f"""
{ban}
{stub}
{memo_block}
{"<div class=\"hyp-page\">Hypothetical round: same home, two what-ifs. Not two real listings.</div>" if left.is_hypothetical or right.is_hypothetical else ""}
<p class="muted">{n} comparisons · ← / → to pick · space to skip · click the details to choose</p>
<p class="why">{_e(why)}</p>
{probe_weighing_row(probe_features)}
<div class="grid">
  {card_html(left, feature_order, "left")}
  <div class="vs">→</div>
  {card_html(right, feature_order, "right")}
</div>
{standings_btn}
<div class="overlay reason-ov" id="elicitationOv" aria-hidden="true">
  <div class="overlay-inner" role="dialog" aria-labelledby="elicitationTitle">
    <h3 id="elicitationTitle">Quick question</h3>
    <p id="elicitationQuestion" style="margin:0 0 16px;font-weight:600;line-height:1.45"></p>
    <div class="elicit-actions">
      <button type="button" class="btn" id="elicitationChoiceA"></button>
      <button type="button" class="btn" id="elicitationChoiceB"></button>
    </div>
  </div>
</div>
<div class="overlay reason-ov" id="surpriseOv" aria-hidden="true">
  <div class="overlay-inner" role="dialog" aria-labelledby="surpriseTitle">
    <h3 id="surpriseTitle">Hold on..</h3>
    <p class="muted" style="margin:0 0 8px">That choice contradicts our preference memo.</p>
    <p id="surpriseReason" style="margin:0 0 14px;font-weight:600;line-height:1.4"></p>
    <p class="muted" style="margin:0 0 8px">What changed? (optional)</p>
    <input type="text" id="surpriseReply" maxlength="240"
      placeholder="e.g. rent mattered more / X is less important than Y" autocomplete="off">
    <div class="reason-actions">
      <button type="button" class="btn secondary" id="surpriseSkip">Skip</button>
      <button type="button" class="btn" id="surpriseSave">Save</button>
    </div>
  </div>
</div>
<div class="overlay reason-ov" id="reasonOv" aria-hidden="true">
  <div class="overlay-inner" role="dialog" aria-labelledby="reasonTitle">
    <h3 id="reasonTitle">Reason (optional)</h3>
    <p class="muted" id="reasonHint">Why this one? Leave blank if you just know.</p>
    <input type="text" id="pickReason" maxlength="240"
      placeholder="e.g. better light / worth the rent for the yard" autocomplete="off">
    <div class="reason-actions">
      <button type="button" class="btn secondary" id="reasonCancel">Cancel</button>
      <button type="button" class="btn" id="reasonConfirm">Continue</button>
    </div>
  </div>
</div>
<div class="overlay" id="ov"><div class="overlay-inner">
  <button type="button" class="ov-close" id="ovClose" aria-label="Close">×</button>
  <img id="ovMain" style="width:100%;max-height:60vh;object-fit:contain" alt="">
  <div class="thumbs" id="ovThumbs"></div>
</div></div>
<div class="busy" id="busy" aria-hidden="true" aria-busy="false">
  <div class="busy-spinner" role="status" aria-label="Loading"></div>
</div>
<script>
const state = {{
  reviewer: {json.dumps(reviewer)},
  left: {json.dumps({"key": left.key, "photos": listing_photos(left), "source": left.source, "photo_count": len(listing_photos(left)), "is_hyp": left.is_hypothetical, "values": left.values, "known": left.known})},
  right: {json.dumps({"key": right.key, "photos": listing_photos(right), "source": right.source, "photo_count": len(listing_photos(right)), "is_hyp": right.is_hypothetical, "values": right.values, "known": right.known})},
  feature_order: {json.dumps(feature_order)},
  strategy: {json.dumps(why)},
  shown_at: new Date().toISOString(),
  overlay: false,
  overlaySide: null,
  photoIdx: 0,
  pendingWinner: null,
  pendingSkip: false,
  reasonOpen: false,
  surpriseOpen: false,
  surpriseReason: {json.dumps((surprise_reason or "").strip() or None)},
  elicitationOpen: false,
  elicitation: {json.dumps(pending_elicitation if pending_elicitation else None)},
}};
let picking = true;
function showBusy() {{
  const el = document.getElementById('busy');
  el.classList.add('open');
  el.setAttribute('aria-busy', 'true');
  el.setAttribute('aria-hidden', 'false');
  document.body.style.overflow = 'hidden';
}}
function openSurprise() {{
  if (!state.surpriseReason) return;
  state.surpriseOpen = true;
  picking = false;
  const ov = document.getElementById('surpriseOv');
  document.getElementById('surpriseReason').textContent = state.surpriseReason;
  document.getElementById('surpriseReply').value = '';
  ov.classList.add('open');
  ov.setAttribute('aria-hidden', 'false');
  document.body.style.overflow = 'hidden';
  setTimeout(() => document.getElementById('surpriseReply').focus(), 0);
}}
async function closeSurprise(reply) {{
  const ov = document.getElementById('surpriseOv');
  ov.classList.remove('open');
  ov.setAttribute('aria-hidden', 'true');
  state.surpriseOpen = false;
  state.surpriseReason = null;
  try {{
    await fetch('/mash/api/surprise', {{
      method: 'POST',
      headers: {{'Content-Type': 'application/json'}},
      body: JSON.stringify({{
        reviewer: state.reviewer,
        reply: (reply || '').trim() || null,
      }}),
    }});
  }} catch (e) {{}}
  if (state.elicitation) {{
    openElicitation();
    return;
  }}
  picking = true;
  document.body.style.overflow = '';
}}
function openElicitation() {{
  if (!state.elicitation) return;
  state.elicitationOpen = true;
  picking = false;
  const ov = document.getElementById('elicitationOv');
  const q = state.elicitation.question || '';
  document.getElementById('elicitationQuestion').textContent = q;
  document.getElementById('elicitationChoiceA').textContent = state.elicitation.choice_a || 'Option A';
  document.getElementById('elicitationChoiceB').textContent = state.elicitation.choice_b || 'Option B';
  ov.classList.add('open');
  ov.setAttribute('aria-hidden', 'false');
  document.body.style.overflow = 'hidden';
  setTimeout(() => document.getElementById('elicitationChoiceA').focus(), 0);
}}
async function answerElicitation(choice) {{
  const ov = document.getElementById('elicitationOv');
  ov.classList.remove('open');
  ov.setAttribute('aria-hidden', 'true');
  state.elicitationOpen = false;
  const question = (state.elicitation && state.elicitation.question) || '';
  state.elicitation = null;
  try {{
    await fetch('/mash/api/elicitation', {{
      method: 'POST',
      headers: {{'Content-Type': 'application/json'}},
      body: JSON.stringify({{
        reviewer: state.reviewer,
        question: question,
        choice: choice,
      }}),
    }});
  }} catch (e) {{}}
  picking = true;
  document.body.style.overflow = '';
}}
document.getElementById('elicitationChoiceA').onclick = () => {{
  answerElicitation(document.getElementById('elicitationChoiceA').textContent || '');
}};
document.getElementById('elicitationChoiceB').onclick = () => {{
  answerElicitation(document.getElementById('elicitationChoiceB').textContent || '');
}};
document.getElementById('surpriseSkip').onclick = () => closeSurprise('');
document.getElementById('surpriseSave').onclick = () => {{
  closeSurprise(document.getElementById('surpriseReply').value || '');
}};
document.getElementById('surpriseOv').addEventListener('click', (e) => {{
  if (e.target.id === 'surpriseOv') closeSurprise('');
}});
if (state.surpriseReason) openSurprise();
else if (state.elicitation) openElicitation();
function openReason(winner, skipped) {{
  if (!picking || state.reasonOpen || state.surpriseOpen || state.elicitationOpen) return;
  state.pendingWinner = skipped ? null : winner;
  state.pendingSkip = !!skipped;
  state.reasonOpen = true;
  picking = false;
  const ov = document.getElementById('reasonOv');
  const input = document.getElementById('pickReason');
  const hint = document.getElementById('reasonHint');
  input.value = '';
  if (skipped) {{
    document.getElementById('reasonTitle').textContent = 'Reason (optional)';
    hint.textContent = 'Why skip this pair? Leave blank if you just want the next one.';
    input.placeholder = "e.g. both feel wrong / too similar / can't tell from photos";
  }} else {{
    document.getElementById('reasonTitle').textContent = 'Reason (optional)';
    hint.textContent = 'Why this one? Leave blank if you just know.';
    input.placeholder = 'e.g. better light / worth the rent for the yard';
  }}
  ov.classList.add('open');
  ov.setAttribute('aria-hidden', 'false');
  document.body.style.overflow = 'hidden';
  setTimeout(() => input.focus(), 0);
}}
function closeReason(resume) {{
  const ov = document.getElementById('reasonOv');
  ov.classList.remove('open');
  ov.setAttribute('aria-hidden', 'true');
  state.reasonOpen = false;
  state.pendingWinner = null;
  state.pendingSkip = false;
  if (resume) {{
    picking = true;
    document.body.style.overflow = '';
  }}
}}
async function commitDecide(winner, skipped, reason) {{
  if (state.surpriseOpen || state.elicitationOpen) return;
  picking = false;
  state.reasonOpen = false;
  showBusy();
  const photoOv = document.getElementById('ov');
  if (photoOv) photoOv.classList.remove('open');
  state.overlay = false;
  const reasonOv = document.getElementById('reasonOv');
  if (reasonOv) {{
    reasonOv.classList.remove('open');
    reasonOv.setAttribute('aria-hidden', 'true');
  }}
  const body = {{
    reviewer: state.reviewer,
    left_key: state.left.key,
    right_key: state.right.key,
    winner: skipped ? null : winner,
    skipped: skipped,
    reason: (reason || '').trim() || null,
    shown_at: state.shown_at,
    decided_at: new Date().toISOString(),
    overlay_opened: state.overlayOpened || false,
    feature_order: state.feature_order,
    left_meta: state.left,
    right_meta: state.right,
    is_hypothetical: !!(state.left.is_hyp || state.right.is_hyp),
    hyp_left: state.left.is_hyp ? state.left : null,
    hyp_right: state.right.is_hyp ? state.right : null,
  }};
  try {{
    await fetch('/mash/api/compare', {{method:'POST', headers:{{'Content-Type':'application/json'}}, body: JSON.stringify(body)}});
    const deadline = Date.now() + 120000;
    while (Date.now() < deadline) {{
      await new Promise(r => setTimeout(r, 400));
      const res = await fetch('/mash/api/pref_status?reviewer=' + encodeURIComponent(state.reviewer));
      const st = await res.json();
      // Advance once memo+probes land; rank may still be running.
      if (st.memo_ready || st.phase === 'rank' || st.phase === 'idle' || st.phase === 'error'
          || st.status === 'idle' || st.status === 'error') break;
    }}
  }} finally {{
    location.reload();
  }}
}}
function setUpdatingVisible(on) {{
  const el = document.getElementById('rankUpdating');
  if (!el) return;
  const wrap = document.getElementById('rankUpdatingWrap');
  if (wrap) wrap.hidden = !on;
  el.hidden = !on;
}}
function startUpdatingPulse() {{
  const el = document.getElementById('rankUpdating');
  if (!el || el.dataset.pulsing) return;
  el.dataset.pulsing = '1';
  setUpdatingVisible(true);
  const frames = ['Updating.', 'Updating..', 'Updating...'];
  let i = 0;
  el._pulse = setInterval(() => {{
    i = (i + 1) % frames.length;
    el.textContent = frames[i];
  }}, 400);
}}
async function maybeShowRankUpdating() {{
  try {{
    const res = await fetch('/mash/api/pref_status?reviewer=' + encodeURIComponent(state.reviewer));
    const st = await res.json();
    if (st.phase === 'rank' || (st.status === 'running' && st.memo_ready && !st.rank_ready)) {{
      startUpdatingPulse();
    }}
  }} catch (e) {{}}
}}
maybeShowRankUpdating();
function confirmReason() {{
  const skipped = !!state.pendingSkip;
  const winner = state.pendingWinner;
  const reason = (document.getElementById('pickReason').value || '').trim();
  if (!skipped && !winner) {{ closeReason(true); return; }}
  commitDecide(skipped ? null : winner, skipped, reason);
}}
document.getElementById('reasonConfirm').onclick = confirmReason;
document.getElementById('reasonCancel').onclick = () => closeReason(true);
document.getElementById('reasonOv').addEventListener('click', (e) => {{
  if (e.target.id === 'reasonOv') closeReason(true);
}});
document.querySelectorAll('[data-pick]').forEach(el => {{
  el.addEventListener('click', (e) => {{
    if (!picking) return;
    if (e.target.closest('a')) return;
    if (e.target.closest('[data-overlay]')) return;
    openReason(el.dataset.pick);
  }});
}});
function openOv(side) {{
  if (!picking) return;
  const meta = side === 'left' ? state.left : state.right;
  const photos = meta.photos || [];
  if (!photos.length) return;
  state.overlay = true; state.overlayOpened = true; state.overlaySide = side; picking = false;
  state.photoIdx = 0;
  const ov = document.getElementById('ov');
  ov.classList.add('open');
  renderOv(meta);
}}
function renderOv(meta) {{
  const photos = meta.photos || [];
  const main = document.getElementById('ovMain');
  const th = document.getElementById('ovThumbs');
  if (!photos.length) {{
    main.removeAttribute('src');
    main.style.display = 'none';
    th.innerHTML = '<div class="ov-empty">No photos for this listing</div>';
    return;
  }}
  main.style.display = '';
  main.src = photos[state.photoIdx] || photos[0] || '';
  th.innerHTML = photos.map((p,i) => '<img src="'+p+'" data-i="'+i+'">').join('');
  th.querySelectorAll('img').forEach(img => img.onclick = () => {{ state.photoIdx = +img.dataset.i; renderOv(meta); }});
}}
document.querySelectorAll('[data-overlay]').forEach(el => {{
  el.onclick = (e) => {{
    e.preventDefault();
    e.stopPropagation();
    if (!picking && !state.overlay) return;
    openOv(el.dataset.overlay);
  }};
  el.onkeydown = (e) => {{
    if (e.key === 'Enter' || e.key === ' ') {{
      e.preventDefault();
      e.stopPropagation();
      if (!picking && !state.overlay) return;
      openOv(el.dataset.overlay);
    }}
  }};
}});
document.getElementById('ovClose').onclick = () => {{
  document.getElementById('ov').classList.remove('open');
  state.overlay = false; picking = true;
}};
window.addEventListener('keydown', (e) => {{
  if (document.getElementById('busy').classList.contains('open')) {{
    e.preventDefault();
    return;
  }}
  if (state.surpriseOpen) {{
    if (e.key === 'Escape') {{ e.preventDefault(); closeSurprise(''); }}
    if (e.key === 'Enter') {{
      e.preventDefault();
      closeSurprise(document.getElementById('surpriseReply').value || '');
    }}
    return;
  }}
  if (state.elicitationOpen) {{
    if (e.key === '1') {{ e.preventDefault(); document.getElementById('elicitationChoiceA').click(); }}
    if (e.key === '2') {{ e.preventDefault(); document.getElementById('elicitationChoiceB').click(); }}
    return;
  }}
  if (state.reasonOpen) {{
    if (e.key === 'Escape') {{ e.preventDefault(); closeReason(true); }}
    if (e.key === 'Enter') {{ e.preventDefault(); confirmReason(); }}
    return;
  }}
  if (state.overlay) {{
    const meta = state.overlaySide === 'left' ? state.left : state.right;
    if (e.key === 'ArrowLeft') {{ state.photoIdx = Math.max(0, state.photoIdx-1); renderOv(meta); }}
    if (e.key === 'ArrowRight') {{ state.photoIdx = Math.min((meta.photos||[]).length-1, state.photoIdx+1); renderOv(meta); }}
    if (e.key === 'Escape') document.getElementById('ovClose').click();
    return;
  }}
  if (!picking) return;
  if (e.key === 'ArrowLeft') openReason(state.left.key);
  if (e.key === 'ArrowRight') openReason(state.right.key);
  if (e.key === ' ') {{ e.preventDefault(); openReason(null, true); }}
}});
</script>"""
    return page("Compare — CasitaMash", body, who=f"playing as {reviewer} · {n} picks")


def _listing_label(row: dict) -> str:
    return display_text(row.get("title") or row.get("key") or "Listing")


def _source_label(row: dict) -> str:
    if row.get("source"):
        return display_text(row["source"])
    url = row.get("url") or ""
    if url:
        host = url.split("//")[-1].split("/")[0].replace("www.", "")
        name = host.split(".")[0] if host else ""
        if name:
            return display_text(name)
    return "Source"


def _source_link(row: dict) -> str:
    if not row.get("url"):
        return "—"
    return _external_a(row["url"], _source_label(row))


def _rank_table(rows: list[dict], start_i: int = 1, *, show_badge: bool = False) -> str:
    rank_rows = ""
    for i, row in enumerate(rows, start_i):
        badge = (
            ' <span class="badge soft">not shown yet</span>'
            if show_badge and row.get("never_shown")
            else ""
        )
        be = ""
        if row.get("break_even") is not None:
            be = f'<div class="muted">Break-even ≈ ${row["break_even"]:,.0f}/mo</div>'
        reason = (row.get("reason") or "").strip()
        reason_cell = (
            f'<div class="rank-reason">{_e(reason)}</div>' if reason else ""
        )
        rank_rows += (
            f'<tr><td>{i}</td><td>{_e(_listing_label(row))}{badge}{be}{reason_cell}</td>'
            f'<td>{row["score"]:.3f}</td><td>{row.get("n_shown", 0)}</td>'
            f'<td>{_source_link(row)}</td></tr>'
        )
    return (
        '<table class="rank">'
        '<tr><th>#</th><th>Listing</th><th>Score</th><th>Seen</th><th>Source</th></tr>'
        f'{rank_rows}</table>'
    )


def _podium(rows: list[dict]) -> str:
    if len(rows) < 1:
        return ""
    medals = [
        ("gold", "1st"),
        ("silver", "2nd"),
        ("bronze", "3rd"),
    ]
    top = rows[:3]
    # Display order: silver, gold, bronze (classic podium) when we have 3
    if len(top) == 3:
        order = [top[1], top[0], top[2]]
        medal_order = [medals[1], medals[0], medals[2]]
        ranks = [2, 1, 3]
    else:
        order = top
        medal_order = medals[: len(top)]
        ranks = list(range(1, len(top) + 1))
    cards = []
    for row, (cls, label), rank in zip(order, medal_order, ranks):
        cover = row.get("cover_url") or ""
        img = (
            f'<img src="{_e(cover)}" alt="" loading="lazy">'
            if cover
            else '<div class="cover-ph"></div>'
        )
        badge = (
            ' <span class="badge soft">not shown yet</span>'
            if row.get("never_shown")
            else ""
        )
        meta = f'#{rank} · score {row["score"]:.3f} · seen {row.get("n_shown", 0)}'
        inner = (
            f'{img}'
            f'<div class="place">{_e(label)}</div>'
            f'<div class="name">{_e(_listing_label(row))}{badge}</div>'
            f'<div class="meta">{_e(meta)}</div>'
        )
        url = row.get("url") or ""
        if url:
            src_label = _source_label(row)
            cards.append(
                f'<a class="podium-card {cls}" href="{_e(url)}" '
                f'data-external-url="{_e(url)}" data-external-label="{_e(src_label)}">'
                f'{inner}</a>'
            )
        else:
            cards.append(f'<div class="podium-card {cls}">{inner}</div>')
    return f'<div class="podium">{"".join(cards)}</div>'


def results_page(
    reviewer: str,
    compared: list[dict],
    unseen: list[dict],
    movers: list[dict],
    n: int,
    *,
    concluded: bool = False,
    memo_text: str = "",
    mode: str = "stub",
    vertex_configured: bool = False,
    last_error: str | None = None,
) -> str:
    if concluded:
        ranking = list(compared) + list(unseen)
        ranking.sort(key=lambda r: r.get("score", 0), reverse=True)
        podium = _podium(ranking)
        rest = ranking[3:] if len(ranking) > 3 else []
        if ranking:
            compared_html = podium + (
                _rank_table(rest, start_i=4, show_badge=True) if rest else ""
            )
        else:
            compared_html = '<p class="muted">No listings scored yet.</p>'
        unseen_html = ""
    else:
        podium = _podium(compared)
        rest = compared[3:] if len(compared) > 3 else []
        if compared:
            compared_html = podium + (_rank_table(rest, start_i=4) if rest else "")
        else:
            compared_html = '<p class="muted">No compared listings yet — keep picking.</p>'
        unseen_html = ""
        if compared and unseen:
            unseen_html = (
                f'<h3 style="margin-top:28px">Also scoring well (not shown yet)</h3>'
                f'<p class="standings-note">These weren’t in your matchups. '
                f"The preference model ranked them from your picks and memo.</p>"
                f"{_rank_table(unseen, start_i=len(compared) + 1, show_badge=True)}"
            )
    heading = "Your Results" if concluded else "Current Standings"
    stub = _mode_banner(mode=mode, vertex_configured=vertex_configured, last_error=last_error)
    memo_block = (
        '<div class="memo-box"><h3>Preference memo '
        '<span class="updating" id="rankUpdating" hidden>Updating.</span></h3>'
        + (
            _e(memo_text.strip())
            if (memo_text or "").strip()
            else '<span class="muted">(empty — make a pick)</span>'
        )
        + "</div>"
    )
    mover_items = "".join(
        f"<li>{_e(m.get('label') or m.get('feature'))}</li>"
        for m in movers
    )
    movers_block = (
        f"<ul>{mover_items}</ul>"
        if mover_items
        else '<p class="muted">Not enough signal yet — keep comparing (and optionally write a reason).</p>'
    )
    body = f"""
<div class="actions" style="margin:0 0 14px">
  <a class="btn" href="/mash/play">← Return</a>
</div>
{stub}
{memo_block}
<div class="panel">
  <div class="standings-head">
    <h2>{_e(heading)}</h2>
    <p class="muted">{n} comparisons</p>
    <span class="updating" id="standingsUpdating" hidden>Updating.</span>
  </div>
  {compared_html}
  {unseen_html}
</div>
<div class="panel">
  <h2>For nerds</h2>
  <p class="standings-note">Standings come from a preference memo + model rank
  (Gemini via Vertex when configured; deterministic stub otherwise). The memo
  updates from your picks, reasons, and photos; this page ranks the catalog
  from that memo.</p>
  <h3>What the memo is picking up</h3>
  {movers_block}
</div>
<script>
const reviewer = {json.dumps(reviewer)};
function startUpdatingPulse(ids) {{
  const frames = ['Updating.', 'Updating..', 'Updating...'];
  let i = 0;
  ids.forEach(id => {{
    const el = document.getElementById(id);
    if (!el) return;
    el.hidden = false;
    if (el.dataset.pulsing) return;
    el.dataset.pulsing = '1';
    el._pulse = setInterval(() => {{
      i = (i + 1) % frames.length;
      el.textContent = frames[i];
    }}, 400);
  }});
}}
async function watchRank() {{
  let sawPending = false;
  const poll = async () => {{
    try {{
      const res = await fetch('/mash/api/pref_status?reviewer=' + encodeURIComponent(reviewer));
      const st = await res.json();
      const ranking = st.phase === 'rank' || (st.status === 'running' && st.memo_ready && !st.rank_ready);
      if (ranking) {{
        sawPending = true;
        startUpdatingPulse(['rankUpdating', 'standingsUpdating']);
      }} else if (sawPending && st.rank_ready) {{
        location.reload();
        return;
      }}
    }} catch (e) {{}}
    setTimeout(poll, 2000);
  }};
  poll();
}}
watchRank();
</script>"""
    return page(f"{heading} — CasitaMash", body, who=f"playing as {reviewer}")
