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
*{box-sizing:border-box}
html,body{margin:0;padding:0;background:#f6f1ea;color:#1a1a1a;
  font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Helvetica,Arial,sans-serif}
a{color:#920004}
header.mash{
  background:#920004;color:#f6f1ea;padding:14px 18px;
  display:flex;align-items:baseline;justify-content:space-between;gap:12px;
  border-bottom:3px solid #5c0002}
header.mash h1{margin:0;font-size:22px;letter-spacing:0.02em;font-weight:700}
header.mash .tag{opacity:0.85;font-size:13px}
header.mash .who{font-size:12px;opacity:0.8}
main{max-width:1100px;margin:0 auto;padding:18px}
.btn{
  display:inline-block;background:#920004;color:#f6f1ea;border:2px solid #5c0002;
  padding:10px 16px;font-size:14px;font-weight:700;cursor:pointer;text-decoration:none}
.btn.secondary{background:#f6f1ea;color:#920004}
.btn:disabled{opacity:0.4;cursor:not-allowed}
.panel{background:#fff;border:2px solid #1a1a1a;padding:18px;margin:14px 0}
input[type=text]{width:100%;max-width:360px;padding:10px;border:2px solid #1a1a1a;font-size:16px}
.muted{color:#666;font-size:13px}
.banner{
  background:#fff3cd;border:2px solid #920004;padding:12px 14px;margin:12px 0;
  display:flex;gap:12px;align-items:center;justify-content:space-between;flex-wrap:wrap}
.warn-box{
  background:#fff3cd;border:2px solid #920004;padding:12px 14px;margin:12px 0;
  font-size:13px;font-weight:700;line-height:1.4}
.why{font-size:14px;margin:8px 0 14px;font-weight:600}
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
.cover-wrap{position:relative}
.cover-wrap:hover .cover{filter:brightness(0.85)}
.photo-btn{
  position:absolute;right:10px;bottom:10px;z-index:2;
  background:#920004;color:#f6f1ea;border:2px solid #5c0002;
  padding:6px 10px;font-size:12px;font-weight:700;cursor:pointer;
  filter:brightness(1);transition:filter 0.12s ease}
.cover-wrap:hover .photo-btn{filter:brightness(1.18)}
.photo-btn:hover{filter:brightness(1.25)}
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
.feat-list{list-style:none;padding:0;margin:0}
.feat-list li{
  display:flex;align-items:center;gap:10px;padding:8px;border:2px solid #1a1a1a;margin:6px 0;
  background:#fff;cursor:grab}
.feat-list li.picked{background:#fce8e9}
.feat-list li.locked{background:#f7ebe5}
.feat-list li.dragging{opacity:0.45}
.feat-section-label{
  font-size:14px;font-weight:800;margin:18px 0 8px;color:#1a1a1a;
  letter-spacing:0.02em}
.feat-list.drop-zone{
  min-height:56px;padding:8px;border:2px dashed #920004;background:#fff}
.feat-list.drop-zone.drag-over{background:#fce8e9}
.feat-list.available{
  min-height:56px;padding:8px;border:2px dashed #aaa;background:#fafafa}
.feat-list.available.drag-over{border-color:#920004;background:#fce8e9}
.feat-empty{font-size:13px;color:#666;padding:10px;margin:0}
.feat-list li .tog{margin-left:auto;cursor:pointer}
.feat-list li.locked .tog{display:none}
.feat-lock{
  font-size:11px;font-weight:700;color:#920004;margin-left:auto;white-space:nowrap}
.rank-num{width:24px;font-weight:800;color:#920004}
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
.overlay-inner{background:#fff;border:3px solid #1a1a1a;max-width:900px;width:100%;max-height:90vh;overflow:auto;padding:12px}
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
</div>
<script>
const known = document.getElementById('known');
const name = document.getElementById('name');
const resume = document.getElementById('resume');
if (known) known.onchange = () => {{ if (known.value) name.value = known.value; }};
name.oninput = async () => {{
  const r = await fetch('/mash/api/reviewer?name=' + encodeURIComponent(name.value.trim()));
  const j = await r.json();
  if (j.exists) resume.textContent = "You've done " + j.count + " comparisons so far. Continue?";
  else resume.textContent = j.name ? "Welcome! Find your next home." : "";
}};
document.getElementById('go').onclick = async () => {{
  const n = name.value.trim();
  if (!n) return;
  const r = await fetch('/mash/api/reviewer', {{method:'POST', headers:{{'Content-Type':'application/json'}},
    body: JSON.stringify({{name:n}})}});
  const j = await r.json();
  location.href = j.next;
}};
</script>"""
    return page("CasitaMash", body)


def features_page(reviewer: str, order: list[str], estimate: tuple[int, int]) -> str:
    locked = list(RANKABLE_LOCKED)
    order = [f for f in order if f in PICKABLE_FEATURES or f in locked]
    for f in locked:
        if f not in order:
            order.append(f)
    selected = list(order)

    def _item(f: str, *, in_selected: bool, rank: int | None) -> str:
        is_locked = f in locked
        cls = "picked" + (" locked" if is_locked else "")
        rank_txt = str(rank) if in_selected and rank is not None else "·"
        if is_locked:
            tail = '<span class="feat-lock">Always on</span>'
        elif in_selected:
            tail = '<button type="button" class="btn secondary tog">Remove</button>'
        else:
            tail = '<button type="button" class="btn secondary tog">Add</button>'
        return (
            f'<li draggable="true" data-f="{_e(f)}" data-locked="{1 if is_locked else 0}" class="{cls}">'
            f'<span class="rank-num">{rank_txt}</span>'
            f'<span>{_e(FEATURE_LABELS.get(f, f))}</span>'
            f"{tail}</li>"
        )

    selected_html = "".join(_item(f, in_selected=True, rank=i + 1) for i, f in enumerate(selected))
    body = f"""
<div class="panel">
  <h2>What features do you care most about?</h2>
  <p class="muted">Add features to Selected, then drag to rank them (order matters!). {ALWAYS_SHOW_COPY}.</p>
  <div class="warn-box">Warning: these picks cannot be changed for this session. If you want to compare across different features, sign out and start a new session.</div>
  <p id="cost"><strong>0 optional features picked.</strong></p>

  <div class="feat-section-label">Selected features</div>
  <ul class="feat-list drop-zone" id="selected">{selected_html}</ul>

  <div class="feat-section-label">Not selected</div>
  <ul class="feat-list available" id="available"></ul>
  <p class="feat-empty" id="availEmpty" hidden>All features are selected.</p>

  <p class="muted" id="warn"></p>
  <div class="actions">
    <button class="btn" id="start">Start comparing →</button>
    <a class="btn secondary" href="/mash/api/logout">Sign out</a>
  </div>
</div>
<script>
const LOCKED = {json.dumps(locked)};
let order = {json.dumps(selected)};
const labels = {json.dumps(FEATURE_LABELS)};
const allPickable = {json.dumps(list(PICKABLE_FEATURES))};

function optionalCount() {{
  return order.filter(f => !LOCKED.includes(f)).length;
}}
function renderCost() {{
  const n = optionalCount();
  let t;
  if (n === 0) {{
    t = '<strong>0 optional features picked.</strong> $ / bed and $ / sqft stay selected. Add more features if you want — each adds roughly 10 comparisons.';
  }} else {{
    const lo = 20 + n * 10;
    const hi = lo + n * 10;
    t = '<strong>' + n + ' optional feature(s) picked</strong> (plus $ / bed and $ / sqft). About ' + lo + '–' + hi +
      ' comparisons before the numbers hold still.';
  }}
  document.getElementById('cost').innerHTML = t;
  document.getElementById('warn').textContent = n > 8 ?
    "Past eight optional features, the numbers probably won't hold still in a normal sitting." : "";
}}

function liHtml(f, inSelected, rank) {{
  const locked = LOCKED.includes(f);
  const cls = 'picked' + (locked ? ' locked' : '');
  const rankTxt = inSelected ? String(rank) : '·';
  let tail;
  if (locked) tail = '<span class="feat-lock">Always on</span>';
  else if (inSelected) tail = '<button type="button" class="btn secondary tog">Remove</button>';
  else tail = '<button type="button" class="btn secondary tog">Add</button>';
  return '<li draggable="true" data-f="'+f+'" data-locked="'+(locked?1:0)+'" class="'+cls+'">' +
    '<span class="rank-num">'+rankTxt+'</span><span>'+(labels[f]||f)+'</span>'+tail+'</li>';
}}

function sync() {{
  // keep locked present
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

let dragF = null;
function bindItems() {{
  document.querySelectorAll('#selected li, #available li').forEach(li => {{
    li.addEventListener('dragstart', (e) => {{
      dragF = li.dataset.f;
      li.classList.add('dragging');
      e.dataTransfer.effectAllowed = 'move';
      e.dataTransfer.setData('text/plain', dragF);
    }});
    li.addEventListener('dragend', () => {{
      li.classList.remove('dragging');
      dragF = null;
      document.querySelectorAll('.drag-over').forEach(el => el.classList.remove('drag-over'));
    }});
    const tog = li.querySelector('.tog');
    if (tog) tog.onclick = () => {{
      const f = li.dataset.f;
      if (LOCKED.includes(f)) return;
      const i = order.indexOf(f);
      if (i >= 0) order.splice(i, 1);
      else order.push(f);
      sync();
    }};
  }});
  // reorder within selected by dragging over items
  document.querySelectorAll('#selected li').forEach(li => {{
    li.addEventListener('dragover', (e) => {{
      e.preventDefault();
      e.dataTransfer.dropEffect = 'move';
    }});
    li.addEventListener('drop', (e) => {{
      e.preventDefault();
      e.stopPropagation();
      const f = dragF || e.dataTransfer.getData('text/plain');
      if (!f) return;
      const target = li.dataset.f;
      if (f === target) return;
      const from = order.indexOf(f);
      if (from >= 0) order.splice(from, 1);
      const to = order.indexOf(target);
      order.splice(to < 0 ? order.length : to, 0, f);
      sync();
    }});
  }});
}}

function wireZone(el, onDrop) {{
  el.addEventListener('dragover', (e) => {{
    e.preventDefault();
    el.classList.add('drag-over');
    e.dataTransfer.dropEffect = 'move';
  }});
  el.addEventListener('dragleave', () => el.classList.remove('drag-over'));
  el.addEventListener('drop', (e) => {{
    e.preventDefault();
    el.classList.remove('drag-over');
    const f = dragF || e.dataTransfer.getData('text/plain');
    if (f) onDrop(f);
  }});
}}
wireZone(document.getElementById('selected'), (f) => {{
  if (!order.includes(f)) order.push(f);
  sync();
}});
wireZone(document.getElementById('available'), (f) => {{
  if (LOCKED.includes(f)) return;
  const i = order.indexOf(f);
  if (i >= 0) order.splice(i, 1);
  sync();
}});

document.getElementById('start').onclick = async () => {{
  LOCKED.forEach(f => {{ if (!order.includes(f)) order.push(f); }});
  await fetch('/mash/api/features', {{method:'POST', headers:{{'Content-Type':'application/json'}},
    body: JSON.stringify({{reviewer: {json.dumps(reviewer)}, order}})}});
  location.href = '/mash/play';
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
    cover = _e(feats.cover_url or "")
    n_photos = len(feats.photos) or (1 if feats.cover_url else 0)
    return f"""
<article class="card{" hyp" if feats.is_hypothetical else ""}" data-side="{side}" data-key="{_e(feats.key)}">
  {hyp}
  <div class="cover-wrap">
    <img class="cover" src="{cover}" alt="">
    <button type="button" class="photo-btn" data-overlay="{side}">View photos ({n_photos})</button>
  </div>
  <div class="body" data-pick="{_e(feats.key)}">{rows}</div>
</article>"""


def play_page(
    reviewer: str,
    left: ListingFeatures,
    right: ListingFeatures,
    why: str,
    n: int,
    banner: str | None,
    feature_order: list[str],
) -> str:
    ban = ""
    if banner:
        ban = f"""<div class="banner"><div>{_e(banner)}</div>
          <a class="btn" href="/mash/results?done=1">See Results →</a></div>"""
    standings_btn = ""
    if not banner:
        standings_btn = """
<div class="actions end">
  <a class="btn" href="/mash/results">Current Standings</a>
</div>"""
    body = f"""
{ban}
{"<div class=\"hyp-page\">Hypothetical round: same home, two what-ifs. Not two real listings.</div>" if left.is_hypothetical or right.is_hypothetical else ""}
<p class="muted">{n} comparisons · ← / → to pick · space to skip · click the details to choose</p>
<p class="why">{_e(why)}</p>
<div class="grid">
  {card_html(left, feature_order, "left")}
  <div class="vs">→</div>
  {card_html(right, feature_order, "right")}
</div>
{standings_btn}
<div class="overlay" id="ov"><div class="overlay-inner">
  <button class="btn secondary" id="ovClose">Close</button>
  <img id="ovMain" style="width:100%;max-height:60vh;object-fit:contain;margin-top:8px" alt="">
  <div class="thumbs" id="ovThumbs"></div>
  <div id="ovExtra" class="muted" style="margin-top:10px"></div>
</div></div>
<div class="busy" id="busy" aria-hidden="true" aria-busy="false">
  <div class="busy-spinner" role="status" aria-label="Loading"></div>
</div>
<script>
const state = {{
  reviewer: {json.dumps(reviewer)},
  left: {json.dumps({"key": left.key, "photos": left.photos, "source": left.source, "photo_count": left.photo_count, "is_hyp": left.is_hypothetical, "values": left.values, "known": left.known})},
  right: {json.dumps({"key": right.key, "photos": right.photos, "source": right.source, "photo_count": right.photo_count, "is_hyp": right.is_hypothetical, "values": right.values, "known": right.known})},
  feature_order: {json.dumps(feature_order)},
  strategy: {json.dumps(why)},
  shown_at: new Date().toISOString(),
  overlay: false,
  overlaySide: null,
  photoIdx: 0,
}};
let picking = true;
function showBusy() {{
  const el = document.getElementById('busy');
  el.classList.add('open');
  el.setAttribute('aria-busy', 'true');
  el.setAttribute('aria-hidden', 'false');
  document.body.style.overflow = 'hidden';
}}
async function decide(winner, skipped=false) {{
  if (!picking) return;
  picking = false;
  showBusy();
  const ov = document.getElementById('ov');
  if (ov) ov.classList.remove('open');
  state.overlay = false;
  const body = {{
    reviewer: state.reviewer,
    left_key: state.left.key,
    right_key: state.right.key,
    winner: skipped ? null : winner,
    skipped: skipped,
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
  }} finally {{
    location.reload();
  }}
}}
document.querySelectorAll('[data-pick]').forEach(el => {{
  el.addEventListener('click', (e) => {{
    if (!picking) return;
    if (e.target.closest('a')) return;
    decide(el.dataset.pick);
  }});
}});
function openOv(side) {{
  if (!picking) return;
  state.overlay = true; state.overlayOpened = true; state.overlaySide = side; picking = false;
  const meta = side === 'left' ? state.left : state.right;
  state.photoIdx = 0;
  const ov = document.getElementById('ov');
  ov.classList.add('open');
  renderOv(meta);
}}
function renderOv(meta) {{
  const photos = meta.photos || [];
  document.getElementById('ovMain').src = photos[state.photoIdx] || '';
  const th = document.getElementById('ovThumbs');
  th.innerHTML = photos.map((p,i) => '<img src="'+p+'" data-i="'+i+'">').join('');
  th.querySelectorAll('img').forEach(img => img.onclick = () => {{ state.photoIdx = +img.dataset.i; renderOv(meta); }});
  const known = Object.entries(meta.known||{{}}).filter(([,v])=>v).map(([k])=>k);
  document.getElementById('ovExtra').textContent = 'Fields known: ' + known.join(', ');
}}
document.querySelectorAll('[data-overlay]').forEach(el => el.onclick = (e) => {{
  e.preventDefault();
  e.stopPropagation();
  if (!picking && !state.overlay) return;
  openOv(el.dataset.overlay);
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
  if (state.overlay) {{
    const meta = state.overlaySide === 'left' ? state.left : state.right;
    if (e.key === 'ArrowLeft') {{ state.photoIdx = Math.max(0, state.photoIdx-1); renderOv(meta); }}
    if (e.key === 'ArrowRight') {{ state.photoIdx = Math.min((meta.photos||[]).length-1, state.photoIdx+1); renderOv(meta); }}
    if (e.key === 'Escape') document.getElementById('ovClose').click();
    return;
  }}
  if (!picking) return;
  if (e.key === 'ArrowLeft') decide(state.left.key);
  if (e.key === 'ArrowRight') decide(state.right.key);
  if (e.key === ' ') {{ e.preventDefault(); decide(null, true); }}
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
        rank_rows += (
            f'<tr><td>{i}</td><td>{_e(_listing_label(row))}{badge}{be}</td>'
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
        inner = (
            f'{img}'
            f'<div class="place">{_e(label)}</div>'
            f'<div class="name">{_e(_listing_label(row))}{badge}</div>'
            f'<div class="meta">#{rank} · score {row["score"]:.3f} · seen {row.get("n_shown", 0)}</div>'
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


def _movers_blurb(movers: list[dict]) -> str:
    if len(movers) >= 2:
        a, b = movers[0], movers[1]
        return f"{a['line']} more than {b['label'].lower()} right now."
    if len(movers) == 1:
        return f"{movers[0]['line']}."
    return "Not enough signal yet to see which features move rankings."


def results_page(
    reviewer: str,
    compared: list[dict],
    unseen: list[dict],
    movers: list[dict],
    n: int,
    *,
    concluded: bool = False,
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
        if unseen:
            unseen_html = (
                f'<h3 style="margin-top:28px">Also scoring well (not shown yet)</h3>'
                f'<p class="standings-note">These weren’t in your matchups. '
                f"We scored them from the patterns in your picks.</p>"
                f"{_rank_table(unseen, start_i=len(compared) + 1, show_badge=True)}"
            )
    heading = "Your Results" if concluded else "Current Standings"
    compared_intro = ""
    if not concluded:
        compared_intro = (
            "<h3>You’ve compared</h3>"
            '<p class="standings-note">Every listing that showed up in at least one of your matchups, ranked by fit.</p>'
        )
    mover_rows = "".join(
        f"<tr><td>{i}</td><td>{_e(m['label'])}</td>"
        f"<td>{_e(m['line'])}</td><td>{m['share']:.0%}</td></tr>"
        for i, m in enumerate(movers, 1)
    )
    movers_table = (
        f'<table class="rates"><tr><th>#</th><th>Feature</th><th>Effect</th><th>|w| share</th></tr>'
        f"{mover_rows}</table>"
        if mover_rows
        else '<p class="muted">No feature weights yet — keep comparing.</p>'
    )
    body = f"""
<div class="actions" style="margin:0 0 14px">
  <a class="btn" href="/mash/play">← Return</a>
</div>
<div class="panel">
  <div class="standings-head">
    <h2>{_e(heading)}</h2>
    <p class="muted">{n} comparisons</p>
  </div>
  {compared_intro}
  {compared_html}
  {unseen_html}
</div>
<div class="panel">
  <h2>For nerds</h2>
  <h3>How the score works</h3>
  <p>Score = feature fit (<code>w·x</code>) + leftover <code>u</code> (photos / vibe / unmodeled).
  Rankings sort by that score.</p>
  <h3>What moves rankings</h3>
  <p class="standings-note">{_e(_movers_blurb(movers))}</p>
  {movers_table}
</div>"""
    return page(f"{heading} — CasitaMash", body, who=f"playing as {reviewer}")