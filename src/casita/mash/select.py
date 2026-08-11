from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass

import numpy as np

from .features import (
    ALWAYS_SHOW,
    FEATURE_LABELS,
    ListingFeatures,
    clone_hypothetical,
    eligible,
)
from .model import FitResult, predict_proba, score_listing


@dataclass
class PairOffer:
    left: ListingFeatures
    right: ListingFeatures
    strategy: str
    why_line: str
    is_hypothetical: bool = False


def _pair_key(a: str, b: str) -> tuple[str, str]:
    return (a, b) if a <= b else (b, a)


def _visible_features(feature_order: list[str]) -> list[str]:
    return list(ALWAYS_SHOW) + [f for f in feature_order if f not in ALWAYS_SHOW]


def _feature_delta(a: ListingFeatures, b: ListingFeatures, name: str) -> float | None:
    if not a.known.get(name) or not b.known.get(name):
        return None
    va, vb = a.values.get(name), b.values.get(name)
    if va is None or vb is None:
        return None
    return float(va - vb)


HIGHER_BETTER = {
    "beds", "baths", "sqft", "outdoor", "laundry", "parking", "dogs",
    "light", "condition", "view", "is_sf",
}
LOWER_BETTER = {
    "price", "price_per_bed", "price_per_sqft",
    "trail", "beach", "bakery", "grocery", "premium_grocery", "bar", "farmers_market", "ferry",
}
COST_FEATURES = {"price", "price_per_bed", "price_per_sqft"}


def _meaningful(d: float, a: ListingFeatures, b: ListingFeatures, name: str, tol: float = 0.04) -> bool:
    scale = max(abs(a.values.get(name) or 0), abs(b.values.get(name) or 0), 1.0)
    if name in COST_FEATURES:
        return abs(d) >= max(25.0 if name == "price" else 0.05, tol * scale)
    if name in LOWER_BETTER:
        return abs(d) >= max(2.0, tol * scale)
    return abs(d) / scale > tol


def advantage(a: ListingFeatures, b: ListingFeatures, name: str) -> int:
    """1 if a is better on name, -1 if b is better, else 0."""
    d = _feature_delta(a, b, name)
    if d is None or not _meaningful(d, a, b, name):
        return 0
    if name in HIGHER_BETTER:
        return 1 if d > 0 else -1
    if name in LOWER_BETTER:
        return 1 if d < 0 else -1
    return 0


def has_tradeoff(a: ListingFeatures, b: ListingFeatures, names: list[str]) -> bool:
    """True only if each side is better on at least one visible feature."""
    a_wins = False
    b_wins = False
    for name in names:
        adv = advantage(a, b, name)
        if adv > 0:
            a_wins = True
        elif adv < 0:
            b_wins = True
        if a_wins and b_wins:
            return True
    return False


def dominated(a: ListingFeatures, b: ListingFeatures, names: list[str]) -> bool:
    """True if a is better-or-equal on every comparable feature and strictly better on one."""
    saw_strict = False
    saw_any = False
    for name in names:
        d = _feature_delta(a, b, name)
        if d is None:
            continue
        saw_any = True
        adv = advantage(a, b, name)
        if adv < 0:
            return False
        if adv > 0:
            saw_strict = True
        elif _meaningful(d, a, b, name):
            return False
    return saw_any and saw_strict


def opposing_focus(a: ListingFeatures, b: ListingFeatures, names: list[str], focus: str) -> str | None:
    """Feature where the other side wins, opposite the focus winner."""
    focus_adv = advantage(a, b, focus)
    if focus_adv == 0:
        return None
    winner_is_a = focus_adv > 0
    best = None
    best_mag = -1.0
    for name in names:
        if name == focus:
            continue
        adv = advantage(a, b, name)
        if winner_is_a and adv >= 0:
            continue
        if (not winner_is_a) and adv <= 0:
            continue
        d = _feature_delta(a, b, name)
        if d is None:
            continue
        mag = abs(d)
        if mag > best_mag:
            best_mag = mag
            best = name
    return best


def varying_count(a: ListingFeatures, b: ListingFeatures, names: list[str], tol: float = 0.05) -> int:
    n = 0
    for name in names:
        d = _feature_delta(a, b, name)
        if d is None:
            continue
        va = a.values[name]
        scale = max(abs(va or 0), abs(b.values[name] or 0), 1.0)
        if abs(d) / scale > tol:
            n += 1
    return n


def cold_start_score(a: ListingFeatures, b: ListingFeatures, names: list[str], coverage: dict[str, int]) -> float:
    var = varying_count(a, b, names)
    if var == 0:
        return -1e9
    gain = 0.0
    for name in names:
        d = _feature_delta(a, b, name)
        if d is None or abs(d) < 1e-9:
            continue
        gain += 1.0 / (1 + coverage.get(name, 0))
    return gain * (1.0 / max(var, 1))


def expected_weight_info(fit: FitResult, a: ListingFeatures, b: ListingFeatures) -> float:
    p = predict_proba(fit, a, b)
    p = min(max(p, 1e-6), 1 - 1e-6)
    entropy = -(p * math.log(p) + (1 - p) * math.log(1 - p))
    xa = np.array([1.0])
    from .features import vectorize
    da = vectorize(a, fit.names, fit.knots, fit.means, fit.scales)
    db = vectorize(b, fit.names, fit.knots, fit.means, fit.scales)
    d = da - db
    moving = float(np.sum(np.abs(fit.w) * np.abs(d)))
    return entropy * moving / (1.0 + varying_count(a, b, fit.active_features))


def why_line(fit: FitResult | None, a: ListingFeatures, b: ListingFeatures, feature_order: list[str], strategy: str, *, probe_features: list[str] | None = None) -> str:
    def _label(name: str) -> str:
        text = FEATURE_LABELS.get(name, name.replace("_", " ")).lower()
        return f'"{text}"'

    def _join_labels(names: list[str]) -> str:
        labels = [_label(n) for n in names]
        if len(labels) == 1:
            return labels[0]
        if len(labels) == 2:
            return f"{labels[0]} and {labels[1]}"
        return f"{labels[0]}, {labels[1]}, and {labels[2]}"

    def _both_known(name: str) -> bool:
        return bool(a.known.get(name) and b.known.get(name)
                    and a.values.get(name) is not None and b.values.get(name) is not None)

    probes = [p for p in (probe_features or []) if p]
    if probes:
        active = [p for p in probes if _both_known(p) or p in _visible_features(feature_order)]
        if not active:
            active = probes[:2]
        if len(active) >= 2:
            return (
                f"Probing {_join_labels(active[:2])} from your preference memo. "
                f"Your pick here will show us how you trade those off."
            )
        return (
            f"Probing {_label(active[0])} from your preference memo. "
            f"Your pick here will show us how much that matters to you."
        )

    visible = _visible_features(feature_order)
    comparable = [n for n in visible if _both_known(n)]

    if fit is None:
        deltas = []
        for name in comparable:
            d = _feature_delta(a, b, name)
            if d is None or not _meaningful(d, a, b, name):
                continue
            vals = [a.values[name], b.values[name]]
            sd = float(np.std(vals)) if len(vals) > 1 else 1.0
            sd = sd or 1.0
            deltas.append((abs(d) / sd, name))
        deltas.sort(reverse=True)
        tops = [n for _, n in deltas[:3]]
        if len(tops) > 2:
            # cold-start: keep 3rd only if its relative delta is meaningful vs #1
            if deltas[2][0] < 0.25 * deltas[0][0]:
                tops = tops[:2]
        if not tops:
            return "Still learning your taste."
        return (
            f"Still learning your taste. These listings differ most on {_join_labels(tops)}."
        )

    contrib = []
    from .features import vectorize
    da = vectorize(a, fit.names, fit.knots, fit.means, fit.scales)
    db = vectorize(b, fit.names, fit.knots, fit.means, fit.scales)
    for name in comparable:
        if name not in fit.names:
            continue
        d = _feature_delta(a, b, name)
        if d is None or not _meaningful(d, a, b, name):
            continue
        i = fit.names.index(name)
        contrib.append((abs(fit.w[i] * (da[i] - db[i])), name))
    contrib.sort(reverse=True)
    tops = []
    top_c = 0.0
    for c, n in contrib:
        if c <= 0:
            continue
        if not tops:
            tops.append(n)
            top_c = c
            continue
        if len(tops) == 1:
            tops.append(n)
            continue
        if len(tops) == 2 and c >= 0.25 * top_c:
            tops.append(n)
        break
    if strategy.startswith("staircase"):
        focus_name = None
        if ":" in strategy:
            focus_name = strategy.split(":", 1)[1]
        if focus_name and not _both_known(focus_name):
            focus_name = None
        other = opposing_focus(a, b, comparable, focus_name) if focus_name else None
        if focus_name and other:
            return (
                f"These trade off {_label(focus_name)} and {_label(other)}. "
                f"Your pick here will show us how much they matter to you."
            )
        focus = (
            _label(focus_name) if focus_name
            else (_label(tops[0]) if tops else '"one feature"')
        )
        return (
            f"These listings look alike except for {focus}. "
            f"Your pick here will show us how much that matters to you."
        )
    if not tops:
        return "These listings are close. Your pick here will help us order them."
    return (
        f"These listings differ most on {_join_labels(tops)}. "
        f"Your pick here will show us how you trade those off."
    )


def _staircase_real(
    pool: list[ListingFeatures],
    feature_order: list[str],
    seen: set[tuple[str, str]],
    focus: str,
) -> PairOffer | None:
    visible = _visible_features(feature_order)
    if focus in COST_FEATURES:
        return None
    best = None
    best_score = -1e18
    for i, a in enumerate(pool):
        for b in pool[i + 1:]:
            pk = _pair_key(a.key, b.key)
            if pk in seen or a.key == b.key:
                continue
            if not has_tradeoff(a, b, visible):
                continue
            if advantage(a, b, focus) == 0:
                continue
            if opposing_focus(a, b, visible, focus) is None:
                continue
            price_d = _feature_delta(a, b, "price")
            beds_d = _feature_delta(a, b, "beds")
            if price_d is not None and abs(price_d) > 800:
                continue
            if beds_d is not None and abs(beds_d) > 0.6:
                continue
            var = varying_count(a, b, visible)
            score = abs(_feature_delta(a, b, focus) or 0) / (1 + max(0, var - 2))
            if score > best_score:
                best_score = score
                best = (a, b)
    if not best:
        return None
    a, b = best
    return PairOffer(a, b, f"staircase_real:{focus}", "", False)


def _staircase_hyp(
    pool: list[ListingFeatures],
    feature_order: list[str],
    focus: str,
    rng: random.Random,
) -> PairOffer | None:
    if focus in COST_FEATURES:
        return None
    cands = [f for f in pool if f.known.get(focus) and f.known.get("price")]
    if not cands:
        return None
    base = rng.choice(cands)
    v = float(base.values[focus])
    p = float(base.values["price"])
    step = max(10.0, round(0.35 * abs(v)))
    new_v = max(1.0, v - step)
    actual_step = v - new_v
    if actual_step < 8:
        return None
    rent_bump = 250
    label = FEATURE_LABELS.get(focus, focus.replace("_", " ")).lower()
    left = clone_hypothetical(base, {focus: v, "price": p}, "as listed")
    right = clone_hypothetical(
        base,
        {focus: new_v, "price": p + rent_bump},
        f"{actual_step:.0f} min closer to {label.replace('distance to ', '')}, +${rent_bump}/mo",
    )
    left.key = f"hypA:{base.key}"
    right.key = f"hypB:{base.key}"
    if not has_tradeoff(left, right, _visible_features(feature_order)):
        return None
    return PairOffer(
        left, right, f"staircase_hyp:{focus}",
        (
            f"Not a real choice between two homes — same place, two what-ifs. "
            f"They differ on {label} and rent."
        ),
        True,
    )


def _probe_boost(a: ListingFeatures, b: ListingFeatures, probe_features: list[str]) -> float:
    if not probe_features:
        return 0.0
    boost = 0.0
    for name in probe_features:
        if advantage(a, b, name) != 0:
            boost += 3.0
    if len(probe_features) >= 2:
        signs = [advantage(a, b, n) for n in probe_features[:3]]
        if 1 in signs and -1 in signs:
            boost += 5.0
    return boost


def select_pair(
    pool: list[ListingFeatures],
    feature_order: list[str],
    seen: set[tuple[str, str]],
    fit: FitResult | None,
    *,
    n_comparisons: int,
    recent_hyp: int,
    coverage: dict[str, int] | None = None,
    probe_features: list[str] | None = None,
    rng: random.Random | None = None,
) -> PairOffer | None:
    """Pick next pair. Play path passes fit=None and uses memo probe_features."""
    rng = rng or random.Random()
    coverage = coverage or {}
    probes = [p for p in (probe_features or []) if p]
    visible = _visible_features(feature_order)
    elig = [f for f in pool if eligible(f)[0]]
    if len(elig) < 2:
        return None

    use_hyp = (
        n_comparisons >= 10
        and recent_hyp < 1
        and rng.random() < 0.18
        and feature_order
    )
    focus_order = [f for f in probes if f not in COST_FEATURES]
    focus_order += [
        f for f in (list(feature_order) or ["trail", "beach", "grocery"])
        if f not in COST_FEATURES and f not in focus_order
    ]
    if not focus_order:
        focus_order = ["trail", "beach", "grocery"]
    if not probes:
        rng.shuffle(focus_order)

    def _finish(offer: PairOffer) -> PairOffer:
        offer.why_line = why_line(
            fit, offer.left, offer.right, feature_order, offer.strategy,
            probe_features=probes,
        )
        return offer

    if use_hyp:
        for focus in focus_order[:3]:
            offer = _staircase_hyp(elig, feature_order, focus, rng)
            if offer and has_tradeoff(offer.left, offer.right, visible):
                return _finish(offer)

    if n_comparisons >= 8 and feature_order and (probes or rng.random() < 0.45):
        for focus in focus_order:
            offer = _staircase_real(elig, feature_order, seen, focus)
            if offer and has_tradeoff(offer.left, offer.right, visible):
                return _finish(offer)

    best = None
    best_score = -1e18
    sample = elig if len(elig) <= 40 else rng.sample(elig, 40)
    for i, a in enumerate(sample):
        for b in sample[i + 1:]:
            if a.key == b.key:
                continue
            pk = _pair_key(a.key, b.key)
            if pk in seen:
                continue
            if not has_tradeoff(a, b, visible):
                continue
            if probes or fit is None:
                score = cold_start_score(a, b, visible, coverage) + _probe_boost(a, b, probes)
            else:
                p = predict_proba(fit, a, b)
                near = 1.0 - abs(p - 0.5) * 2
                info = expected_weight_info(fit, a, b)
                score = near * info
            if score > best_score:
                best_score = score
                best = (a, b)
    if not best:
        sample = elig
        for i, a in enumerate(sample):
            for b in sample[i + 1:]:
                if a.key == b.key:
                    continue
                pk = _pair_key(a.key, b.key)
                if pk in seen:
                    continue
                if not has_tradeoff(a, b, visible):
                    continue
                score = cold_start_score(a, b, visible, coverage) + _probe_boost(a, b, probes)
                if score > best_score:
                    best_score = score
                    best = (a, b)
    if not best:
        return None
    a, b = best
    strategy = "memo_probe" if probes else ("cold_start" if fit is None else "info_gain")
    return PairOffer(
        a, b, strategy,
        why_line(fit, a, b, feature_order, strategy, probe_features=probes),
        False,
    )



def params_per_feature() -> int:
    return 3


def comparison_estimate(n_features: int, comps_per_param: float = 4.0) -> tuple[int, int]:
    params = max(1, n_features) * params_per_feature() + 3
    mid = int(params * comps_per_param)
    return mid, mid + n_features * 10
