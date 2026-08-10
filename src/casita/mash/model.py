from __future__ import annotations

import json
from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit

from .features import (
    ALWAYS_SHOW,
    HINGE_FEATURES,
    NUMERIC_FEATURES,
    ListingFeatures,
    build_design_names,
    vectorize,
)


@dataclass
class FitResult:
    names: list[str]
    w: np.ndarray
    u: dict[str, float]
    keys: list[str]
    knots: dict[str, list[float]]
    means: dict[str, float]
    scales: dict[str, float]
    lambda_w: float
    lambda_u: float
    heldout_acc: float
    n_comparisons: int
    active_features: list[str]

    def to_dict(self) -> dict:
        return {
            "names": self.names,
            "w": self.w.tolist(),
            "u": self.u,
            "keys": self.keys,
            "knots": self.knots,
            "means": self.means,
            "scales": self.scales,
            "lambda_w": self.lambda_w,
            "lambda_u": self.lambda_u,
            "heldout_acc": self.heldout_acc,
            "n_comparisons": self.n_comparisons,
            "active_features": self.active_features,
        }

    @classmethod
    def from_dict(cls, d: dict) -> FitResult:
        return cls(
            names=d["names"],
            w=np.array(d["w"], dtype=float),
            u={k: float(v) for k, v in d["u"].items()},
            keys=d["keys"],
            knots=d["knots"],
            means=d["means"],
            scales=d["scales"],
            lambda_w=float(d["lambda_w"]),
            lambda_u=float(d["lambda_u"]),
            heldout_acc=float(d["heldout_acc"]),
            n_comparisons=int(d["n_comparisons"]),
            active_features=d["active_features"],
        )


def _pairs_from_rows(rows: list, feat_map: dict[str, ListingFeatures]) -> list[tuple[str, str, float]]:
    pairs = []
    for r in rows:
        if r["skipped"] or not r["winner"]:
            continue
        a, b, w = r["left_key"], r["right_key"], r["winner"]
        weight = float(r["weight"] or 1.0)
        if a not in feat_map or b not in feat_map:
            if str(a).startswith("hyp:") or str(b).startswith("hyp:"):
                continue
            continue
        if w == a:
            pairs.append((a, b, weight))
        elif w == b:
            pairs.append((b, a, weight))
    return pairs


def _choose_knots(feat_map: dict[str, ListingFeatures], names: list[str]) -> dict[str, list[float]]:
    knots: dict[str, list[float]] = {}
    for name in HINGE_FEATURES:
        vals = [f.values[name] for f in feat_map.values() if f.known.get(name) and f.values.get(name) is not None]
        if len(vals) < 5:
            continue
        arr = np.array(vals, dtype=float)
        knots[name] = [float(np.percentile(arr, 60))]
    return knots


def _standardize(feat_map: dict[str, ListingFeatures], active: list[str]) -> tuple[dict[str, float], dict[str, float]]:
    means, scales = {}, {}
    fit_set = set(ALWAYS_SHOW) | set(active)
    for name in list(NUMERIC_FEATURES) + ["outdoor", "laundry", "parking", "dogs", "light", "condition", "view", "is_sf"]:
        if name not in fit_set and name not in ALWAYS_SHOW:
            continue
        vals = [f.values[name] for f in feat_map.values() if f.known.get(name) and f.values.get(name) is not None]
        if not vals:
            means[name] = 0.0
            scales[name] = 1.0
            continue
        arr = np.array(vals, dtype=float)
        means[name] = float(arr.mean())
        scales[name] = float(arr.std()) if arr.std() > 1e-8 else 1.0
    return means, scales


def _design(keys: list[str], feat_map: dict[str, ListingFeatures], names, knots, means, scales):
    X = np.vstack([vectorize(feat_map[k], names, knots, means, scales) for k in keys])
    idx = {k: i for i, k in enumerate(keys)}
    return X, idx


def _nll(theta, X, idx, pairs, p, lambda_w, lambda_u, n_keys):
    w = theta[:p]
    u = theta[p:]
    loss = 0.0
    for win, lose, weight in pairs:
        if win not in idx or lose not in idx:
            continue
        s = float(X[idx[win]] @ w + u[idx[win]] - (X[idx[lose]] @ w + u[idx[lose]]))
        loss -= weight * (s - np.logaddexp(0.0, s))
    loss += 0.5 * lambda_w * float(w @ w)
    loss += 0.5 * lambda_u * float(u @ u)
    return loss


def _fit_once(X, idx, pairs, lambda_w, lambda_u):
    n_keys, p = X.shape
    theta0 = np.zeros(p + n_keys)
    res = minimize(
        _nll,
        theta0,
        args=(X, idx, pairs, p, lambda_w, lambda_u, n_keys),
        method="L-BFGS-B",
        options={"maxiter": 80},
    )
    w = res.x[:p]
    u = res.x[p:]
    return w, u


def _accuracy(w, u, X, idx, pairs) -> float:
    if not pairs:
        return 0.5
    correct = 0.0
    total = 0.0
    for win, lose, weight in pairs:
        if win not in idx or lose not in idx:
            continue
        s = float(X[idx[win]] @ w + u[idx[win]] - (X[idx[lose]] @ w + u[idx[lose]]))
        correct += weight * (1.0 if s > 0 else 0.0)
        total += weight
    return float(correct / total) if total else 0.5


def fit(
    rows: list,
    feat_map: dict[str, ListingFeatures],
    active_features: list[str],
    *,
    lambda_grid: list[float] | None = None,
) -> FitResult | None:
    pairs = _pairs_from_rows(rows, feat_map)
    if len(pairs) < 3:
        return None
    keys = sorted({k for pair in pairs for k in pair[:2] if k in feat_map})
    if len(keys) < 2:
        return None

    knots = _choose_knots(feat_map, active_features)
    names = build_design_names(active_features, {k: len(v) for k, v in knots.items()})
    means, scales = _standardize(feat_map, active_features)
    X, idx = _design(keys, feat_map, names, knots, means, scales)

    lambda_grid = lambda_grid or [0.5, 2.0, 10.0]
    rng = np.random.default_rng(0)
    # Cap pairs for CV speed; final fit uses all.
    if len(pairs) > 200:
        pick = rng.choice(len(pairs), size=200, replace=False)
        cv_pairs = [pairs[i] for i in pick]
    else:
        cv_pairs = pairs
    order = np.arange(len(cv_pairs))
    rng.shuffle(order)
    folds = np.array_split(order, min(3, max(2, len(cv_pairs) // 10)))

    best = None
    best_score = -1.0
    for lw in lambda_grid:
        for lu in lambda_grid:
            accs = []
            for fi, fold in enumerate(folds):
                if len(fold) == 0:
                    continue
                test = [cv_pairs[i] for i in fold]
                train = [cv_pairs[i] for i in order if i not in set(fold.tolist())]
                if len(train) < 2:
                    continue
                w, u = _fit_once(X, idx, train, lw, lu)
                accs.append(_accuracy(w, u, X, idx, test))
            score = float(np.mean(accs)) if accs else 0.0
            if score > best_score:
                best_score = score
                best = (lw, lu)

    assert best is not None
    lw, lu = best
    w, u_arr = _fit_once(X, idx, pairs, lw, lu)
    u = {k: float(u_arr[idx[k]]) for k in keys}
    return FitResult(
        names=names,
        w=w,
        u=u,
        keys=keys,
        knots=knots,
        means=means,
        scales=scales,
        lambda_w=lw,
        lambda_u=lu,
        heldout_acc=best_score,
        n_comparisons=len(pairs),
        active_features=list(active_features),
    )


def score_listing(fit: FitResult, feats: ListingFeatures) -> float:
    x = vectorize(feats, fit.names, fit.knots, fit.means, fit.scales)
    return float(x @ fit.w + fit.u.get(feats.key, 0.0))


def predict_proba(fit: FitResult, a: ListingFeatures, b: ListingFeatures) -> float:
    return float(expit(score_listing(fit, a) - score_listing(fit, b)))


def feature_importance(fit: FitResult, top_n: int = 10) -> list[dict]:
    """Rank base features by |w| (standardized design → comparable magnitudes)."""
    from .features import FEATURE_LABELS, ROUTE_FEATURES

    lower_better = {
        "price", "price_per_bed", "price_per_sqft",
        *ROUTE_FEATURES,
    }
    rows: list[tuple[float, str, float]] = []
    for i, name in enumerate(fit.names):
        if "__hinge" in name:
            continue
        w = float(fit.w[i])
        aw = abs(w)
        if aw < 1e-9:
            continue
        rows.append((aw, name, w))
    rows.sort(reverse=True)
    total = sum(a for a, _, _ in rows) or 1.0
    out = []
    for aw, name, w in rows[:top_n]:
        label = FEATURE_LABELS.get(name, name.replace("_", " "))
        low = label.lower()
        if name in ROUTE_FEATURES:
            place = low.replace("distance to ", "")
            phrase = (
                f"Closer {place} pulls scores up"
                if w < 0
                else f"Farther {place} pulls scores up"
            )
        elif name in lower_better:
            phrase = (
                f"Lower {low} pulls scores up"
                if w < 0
                else f"Higher {low} pulls scores up"
            )
        else:
            phrase = (
                f"Better {low} pulls scores up"
                if w > 0
                else f"Worse {low} pulls scores up"
            )
        out.append({
            "feature": name,
            "label": label,
            "abs_w": aw,
            "share": aw / total,
            "weight": w,
            "line": phrase,
        })
    return out


def exchange_rates(fit: FitResult, n_boot: int = 40) -> list[dict]:
    out = []
    try:
        price_i = fit.names.index("price_per_bed")
    except ValueError:
        try:
            price_i = fit.names.index("price")
        except ValueError:
            return out
    sd_price = fit.scales.get("price_per_bed") or fit.scales.get("price") or 1.0
    w_price = fit.w[price_i]
    if abs(w_price) < 1e-12:
        return out

    rng = np.random.default_rng(1)
    for name in fit.active_features:
        if name not in fit.names or name in ("price", "price_per_bed", "price_per_sqft"):
            continue
        i = fit.names.index(name)
        sd_f = fit.scales.get(name, 1.0) or 1.0
        ratio = float(fit.w[i] / w_price)
        rate = ratio * (sd_price / sd_f)
        boots = []
        for _ in range(n_boot):
            noise = rng.normal(0, 0.15, size=fit.w.shape)
            wp = fit.w[price_i] + noise[price_i]
            if abs(wp) < 1e-12:
                continue
            boots.append(float((fit.w[i] + noise[i]) / wp * (sd_price / sd_f)))
        lo = float(np.percentile(boots, 10)) if boots else rate
        hi = float(np.percentile(boots, 90)) if boots else rate
        spans_zero = lo * hi < 0
        out.append({
            "feature": name,
            "rate": rate,
            "weight_ratio": ratio,
            "lo": lo,
            "hi": hi,
            "spans_zero": spans_zero,
        })
    out.sort(key=lambda d: abs(d["rate"]), reverse=True)
    return out


def hinge_sentences(fit: FitResult) -> list[str]:
    lines = []
    for name, ks in fit.knots.items():
        for i, k in enumerate(ks):
            hname = f"{name}__hinge{i}"
            if hname not in fit.names:
                continue
            hi = fit.names.index(hname)
            base_i = fit.names.index(name) if name in fit.names else None
            if base_i is None:
                continue
            w0, w1 = fit.w[base_i], fit.w[hi]
            if abs(w0) < 1e-9:
                continue
            mult = (w0 + w1) / w0
            if abs(mult - 1.0) < 0.15:
                continue
            strength = abs(mult)
            if "price" in name:
                if "bed" in name:
                    thresh = f"${k:,.0f}/bed"
                elif "sqft" in name:
                    thresh = f"${k:,.2f}/sqft"
                else:
                    thresh = f"${k:,.0f}/mo"
                if mult > 1:
                    lines.append(
                        f"Past {thresh}, rent starts to matter more (about {strength:.1f}×)."
                    )
                else:
                    lines.append(
                        f"Past {thresh}, the hit from higher rent eases a bit (about {strength:.1f}×)."
                    )
            else:
                label = name.replace("_", " ")
                if mult > 1:
                    lines.append(
                        f"Past {k:.0f} min to {label}, each extra minute hurts more "
                        f"(about {strength:.1f}×)."
                    )
                else:
                    lines.append(
                        f"Past {k:.0f} min to {label}, each extra minute hurts less "
                        f"(about {strength:.1f}×)."
                    )
    return lines


def top_stability(prev_top: list[str] | None, new_top: list[str], k: int = 20) -> float:
    if not prev_top:
        return 0.0
    a, b = set(prev_top[:k]), set(new_top[:k])
    if not a and not b:
        return 1.0
    return len(a & b) / len(a | b)


def plain_bradley_terry(pairs: list[tuple[str, str, float]], keys: list[str], lambda_u: float = 1.0):
    idx = {k: i for i, k in enumerate(keys)}
    n = len(keys)

    def nll(u):
        loss = 0.0
        for win, lose, weight in pairs:
            if win not in idx or lose not in idx:
                continue
            s = u[idx[win]] - u[idx[lose]]
            loss -= weight * (s - np.logaddexp(0.0, s))
        loss += 0.5 * lambda_u * float(u @ u)
        return loss

    res = minimize(nll, np.zeros(n), method="L-BFGS-B", options={"maxiter": 200})
    return {k: float(res.x[idx[k]]) for k in keys}


def break_even_price(fit: FitResult, feats: ListingFeatures, target_score: float) -> float | None:
    if feats.known.get("price"):
        return None
    if "price" not in fit.names:
        return None
    i = fit.names.index("price")
    w = fit.w[i]
    if abs(w) < 1e-12:
        return None
    sd = fit.scales.get("price", 1.0) or 1.0
    mu = fit.means.get("price", 0.0)
    # score contrib from standardized price ≈ w * (price - mu) / sd
    delta = target_score - score_listing(fit, feats)
    price = mu + (delta * sd) / w
    if not (800.0 <= price <= 15_000.0):
        return None
    return float(price)
