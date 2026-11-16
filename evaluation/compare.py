"""Execution-accuracy comparison between a predicted result set and a gold result set.

Rules (documented in the README):
- Column names and column order do not matter. Each gold column must match a distinct predicted
  column; extra predicted columns are allowed.
- Row order does not matter unless the question sets order_matters.
- The number of rows must be equal.
- A country may be given as its code (`DE`) or its name (`Germany`).
- A share may be given as a fraction (0.0412) or a percentage (4.12).
- Numbers: integers must match exactly; other numbers within a relative tolerance of 0.1 %, or within
  half a unit of the predicted value's rounding when that rounding keeps the value within 3 %.
- NULL matches only NULL. Strings are compared case-insensitively after trimming.
"""
from __future__ import annotations

import itertools
import math
from decimal import Decimal
from typing import Any, Sequence

Row = Sequence[Any]

# Country name -> geo code, filled from dim_country by set_country_names().
_COUNTRY_ALIASES: dict[str, str] = {}


def set_country_names(pairs: list[tuple[str, str]]) -> None:
    """pairs: (geo, country_name)."""
    _COUNTRY_ALIASES.clear()
    for geo, name in pairs:
        _COUNTRY_ALIASES[name.strip().lower()] = geo.lower()
        _COUNTRY_ALIASES[name.strip().lower().rstrip("*")] = geo.lower()
    _COUNTRY_ALIASES.update({"greece": "el", "united kingdom": "uk", "gr": "el", "gb": "uk",
                             "eu27": "eu27_2020", "eu-27": "eu27_2020", "eu": "eu27_2020"})


def _norm(v: Any) -> Any:
    if v is None:
        return None
    if isinstance(v, bool):
        return v
    if isinstance(v, Decimal):
        v = float(v)
    if isinstance(v, (int, float)):
        if isinstance(v, float) and math.isnan(v):
            return None
        return float(v)
    s = str(v).strip().lower()
    if s in ("true", "false"):
        return s == "true"
    return _COUNTRY_ALIASES.get(s, s)


def _decimals(x: float) -> int | None:
    """Number of decimals if x looks rounded (at most 4 decimals), else None."""
    for d in range(0, 5):
        if abs(x - round(x, d)) < 1e-12:
            return d
    return None


def values_match(pred: Any, gold: Any) -> bool:
    p, g = _norm(pred), _norm(gold)
    if p is None or g is None:
        return p is None and g is None
    if isinstance(g, bool) or isinstance(p, bool):
        return p == g
    if isinstance(g, float) and isinstance(p, float):
        if g.is_integer() and p.is_integer():
            return p == g
        tol = max(1e-9, 1e-3 * abs(g))
        d = _decimals(p)
        if d is not None and 10 ** -d <= 0.03 * abs(g):
            tol = max(tol, 0.5 * 10 ** -d + 1e-12)
        return abs(p - g) <= tol
    if isinstance(g, float) or isinstance(p, float):
        return False
    return p == g


def _column(rows: list[Row], j: int) -> list[Any]:
    return [r[j] for r in rows]


def _scaled(col: list[Any], factor: float) -> list[Any]:
    return [v * factor if isinstance(v, (int, float)) and not isinstance(v, bool) and v is not None else v for v in col]


def _multiset_match(pred: list[Any], gold: list[Any]) -> bool:
    used = [False] * len(pred)
    for g in gold:
        for i, p in enumerate(pred):
            if not used[i] and values_match(p, g):
                used[i] = True
                break
        else:
            return False
    return True


def results_match(pred_cols: list[str], pred_rows: list[Row], gold_cols: list[str], gold_rows: list[Row],
                  order_matters: bool = False) -> bool:
    if len(pred_rows) != len(gold_rows):
        return False
    if not gold_rows:
        return True
    n_gold, n_pred = len(gold_cols), len(pred_cols)
    if n_pred < n_gold:
        return False
    # Candidate predicted columns (with a scale factor) for each gold column.
    pcols = [_column(pred_rows, k) for k in range(n_pred)]
    candidates: list[list[tuple[int, float]]] = []
    for j in range(n_gold):
        g = _column(gold_rows, j)
        cands = []
        for k in range(n_pred):
            for factor in (1.0, 0.01, 100.0):
                col = pcols[k] if factor == 1.0 else _scaled(pcols[k], factor)
                if (all(values_match(a, b) for a, b in zip(col, g)) if order_matters else _multiset_match(col, g)):
                    cands.append((k, factor))
                    break
        if not cands:
            return False
        candidates.append(cands)
    # Try combinations of distinct predicted columns; check whole rows match.
    for combo in itertools.islice(itertools.product(*candidates), 5000):
        ks = [k for k, _ in combo]
        if len(set(ks)) != len(ks):
            continue
        proj = [[(r[k] * f if f != 1.0 and isinstance(r[k], (int, float)) and not isinstance(r[k], bool) else r[k])
                 for k, f in combo] for r in pred_rows]
        if order_matters:
            if all(all(values_match(a, b) for a, b in zip(pr, gr)) for pr, gr in zip(proj, gold_rows)):
                return True
        else:
            used = [False] * len(proj)
            ok = True
            for gr in gold_rows:
                for i, pr in enumerate(proj):
                    if not used[i] and all(values_match(a, b) for a, b in zip(pr, gr)):
                        used[i] = True
                        break
                else:
                    ok = False
                    break
            if ok:
                return True
    return False
