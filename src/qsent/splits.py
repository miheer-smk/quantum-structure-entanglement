"""
Fit/eval splits with runtime disjointness assertions.

Brief Part 8 item 7. Probes and readouts must never be fit on data they are evaluated on.
Two notions are needed here and they are NOT the same:

* **realization-disjoint** — no `h` VECTOR appears in both sides. This is the operative one
  for the disordered ensemble: every realization is a distinct 8-dimensional vector.
* **field-value-disjoint** — no scalar `h` value appears in both sides. Required for any
  uniform-field scan, where a realization is characterised by one number.

Per-site disorder makes the distinction sharp: two different realizations routinely share
individual `h_i` values while being entirely distinct vectors, so realization-disjointness
does not imply field-value-disjointness and vice versa. Both are asserted explicitly rather
than assumed from a shuffle.
"""

from __future__ import annotations

import numpy as np

__all__ = ["assert_realization_disjoint", "assert_field_value_disjoint",
           "delta_stratified_split"]


def _rows_as_keys(a: np.ndarray) -> set[bytes]:
    a = np.ascontiguousarray(np.asarray(a, dtype=np.float64))
    return {row.tobytes() for row in a}


def assert_realization_disjoint(fit: np.ndarray, evl: np.ndarray) -> None:
    """Raise if any realization VECTOR appears on both sides."""
    shared = _rows_as_keys(fit) & _rows_as_keys(evl)
    if shared:
        raise AssertionError(
            f"split leakage: {len(shared)} realization vector(s) appear in both fit and eval")


def assert_field_value_disjoint(fit: np.ndarray, evl: np.ndarray, decimals: int = 12) -> None:
    """Raise if any scalar field VALUE appears on both sides.

    Only meaningful for uniform-field scans. On i.i.d. per-site disorder this will almost
    always fail by construction, which is correct behaviour: it says the notion does not
    apply, rather than silently passing.
    """
    f = set(np.round(np.asarray(fit, dtype=np.float64).ravel(), decimals))
    e = set(np.round(np.asarray(evl, dtype=np.float64).ravel(), decimals))
    shared = f & e
    if shared:
        raise AssertionError(
            f"split leakage: {len(shared)} scalar field value(s) appear in both fit and eval")


def delta_stratified_split(
    h_fields: np.ndarray, delta: np.ndarray, n_bins: int = 10,
    frac_fit: float = 0.5, seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Realization-disjoint split, stratified so both sides span delta_r equally.

    An unstratified random split leaves the fit and eval sides with different delta_r
    distributions, which would confound any delta-dependent claim with a split artifact.
    Returns (fit_indices, eval_indices); disjointness is asserted before returning.
    """
    h = np.asarray(h_fields)
    d = np.asarray(delta)
    if len(h) != len(d):
        raise ValueError("h_fields and delta must have the same length")

    rng = np.random.default_rng(seed)
    edges = np.quantile(d, np.linspace(0, 1, n_bins + 1))
    edges[-1] = np.nextafter(edges[-1], np.inf)
    fit_idx: list[int] = []
    eval_idx: list[int] = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        members = np.flatnonzero((d >= lo) & (d < hi))
        rng.shuffle(members)
        k = int(round(frac_fit * len(members)))
        fit_idx.extend(members[:k].tolist())
        eval_idx.extend(members[k:].tolist())

    fit_idx = np.array(sorted(fit_idx), dtype=int)
    eval_idx = np.array(sorted(eval_idx), dtype=int)
    assert not set(fit_idx.tolist()) & set(eval_idx.tolist())
    assert_realization_disjoint(h[fit_idx], h[eval_idx])
    return fit_idx, eval_idx
