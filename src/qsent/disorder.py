"""
Realization-level criticality parameter delta_r and delta-stratification.

The infinite-randomness fixed point of the random transverse-field Ising chain sits at
[ln J] = [ln h]. The training ensemble does not: with J == 1 and h ~ U(0.1, 2.0),
E[ln h] = -0.149183, so the ensemble mean lies on the *ordered* side. delta_r orders
individual realizations so the critical sub-ensemble can be selected in-distribution.

    delta_r = ( sum_i ln J_i  -  sum_i ln h_i ) / ( sqrt(L) * sigma_lnh )

Normalisation fixed by the author: sigma = sqrt(L) * sigma_lnh. This is not an arbitrary
unit-variance rescaling. The IRFP correlation-length exponent is nu = 2, so the finite-size
scaling variable is delta * L^(1/nu) = delta * sqrt(L); writing
sum_i ln h_i = L E[ln h] + sqrt(L) sigma_lnh xi with xi ~ N(0,1) gives

    delta_r = sqrt(L) * ( -E[ln h] / sigma_lnh )  -  xi  =  delta_ens * L^(1/nu)  -  xi

so delta_r *is* the scaling variable, with an O(1) realization-level fluctuation on top.

Sign convention: delta_r > 0 means couplings dominate -> ORDERED (ferromagnetic);
delta_r < 0 -> PARAMAGNETIC. Small h therefore gives large positive delta_r.
"""

from __future__ import annotations

import numpy as np

__all__ = ["sigma_lnh_uniform", "delta_r", "stratify", "select_spanning_realizations"]


def sigma_lnh_uniform(h_min: float = 0.1, h_max: float = 2.0) -> tuple[float, float]:
    """Exact (E[ln h], sd[ln h]) for h ~ Uniform(h_min, h_max)."""
    a, b = float(h_min), float(h_max)
    m1 = ((b * np.log(b) - b) - (a * np.log(a) - a)) / (b - a)
    f = lambda x: x * (np.log(x) ** 2 - 2 * np.log(x) + 2)
    m2 = (f(b) - f(a)) / (b - a)
    return float(m1), float(np.sqrt(m2 - m1**2))


def delta_r(
    h_fields: np.ndarray,
    J_fields: np.ndarray | None = None,
    sigma_lnh: float | None = None,
) -> np.ndarray:
    """delta_r for each realization. `h_fields` is (N, L); returns (N,).

    `J_fields` is (N, L-1) if bond disorder is present. When omitted, J == 1 is assumed
    and sum_i ln J_i vanishes identically -- true of the current pipeline, but the general
    path is implemented and unit-tested so this is not silently specialised to J == 1.
    """
    h = np.asarray(h_fields, dtype=np.float64)
    if h.ndim == 1:
        h = h[None, :]
    N, L = h.shape
    if sigma_lnh is None:
        sigma_lnh = sigma_lnh_uniform()[1]

    sum_ln_h = np.log(h).sum(axis=1)
    if J_fields is None:
        sum_ln_J = np.zeros(N)
    else:
        J = np.asarray(J_fields, dtype=np.float64)
        if J.ndim == 1:
            J = J[None, :]
        sum_ln_J = np.log(J).sum(axis=1)
    return (sum_ln_J - sum_ln_h) / (np.sqrt(L) * sigma_lnh)


def stratify(d: np.ndarray, edges: np.ndarray) -> dict[str, int]:
    """Occupancy of delta_r bins defined by `edges`."""
    d = np.asarray(d)
    out: dict[str, int] = {}
    for lo, hi in zip(edges[:-1], edges[1:]):
        out[f"[{lo:+.2f},{hi:+.2f})"] = int(((d >= lo) & (d < hi)).sum())
    return out


def select_spanning_realizations(
    h_fields: np.ndarray, targets=(2.0, 1.0, 0.0, -1.0, -2.0), sigma_lnh=None
) -> tuple[np.ndarray, np.ndarray]:
    """Pick the realizations closest to each target delta_r.

    Used to choose the ED/free-fermion cross-validation cases so they *span* the
    criticality axis rather than clustering in one phase. Returns (indices, delta_values).
    """
    d = delta_r(h_fields, sigma_lnh=sigma_lnh)
    idx = np.array([int(np.argmin(np.abs(d - t))) for t in targets])
    return idx, d[idx]
