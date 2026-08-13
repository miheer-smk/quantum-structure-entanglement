"""
Entropy units, boundary conventions, and the Calabrese-Cardy fit.

`free_fermions.py` has referred to this module since Stage 0; it did not exist until
2026-08-11. Everything here was previously either inlined in a test or written only as prose,
which is the state PLAN.md S0.7 explicitly forbade: the convention audit was to be "executed
as a test, not a comment".

THE CONVERSION, AND WHY IT IS A FACTOR OF 2 * ln2
------------------------------------------------
Refael & Moore, PRL 93, 260602 (2004), arXiv:cond-mat/0406737, differs from this repository
in two independent ways that compound:

|            | Refael-Moore (2004)                         | this repository                    |
|------------|---------------------------------------------|------------------------------------|
| Log base   | **bits** (S = -Tr rho log2 rho, their Eq. 1) | **nats** (`np.log`)                |
| Geometry   | segment inside a chain -> **two** boundaries | open chain cut once -> **one**     |
|            | -> `c~/3` convention (their Eq. 3)          | -> `c~/6` convention               |

So a printed Refael-Moore constant and anything this codebase prints differ by `2 * ln2`.

The RTFIM infinite-randomness value is `c~ = (ln2)/2 ~ 0.34657` (their Eq. 22, and the text:
"the effective central charge of the random quantum Ising model is c~ = 1/2 ln 2"). The
random-singlet / Heisenberg value `c~ = ln2` (their Eq. 18) is a DIFFERENT constant and is
wrong for this model -- an error that was made once in planning and caught by the author.

In this repository's units the pre-registered form is

    S(l) = (c~/6) * ln[(2L/pi) * sin(pi*l/L)] + k        open BC, nats, one boundary

so the predicted slope against `ln[(2L/pi) sin(pi*l/L)]` is `c~/6 = ln2/12 ~ 0.05776`.

WHAT THE c_eff FIT CANNOT DO
----------------------------
`fit_c_eff` is symmetric under `l -> L-l`, because `sin(pi*l/L)` is. A fully mirrored profile
fits with identical `c_eff` and identical residuals. The fit therefore **cannot** serve as an
orientation check; `tests/test_orientation.py` exists for that and proves this blindness
explicitly. See also the finite-size bias measured in `PLAN.md` §A0: at L <= 12 the fit cannot
distinguish clean Ising from the IRFP, and no universality-class claim is made from it.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "NATS_PER_BIT", "C_TILDE_RTFIM_ISING", "C_TILDE_RANDOM_SINGLET",
    "PREDICTED_SLOPE_NATS_ONE_BOUNDARY", "REFAEL_MOORE_FACTOR",
    "bits_to_nats", "nats_to_bits", "refael_moore_to_repo_units",
    "cc_design_matrix", "fit_c_eff",
]

NATS_PER_BIT: float = float(np.log(2.0))

#: RTFIM infinite-randomness effective central charge, Refael-Moore Eq. (22).
C_TILDE_RTFIM_ISING: float = float(np.log(2.0) / 2.0)

#: Random-singlet (Heisenberg) value, Refael-Moore Eq. (18). NOT the RTFIM value.
C_TILDE_RANDOM_SINGLET: float = float(np.log(2.0))

#: Slope of S against ln[(2L/pi) sin(pi l/L)] predicted for the RTFIM in repo units (nats,
#: one boundary): c~/6 = ln2/12.
PREDICTED_SLOPE_NATS_ONE_BOUNDARY: float = float(np.log(2.0) / 12.0)

#: The compounded bits->nats and two-boundary->one-boundary factor.
REFAEL_MOORE_FACTOR: float = float(2.0 * np.log(2.0))


def bits_to_nats(s: float | np.ndarray) -> float | np.ndarray:
    """Convert an entropy in bits to nats."""
    return np.asarray(s, dtype=np.float64) * NATS_PER_BIT


def nats_to_bits(s: float | np.ndarray) -> float | np.ndarray:
    """Convert an entropy in nats to bits."""
    return np.asarray(s, dtype=np.float64) / NATS_PER_BIT


def refael_moore_to_repo_units(c_tilde_bits_two_boundary: float) -> float:
    """Convert a Refael-Moore constant (bits, two boundaries) to repo units (nats, one).

    The two differences compound to `2 * ln2`; this is the single documented conversion any
    comparison to that paper must route through, rather than being re-derived at each use.
    """
    return float(c_tilde_bits_two_boundary) / REFAEL_MOORE_FACTOR


def cc_design_matrix(L: int) -> np.ndarray:
    """Design matrix [x, 1] for S(l) = (c/6) x + k with x = ln[(2L/pi) sin(pi l/L)].

    Rows run over internal cuts l = 1 .. L-1, matching the profile convention fixed in
    `qsent.exact`: profile index i corresponds to cut l = i + 1.
    """
    l = np.arange(1, L)
    x = np.log((2.0 * L / np.pi) * np.sin(np.pi * l / L))
    return np.vstack([x, np.ones_like(x)]).T


def fit_c_eff(S: np.ndarray, L: int) -> tuple[float, float]:
    """Least-squares (c_eff, SSR) for an entropy profile in **nats** on an open chain.

    Parameters
    ----------
    S : (L-1,) entropy at every internal cut, nats, block A = sites [0, l).
    L : chain length.

    Returns
    -------
    (c_eff, ssr) -- `c_eff = 6 * slope`, and the sum of squared residuals.

    Note this returns `c_eff` in the ONE-boundary convention, so a clean critical Ising chain
    gives ~0.5 (from above at finite L) and the RTFIM asymptote is `ln2/2 ~ 0.347`.
    """
    S = np.asarray(S, dtype=np.float64)
    if S.shape[0] != L - 1:
        raise ValueError(f"expected a profile of length L-1={L - 1}, got {S.shape[0]}")
    A = cc_design_matrix(L)
    coef, *_ = np.linalg.lstsq(A, S, rcond=None)
    return float(6.0 * coef[0]), float(np.sum((S - A @ coef) ** 2))
