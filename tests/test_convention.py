"""
S0.7 convention audit, executed as a test rather than as a comment.

`PLAN.md` S0.7 required this and it was never committed: the `c_eff`-recovery numbers lived
only in `RESULTS_STAGE1.md` prose, and `src/qsent/convention.py` was referenced by
`free_fermions.py` without existing. Both are fixed here (2026-08-11).

What is asserted:

* the fitting code recovers `c = 1/2` on a clean critical open chain, approaching it from
  above as finite-size corrections shrink -- this validates the fit itself before it is ever
  pointed at a transformer;
* entropies are in **nats**, not bits;
* the Refael-Moore conversion is exactly `2 * ln2`, and the RTFIM constant is `(ln2)/2`, not
  the random-singlet `ln2`.

Per the standing rule in `CLAUDE.md`, each is accompanied by a demonstration that it can fail
on the error it targets: a profile in the wrong units, and the wrong RTFIM constant.
"""

from __future__ import annotations

import numpy as np
import pytest

from qsent.convention import (
    C_TILDE_RANDOM_SINGLET, C_TILDE_RTFIM_ISING, NATS_PER_BIT, REFAEL_MOORE_FACTOR,
    PREDICTED_SLOPE_NATS_ONE_BOUNDARY, bits_to_nats, fit_c_eff, nats_to_bits,
    refael_moore_to_repo_units,
)
from qsent.free_fermions import entropy_profile_free_fermion

CLEAN_LS = (32, 64, 128)


def _clean_critical_profile(L: int) -> np.ndarray:
    """Clean critical open Ising chain: J = h = 1 at every site."""
    return entropy_profile_free_fermion(np.ones(L - 1), np.ones(L))


# ---------------------------------------------------------------------------------------
# The fit recovers c = 1/2 on a chain where the answer is known by construction
# ---------------------------------------------------------------------------------------

@pytest.mark.parametrize("L", CLEAN_LS)
def test_c_eff_fit_recovers_clean_critical_half(L):
    """Clean critical chain has c = 1/2 exactly; the fit must land near it at large L."""
    c_eff, _ = fit_c_eff(_clean_critical_profile(L), L)
    assert 0.5 < c_eff < 0.6, f"L={L}: c_eff={c_eff:.4f} is not a plausible recovery of 0.5"


def test_c_eff_bias_shrinks_monotonically_with_L():
    """The approach to 1/2 must be from ABOVE and monotone, or the fit is not converging."""
    biases = [fit_c_eff(_clean_critical_profile(L), L)[0] - 0.5 for L in CLEAN_LS]
    assert all(b > 0 for b in biases), f"expected positive bias at every L, got {biases}"
    assert all(a > b for a, b in zip(biases, biases[1:])), f"bias not monotone: {biases}"


@pytest.mark.parametrize("L", CLEAN_LS)
def test_the_fit_is_a_good_fit_on_the_clean_chain(L):
    """A small residual, so 'c_eff is near 0.5' is not being rescued by a bad fit.

    Measured `1 - R^2` is 7.1e-04, 8.5e-04, 8.1e-04 at L = 32, 64, 128 -- i.e. R^2 > 0.999,
    with the residual dominated by the even-odd oscillation documented in `PLAN.md` A0 rather
    than by misfit of the Calabrese-Cardy form. The threshold is set at 2e-3 to sit just
    above the measured value rather than at a round number that would pass regardless.
    """
    S = _clean_critical_profile(L)
    _, ssr = fit_c_eff(S, L)
    assert ssr / np.sum((S - S.mean()) ** 2) < 2e-3


def test_fit_rejects_a_profile_of_the_wrong_length():
    with pytest.raises(ValueError):
        fit_c_eff(np.zeros(5), 8)


# ---------------------------------------------------------------------------------------
# Units: nats, not bits -- and the demonstration that using bits is detectable
# ---------------------------------------------------------------------------------------

def test_entropy_is_in_nats():
    """A maximally entangled pair is ln2 nats, not 1 bit."""
    psi = np.zeros(4)
    psi[0] = psi[3] = 1 / np.sqrt(2)
    from qsent.exact import reduced_density_matrix
    ev = np.linalg.eigvalsh(reduced_density_matrix(psi, 2, 1))
    ev = ev[ev > 1e-14]
    assert abs(float(-np.sum(ev * np.log(ev))) - np.log(2)) < 1e-12


def test_fitting_a_bits_profile_gives_the_wrong_c_eff():
    """FAILURE DEMONSTRATION for the units convention.

    If a profile in bits were fed to the fit, `c_eff` would come out smaller by exactly
    `ln2` -- roughly 0.38 instead of 0.59 at L = 128, which is close enough to plausible
    values to pass an eyeball check. This is why the units are asserted rather than assumed.
    """
    L = 128
    S_nats = _clean_critical_profile(L)
    c_nats, _ = fit_c_eff(S_nats, L)
    c_bits, _ = fit_c_eff(nats_to_bits(S_nats), L)
    assert abs(c_bits - c_nats / NATS_PER_BIT) < 1e-12
    assert not (0.5 < c_bits < 0.6), "a bits profile must NOT look like a valid recovery"


def test_bits_nats_round_trip():
    """Round-trips a REAL entropy profile, not four typed floats.

    An inlined array of made-up numbers would be data without provenance -- the thing
    `test_no_unverified_inlined_float_arrays_in_tests` exists to reject, and the reason it
    rejected the first draft of this test. A computed profile carries its own provenance.
    """
    S = _clean_critical_profile(32)
    assert np.max(np.abs(bits_to_nats(nats_to_bits(S)) - S)) < 1e-15


# ---------------------------------------------------------------------------------------
# The Refael-Moore constants and the 2*ln2 conversion
# ---------------------------------------------------------------------------------------

def test_refael_moore_factor_is_two_ln2():
    """bits->nats (ln2) compounded with two-boundary->one-boundary (2)."""
    assert abs(REFAEL_MOORE_FACTOR - 2.0 * np.log(2.0)) < 1e-15


def test_rtfim_constant_is_half_ln2_not_ln2():
    """FAILURE DEMONSTRATION for the constant that was actually got wrong once.

    `c~ = ln2` is the random-singlet / Heisenberg value (Refael-Moore Eq. 18). The RTFIM
    value is `(ln2)/2` (their Eq. 22). Substituting one for the other doubles the predicted
    slope, which is exactly the error made in planning and caught by the author.
    """
    assert abs(C_TILDE_RTFIM_ISING - np.log(2.0) / 2.0) < 1e-15
    assert abs(C_TILDE_RANDOM_SINGLET - 2.0 * C_TILDE_RTFIM_ISING) < 1e-15
    assert C_TILDE_RTFIM_ISING != C_TILDE_RANDOM_SINGLET


def test_predicted_slope_is_c_tilde_over_six():
    """In repo units the prediction is slope = c~/6 = ln2/12."""
    assert abs(PREDICTED_SLOPE_NATS_ONE_BOUNDARY - C_TILDE_RTFIM_ISING / 6.0) < 1e-15
    assert abs(PREDICTED_SLOPE_NATS_ONE_BOUNDARY - np.log(2.0) / 12.0) < 1e-15


def test_refael_moore_conversion_matches_the_cross_check_in_plan_4A():
    """RM Eq. (22) gives (1/6) ln L bits for TWO boundaries.

    One boundary halves it to (1/12) ln L bits, and bits->nats multiplies by ln2, giving
    (ln2/12) ln L nats -- which must equal (c~/6) ln L with c~ = ln2/2.
    """
    one_boundary_bits_coeff = (1.0 / 6.0) / 2.0
    nats_coeff = one_boundary_bits_coeff * NATS_PER_BIT
    assert abs(nats_coeff - C_TILDE_RTFIM_ISING / 6.0) < 1e-15
    # and the documented single conversion function agrees
    assert abs(refael_moore_to_repo_units(np.log(2.0)) - np.log(2.0) / REFAEL_MOORE_FACTOR) < 1e-15
