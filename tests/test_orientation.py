"""
Orientation gate: every entropy profile in this arm must be LEFT-block oriented.

WHY THIS IS A SEPARATE GATE FROM THE c_eff FIT
----------------------------------------------
The Calabrese-Cardy form goes as sin(pi*l/L), which is symmetric under l -> L-l. A fully
mirrored profile therefore fits with **identical c_eff and identical residuals**, so the
Stage 1 c_eff recovery check passes green on a reflected profile and cannot serve as an
orientation check. Demonstrated below in `test_ceff_fit_is_blind_to_mirroring`, which is
here to prove the *other* check is incapable of catching this error.

A consistent mirror applied to both the exact reference and the model-side path would
cancel in the paired per-realization test (H2 primary) and still be wrong in every figure
and every per-cut number. So orientation is asserted against ED directly, per provider.

CONVENTION (binding on every profile producer in this arm)
----------------------------------------------------------
    site j  <->  bit j of the state index   (site 0 is the least significant bit)
    cut l   ->  block A = sites [0, l)      (the LEFT block)
    profile index i (0-based) corresponds to cut l = i + 1

The inherited `qsae.observables._reduce_density_matrix` uses the opposite orientation
(subsystem A = the HIGH bits = the right block) while documenting the left block. This arm
does not call it: `grep -rn qsae src/ tests/` returns docstring mentions only.
"""

from __future__ import annotations

import inspect
import os

import numpy as np
import pytest

import qsent
from qsent.exact import entropy_profile_ed
from qsent.free_fermions import entropy_profile_free_fermion

# ---------------------------------------------------------------------------------------
# The maximally asymmetric realization found over 4000 sampled members of the pinned
# seed-1 ensemble. Its asymmetry is the test's power margin: an undetected mirror would
# show up as a 0.5695-nat error, roughly 5e9 times the 1e-10 agreement gate.
# ---------------------------------------------------------------------------------------
ASYM_INDEX = 46009
ASYM_DELTA_R = 1.716380
ASYM_MARGIN = 0.569541          # max |S(l) - S(L-l)| in nats, measured
ASYM_H = [1.9943493604660034, 1.9512919187545776, 0.57616066932678223,
          0.46158456802368164, 0.27974054217338562, 1.8719867467880249,
          0.55959624052047729, 0.10547444969415665]

# Every profile producer in this arm must appear here. Construction B registers when built;
# `test_every_profile_provider_is_registered` fails if one is added without registering.
PROFILE_PROVIDERS = {
    "exact_ed": entropy_profile_ed,
    "free_fermion": entropy_profile_free_fermion,
}

GATE = 1e-10
MIRROR_MIN_SEPARATION = 0.1     # a mirrored profile must differ by at least this


def _asym_profile_ed() -> np.ndarray:
    return entropy_profile_ed(np.ones(7), np.array(ASYM_H))


def test_asymmetry_margin_is_real():
    """The chosen realization must actually be strongly asymmetric, else the test is toothless."""
    S = _asym_profile_ed()
    measured = float(np.max(np.abs(S - S[::-1])))
    assert abs(measured - ASYM_MARGIN) < 1e-5, f"asymmetry drifted: {measured}"
    assert measured > 10_000 * GATE


@pytest.mark.parametrize("name", sorted(PROFILE_PROVIDERS))
def test_profile_orientation(name):
    """Each provider must match ED unreversed AND clearly fail to match reversed."""
    S_ref = _asym_profile_ed()
    S = np.asarray(PROFILE_PROVIDERS[name](np.ones(7), np.array(ASYM_H)))

    forward = float(np.max(np.abs(S - S_ref)))
    reversed_ = float(np.max(np.abs(S - S_ref[::-1])))

    assert forward < GATE, f"{name} disagrees with ED unreversed ({forward:.3e})"
    assert reversed_ > MIRROR_MIN_SEPARATION, (
        f"{name} also matches the REVERSED reference ({reversed_:.3e}); the realization is "
        "not asymmetric enough for this test to have power")


def test_left_block_convention_is_the_one_implemented():
    """Pin the convention itself: cut l must be the LEFT block, sites [0, l).

    Built from an explicitly asymmetric product state -- site 0 unentangled, entanglement
    concentrated at the right end -- so left- and right-block readings differ maximally.
    """
    from qsent.exact import reduced_density_matrix
    # |0> (x) Bell(sites 1,2) ... on 3 sites: site 0 is a pure product, sites 1-2 maximally mixed
    psi = np.zeros(8)
    psi[0b000] = psi[0b110] = 1 / np.sqrt(2)     # bit j = site j
    S1 = np.linalg.eigvalsh(reduced_density_matrix(psi, 3, 1))
    S1 = -float(np.sum(S1[S1 > 1e-14] * np.log(S1[S1 > 1e-14])))
    assert S1 < 1e-12, "cut=1 must isolate SITE 0 (the left block), which here is unentangled"


def test_ceff_fit_is_blind_to_mirroring():
    """Proof that the c_eff fit CANNOT serve as the orientation check.

    Required by the standing rule in CLAUDE.md: a check counts as a gate only once it is
    shown capable of failing on the error it targets. This one demonstrably is not, which
    is precisely why `test_profile_orientation` exists as a separate gate.
    """
    def fit(S, L):
        l = np.arange(1, L)
        x = np.log((2 * L / np.pi) * np.sin(np.pi * l / L))
        A = np.vstack([x, np.ones_like(x)]).T
        coef, *_ = np.linalg.lstsq(A, S, rcond=None)
        return 6 * coef[0], float(np.sum((S - A @ coef) ** 2))

    S = _asym_profile_ed()
    c_fwd, ssr_fwd = fit(S, 8)
    c_rev, ssr_rev = fit(S[::-1], 8)
    assert abs(c_fwd - c_rev) < 1e-12, "unexpected: the fit distinguished the mirror"
    assert abs(ssr_fwd - ssr_rev) < 1e-12
    assert np.max(np.abs(S - S[::-1])) > MIRROR_MIN_SEPARATION   # while the data IS asymmetric


def test_every_profile_provider_is_registered():
    """Any new `entropy_profile_*` in qsent must be added to PROFILE_PROVIDERS.

    This is what makes the Construction-B path unable to bypass the orientation gate: when
    it lands as `entropy_profile_decoded`, this test fails until it is registered.
    """
    import pkgutil
    import importlib

    found = set()
    for mod in pkgutil.iter_modules(qsent.__path__):
        m = importlib.import_module(f"qsent.{mod.name}")
        for fname, fn in inspect.getmembers(m, inspect.isfunction):
            if fname.startswith("entropy_profile_") and fn.__module__ == m.__name__:
                found.add(fname)
    registered = {f.__name__ for f in PROFILE_PROVIDERS.values()}
    missing = found - registered
    assert not missing, (
        f"unregistered entropy profile producers: {sorted(missing)}. Add them to "
        "PROFILE_PROVIDERS so the orientation gate covers them.")


@pytest.mark.skipif(not os.environ.get("QSAE_ARTIFACTS"), reason="QSAE_ARTIFACTS not configured")
def test_asym_realization_matches_pinned_ensemble():
    """Provenance, same discipline as the spanning realizations."""
    from qsent.disorder import delta_r
    from qsent.pins import load_ensemble
    h_all = np.asarray(load_ensemble(1)["h_train"], dtype=np.float64)
    assert np.max(np.abs(np.array(ASYM_H) - h_all[ASYM_INDEX])) < 1e-12
    assert abs(float(delta_r(h_all[ASYM_INDEX])[0]) - ASYM_DELTA_R) < 1e-5


def test_orientation_gate_catches_a_mirrored_provider():
    """The gate must FAIL on the specific error it targets, or it is not a gate.

    A deliberately mirrored provider is run through the same assertions
    `test_profile_orientation` applies. It must be rejected.
    """
    mirrored = lambda J, h: entropy_profile_free_fermion(J, h)[::-1]

    S_ref = _asym_profile_ed()
    S = np.asarray(mirrored(np.ones(7), np.array(ASYM_H)))

    forward = float(np.max(np.abs(S - S_ref)))
    reversed_ = float(np.max(np.abs(S - S_ref[::-1])))

    assert forward > MIRROR_MIN_SEPARATION, "gate would NOT catch a mirrored provider"
    assert reversed_ < GATE, "the mirrored provider is not in fact a clean mirror"
