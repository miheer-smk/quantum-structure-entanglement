"""Stage 0 gate: ED and free-fermion entropy must agree to < 1e-10 at every cut."""

from __future__ import annotations

import numpy as np
import pytest

from qsent.exact import entropy_profile_ed, ground_state, reduced_density_matrix
from qsent.free_fermions import (
    binary_entropy,
    build_majorana_matrix,
    entropy_profile_free_fermion,
    ground_state_covariance,
)

GATE = 1e-10

# Five L=8 realizations chosen to SPAN delta_r, drawn from the pinned seed-1 ensemble
# (indices recorded so the choice is reproducible, values inlined so the test does not
# depend on $QSAE_ARTIFACTS being present).
SPANNING = {
    # index 29086, delta_r = +1.999946 (ordered)
    "r0_ordered": [0.76286357641220093, 1.0945854187011719, 1.6707371473312378,
                   0.38778185844421387, 0.11774140596389771, 0.23507729172706604,
                   0.8933413028717041, 1.3541210889816284],
    # index 40316, delta_r = +0.999991
    "r1": [0.25951644778251648, 0.19717611372470856, 1.8613616228103638,
           1.5017951726913452, 1.1894482374191284, 1.7348805665969849,
           0.24230317771434784, 1.8817124366760254],
    # index 49390, delta_r = +0.000000 (near-critical)
    "r2_critical": [0.39817598462104797, 1.0879230499267578, 1.5586049556732178,
                    1.3536075353622437, 1.4377062320709229, 0.61117672920227051,
                    1.2173998355865479, 1.022887110710144],
    # index 28980, delta_r = -1.000099
    "r3": [0.75971966981887817, 1.7865350246429443, 1.6075568199157715,
           1.0476822853088379, 0.96218866109848022, 1.2275468111038208,
           1.8394546508789062, 1.4964354038238525],
    # index 11303, delta_r = -2.000674 (paramagnetic)
    "r4_paramagnetic": [1.7211217880249023, 1.8979150056838989, 1.7614049911499023,
                        1.2373764514923096, 1.6359497308731079, 1.5280174016952515,
                        1.95789635181427, 1.5866810083389282],
}
SPANNING_INDICES = {"r0_ordered": 29086, "r1": 40316, "r2_critical": 49390,
                    "r3": 28980, "r4_paramagnetic": 11303}
SPANNING_DELTA = {"r0_ordered": 1.999946, "r1": 0.999991, "r2_critical": 0.0,
                  "r3": -1.000099, "r4_paramagnetic": -2.000674}


@pytest.mark.parametrize("L", [8, 10, 12])
@pytest.mark.parametrize("hv", [0.5, 1.0, 2.0])
def test_uniform_field_agreement(L, hv):
    """Necessary but NOT sufficient -- see test_uniform_cases_are_degenerate."""
    h = np.full(L, hv)
    J = np.ones(L - 1)
    assert np.max(np.abs(entropy_profile_ed(J, h) - entropy_profile_free_fermion(J, h))) < GATE


@pytest.mark.parametrize("name", sorted(SPANNING))
def test_disordered_field_agreement(name):
    """The load-bearing gate: only per-site disorder can detect a site-blind solver."""
    h = np.array(SPANNING[name])
    J = np.ones(len(h) - 1)
    assert np.max(np.abs(entropy_profile_ed(J, h) - entropy_profile_free_fermion(J, h))) < GATE


def test_uniform_cases_are_degenerate():
    """Uniform fields CANNOT distinguish a correct solver from a site-blind one.

    A solver that ignores per-site structure and uses h[0] everywhere passes every uniform
    test and fails every disordered one. This is why the disordered cases exist, and it is
    recorded here as an executable fact rather than a claim in a document.
    """
    site_blind = lambda J, h: entropy_profile_free_fermion(J, np.full(len(h), h[0]))

    for hv in (0.5, 1.0, 2.0):                       # uniform: bug is INVISIBLE
        h, J = np.full(8, hv), np.ones(7)
        assert np.max(np.abs(entropy_profile_ed(J, h) - site_blind(J, h))) < GATE

    for name, vals in SPANNING.items():              # disordered: bug is CAUGHT
        h, J = np.array(vals), np.ones(len(vals) - 1)
        assert np.max(np.abs(entropy_profile_ed(J, h) - site_blind(J, h))) > 1e-3, name


def test_bond_disorder_path():
    """J is uniform in the current pipeline; the general J_j path must still be correct."""
    rng = np.random.default_rng(0)
    h = rng.uniform(0.1, 2.0, 8)
    J = rng.uniform(0.3, 1.7, 7)
    assert np.max(np.abs(entropy_profile_ed(J, h) - entropy_profile_free_fermion(J, h))) < GATE


def test_free_fermion_energy_matches_ed():
    """Independent check that the covariance is the GROUND state, not some other state."""
    rng = np.random.default_rng(1)
    for _ in range(5):
        h = rng.uniform(0.1, 2.0, 8)
        J = np.ones(7)
        _, e0 = ground_state(J, h)
        e_ff = 0.25 * np.trace(build_majorana_matrix(J, h) @ ground_state_covariance(J, h))
        assert abs(e0 - e_ff) < 1e-10


def test_entropy_is_in_nats_not_bits():
    """GHZ-like check: a maximally entangled qubit pair gives ln 2 nats, not 1 bit."""
    psi = np.zeros(4)
    psi[0] = psi[3] = 1 / np.sqrt(2)               # (|00> + |11>)/sqrt2
    ev = np.linalg.eigvalsh(reduced_density_matrix(psi, 2, 1))
    ev = ev[ev > 1e-14]
    assert abs(float(-np.sum(ev * np.log(ev))) - np.log(2)) < 1e-12


def test_untruncated_spectrum():
    """No bond-dimension ceiling: the full Schmidt spectrum is used at every cut."""
    h = np.array(SPANNING["r2_critical"])
    psi, _ = ground_state(np.ones(7), h)
    for c in range(1, 8):
        rho = reduced_density_matrix(psi, 8, c)
        assert rho.shape == (1 << c, 1 << c)
        assert abs(np.trace(rho).real - 1.0) < 1e-12


def test_binary_entropy_endpoints():
    assert binary_entropy(0.0) == 0.0
    assert binary_entropy(1.0) == 0.0
    assert abs(binary_entropy(0.5) - np.log(2)) < 1e-14


def test_spanning_realizations_match_pinned_ensemble():
    """Provenance: the inlined vectors must BE the pinned realizations at those indices.

    Without this, the ED/free-fermion gate still passes on any h vector, so a fabricated
    or mistyped realization would be invisible -- the delta_r labels would be false while
    every physics assertion stayed green.
    """
    import os
    root = os.environ.get("QSAE_ARTIFACTS")
    if not root or not os.path.exists(f"{root}/data/tfim_L8_N50k_seed1.pt"):
        pytest.skip("QSAE_ARTIFACTS not configured")
    import torch
    from qsent.disorder import delta_r

    h_all = np.asarray(
        torch.load(f"{root}/data/tfim_L8_N50k_seed1.pt", map_location="cpu",
                   weights_only=False)["h_train"], dtype=np.float64)
    for name, vals in SPANNING.items():
        pinned = h_all[SPANNING_INDICES[name]]
        assert np.max(np.abs(np.array(vals) - pinned)) < 1e-12, f"{name} is not the pinned vector"
        assert abs(float(delta_r(pinned)[0]) - SPANNING_DELTA[name]) < 1e-5, name


def test_spanning_realizations_actually_span():
    """One clearly ordered, one near-critical, one clearly paramagnetic."""
    from qsent.disorder import delta_r
    d = {n: float(delta_r(np.array(v))[0]) for n, v in SPANNING.items()}
    assert d["r0_ordered"] > 1.5
    assert abs(d["r2_critical"]) < 0.05
    assert d["r4_paramagnetic"] < -1.5
