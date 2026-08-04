"""Stage 1: closed-form toy cases, Construction-B path validation, and split disjointness."""

from __future__ import annotations

import numpy as np
import pytest

from qsent.exact import entropy_profile_ed, ground_state, reduced_density_matrix
from qsent.free_fermions import entropy_profile_free_fermion
from qsent.splits import (
    assert_field_value_disjoint, assert_realization_disjoint, delta_stratified_split,
)

TOL = 1e-10
LN2 = np.log(2.0)


def _entropy(psi: np.ndarray, L: int, cut: int) -> float:
    ev = np.linalg.eigvalsh(reduced_density_matrix(psi, L, cut))
    ev = ev[ev > 1e-14]
    return -float(np.sum(ev * np.log(ev)))


def _profile(psi: np.ndarray, L: int) -> np.ndarray:
    return np.array([_entropy(psi, L, c) for c in range(1, L)])


# --------------------------------------------------------------------------------------
# Closed-form toy cases
# --------------------------------------------------------------------------------------

@pytest.mark.parametrize("L", [4, 6, 8])
def test_product_state_zero_entropy(L):
    for psi in (np.eye(1 << L)[0],                                  # |0...0>
                np.full(1 << L, (1 << L) ** -0.5)):                 # |+...+>
        assert np.max(np.abs(_profile(psi, L))) < TOL


def test_bell_pair_times_product():
    """Bell on sites 0,1 tensored with |0> elsewhere: ln2 at cut 1, ~0 at cuts 2,3."""
    L = 4
    psi = np.zeros(1 << L)
    psi[0b0000] = psi[0b0011] = 1 / np.sqrt(2)      # bit j = site j
    S = _profile(psi, L)
    assert abs(S[0] - LN2) < TOL                    # cut=1 splits the pair
    assert abs(S[1]) < TOL and abs(S[2]) < TOL      # cuts 2,3 do not


@pytest.mark.parametrize("L", [4, 6, 8])
def test_ghz_ln2_at_every_internal_cut(L):
    psi = np.zeros(1 << L)
    psi[0] = psi[-1] = 1 / np.sqrt(2)
    assert np.max(np.abs(_profile(psi, L) - LN2)) < TOL


@pytest.mark.parametrize("L", [4, 6, 8])
def test_w_state_binary_entropy_profile(L):
    """W state has the closed form S(l) = H2(l/L)."""
    psi = np.zeros(1 << L)
    for j in range(L):
        psi[1 << j] = 1 / np.sqrt(L)
    l = np.arange(1, L) / L
    expected = -(l * np.log(l) + (1 - l) * np.log(1 - l))
    assert np.max(np.abs(_profile(psi, L) - expected)) < TOL


def test_maximally_mixed_half_is_the_ceiling():
    """S = (L/2) ln 2 is the maximum attainable at the half cut."""
    L = 6
    d = 1 << (L // 2)
    psi = (np.eye(d).ravel() / np.sqrt(d))
    assert abs(_entropy(psi, L, L // 2) - (L // 2) * LN2) < TOL


def test_tfim_toy_matches_stage0_golden():
    from test_exact_entropy import SPANNING
    for vals in SPANNING.values():
        h = np.array(vals)
        J = np.ones(len(h) - 1)
        assert np.max(np.abs(entropy_profile_ed(J, h) - entropy_profile_free_fermion(J, h))) < TOL


# --------------------------------------------------------------------------------------
# Construction B: the pipeline must be validated before any model touches it
# --------------------------------------------------------------------------------------

def _decoded_profile(psi_hat: np.ndarray, L: int) -> np.ndarray:
    """Construction-B entropy path: L2-normalise the decoded amplitudes, then profile."""
    psi_hat = np.asarray(psi_hat, dtype=np.float64)
    n = np.linalg.norm(psi_hat)
    if n < 1e-300:
        raise ValueError("decoded amplitude vector has zero norm")
    return _profile(psi_hat / n, L)


def test_construction_b_perfect_readout_recovers_exact():
    """A perfect readout must return S_exact -- separates 'pipeline wrong' from 'model doesn't encode it'."""
    # Generated, not inlined: this is an arbitrary test input, not a pinned realization,
    # and inlining it would make it indistinguishable from data that DOES claim provenance.
    h = np.random.default_rng(11).uniform(0.1, 2.0, 8)
    L, J = 8, np.ones(7)
    psi, _ = ground_state(J, h)
    assert np.max(np.abs(_decoded_profile(psi, L) - entropy_profile_ed(J, h))) < TOL
    assert np.max(np.abs(_decoded_profile(psi * -3.7, L) - entropy_profile_ed(J, h))) < TOL


def test_construction_b_rank1_readout_gives_zero():
    """A rank-1 (product) decoded state must return S = 0 at every cut."""
    L = 8
    psi = np.zeros(1 << L)
    psi[0b01011010] = 1.0
    assert np.max(np.abs(_decoded_profile(psi, L))) < TOL


def test_construction_b_schmidt_rank_bound_is_flagged():
    """rank <= min(feature_dim, 2^min(l, L-l)); censored=True when the first term binds."""
    def censored(feature_dim: int, L: int, cut: int) -> bool:
        return feature_dim < (1 << min(cut, L - cut))
    assert not censored(64, 8, 4)      # min(64, 16) = 16, not binding
    assert not censored(64, 12, 6)     # min(64, 64) = 64, exactly full, not binding
    assert censored(64, 16, 8)         # min(64, 256) = 64 -> entropy capped at ln 64
    assert abs(np.log(64) - 4.1589) < 1e-3


# --------------------------------------------------------------------------------------
# Split disjointness -- written now, tested with deliberately leaky splits
# --------------------------------------------------------------------------------------

def test_realization_disjointness_catches_a_leaky_split():
    rng = np.random.default_rng(0)
    h = rng.uniform(0.1, 2.0, (200, 8))
    assert_realization_disjoint(h[:100], h[100:])          # clean
    with pytest.raises(AssertionError):
        assert_realization_disjoint(h[:100], h[99:])       # one shared row


def test_field_value_disjointness_catches_a_leaky_split():
    fit = np.array([[0.5], [1.0], [1.5]])
    assert_field_value_disjoint(fit, np.array([[0.7], [1.2]]))
    with pytest.raises(AssertionError):
        assert_field_value_disjoint(fit, np.array([[0.7], [1.0]]))


def test_per_site_disorder_separates_the_two_notions():
    """Realization-disjoint does NOT imply field-value-disjoint under per-site disorder."""
    fit = np.array([[0.5, 1.2], [0.9, 1.4]])
    evl = np.array([[0.5, 1.4]])                  # a NEW vector reusing both scalar values
    assert_realization_disjoint(fit, evl)          # passes
    with pytest.raises(AssertionError):
        assert_field_value_disjoint(fit, evl)      # and must still fail


def test_delta_stratified_split_is_disjoint_and_balanced():
    from qsent.disorder import delta_r
    rng = np.random.default_rng(1)
    h = rng.uniform(0.1, 2.0, (4000, 8))
    d = delta_r(h)
    fit, evl = delta_stratified_split(h, d, seed=3)
    assert not set(fit.tolist()) & set(evl.tolist())
    assert len(fit) + len(evl) == len(h)
    assert abs(d[fit].mean() - d[evl].mean()) < 0.05      # strata balance delta_r
    assert abs(d[fit].std() - d[evl].std()) < 0.05


# --------------------------------------------------------------------------------------
# Estimator bias harness
# --------------------------------------------------------------------------------------

def test_plugin_entropy_is_biased_down_and_miller_madow_reduces_it():
    """Characterise the bias before relying on any corrected estimate."""
    rng = np.random.default_rng(5)
    K = 16
    p = np.full(K, 1.0 / K)
    true_H = np.log(K)
    for N in (64, 256):
        plug, mm = [], []
        for _ in range(400):
            counts = rng.multinomial(N, p)
            q = counts / N
            nz = q[q > 0]
            h = -float(np.sum(nz * np.log(nz)))
            plug.append(h)
            mm.append(h + (len(nz) - 1) / (2 * N))
        plug_bias = np.mean(plug) - true_H
        mm_bias = np.mean(mm) - true_H
        assert plug_bias < 0, f"plug-in should be biased DOWN at N={N}"
        assert abs(mm_bias) < abs(plug_bias), f"Miller-Madow should reduce bias at N={N}"
