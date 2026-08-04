"""
Free-fermion (Peschel) entanglement entropy for the open transverse-field Ising chain.

Ground-truth Method 2 of the brief. Independent of exact diagonalisation: the two must
agree to < 1e-10 at every cut before any transformer activation is touched.

Convention (fixed here, asserted in tests, and referenced by PREREGISTRATION.md)
-------------------------------------------------------------------------------
Hamiltonian, **open** boundaries, site-dependent fields *and* couplings:

    H = - sum_{j=0}^{L-2} J_j Z_j Z_{j+1}  -  sum_{j=0}^{L-1} h_j X_j

Entropy is returned in **nats** (natural log), matching
`qsae.observables.half_chain_entanglement_entropy`. Note that Refael & Moore (2004) work
in **bits** and for a **two-boundary segment**; converting to their published constants
costs a factor of 2 * ln 2. See `convention.py`.

Method
------
Jordan-Wigner to Majoranas c_{2j}, c_{2j+1} with

    X_j            =  i c_{2j} c_{2j+1}
    Z_j Z_{j+1}    =  i c_{2j+1} c_{2j+2}

so that H = (i/4) sum_{mn} A_{mn} c_m c_n with A real antisymmetric,

    A[2j,   2j+1] = -2 h_j
    A[2j+1, 2j+2] = -2 J_j        (and A antisymmetric)

The real Schur form block-diagonalises A into 2x2 blocks; the ground-state Majorana
covariance Gamma is rebuilt from the canonical blocks and restricted to the sites in
block A. Its spectrum comes in pairs +/- i nu_k, and

    S = sum_k H2((1 + nu_k)/2),   H2(x) = -x ln x - (1-x) ln(1-x)

Because H2(x) = H2(1-x), the entropy is invariant under a global sign flip of Gamma, so
the overall ground-state-vs-highest-state convention cannot corrupt the result; only
per-block *inconsistency* could, and the Schur routine cannot produce that.
"""

from __future__ import annotations

import numpy as np


__all__ = [
    "build_majorana_matrix",
    "ground_state_covariance",
    "entanglement_entropy_free_fermion",
    "entropy_profile_free_fermion",
    "binary_entropy",
]

_EPS = 1e-12


def binary_entropy(x: np.ndarray | float) -> np.ndarray | float:
    """H2(x) = -x ln x - (1-x) ln(1-x), in nats, safe at the endpoints."""
    x = np.asarray(x, dtype=np.float64)
    out = np.zeros_like(x)
    m = (x > _EPS) & (x < 1.0 - _EPS)
    xm = x[m]
    out[m] = -xm * np.log(xm) - (1.0 - xm) * np.log(1.0 - xm)
    return out if out.ndim else float(out)


def build_majorana_matrix(J: np.ndarray, h: np.ndarray) -> np.ndarray:
    """Real antisymmetric A (2L x 2L) with H = (i/4) sum A_mn c_m c_n.

    Parameters
    ----------
    J : (L-1,) nearest-neighbour couplings, open chain
    h : (L,)   transverse fields
    """
    h = np.asarray(h, dtype=np.float64)
    J = np.asarray(J, dtype=np.float64)
    L = h.shape[0]
    if J.shape[0] != L - 1:
        raise ValueError(f"open chain needs L-1={L - 1} couplings, got {J.shape[0]}")

    A = np.zeros((2 * L, 2 * L), dtype=np.float64)
    for j in range(L):
        A[2 * j, 2 * j + 1] = -2.0 * h[j]
        A[2 * j + 1, 2 * j] = +2.0 * h[j]
    for j in range(L - 1):
        A[2 * j + 1, 2 * j + 2] = -2.0 * J[j]
        A[2 * j + 2, 2 * j + 1] = +2.0 * J[j]
    return A


def ground_state_covariance(J: np.ndarray, h: np.ndarray) -> np.ndarray:
    """Ground-state Majorana covariance Gamma, real antisymmetric (2L x 2L).

    <c_m c_n> = delta_mn + i Gamma_mn.
    """
    A = build_majorana_matrix(J, h)

    # Gamma = -i sign(iA).  iA is Hermitian (A real antisymmetric), its spectrum is real
    # and symmetric about 0, and this form needs no assumption about how a Schur
    # factorisation happens to align its 2x2 blocks -- an assumption that silently fails
    # for site-dependent fields and is invisible to uniform-field tests.
    #
    # Energy check implied by this choice: E0 = (1/4) tr(A Gamma) = -(1/2) sum_{lam>0} lam,
    # which `tests/test_exact_entropy.py` asserts against exact diagonalisation.
    M = 1j * A
    lam, V = np.linalg.eigh(M)
    s = np.sign(lam)
    s[np.abs(lam) < _EPS] = 0.0              # exact zero modes -> maximally mixed
    Gamma = (-1j * (V * s) @ V.conj().T)

    if np.max(np.abs(Gamma.imag)) > 1e-9:
        raise RuntimeError("covariance acquired an imaginary part; check A construction")
    return np.ascontiguousarray(Gamma.real)


def entanglement_entropy_free_fermion(
    J: np.ndarray, h: np.ndarray, cut: int, Gamma: np.ndarray | None = None
) -> float:
    """Von Neumann entropy in **nats** of sites [0, cut) against the rest.

    `Gamma` may be passed in to avoid re-diagonalising when sweeping cuts.
    """
    h = np.asarray(h, dtype=np.float64)
    L = h.shape[0]
    if not 0 < cut < L:
        raise ValueError(f"cut must lie in (0, {L}), got {cut}")
    if Gamma is None:
        Gamma = ground_state_covariance(J, h)

    GA = Gamma[: 2 * cut, : 2 * cut]         # Majoranas of sites 0..cut-1
    # Real antisymmetric -> eigenvalues are purely imaginary, in +/- i nu pairs.
    # Eigenvalues come in +/- i nu pairs; sorting gives each nu twice, so step by 2
    # to take exactly one representative per pair (taking the first `cut` entries
    # would double-count the largest and silently drop the smallest).
    nu = np.sort(np.abs(np.linalg.eigvals(GA).imag))[::-1][0::2]
    nu = np.clip(nu[:cut], 0.0, 1.0)
    return float(np.sum(binary_entropy((1.0 + nu) / 2.0)))


def entropy_profile_free_fermion(J: np.ndarray, h: np.ndarray) -> np.ndarray:
    """Entropy at every cut 1..L-1, in nats. Diagonalises once."""
    h = np.asarray(h, dtype=np.float64)
    L = h.shape[0]
    Gamma = ground_state_covariance(J, h)
    return np.array(
        [entanglement_entropy_free_fermion(J, h, c, Gamma=Gamma) for c in range(1, L)]
    )
