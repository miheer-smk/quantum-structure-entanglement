"""
Exact diagonalisation ground truth, with an explicitly fixed site/bit convention.

Ground-truth Method 1 of the brief.

WHY THIS EXISTS RATHER THAN CALLING THE INHERITED FUNCTION
----------------------------------------------------------
`qsae.observables.half_chain_entanglement_entropy` is documented as returning the entropy
of "qubits 0..n_A-1 (left block)", but it computes `state.reshape(dim_A, dim_B)`, which
makes subsystem A the **high** bits of the state index. The predecessor's Hamiltonian
builder (`qsae.reverse_arrow.data._build_zz_xx_dense`) uses `bit i = site i`, so site 0 is
the **low** bit. The two conventions are reflections of each other: that function actually
returns the entropy of sites [n - n_A, n), the RIGHT block.

Impact on the predecessor's published results: **none**. Every inherited result uses the
half cut, and for a pure state S(left half) == S(right half) exactly, so the reflection is
invisible there. It is *not* invisible for this arm, which needs the entropy profile at
every cut on disordered chains, where the profile is asymmetric.

Convention fixed here and asserted in tests:

    site j  <->  bit j of the state index  (site 0 is the least significant bit)
    block A =  sites [0, cut)             (the low bits)

`entropy_profile_ed` and `qsent.free_fermions.entropy_profile_free_fermion` both follow it,
which is what makes their < 1e-10 agreement meaningful rather than coincidental.
"""

from __future__ import annotations

import numpy as np

__all__ = ["build_hamiltonian", "ground_state", "entropy_profile_ed", "reduced_density_matrix"]

_EPS = 1e-14


def build_hamiltonian(J: np.ndarray, h: np.ndarray) -> np.ndarray:
    """Dense H = -sum J_j Z_j Z_{j+1} - sum h_j X_j, open chain, site j = bit j."""
    h = np.asarray(h, dtype=np.float64)
    J = np.asarray(J, dtype=np.float64)
    L = h.shape[0]
    if J.shape[0] != L - 1:
        raise ValueError(f"open chain needs L-1={L - 1} couplings, got {J.shape[0]}")

    dim = 1 << L
    idx = np.arange(dim)
    diag = np.zeros(dim, dtype=np.float64)
    for j in range(L - 1):
        sj = 1.0 - 2.0 * ((idx >> j) & 1)
        sj1 = 1.0 - 2.0 * ((idx >> (j + 1)) & 1)
        diag += -J[j] * sj * sj1
    H = np.diag(diag)
    for j in range(L):
        H[idx, idx ^ (1 << j)] += -h[j]
    return H


def ground_state(J: np.ndarray, h: np.ndarray) -> tuple[np.ndarray, float]:
    """Ground state vector and energy by dense symmetric eigendecomposition."""
    w, v = np.linalg.eigh(build_hamiltonian(J, h))
    return v[:, 0], float(w[0])


def reduced_density_matrix(state: np.ndarray, L: int, cut: int) -> np.ndarray:
    """rho_A for block A = sites [0, cut) under the little-endian convention above.

    Sites [0, cut) are the LOW bits, so A is the fast-varying index and the state must be
    reshaped as (dim_B, dim_A) -- not (dim_A, dim_B), which would silently select the
    opposite block.
    """
    dim_A = 1 << cut
    dim_B = 1 << (L - cut)
    psi = np.asarray(state).reshape(dim_B, dim_A)
    return psi.conj().T @ psi


def entropy_profile_ed(J: np.ndarray, h: np.ndarray) -> np.ndarray:
    """Von Neumann entropy in nats at every cut 1..L-1, block A = sites [0, cut)."""
    h = np.asarray(h, dtype=np.float64)
    L = h.shape[0]
    psi, _ = ground_state(J, h)
    out = np.empty(L - 1)
    for c in range(1, L):
        ev = np.linalg.eigvalsh(reduced_density_matrix(psi, L, c))
        ev = ev[ev > _EPS]
        out[c - 1] = -float(np.sum(ev * np.log(ev)))
    return out
