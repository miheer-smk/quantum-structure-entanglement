# RESULTS_STAGE0.md — Inventory and exact-solver validation

**Gate: PASS.** 22/22 tests green. Worst ED vs free-fermion disagreement across all 14
cross-validation cases: **1.648e-11** (gate: < 1e-10).

Provenance: submodule pin `0c4e6e4`; artifacts under `$QSAE_ARTIFACTS`, hashes in
`pins/`; entropy in **nats** throughout; open boundaries; `J ≡ 1`.
`PREREGISTRATION.md` does not exist yet by design — its SHA will be added to every
downstream results file once Stage 1.5 clears.

---

## 1. Did the uniform golden values pass for the right reason, or by degeneracy?

**By degeneracy. The uniform cases are not load-bearing, and this is now proven, not
asserted.**

A deliberately site-blind solver — one that reads `h[0]` and applies it to every site,
ignoring per-site structure entirely — was run against the same gate:

| Case | max abs difference vs ED | Verdict |
|---|---|---|
| uniform h = 0.5, L = 8 | 1.648e-11 | **passes — bug invisible** |
| uniform h = 1.0, L = 8 | 3.225e-14 | **passes — bug invisible** |
| uniform h = 2.0, L = 8 | 5.045e-13 | **passes — bug invisible** |
| disordered r0 (δ = +2.00) | 1.515e-01 | caught |
| disordered r1 (δ = +1.00) | 4.596e-01 | caught |
| disordered r2 (δ = +0.00) | 4.636e-01 | caught |
| disordered r3 (δ = −1.00) | 3.613e-01 | caught |
| disordered r4 (δ = −2.00) | 3.859e-02 | caught |

With a uniform field every `h_j` is equal, so "uses per-site `h_j`" and "uses `h[0]`
everywhere" are the same computation. The uniform gate cannot separate them **in
principle**, not merely in practice. This is frozen as an executable test
(`test_uniform_cases_are_degenerate`) so the degeneracy cannot be forgotten and the
disordered cases cannot be quietly dropped.

### Two real defects that only the disordered cases exposed

Both were invisible to every uniform test and to the energy check.

**(a) Eigenvalue pairing.** The restricted covariance has spectrum `± i ν_k`. Sorting
`|imag|` descending and taking the first `cut` entries double-counts the largest `ν` and
drops the smallest. Correct is to step by 2 and take one representative per pair. Uniform
chains masked this through their degenerate spectra.

**(b) Site-ordering (endianness) mismatch — the more serious one.** The free-fermion
profile matched ED **reversed** to 4.86e-12 while differing from it by 4.2e-01 unreversed.
Root cause, in the inherited code:

- `qsae.reverse_arrow.data._build_zz_xx_dense` builds the Hamiltonian with
  `bit i = site i` → **site 0 is the least significant bit**.
- `qsae.observables._reduce_density_matrix` computes `state.reshape(dim_A, dim_B)`, which
  makes subsystem A the **high** bits → the block it returns is sites `[n − n_A, n)`, the
  **right** block. Its docstring says *"qubits 0..n_A-1 (left block)"* — the opposite block.

**Impact on the predecessor's published results: none.** Every inherited result uses the
half cut, and for a pure state `S(left half) = S(right half)` exactly, so the reflection is
unobservable there. It is *not* unobservable for this arm, which needs the profile at every
cut on asymmetric disordered chains.

**Resolution.** This arm fixes one convention explicitly — `site j ↔ bit j`, block A =
sites `[0, cut)` — in `qsent/exact.py`, and does not call the inherited entropy function for
profiles. A reflection-asymmetry test guards it. Recorded in `DEVIATIONS.md`.

---

## 2. Are the 8 hook points the same tensor family as the published `layers[-1]` hook?

**Yes for seven of them; the eighth was excluded precisely because it is not.** The family
is **7 points, not 8**.

### Precise definition of the published hook

`qsae.analysis.extract.last_layer_pooled` registers a forward hook on
`model.encoder.layers[-1]` and keeps the block's forward **output**, then mean-pools over
sites. Two facts fix what that tensor is:

1. `nn.TransformerEncoder(...)` is constructed with **no `norm=` argument**, so
   `encoder.norm is None` — **verified at runtime: `True`**. The encoder output is exactly
   the last block's output.
2. `model.final_norm` is applied in `TFIMTransformer.forward` *after* the encoder, so the
   published tensor is **pre-`final_norm`**.

With `norm_first=True` a block computes `x = x + attn(norm1(x))`, then
`x = x + ff(norm2(x))`. So the published tensor is a **post-residual-add, un-normalised**
residual-stream state of shape `(N, L, d_model)`, mean-pooled to `(N, d_model)`.

### The family used here

| k | name | position | same family? |
|---|---|---|---|
| 0 | `embed` | embedding + positional, input to block 0 | yes |
| 1 | `block0_attn` | after attention residual add | yes |
| 2 | `block0_mlp` | after MLP residual add (= block 0 output) | yes |
| 3 | `block1_attn` | after attention residual add | yes |
| 4 | `block1_mlp` | after MLP residual add (= block 1 output) | yes |
| 5 | `block2_attn` | after attention residual add | yes |
| 6 | `block2_mlp` | **= `encoder.layers[-1]` output = THE PUBLISHED HOOK** | **identical** |
| — | `post_final_norm` | after `final_norm` | **NO — excluded** |

All seven are post-residual-add, pre-`final_norm`, `(N, L, d_model)`, no pooling.

### Verified equality, not assumed

On 512 pinned test realizations with checkpoint `ms_trained/seed1`:

```
max |mean_pool(k=6)  −  last_layer_pooled(...)|  =  8.94e-07
max |mean_pool(post_final_norm) − last_layer_pooled(...)|  =  2.39e+00
```

`8.94e-07` is float32 machine precision — the published path stays in float32 while this
one casts to float64 at the end. So `k=6` **is** the published tensor.
`post_final_norm` differs by 2.39, which is why it is not in the layer axis: mixing it in
would compare unlike normalisations across the very axis H4's rank correlation runs along.

RMS magnitude grows monotonically through the stack (0.031 → 0.127 → 0.193 → 0.799 → 0.948
→ 1.405 → 1.635), as expected for an un-normalised Pre-LN residual stream. This matters for
Brief Part 8 item 5 (massive activations) and is why control C8 is not optional.

**Consequence for R1.** R1 will be run at `k=6`, which is provably the published tensor, so
a pass means the extraction stack is commensurable rather than accidentally landing in range.

---

## 3. Do the 5 cross-validation realizations span δ_r, or cluster?

**They span, by construction.** Selected by nearest-δ_r match to targets
`(+2, +1, 0, −1, −2)` from the pinned `seed1` training ensemble.

| label | ensemble index | δ_r | phase |
|---|---|---|---|
| `r0_ordered` | 29086 | **+1.999946** | clearly ordered |
| `r1` | 40316 | +0.999991 | ordered side |
| `r2_critical` | 49390 | **+0.000000** | near-critical |
| `r3` | 28980 | −1.000099 | paramagnetic side |
| `r4_paramagnetic` | 11303 | **−2.000674** | clearly paramagnetic |

`δ_r = (Σ ln J_i − Σ ln h_i)/(√L σ_lnh)`, with `σ_lnh = 0.709086` exact for
`h ~ U(0.1, 2.0)`. Since `J ≡ 1`, `Σ ln J_i ≡ 0`. Positive δ_r = couplings dominate =
ordered.

The exact `h` vectors are inlined in `tests/test_exact_entropy.py` so the suite runs without
`$QSAE_ARTIFACTS`, and a provenance test asserts they are byte-identical to the pinned
realizations at those indices, with δ_r matching to 1e-5.

> **Process note, recorded deliberately.** In the first draft of that test four of the five
> inlined vectors were plausible-looking numbers rather than the actual pinned realizations.
> All physics assertions still passed — the ED/free-fermion gate holds for *any* `h` — so the
> error was invisible to every test that existed at that moment, and only the provenance
> check caught it. The δ_r labels would have been false while the suite stayed green. Both
> the provenance test and the spanning test are now permanent.

---

## 4. Disorder-ensemble characterization (S0.5)

Exact for `h ~ U(0.1, 2.0)`, confirmed against the pinned tensors (`seed{1,2,3}`, 150,000
realizations): per-seed `E[ln h]` = −0.1498, −0.1490, −0.1480.

| quantity | value |
|---|---|
| `E[ln h]` | **−0.149183** |
| `sd[ln h]` | **0.709086** |
| `[ln J] − [ln h]` | **+0.149183** → **ordered side** |

Mean δ_r drifts **further from criticality as L grows**: **+0.595** (L=8), **+0.665**
(L=10), **+0.729** (L=12). Worth carrying into H1: the ensemble is not merely off-critical,
it is increasingly off-critical in the direction the scaling test needs to control.

δ_r occupancy, pooled L=8 across seeds 1–3 (N = 150,000):

| band | count | share | ≈ per seed |
|---|---|---|---|
| \|δ_r\| < 0.05 | 5,503 | 3.67 % | ~1,834 |
| \|δ_r\| < 0.10 | 10,980 | 7.32 % | ~3,660 |
| \|δ_r\| < 0.25 | 27,154 | 18.10 % | ~9,051 |
| \|δ_r\| < 0.50 | 52,545 | 35.03 % | ~17,515 |
| δ_r < 0 (paramagnetic) | 44,079 | 29.39 % | — |

The critical sub-ensemble is in-distribution and well populated; no retraining is needed.

---

## 5. Which Part 4 construction is available

- **A (amplitude space): NOT AVAILABLE.** The model maps `h ∈ R^L` → one scalar energy and
  never sees a spin configuration. Author declined to attach a head.
- **B (layer-resolved decodable): AVAILABLE**, as the workhorse, via a linear readout
  `residual_k(h) → ψ_h` with `ψ_h` from ED. Runtime Schmidt-rank assertion
  `rank ≤ min(feature_dim, 2^min(ℓ, L−ℓ))`; non-binding at L = 8 and L = 12, binding at
  L = 16.
- **C (`d_model` reshape MPS): AVAILABLE**, `d_model = 64 = 2^6`. Requires C4/C5 nulls.
- **Ensemble covariance on the token axis: AVAILABLE and promoted** to primary
  basis-independent measure — token i ↔ site i makes that cut canonical.

---

## 6. Test inventory (22 passed)

`test_uniform_field_agreement` (9 cases) · `test_disordered_field_agreement` (5) ·
`test_uniform_cases_are_degenerate` · `test_bond_disorder_path` (general `J_j`, not
specialised to `J ≡ 1`) · `test_free_fermion_energy_matches_ed` (ground state, not merely a
stationary state; matches to < 1e-10) · `test_entropy_is_in_nats_not_bits` ·
`test_untruncated_spectrum` (full `2^cut` spectrum, trace 1 — the bond-dimension trap is
structurally avoided at L ≤ 16, not merely reported) · `test_binary_entropy_endpoints` ·
`test_spanning_realizations_match_pinned_ensemble` · `test_spanning_realizations_actually_span`.

## 7. Open, carried into later stages

- **K2:** 3 blocks / 7 hook points; H4 underpowered by construction. Power calculation must
  be pre-registered, not discovered.
- **K3:** no L = 10 or L = 12 checkpoints survive; scaling claims need new training.
- Refael–Moore convention conversion (`2 × ln2`) is documented but its dedicated
  `c_eff`-recovery fit test is Stage 1, not Stage 0.
- Qi & Earls (arXiv:2607.01336) still unverified; appears nowhere in any output.
