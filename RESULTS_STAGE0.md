# RESULTS_STAGE0.md — Inventory and exact-solver validation

**Gate: PASS.** 22/22 tests green. Worst ED vs free-fermion disagreement across all 14
cross-validation cases:
<!--prov id=ed_vs_ff_worst script=scripts/regen_stage0.py array=data/tfim_L8_N50k_seed1.pt seed=none sha256=10aacd0f50a4 kind=value md=1.648e-11 -->
**1.648e-11**, spread **2.0e-15** across BLAS thread counts (gate: < 1e-10).

Provenance: submodule pin `0c4e6e4`; artifacts under `$QSAE_ARTIFACTS`, hashes in
`pins/`; entropy in **nats** throughout; open boundaries; `J ≡ 1`.
`PREREGISTRATION.md` does not exist yet by design — its SHA will be added to every
downstream results file once Stage 1.5 clears.

> **Regenerable, and provenance-checked (2026-08-11).** Every number below is produced by
> `scripts/regen_stage0.py` from pinned artifacts and carries a machine-readable provenance
> tag naming the script, the input array, the seed, and that array's hash.
> `scripts/check_provenance.py` fails if a tagged number's stated source is not the source it
> was computed from. See `DEVIATIONS.md` (2026-08-11) — this file previously mis-stated the
> source of the section-2 measurement, the third instance of that error class.
>
> **Every value is quoted to the number of significant figures measured to be stable, with
> its spread.** `scripts/audit_precision.py` recomputes each quantity under BLAS thread
> counts — which cannot change a physical result — and counts how far down the digits agree.
> A quantity is quoted to exactly that many figures and never more; below two stable figures
> it is stated as a bound instead, because one stable digit is an order-of-magnitude claim and
> an inequality makes that claim more honestly.
>
> **Rule corrected by the author on 2026-08-13** (`DEVIATIONS.md`). The previous rule —
> *moves at all under thread count → bound* — was too crude, and this file was the evidence:
> it filed the headline agreement, whose leading four digits are identical in every
> configuration, under the same verdict as `|Δc_eff|` under mirroring, which does not
> reproduce its first digit. That discarded honestly measured precision from the number
> carrying the two-solver ground-truth claim. The headline is restored to **1.648e-11**, at
> the four figures it actually has.

---

## 1. Did the uniform golden values pass for the right reason, or by degeneracy?

**By degeneracy. The uniform cases are not load-bearing, and this is now proven, not
asserted.**

A deliberately site-blind solver — one that reads `h[0]` and applies it to every site,
ignoring per-site structure entirely — was run against the same gate:

On a uniform field the site-blind solver performs the **identical computation**, so these rows
detect nothing about site-blindness. That is section 1's finding, and it is now measured rather
than argued: each uniform row below is **bit-identical** to the ED-vs-free-fermion
cross-validation case at the same `(L = 8, h)` — same number, arrived at by a second code path.

Their *reproducibility* is a separate question from their *meaning*, and the corrected
precision rule answers only the first: `h = 0.5` is stable to four significant figures and
`h = 2.0` to two, so both are quoted as values; `h = 1.0` is stable to one and stays a bound.
**A figure that reproduces is not thereby a measurement of anything** — all three remain
evidence of degeneracy, not of detection.

The disordered rows are values in both senses: a real physical separation roughly 10¹³ times
the noise floor, stable to 11–14 significant figures, quoted here to five.

<!--prov id=site_blind_uniform_h0.5_L8 script=scripts/regen_stage0.py array=none seed=none sha256=none kind=value md=1.648e-11 -->
<!--prov id=site_blind_uniform_h1.0_L8 script=scripts/regen_stage0.py array=none seed=none sha256=none kind=bound md=1e-13 -->
<!--prov id=site_blind_uniform_h2.0_L8 script=scripts/regen_stage0.py array=none seed=none sha256=none kind=value md=4.9e-13 -->
<!--prov id=site_blind_disordered_r0_ordered script=scripts/regen_stage0.py array=data/tfim_L8_N50k_seed1.pt seed=none sha256=10aacd0f50a4 kind=value md=0.15145 -->
<!--prov id=site_blind_disordered_r1 script=scripts/regen_stage0.py array=data/tfim_L8_N50k_seed1.pt seed=none sha256=10aacd0f50a4 kind=value md=0.45961 -->
<!--prov id=site_blind_disordered_r2_critical script=scripts/regen_stage0.py array=data/tfim_L8_N50k_seed1.pt seed=none sha256=10aacd0f50a4 kind=value md=0.46357 -->
<!--prov id=site_blind_disordered_r3 script=scripts/regen_stage0.py array=data/tfim_L8_N50k_seed1.pt seed=none sha256=10aacd0f50a4 kind=value md=0.36126 -->
<!--prov id=site_blind_disordered_r4_paramagnetic script=scripts/regen_stage0.py array=data/tfim_L8_N50k_seed1.pt seed=none sha256=10aacd0f50a4 kind=value md=0.038587 -->

| Case | max abs difference vs ED | stable s.f. (spread) | Verdict |
|---|---|---|---|
| uniform h = 0.5, L = 8 | 1.648e-11 | 4 (2.0e-15) | **passes — bug invisible** |
| uniform h = 1.0, L = 8 | < 1e-13 | 1 (3.3e-15) | **passes — bug invisible** |
| uniform h = 2.0, L = 8 | 4.9e-13 | 2 (1.7e-15) | **passes — bug invisible** |
| disordered r0 (δ = +2.00) | 0.15145 | 13 (2.5e-15) | caught |
| disordered r1 (δ = +1.00) | 0.45961 | 14 (5.0e-16) | caught |
| disordered r2 (δ = +0.00) | 0.46357 | 11 (6.9e-15) | caught |
| disordered r3 (δ = −1.00) | 0.36126 | 14 (4.5e-15) | caught |
| disordered r4 (δ = −2.00) | 0.038587 | 13 (4.7e-15) | caught |

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

### Verified equality, not assumed — measured on the arrays R1 actually consumes

> **CORRECTED 2026-08-11.** This section previously reported `8.94e-07` on "512 pinned **test**
> realizations". That value is **not** reproducible on `h_test`; it reproduces exactly on
> `h_val[:512]` and `h_train[:512]`. The number was right and the stated source was wrong —
> the third instance of that error class in this repository (`DEVIATIONS.md`, 2026-08-11).
> Re-measured here on `data/ra03_states_L8_N800_s{42,43,44}.pt`, the evaluation arrays
> `phase06` — and therefore R1 — actually uses, so R1's premise is asserted on R1's own data.

Checkpoint `ms_trained/seed1`, all 800 realizations of each pinned eval array:

<!--prov id=hook_k6_vs_published_eval_s42 script=scripts/regen_stage0.py array=data/ra03_states_L8_N800_s42.pt seed=none sha256=b605c43da217 kind=bound md=2e-06 -->
<!--prov id=hook_k6_vs_published_eval_s43 script=scripts/regen_stage0.py array=data/ra03_states_L8_N800_s43.pt seed=none sha256=47a0e6afacae kind=bound md=2e-06 -->
<!--prov id=hook_k6_vs_published_eval_s44 script=scripts/regen_stage0.py array=data/ra03_states_L8_N800_s44.pt seed=none sha256=cc7d8ba56e25 kind=bound md=2e-06 -->
<!--prov id=hook_postnorm_vs_published_eval_s42 script=scripts/regen_stage0.py array=data/ra03_states_L8_N800_s42.pt seed=none sha256=b605c43da217 kind=bound md=3.0 -->
<!--prov id=hook_postnorm_vs_published_eval_s43 script=scripts/regen_stage0.py array=data/ra03_states_L8_N800_s43.pt seed=none sha256=47a0e6afacae kind=bound md=3.0 -->
<!--prov id=hook_postnorm_vs_published_eval_s44 script=scripts/regen_stage0.py array=data/ra03_states_L8_N800_s44.pt seed=none sha256=cc7d8ba56e25 kind=bound md=3.0 -->

| eval array | `max｜mean_pool(k=6) − last_layer_pooled｜` | in float32 ULPs | `post_final_norm` |
|---|---|---|---|
| `…s42.pt` | < 2e-06 | 14 × 2⁻²⁴ | > 2.4, < 3.0 |
| `…s43.pt` | < 2e-06 | 17 × 2⁻²⁴ | > 2.4, < 3.0 |
| `…s44.pt` | < 2e-06 | 15 × 2⁻²⁴ | > 2.4, < 3.0 |

**Reported as a bound, deliberately — and the corrected precision rule does not change that.**
The rule quotes the figures that survive reconfiguration, and here the binding axis is not the
one the thread-count audit sweeps: each per-array value is bit-stable — zero spread across
BLAS thread counts — but differs *between* arrays, because it is simply a count of float32
ULPs: exactly 14, 17 and 15 times 2⁻²⁴. Quoting any one of them to three significant figures
reports the ULP count of whichever array was picked, which is why the original `8.94e-07`
(= 15 × 2⁻²⁴) looked like a measurement and was really a coincidence of array choice. The
claim that survives every array is the one worth making:

> `k=6` agrees with the published `last_layer_pooled` tensor to **< 2e-06 on every array
> tested** — float32 machine precision, the published path staying in float32 while this one
> casts to float64 at the end. **`k=6` is the published tensor.**

`post_final_norm` differs by more than 2.4 on every eval array — six orders of magnitude
above the agreement bound — which is why it is not on the layer axis: mixing it in would
compare unlike normalisations across the very axis H4's rank correlation runs along.

<!--prov id=rms_embed script=scripts/regen_stage0.py array=data/ra03_states_L8_N800_s42.pt seed=none sha256=b605c43da217 kind=value md=0.0311 -->
<!--prov id=rms_block0_attn script=scripts/regen_stage0.py array=data/ra03_states_L8_N800_s42.pt seed=none sha256=b605c43da217 kind=value md=0.1287 -->
<!--prov id=rms_block0_mlp script=scripts/regen_stage0.py array=data/ra03_states_L8_N800_s42.pt seed=none sha256=b605c43da217 kind=value md=0.1929 -->
<!--prov id=rms_block1_attn script=scripts/regen_stage0.py array=data/ra03_states_L8_N800_s42.pt seed=none sha256=b605c43da217 kind=value md=0.8025 -->
<!--prov id=rms_block1_mlp script=scripts/regen_stage0.py array=data/ra03_states_L8_N800_s42.pt seed=none sha256=b605c43da217 kind=value md=0.9529 -->
<!--prov id=rms_block2_attn script=scripts/regen_stage0.py array=data/ra03_states_L8_N800_s42.pt seed=none sha256=b605c43da217 kind=value md=1.4044 -->
<!--prov id=rms_block2_mlp script=scripts/regen_stage0.py array=data/ra03_states_L8_N800_s42.pt seed=none sha256=b605c43da217 kind=value md=1.6390 -->

RMS magnitude grows monotonically through the stack — 0.0311 → 0.1287 → 0.1929 → 0.8025 →
0.9529 → 1.4044 → 1.6390 on `…s42.pt` — as expected for an un-normalised Pre-LN residual
stream. (These are restated on the eval array rather than corrected; the earlier figures were
measured on a training-ensemble split and are consistent to the two digits they were given
to.) This matters for Brief Part 8 item 5 (massive activations) and is why control C8 is not
optional.

**Consequence for R1.** R1 will be run at `k=6`, which is provably the published tensor on
the very arrays R1 evaluates, so a pass means the extraction stack is commensurable rather
than accidentally landing in range.

---

## 3. Do the 5 cross-validation realizations span δ_r, or cluster?

**They span, by construction.** Selected by nearest-δ_r match to targets
`(+2, +1, 0, −1, −2)` from the pinned `seed1` training ensemble.

<!--prov id=index_r0_ordered script=scripts/regen_stage0.py array=data/tfim_L8_N50k_seed1.pt seed=none sha256=10aacd0f50a4 kind=value md=29086 -->
<!--prov id=delta_r_r0_ordered script=scripts/regen_stage0.py array=data/tfim_L8_N50k_seed1.pt seed=none sha256=10aacd0f50a4 kind=value md=1.999946 -->
<!--prov id=index_r1 script=scripts/regen_stage0.py array=data/tfim_L8_N50k_seed1.pt seed=none sha256=10aacd0f50a4 kind=value md=40316 -->
<!--prov id=delta_r_r1 script=scripts/regen_stage0.py array=data/tfim_L8_N50k_seed1.pt seed=none sha256=10aacd0f50a4 kind=value md=0.999991 -->
<!--prov id=index_r2_critical script=scripts/regen_stage0.py array=data/tfim_L8_N50k_seed1.pt seed=none sha256=10aacd0f50a4 kind=value md=49390 -->
<!--prov id=delta_r_r2_critical script=scripts/regen_stage0.py array=data/tfim_L8_N50k_seed1.pt seed=none sha256=10aacd0f50a4 kind=bound md=1e-7 -->
<!--prov id=index_r3 script=scripts/regen_stage0.py array=data/tfim_L8_N50k_seed1.pt seed=none sha256=10aacd0f50a4 kind=value md=28980 -->
<!--prov id=delta_r_r3 script=scripts/regen_stage0.py array=data/tfim_L8_N50k_seed1.pt seed=none sha256=10aacd0f50a4 kind=value md=-1.000099 -->
<!--prov id=index_r4_paramagnetic script=scripts/regen_stage0.py array=data/tfim_L8_N50k_seed1.pt seed=none sha256=10aacd0f50a4 kind=value md=11303 -->
<!--prov id=delta_r_r4_paramagnetic script=scripts/regen_stage0.py array=data/tfim_L8_N50k_seed1.pt seed=none sha256=10aacd0f50a4 kind=value md=-2.000674 -->

| label | ensemble index | δ_r | phase |
|---|---|---|---|
| `r0_ordered` | 29086 | **+1.999946** | clearly ordered |
| `r1` | 40316 | +0.999991 | ordered side |
| `r2_critical` | 49390 | **< 1e-7 in magnitude** | near-critical |
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
realizations):

<!--prov id=E_ln_h_seed1 script=scripts/regen_stage0.py array=data/tfim_L8_N50k_seed1.pt seed=none sha256=10aacd0f50a4 kind=value md=-0.1498 -->
<!--prov id=E_ln_h_seed2 script=scripts/regen_stage0.py array=data/tfim_L8_N50k_seed2.pt seed=none sha256=2bfc7dab9e5d kind=value md=-0.1490 -->
<!--prov id=E_ln_h_seed3 script=scripts/regen_stage0.py array=data/tfim_L8_N50k_seed3.pt seed=none sha256=bf0bfa8597be kind=value md=-0.1480 -->
per-seed `E[ln h]` = −0.1498, −0.1490, −0.1480.

<!--prov id=E_ln_h script=scripts/regen_stage0.py array=data/tfim_L8_N50k_seed1.pt seed=none sha256=10aacd0f50a4 kind=value md=-0.149183 -->
<!--prov id=sd_ln_h script=scripts/regen_stage0.py array=data/tfim_L8_N50k_seed1.pt seed=none sha256=10aacd0f50a4 kind=value md=0.709086 -->

| quantity | value |
|---|---|
| `E[ln h]` | **−0.149183** |
| `sd[ln h]` | **0.709086** |
| `[ln J] − [ln h]` | **+0.149183** → **ordered side** |

(`E[ln h]` and `sd[ln h]` are closed forms in `h_min`/`h_max` — but `h_min`/`h_max` are read
from the pinned artifact's own `meta`, never hardcoded, because they determine every `δ_r`.
The provenance tags therefore name `seed1` rather than claiming these are source-free.)

<!--prov id=mean_delta_r_L8 script=scripts/regen_stage0.py array=data/tfim_L8_N50k_seed1.pt seed=none sha256=10aacd0f50a4 kind=value md=0.595 -->
<!--prov id=mean_delta_r_L10 script=scripts/regen_stage0.py array=data/tfim_L8_N50k_seed1.pt seed=none sha256=10aacd0f50a4 kind=value md=0.665 -->
<!--prov id=mean_delta_r_L12 script=scripts/regen_stage0.py array=data/tfim_L8_N50k_seed1.pt seed=none sha256=10aacd0f50a4 kind=value md=0.729 -->

Mean δ_r drifts **further from criticality as L grows**: **+0.595** (L=8), **+0.665**
(L=10), **+0.729** (L=12). Worth carrying into H1: the ensemble is not merely off-critical,
it is increasingly off-critical in the direction the scaling test needs to control.
(Analytic — `√L · (−E[ln h]/σ_lnh)` — so the L = 10 and L = 12 entries need no ensemble, which
matters because neither has a pinned one.)

δ_r occupancy, pooled L=8 across seeds 1–3 (N = 150,000):

<!--prov id=occupancy_0.05_count script=scripts/regen_stage0.py array=data/tfim_L8_N50k_seed1.pt,data/tfim_L8_N50k_seed2.pt,data/tfim_L8_N50k_seed3.pt seed=none sha256=10aacd0f50a4,2bfc7dab9e5d,bf0bfa8597be kind=value md=5503 -->
<!--prov id=occupancy_0.10_count script=scripts/regen_stage0.py array=data/tfim_L8_N50k_seed1.pt,data/tfim_L8_N50k_seed2.pt,data/tfim_L8_N50k_seed3.pt seed=none sha256=10aacd0f50a4,2bfc7dab9e5d,bf0bfa8597be kind=value md=10980 -->
<!--prov id=occupancy_0.25_count script=scripts/regen_stage0.py array=data/tfim_L8_N50k_seed1.pt,data/tfim_L8_N50k_seed2.pt,data/tfim_L8_N50k_seed3.pt seed=none sha256=10aacd0f50a4,2bfc7dab9e5d,bf0bfa8597be kind=value md=27154 -->
<!--prov id=occupancy_0.50_count script=scripts/regen_stage0.py array=data/tfim_L8_N50k_seed1.pt,data/tfim_L8_N50k_seed2.pt,data/tfim_L8_N50k_seed3.pt seed=none sha256=10aacd0f50a4,2bfc7dab9e5d,bf0bfa8597be kind=value md=52545 -->
<!--prov id=occupancy_neg_count script=scripts/regen_stage0.py array=data/tfim_L8_N50k_seed1.pt,data/tfim_L8_N50k_seed2.pt,data/tfim_L8_N50k_seed3.pt seed=none sha256=10aacd0f50a4,2bfc7dab9e5d,bf0bfa8597be kind=value md=44079 -->

| band | count | share | ≈ per seed |
|---|---|---|---|
| \|δ_r\| < 0.05 | 5503 | 3.67 % | ~1,834 |
| \|δ_r\| < 0.10 | 10980 | 7.32 % | ~3,660 |
| \|δ_r\| < 0.25 | 27154 | 18.10 % | ~9,051 |
| \|δ_r\| < 0.50 | 52545 | 35.03 % | ~17,515 |
| δ_r < 0 (paramagnetic) | 44079 | 29.39 % | — |

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
