# PLAN.md — Tensor Network Arm, Stages 0 and 1

**Status:** awaiting author approval. No repo created, nothing pushed, no experiment run,
no commit made. This file and `TENSOR_NETWORK_ARM_BRIEF.md` are the only two files
written, both untracked in the working tree of the existing repo.

**Revision 2 (2026-08-03):** D1/D5 resolved (author is Miheer Kulkarni; existing identity
and LICENSE stand). K1 resolved in favour of re-pre-registering against the
infinite-randomness fixed point, with four author amendments now discharged or scoped in
§4A. The RTFIM constant is `c̃ = (ln2)/2 ≈ 0.347`, verified against the primary source —
my earlier `ln2` was the random-singlet value and was wrong.

**Revision 3 (2026-08-03):** D3 overridden by the author — **new private repo**, not a
branch, with a six-point cross-repo pinning contract (§4B). Two blockers surfaced there:
the intended pin target is unpushed, and the ensemble/checkpoints are gitignored in the
source repo so no git mechanism can pin them without a separate content-hash manifest.

**Revision 4 (2026-08-04):** Author set the order of operations — back up the artifacts
before anything else, including `git init`. Repo name **`quantum-structure-entanglement`**.
Pin target: push `quantum-structure-sae`, pin `0c4e6e4`. `$QSAE_ARTIFACTS` sits outside both
repos. `δ_r` normalization confirmed (`σ = √L σ_lnh`) with the ν = 2 justification and an
FSS-collapse upgrade to H1 (§4A(c)). D4 resolved: no amplitude head (§5). Per-layer SAE work
moved out of Stage 0 into its own **Stage 1.5** with a reproduction gate (§3.5) — where the
gate as specified turned out not to be well-posed; see §3.5.0.

**Scope of this document:** Stage 0 and Stage 1 only, per Brief Part 10. Stages 2–4 are
sketched only where a Stage 0/1 design decision forecloses them.

---

## 0. Inventory findings (Brief Part 2 deliverable, done first because everything branches on it)

Read from the `quantum-structure-sae` working tree at `HEAD = 0c4e6e4` (now pinned as
`submodules/quantum-structure-sae`; machine-local paths live in the gitignored `.env.local`).

### 0.1 The model

`src/qsae/reverse_arrow/transformer.py` — `TFIMTransformer`:

| Property | Value |
|---|---|
| Input | `h ∈ R^L` — the vector of **per-site transverse fields** (one disorder realization) |
| Output | `E_0 ∈ R` — a **single scalar** ground-state energy |
| Head | `Linear(d_model → d_model/2) → GELU → Linear(→ 1)`, after mean-pool over sites |
| Layers | **3** (`n_layers=3`) |
| Width | `d_model = 64`, `n_heads = 4`, `d_ff = 256` |
| Norm | Pre-LN (`norm_first=True`), plus `final_norm` before pooling |
| Positional | Learned `pos_emb`, shape `(L, d_model)` |

**This is the single most consequential fact in the inventory: the model does not emit a
wavefunction. It does not even take a spin configuration as input.** It maps a disorder
realization to one number.

### 0.2 The data

`src/qsae/reverse_arrow/data.py` — `make_splits()`:

- `H = −J Σ Z_i Z_{i+1} − Σ_i h_i X_i`, **open** boundary conditions, `J = 1.0`.
- Fields are **i.i.d. per-site disorder**: `h_i ~ Uniform(0.1, 2.0)`, drawn independently
  for every site of every sample. The chain is never uniform.
- `compute_ground_states_sparse()` can return **ground-state vectors**, not just energies.
  This is what makes Construction B reachable at all.
- Cached datasets present: `data/tfim_L8_N50k_seed{1..10}.pt`, plus `tfim_L8_N50k.pt` and
  `tfim_L8_N50k_hcrit.pt`. **L = 8 only.**

### 0.3 Checkpoints

- `runs/ms_trained/seed{1..10}/best.pt` — 10 seeds, **all L = 8**, `n_layers=3`,
  `d_model=64`, val `R² ≈ 0.9996` on energy.
- `runs/ra01_wide/`, `runs/ra01b_narrow/` — earlier single models.
- `runs/ra08_scaling/` — the L = 8/10/12 scaling study kept **`scaling_results.json` only**.
  **No L = 10 or L = 12 checkpoint survives anywhere in the tree.**

### 0.4 Physics code already present

`src/qsae/observables.py` already implements, exactly and tested:

- `half_chain_entanglement_entropy(state, n, cut)` — von Neumann entropy in nats from the
  reduced density matrix, arbitrary cut.
- `entanglement_spectrum(state, n, cut)` — Schmidt values.

So Brief Part 5 **Method 1 (exact diagonalization) already exists**. **Method 2 (Peschel
free-fermion / Majorana correlation matrix) does not exist** and is the real Stage 0 build.

`src/qsae/physics/hamiltonians.py` has validated XXZ and ANNNI builders (Phase 0.7).

### 0.5 Analysis machinery already present

- `src/qsae/analysis/extract.py` — `last_layer_pooled()` hooks **only**
  `model.encoder.layers[-1]` and mean-pools. There is **no per-layer, per-token extraction
  yet**; it must be built.
- `src/qsae/analysis/input_control.py` — the full-input (degree-2 polynomial)
  recoverability control. This is the most important inherited tool for this arm; see §3.
- `src/qsae/analysis/fdr.py` — Benjamini–Hochberg.
- `src/qsae/sae.py` — SAE. `runs/ra04_sae_grid/` holds the grid, **not** a per-layer gain
  curve.

### 0.6 Two factual corrections to Brief Part 2

I am flagging these rather than silently building on the brief's wording.

1. **The +0.028 is not an "SAE reconstruction gain."** Tracing it to
   `runs/ra08_scaling/scaling_results.json` and `docs/week3_results.md:248`, it is
   `learned_gain = R²(probe on trained activations) − max(R² untrained, R² raw-h,
   R² mean-h)` for the **long-range ZZ order proxy**, measured on the **final layer only**.
   Values: L=8 → +0.0294, L=10 → +0.0275, L=12 → (row present, same ballpark). The
   "stable ≈ +0.028 across L" claim is correct; the attribution to SAE reconstruction is
   not. A separate `0.0283±0.0030` appears in `results/phase06_multiseed_trained.md` as the
   incremental-R² beyond the poly-2 input control, which is a different quantity that
   happens to round the same. Stage 4 must not conflate them.
2. **The mixed-field null is on the same probe measure**, not SAE reconstruction:
   `runs/ra09_mixedfield/summary.md` reports learned gain **−0.018 (L=8)** and **−0.007
   (L=10)** — collapsed, consistent with the brief's description of the finding.

Consequence for **H4**: the pre-registered x-axis ("SAE feature gain per layer") **does not
exist in the repo today**. It has to be produced by this arm before H4 can be tested at all.

---

## 1. Which of the three constructions is available (Brief Part 4, explicit decision)

**Construction A — amplitude-space entanglement: NOT AVAILABLE.**
The model has no amplitude head and no configuration input. `ψ_θ(σ)` is undefined for this
architecture. It could be created — attach a `d_model → 2^L` head (256 outputs at L=8,
4096 at L=12) and fit it — but Brief Part 4 requires me to ask before adding a head. See
Decision D4 in §5.

**Construction B — layer-resolved decodable entanglement: AVAILABLE, in a modified form,
and it is the workhorse.**
The modification matters and I want it on the record. The brief says "fit a readout from
the layer-k residual stream to the amplitude / target." Here the *target* is a scalar
energy, so a readout to the target gives no wavefunction. Instead:

> For each disorder realization `h`, exact diagonalization gives the true ground state
> `ψ_h ∈ R^{2^L}` (already computable via `compute_ground_states_sparse(..., return_vectors=True)`).
> Fit a readout `R_k : residual_k(h) ∈ R^{L × d_model} → ψ̂_h ∈ R^{2^L}` on a
> field-disjoint split, L2-normalize `ψ̂_h`, then compute its entanglement exactly as in
> Construction A and compare to `S_exact(ℓ; h)`.

That is a well-defined "how much of the true entanglement structure is linearly decodable
at depth k," and it is the only route to a depth curve for this architecture.

**Construction C — `d_model` reshape MPS: AVAILABLE immediately.**
`d_model = 64 = 2^6` factorizes cleanly (`2^6`, `4×4×4`, or `8×8`). Requires C4/C5 nulls as
specified. I expect this to sit inside its own basis-rotation null and end up in an
appendix, and I will report that outcome plainly if it happens.

**Ensemble-covariance variant: AVAILABLE immediately, and better here than the brief
assumes.**
This architecture has a genuinely canonical cut that the brief's Construction C lacks: the
residual stream is `(batch, L, d_model)`, and **token index i corresponds to physical site
i**. Cutting the token axis at ℓ is a physically meaningful bipartition — not an arbitrary
reshaping. So the cross-covariance construction between block-A tokens and block-B tokens
at cut ℓ inherits the locality that makes the spin-chain tensor product canonical. I intend
to promote this from "cheap sanity companion" to the **primary basis-independent measure**,
with Construction B as the primary physics-grounded measure. Flagging it as a deliberate
deviation from the brief's ranking rather than doing it silently.

At `L=8`, `d_model=64`, `N_test=5000`: block dimension is `4×64=256` at the half cut, so
`N/dim ≈ 20`. Not comfortably `≫`. Ledoit–Wolf shrinkage plus Miller–Madow bias correction
plus a matched-Gaussian surrogate, all three, per Brief Part 8 item 3.

---

## 2. Stage 0 — Inventory and exact-solver validation

**Deliverables**

| Item | Path | Notes |
|---|---|---|
| Free-fermion solver | `src/qsae/physics/free_fermions.py` | new |
| Cross-validation test | `tests/test_exact_entropy.py` | new, golden values frozen |
| Per-layer extraction | `src/qsae/analysis/extract.py` | extend, do not replace `last_layer_pooled` |
| Stage results | `RESULTS_STAGE0.md` + mirror to `results/` | per standing archive convention |

**S0.1 — Free-fermion (Peschel) solver.** Jordan–Wigner the open TFIM with site-dependent
`h_i` to a quadratic Majorana form; build `A = J-coupling`, `B = h-field` matrices; SVD the
`(A−B)`-type matrix to get the canonical modes; restrict the Majorana correlation matrix to
block `A = [0, ℓ)`; its spectrum comes in pairs `± i ν_k`; then
`S_A = Σ_k H₂((1+ν_k)/2)` with `H₂(x) = −x log x − (1−x) log(1−x)`.
Site-dependent `h_i` is handled natively by this formalism — no uniform-field assumption —
which is what makes it usable for the disordered arm too.

**S0.2 — Agreement gate.** ED (`observables.half_chain_entanglement_entropy`) vs
free-fermion, `< 1e-10`, at **every cut** `ℓ ∈ {1..L−1}`, for `L ∈ {8, 10, 12}` at
`h ∈ {0.5, 1.0, 2.0}` **uniform**, plus — added by me, because it is the case this project
actually trains on — **5 fixed disorder realizations** drawn from `U(0.1, 2.0)` at each L.
Freeze all of these as golden values.

**S0.3 — Independent analytical check.** Reproduce a known closed-form value not routed
through either solver. Candidate: the `h → ∞` paramagnetic limit gives the exact product
state `|+⟩^L`, `S = 0` at every cut; and `h → 0` with open boundaries gives the GHZ-like
degenerate manifold. I will also fit `c_eff` from the ED profile of the **clean, critical**
open chain at `L = 12, 14, 16` against `S(ℓ) = (c/6) log[(2L/π) sin(πℓ/L)] + const` and
confirm recovery of `c = 1/2` — this validates the fitting code itself before it is ever
pointed at a transformer, and it is where the even–odd oscillation handling (Brief Part 5)
gets built and tested.

**S0.4 — Per-layer, per-token extraction.** Hook the residual stream at every point that
exists in a 3-block Pre-LN encoder, giving more than 3 measurement points:
`k=0` embedding+pos, then per block `b ∈ {0,1,2}`: post-attention residual write,
post-MLP residual write → **7 hook points**, plus `k=7` post-`final_norm`. Shape preserved
as `(N, L, d_model)`; no mean-pooling (pooling destroys the token axis this arm needs).
Unit-test that hooks fire in order and that the `k=7` value reproduces the existing
`last_layer_pooled` output after mean-pooling — a regression guard on inherited results.

**S0.5 — Disorder-ensemble characterization (amendment (b)).** Compute exactly, for the real
pipeline, at L = 8/10/12 and every available seed: `E[ln h]`, `σ_lnh`, `[lnJ]−[lnh]`, which
side of criticality the ensemble mean sits on, and the drift of that offset with L. Report in
`RESULTS_STAGE0.md`. Analytic values are pre-computed in §4A(b); Stage 0 re-derives them from
the cached tensors rather than trusting the closed form.

**S0.6 — `δ_r` stratification (amendment (c)).** Implement `δ_r` with the normalization fixed
in `PREREGISTRATION.md`, assign every realization a `δ_r`, and report bin occupancy per L per
seed. Unit-test `δ_r` on synthetic ensembles with known `[lnJ]−[lnh]`, including a bond-disorder
case where `Σ ln J_i ≠ 0`, so the implementation is not silently specialized to `J ≡ 1`.

**S0.7 — Convention audit, executed as a test, not a comment.** §4A(a) identifies a
`2 × ln2` factor between the literature convention and this repo's. Freeze it: a test that
takes the clean critical open chain, fits `S(ℓ) = (c/6) ln[(2L/π) sin(πℓ/L)] + k` in nats,
and asserts recovery of `c = 1/2`; plus a units assertion that
`half_chain_entanglement_entropy` returns nats (GHZ → `ln 2 ≈ 0.6931`, not `1.0`). Any
future comparison to Refael–Moore routes through one documented conversion function.

**Gate:** `pytest tests/` green including the new `< 1e-10` test; `RESULTS_STAGE0.md`
written and read by the author. Nothing touches a transformer activation for a *scientific*
claim before this is green (S0.4 is plumbing and is unit-tested, not claim-generating).

**Cost:** CPU only, minutes. No GPU, no SLURM.

---

## 3. Stage 1 — Toy-case pipeline validation

**Deliverables:** `src/qsae/entanglement/` (constructions B, C, covariance),
`tests/test_entanglement_toy.py`, `RESULTS_STAGE1.md`.

**S1.1 — Closed-form toy cases**, each to `1e-10`:

| Case | Expected |
|---|---|
| Product state `|0⟩^L`, `|+⟩^L` | `S(ℓ) = 0` ∀ℓ |
| Bell pair ⊗ product | `S = log 2` at the cut splitting the pair, `0` elsewhere |
| GHZ, `L = 8` | `S = log 2` at every internal cut |
| W state, `L = 8` | closed form `S(ℓ) = H₂(ℓ/L)` |
| Exact TFIM ground state, `L = 8`, `h ∈ {0.5,1,2}` | matches Stage 0 golden values |
| Maximally mixed reduced state at ℓ=L/2 | `S = (L/2) log 2`, the ceiling |

**S1.2 — Untruncated-spectrum assertion.** Assert in code that the Schmidt spectrum used
has full length `2^min(ℓ, L−ℓ)` and that `Σλ² = 1` to `1e-12`. At `L ≤ 16` the max cut
dimension is 256, so **no truncation is ever needed** and the Brief Part 8 item 1
bond-dimension trap is structurally avoided rather than merely reported. The code will
raise if anyone later introduces a χ cap without setting an explicit `censored=True` flag.

**S1.3 — Construction B validated on a known-answer readout.** Before any trained model:
feed the *exact* `ψ_h` through the Construction B code path as a perfect readout and
confirm it returns `S_exact` to `1e-10`. Then feed a deliberately rank-1 readout and
confirm it returns `S = 0`. This separates "the pipeline is wrong" from "the model doesn't
encode it," which is the failure mode that wastes the most time downstream.

**S1.4 — Split-disjointness test, written now, not later.** Brief Part 8 item 7 is the one
I rate highest-risk here, because the disorder is per-site: two realizations can share
individual `h_i` values while being distinct vectors. I will define disjointness explicitly
as **realization-disjoint** (no `h` vector in both fit and eval) and, for any future
uniform-field scan, **field-value-disjoint** (no scalar `h` in both). The runtime assertion
will check both and will be unit-tested with a deliberately leaky split that must fail.

**S1.5 — Estimator-bias harness.** Plug-in vs Miller–Madow vs jackknife on synthetic
spectra with known entropy at the sample counts Stage 2 will actually use, so the bias
correction is characterized before it is relied on.

**Gate:** all toy cases within `1e-10`; `RESULTS_STAGE1.md` written and read by the author.

**Cost:** CPU only, minutes.

---

## 3.5 STAGE 1.5 — Per-layer SAE production and the reproduction gate

Author-directed (2026-08-04), moved out of S0.4: training 8 SAEs is an experiment, not an
inventory step. Own stage, own gate, runs **before** `PREREGISTRATION.md` is committed.

### 3.5.0 The gate as specified is not well-posed — and the reason matters

The instruction is "the newly trained final-layer SAE must reproduce +0.028 within seed
noise." I cannot implement that as written, because **+0.028 never came from an SAE**.
Verified against every tracked summary in the source repo:

- `results/legacy/ra04_sae_grid.md` — the only SAE result — reports **matched cosine
  similarity across seeds, dead fraction, and reconstruction MSE**. It contains **no gain
  quantity of any kind**. There is no published per-layer or final-layer "SAE feature gain."
- `results/legacy/ra08_scaling.md` — `learned_gain = R²(trained) − max(R² untrained, raw-h,
  mean-h)` for the long-range-ZZ probe on **raw activations**: L=8 **+0.029**, L=10 +0.028,
  L=12 +0.027.
- `results/phase06_multiseed_trained.md` — `long_range_zz` **incremental R² beyond poly2-h**
  = **0.0283 ± 0.0030 [0.0231, 0.0320]**, 10 seeds, with per-seed values published.

Asking an SAE-derived number to reproduce a probe-derived number would fail for reasons
having nothing to do with correctness. A perfect SAE would still not return 0.028.

**Second finding, which bears directly on requirement 4.** `ra08`'s +0.029 was measured on
models trained with `n_train=15000, epochs=100, seed=0` (its own `meta` block). The surviving
`ms_trained` checkpoints are `n_train=50000, epochs=200, seeds 1–10` — **a different training
configuration** — and `ra08`'s own checkpoints were never saved. So "the same checkpoints the
gain was measured on" **does not exist for the +0.029 number**. It is unrecoverable, and no
backup fixes that; the artifacts were never written.

The number that *is* anchored to surviving checkpoints is **phase06's 0.0283 ± 0.0030**,
measured on `ms_trained` seeds 1–10 — the exact checkpoints now hashed in `pins/`.

### 3.5.1 Restructured into two checks

**R1 — Commensurability reproduction (this is the real gate, and it is well-posed).**
Re-run the phase06 `long_range_zz` incremental-R²-beyond-poly2 protocol at the final layer,
on `ms_trained` seeds 1–10, through *this arm's* extraction stack (S0.4's hook points rather
than `last_layer_pooled`). This is what protects the credibility concern: it demonstrates
that this repo's pipeline is commensurable with the published one before any new number is
reported next to a published one.

**Pre-registered tolerance, fixed now, before the number is seen** (published: mean 0.0283,
sd 0.0030, range [0.0231, 0.0320]; per-seed s1..s10 = 0.023, 0.029, 0.025, 0.032, 0.027,
0.031, 0.026, 0.032, 0.029, 0.030):

> **PASS** iff *both*: (i) the new 10-seed mean lies within **0.0283 ± 0.0060** (± 2 published
> sd), i.e. **[0.0223, 0.0343]**; and (ii) the paired per-seed difference satisfies
> **|Δ_s| ≤ 0.010 for at least 8 of 10 seeds**, pairing on seed identity.
> **FAIL** otherwise. No third outcome, no post-hoc widening.

**R2 — Per-layer SAE gain is a NEW quantity with no published counterpart.**
It must be *defined* in `PREREGISTRATION.md` from scratch (proposed: R² of the long-range-ZZ
probe on SAE features at layer k, minus R² on raw activations at layer k, same probe family,
same splits), trained at all 8 hook points × ≥5 seeds. The reproduction gate **cannot** apply
to it — there is nothing to reproduce. It is reported as new work, explicitly.

### 3.5.2 H4 restated

H4's text must say plainly that the per-layer SAE gains are produced by this arm, that no
prior per-layer SAE gain exists, that the published +0.028/+0.029 are *probe* gains on raw
activations, and what R1 returned. A reviewer comparing a number here against +0.028 in the
SAE paper must find the explanation in the hypothesis statement, not infer it.

**Pre-registered fallback, also fixed now.** If R1 FAILS: H4 is demoted to a comparison
against the single published final-layer value only, the layer-coincidence claim is dropped
entirely, and the failure is reported with both numbers and the tolerance that was missed.
No retuning of the SAE, the probe, or the splits to recover R1.

### 3.5.3 Cost and gate

8 hook points × 10 seeds SAEs, `d_hidden=256, k=8` (ra04's selected cell). Small — CPU-viable,
GPU optional. **Gate:** R1 verdict reported against the tolerance above, `RESULTS_STAGE1_5.md`
written and read by the author. `PREREGISTRATION.md` is committed only after this.

---

## 3.55 PAPER SPINE — confirmatory vs exploratory split (author-directed, frozen 2026-08-04)

Restructured **before Stage 2**, not discovered after Stage 4. This is pre-registration text.

**The structural fact.** The model has **3 blocks** → 7 residual-stream hook points. H3 (depth
profile) and H4 (layer coincidence) both require a *resolvable depth axis*; 3 blocks do not
provide one, and no amount of pre-registration fixes that. H1 and H2 need no depth axis at
all, are fully powered by the 50,000-realization ensemble, and carry the quantitative result.

### CONFIRMATORY — the spine

> **H1 — FSS collapse in δ_r.** `S_model(δ_r; L)` collapses onto a single scaling function
> when plotted against `δ_r = δ·L^{1/ν}`, `ν = 2`, matching the collapse of `S_exact` on the
> same realizations. Tested on the δ-stratified sub-ensemble, which is in-distribution.
>
> **H2 — paired per-realization entropy agreement**, primary; `c_eff` vs the `ln2/12` slope,
> secondary. Exact ground truth is available per realization, pairing removes
> realization-to-realization variance, and neither depends on asymptotic RG convergence.

Both are powered by realization count, not by depth, so the architecture's shallowness does
not touch them.

### EXPLORATORY — H3 and H4

> **H3 and H4 are EXPLORATORY. The reason is the model's shallowness: 3 blocks give 7 hook
> points, of which only 3 are block outputs, and the 7 are not mutually independent because
> attention and MLP residual writes within a block are strongly coupled. The depth axis is
> therefore not resolvable, and no result along it can be confirmatory.** See §A2a for the
> exact power arithmetic: `ρ ≥ 0.7857` is required for p < 0.05 at n = 7, and at n = 3 or 4
> significance is unreachable at any ρ.

This appears in the pre-registration, **not** in a later limitations section. A depth result
reported as confirmatory and then walked back in limitations is the failure mode this split
exists to prevent.

### What would make H3/H4 confirmatory

Deeper models — enough blocks for a resolvable depth axis. That is **a separate arm**, and it
**breaks checkpoint reuse for H4**: new models are not the `ms_trained` checkpoints the
published probe gain was measured on, so the cross-reference to the SAE line would no longer
be anchored to shared artifacts. Noted here as the honest path forward. **Not to be done in
this arm.**

---

## 3.6 Mandated `PREREGISTRATION.md` text (A1–A3), frozen 2026-08-04

`PREREGISTRATION.md` is **not** created or committed until the author has read the Stage 1.5
reproduction result. This section holds the text that must be lifted into it verbatim, so the
wording is fixed now rather than written after the numbers are seen.

### A0 — `c_eff` finite-size bias at L = 8/10/12: MEASURED, and the decision taken in advance

Measured, not extrapolated, before any model number exists. Ground truth throughout is the
free-fermion solver (validated against ED to < 1e-10 at these L).

**Clean critical open chain, where the true answer is `c = 0.5` by construction:**

| L | plain fit | bias | + parity term | + decaying parity | even-ℓ only |
|---|---|---|---|---|---|
| **8** | 0.5881 | **+0.0881** | 0.5870 | 0.5861 | 0.5718 |
| **10** | 0.5845 | **+0.0845** | 0.5836 | 0.5825 | 0.5691 |
| **12** | 0.5809 | **+0.0809** | 0.5801 | 0.5789 | 0.5662 |
| 32 | 0.5570 | +0.0570 | 0.5568 | 0.5557 | 0.5468 |
| 128 | 0.5289 | +0.0289 | 0.5289 | 0.5283 | 0.5240 |

**Disordered critical sub-ensemble `|δ_r| < 0.05`, asymptotic target `ln2/2 = 0.34657`:**

| L | N | source | disorder-avg `[S]` fit | bias | typical (median) fit | bias |
|---|---|---|---|---|---|---|
| **8** | 1877 | pinned seed-1 | 0.5419 | **+0.1953** | 0.6420 | **+0.2954** |
| **10** | 2000 | reference RNG | 0.5557 | **+0.2091** | 0.6479 | **+0.3013** |
| **12** | 2000 | reference RNG | 0.5229 | **+0.1763** | 0.6025 | **+0.2559** |

*(L = 10/12 disordered chains are generated reference-only for this measurement — a
documented separate RNG, explicitly **not** a pinned ensemble and never used for any model
claim. Only L = 8 has a pinned ensemble.)*

**Even-odd oscillations do not explain the bias.** Adding a parity term moves clean-chain
`c_eff` by ≤ 0.0011 and the disordered fit by ≤ 0.0065. Restricting to even ℓ changes it
erratically and non-monotonically (disordered: 0.4818, 0.5550, 0.4991 at L = 8, 10, 12).
Neither correction removes it. This is a genuine finite-size correction, not an oscillation
artifact.

**The separation the H2 secondary would have to resolve:**

| L | clean `c_eff` | disordered `c_eff` | observed gap | asymptotic gap |
|---|---|---|---|---|
| 8 | 0.5881 | 0.5419 | **+0.0463** | 0.15343 |
| 10 | 0.5845 | 0.5557 | **+0.0288** | 0.15343 |
| 12 | 0.5809 | 0.5229 | **+0.0580** | 0.15343 |

#### DECISION, pre-registered now

The disordered bias (**+0.176 to +0.209**, and **+0.256 to +0.301** for the typical profile)
is **larger than the entire 0.15343 gap** that separates clean Ising from the IRFP. The
surviving clean-vs-disordered separation is 3–5× smaller than asymptotic and **non-monotonic
in L** (0.046 → 0.029 → 0.058), so there is no trend to extrapolate along.

A bias-corrected estimator is **rejected**: the correction is larger than the effect, differs
between the clean and disordered cases (different universality classes have different
finite-size corrections), and is non-monotonic in L in the disordered case. Applying a
clean-chain correction to a disordered fit would amount to fitting the answer.

> **Pre-registered conclusion, fixed before any model number exists.** At L = 8, 10, 12 the
> `c_eff` fit **cannot distinguish the clean Ising value `c = 1/2` from the infinite-randomness
> value `c̃ = ln2/2`.** The finite-size bias (+0.18 to +0.21 disorder-averaged, +0.26 to +0.30
> typical) exceeds the 0.153 gap between them, and neither an oscillatory term nor an even-ℓ
> restriction removes it.
>
> **`c_eff` is therefore reported as a descriptive quantity with a bootstrap CI, accompanied
> by this measured bias table, and NO claim of universality-class identification is made from
> it at these system sizes.** A fitted `c_eff` near 0.5 is **not** evidence for the clean class,
> and one near 0.35 is **not** evidence for the IRFP class. Any such claim would require
> L ≳ 64, which needs models this arm does not have.

#### This does not touch the H2 primary

**H2's primary test is the paired per-realization comparison of `S_model(ℓ; r)` against
`S_exact(ℓ; r)` on the same realization.** It involves no `c_eff` fit, no scaling form, and no
asymptotic RG target, so **the bias documented above does not affect it at all**. The paired
test compares two entropy profiles on identical disorder, where exact ground truth is
available per realization at these sizes.

This sentence must appear in the write-up next to the `c_eff` caveat, so that a reader does not
mistake a limitation of the secondary descriptive for a limitation of the primary test.

### A0b — H1 collapse feasibility on EXACT ground truth: TESTABLE, H1 stands

Same discipline as A0: the hypothesis's machinery tested on data where the answer is known,
**with no model involved**. `S_exact(ℓ = L/2)` computed per realization, binned in δ_r,
N = 12,000 per L.

**Collapse quality `Q`** — bin δ_r on a common grid, take the mean profile per L, remove a
per-L additive offset (the FSS ansatz fixes the shape, not the constant), and report the
residual spread across L divided by the consensus curve's dynamic range. `Q = 0` is perfect
collapse. Bootstrap over realizations, 400 resamples.

| collapse variable | Q | bootstrap 95% CI |
|---|---|---|
| **ν = 2** (`δ·L^{1/2}`, the pre-registered variable) | **0.0091** | **[0.0095, 0.0152]** |
| ν = 1 (`δ`, wrong exponent — control) | 0.0376 | [0.0413, 0.0710] |

**Exact ground truth collapses at L = 8, 10, 12, and the ν = 2 variable outperforms the ν = 1
control by ~4×, with non-overlapping CIs.** The control matters: without it, a small `Q`
could simply mean the binning is coarse. It does not — the wrong exponent gives a visibly
worse collapse on the same bins.

*(The point estimate sits marginally below its own bootstrap CI, 0.0091 vs a [0.0095, …]
lower bound. That is ordinary upward bootstrap bias — resampling injects noise that inflates
a residual-spread statistic. Reported rather than tidied; the comparison against the ν = 1
control is unaffected since both are computed identically.)*

**H1 as written is testable and stands.** The fallback below is therefore NOT triggered, but
is pre-registered anyway so the decision is fixed in advance:

> **Pre-registered H1 fallback (not currently triggered).** Should the collapse test prove
> unusable on model data — e.g. `Q_model` indistinguishable from the ν = 1 control — H1
> reverts to the **paired per-realization statement**: does `S_model` track `S_exact` as a
> function of δ_r **at fixed L**? That needs no cross-L collapse, uses only the pinned L = 8
> ensemble, and is unaffected by everything in A0.

#### Pinned-ensemble gap — explicit answer

**Only L = 8 has a pinned ensemble.** The L = 10 and L = 12 rows above use reference-only
generated chains under a documented separate RNG (`default_rng(20260804 + L)`), which is
legitimate here because **no model is involved** — this measures a property of exact physics
and of the estimator, not of any trained network.

> **For a claim-bearing H1, pinned L = 10 and L = 12 ensembles MUST be generated and hashed
> first**, and added to `pins/ensemble.sha256`, before any cross-L collapse result is
> reported as a finding. This also interacts with K3: no L = 10/12 *checkpoints* exist
> either, so a cross-L H1 needs both new ensembles **and** new training. The fixed-L fallback
> above needs neither.

### A0c — Clean-vs-disordered `c_eff` gap, bootstrapped (descriptive only)

4,000 bootstrap resamples over realizations of the `|δ_r| < 0.05` sub-ensemble; the clean-chain
value is deterministic (a single chain), so the CI comes entirely from the disordered side.

| L | clean | disordered | gap | bootstrap 95% CI | spans 0? |
|---|---|---|---|---|---|
| 8 | 0.5881 | 0.5419 | +0.0463 | [+0.0103, +0.0817] | no |
| 10 | 0.5845 | 0.5557 | +0.0288 | **[−0.0028, +0.0603]** | **YES** |
| 12 | 0.5809 | 0.5229 | +0.0580 | [+0.0285, +0.0869] | no |

**At L = 10 the CI spans zero** — stated plainly. The non-monotonicity flagged in A0 is now
quantified: the three gaps are mutually consistent within their CIs, so the apparent ordering
(0.046 → 0.029 → 0.058) carries no signal. Every gap is far below the asymptotic 0.15343.

**Strictly descriptive. Consistent with the A0 decision: no universality-class claim is made
from `c_eff` at these system sizes**, and these CIs do not license one — a gap distinguishable
from zero is not a gap that identifies a fixed point.

### A1 — R1's scope, stated in the required words

> **What R1 validates.** R1 validates **this arm's extraction stack** against a published
> **probe-gain** number on pinned checkpoints. **It does not validate anything about SAEs.**
> The SAE line of the predecessor work is **not** reproduced here, and no claim about it
> should be inferred from R1 passing.

### A2 — H4 restructured: primary axis is per-layer PROBE gain

The published `+0.028`/`+0.029` are probe gains. The natural per-layer quantity is therefore
the *same* `long_range_zz` incremental-R²-beyond-poly2 measure **extended to all 8 hook
points**. This extends a published quantity rather than inventing one, requires no SAE
training, and keeps H4 anchored to something with a published value at one layer.

> **H4 (primary, confirmatory).** Spearman rank correlation across layers between the
> per-layer probe gain (`long_range_zz` incremental R² beyond poly2-h) and `ΔS_decoded`.
> **H4 (secondary, EXPLORATORY).** The same rank correlation against the per-layer **SAE**
> gain (R2). This is labeled exploratory and cannot carry a confirmatory hypothesis, because
> it is a **new quantity with no published counterpart**.
> **Stated plainly:** the original brief's "SAE cross-reference" was premised on per-layer SAE
> data that **does not exist** in the predecessor repository.

### A2a — H4 power limitation, pre-registered in advance (frozen 2026-08-04)

Stated **before** H4 is run, with the arithmetic done rather than asserted. Exact
enumeration of the Spearman permutation null (`itertools.permutations`, all `n!` orderings):

| n (hook points) | permutations | attainable ρ values | two-sided p at ρ = +1 | ρ needed for p < 0.05 |
|---|---|---|---|---|
| 3 (block outputs only) | 6 | 4 | 0.3333 | **unreachable at any ρ** |
| 4 (+ embedding) | 24 | 11 | 0.0833 | **unreachable at any ρ** |
| **7 (the family used)** | **5040** | **57** | **3.968e-04** | **0.7857** |

**Correction to an earlier estimate.** A working figure of "p ≈ 0.008 at n = 7" was in
circulation. It is wrong: the exact two-sided floor is **3.968e-04** (`2/5040`). The
practical consequence differs from what that figure implied — at 3.968e-04 the test *does*
survive Holm correction for any layer family up to **125 tests** (`m × 3.968e-04 < 0.05`
⟺ `m ≤ 125`). H4 is not killed by multiplicity.

**What actually limits H4**, stated as the pre-registered caveat:

1. **Only near-perfect monotonicity is detectable.** `ρ ≥ 0.7857` is required for p < 0.05
   at n = 7. Any genuine-but-imperfect layerwise correspondence is undetectable at this n.
2. **The p-value is coarsely quantised** — 57 attainable ρ values, so p moves in visible
   jumps and small ρ differences are not resolvable.
3. **The 7 points are not independent.** The attention and MLP residual writes within a
   block are strongly coupled, so the permutation null's exchangeability assumption is
   questionable and the *effective* n lies between 3 and 7. At the conservative end
   (n = 3 or 4, block outputs only) **p < 0.05 is unreachable at any ρ whatsoever**.

**Therefore, pre-registered now:**

> **H4 is reported as an EFFECT SIZE (Spearman ρ) WITH A BOOTSTRAP CI. The significance
> test is declared underpowered and is NOT the basis of any claim.** A p-value may be shown
> for completeness, alongside the n, the floor, and the ρ threshold above; it will not be
> interpreted as support or refutation.
>
> **A null H4 at this n is uninformative — it is NOT evidence of dissociation between
> dictionary-learning features and physics-native entanglement.** The brief's framing of a
> CI spanning zero as "a genuinely interesting dissociation" does **not** apply at n = 7,
> because a CI spanning zero is the expected outcome for any true ρ below ≈ 0.79. Claiming
> dissociation would require either more layers (a deeper model) or a different estimator
> with a declared power analysis.

This is a pre-registration statement, not a Stage 4 finding. It is fixed before the number
exists so it cannot be relaxed after seeing it.

### A3 — H5 AUDITED: the mixed-field null is also a probe result

**Checked, not assumed.** `runs/ra09_mixedfield/scaling_results.json` has row keys
`probe_r2_trained, probe_r2_untrained, probe_r2_raw_h, probe_r2_mean_h, learned_gain`, and a
recursive case-insensitive search for "sae" across `runs/ra09_mixedfield/` and
`results/legacy/ra09_mixedfield.md` returns **nothing**. The mixed-field null is a **probe**
result, identical in kind to the +0.028 — the same defect, second instance, exactly as
suspected.

Values: `learned_gain` = **−0.0175 (L=8)**, **−0.0070 (L=10)**; `meta` =
`n_train=15000, epochs=100, seed=0, Ls=[8,10], g=0.5`.

> **H5 (restated).** In the mixed-field regime, where the **probe** gain collapses
> (−0.0175 at L=8, −0.0070 at L=10), entanglement tracking should collapse too. H5 concerns
> the probe-gain measure throughout. **No SAE quantity is involved in the published
> mixed-field result**, and H5 makes no claim about SAE behaviour in that regime.

**Two further findings from the audit, both recorded rather than absorbed:**

1. **`ra09`'s checkpoints were also never saved** — `meta` is `n_train=15000, epochs=100,
   seed=0`, the same configuration as `ra08` and different from `ms_trained`. So the
   mixed-field null, like +0.029, has **no recoverable artifact**. That is a *second*
   unpinnable published number. It also only reaches L = 10, not L = 12.
2. **`results/legacy/ra09_mixedfield.md` is mislabeled in the predecessor repo.** Its title
   reads "RA-08 — L-scaling of the ⟨Z₀Z_{L-1}⟩ signal" and it carries `ra08`'s caption
   ("the scaling prediction is that this grows with L"), which is meaningless for a null.
   The *numbers* in it are genuinely `ra09`'s (−0.018, −0.007, matching the JSON). This is a
   documentation defect in the source, not a data defect; it is flagged here so a reviewer
   reading that file is not misled, and it must not be silently corrected in the pinned
   submodule.

---

## 4. Design conflicts between the brief and the repo that I could not resolve myself

These do not block Stage 0 or Stage 1 — I can build and gate both regardless. They block
**Stage 2 onward**, and I am surfacing them now, at plan time, because two of them may
change what Stage 0 should freeze as golden values.

**K1 — RESOLVED by the author (2026-08-03).** Keep the disordered system and the existing
checkpoints; re-pre-register against the infinite-randomness fixed point. No clean-field
models are to be trained. The uniform-`h` OOD evaluation is retained as a **labeled
secondary generalization probe, reported as distribution shift**, never as the primary test.
Four amendments were attached; see §4A. `PREREGISTRATION.md` cannot be committed until all
four are discharged.

**K2 — Depth is 3, not 12. H4 cannot reach significance as pre-registered.**
`n_layers = 3`. A Spearman rank correlation over 3 points has a permutation null of `3! = 6`
orderings, so the smallest attainable two-sided p-value is `2/6 = 0.333` — before any
Holm–Bonferroni correction across the layer family. H4 is **not falsifiable at conventional
α with this architecture**; it can only ever return "CI spans zero," which the brief
correctly identifies as the interesting-dissociation branch, but here it would be an
artifact of `n=3`, not a finding. My S0.4 extraction gets this to 8 hook points, which
raises the floor to `2/8! ` territory in principle but the points are not independent
(sub-layer writes within a block are strongly coupled), so the effective n is somewhere
between 3 and 8 and I would not claim otherwise. Honest options: report H4 as
**underpowered by construction** with the power calculation stated, or train deeper models.
The brief's appendix language ("across 12 layers") does not describe this repo.

**K3 — L = 10 and L = 12 checkpoints do not exist.**
Stage 2 as written ("existing trained checkpoints, L = 8, 10, 12, ≥ 5 seeds") cannot run.
Only L = 8 exists, at 10 seeds. Getting to L = 10 and 12 at 5 seeds each means training 10
new models. From `ra08_scaling` timings (~1840 s at L=8, ~2119 s at L=10, including probe
work) this is on the order of a few hours on the local workstation — tractable, but it is **new training,
not reuse**, and it is scope the brief did not budget. It also interacts with K1: if the
answer to K1 is (b), these models should be trained on clean fields, and then there is no
reason to train disordered ones.

**K4 — Hardware.** The brief's Part 9 specifies SLURM on an HPC cluster (A100). This arm runs on a local workstation with no SLURM; exact hardware specifics stay in
gitignored local config. I will write the
`slurm/` scripts as specified so they are submittable elsewhere, but Stages 0 and 1 are
pure CPU and will run locally in minutes. I will not fabricate a SLURM job ID in the
provenance header when there is none — it will be recorded as `null`.

**K5 — Citation not yet verified.** Brief Part 2 cites Qi & Earls, arXiv:2607.01336, with an
explicit instruction to verify before use and not to assert anything unread. I have not
verified it — that needs network access. Until I do, it appears nowhere in any output. I
will verify it as part of Stage 0 if you want it in scope, or leave it for the write-up.

---

## 4A. K1 amendments — the four things that must be discharged before `PREREGISTRATION.md`

Author-directed, 2026-08-03. Items (a) and (b) are already verified below; (c) is verified
as feasible; (d) is a design change carried into Stage 2.

### (a) The RTFIM IRFP constant — VERIFIED, author was correct, I was wrong

Source read directly: G. Refael & J. E. Moore, *Entanglement entropy of random quantum
critical points in one dimension*, PRL **93**, 260602 (2004), arXiv:cond-mat/0406737.

- **Random quantum Ising (RTFIM), Eq. (22):** `S_L = (1/6) ln L + k = (ln2/6) log₂ L + k`,
  with the text stating "the effective central charge of the random quantum Ising model is
  `c̃ = 1/2·ln 2`". → **`c̃_Ising = (ln 2)/2 ≈ 0.34657`.**
- **Random Heisenberg / random-singlet, Eq. (18):** `S_L = (ln2/3) log₂ L + k` →
  `c̃_RS = ln 2 ≈ 0.69315`. This is the value I originally and incorrectly quoted for the
  RTFIM.
- Paper's own statement of the relation: "The central charges we find for the random
  Heisenberg, XX, and quantum Ising chains are those of the pure models times ln 2."
  Hence `c̃ = c · ln2`, and `c = 1/2 → c̃ = (ln2)/2`. Eq. (22) also notes the critical
  quantum Ising chain "has half the entanglement of the random singlet phase."

**Convention, stated explicitly as instructed — this matters more than the constant.**

| | Refael–Moore (2004) | This repo |
|---|---|---|
| Log base | **bits** (`S = −Tr ρ log₂ ρ`, their Eq. 1) | **nats** (`observables.half_chain_entanglement_entropy` uses `np.log`) |
| Geometry | segment of length L inside the chain → **two** boundaries → `c̃/3` convention (their Eq. 3) | **open** chain cut once at ℓ → **one** boundary → `c̃/6` convention |

The two differences compound to a factor of `2 × ln2` between the paper's printed number
and anything this codebase prints. The form to pre-register, in **this repo's units**:

> **`S(ℓ) = (c̃/6) · ln[(2L/π) · sin(πℓ/L)] + k`,  open BC, nats, `c̃ = (ln2)/2`**
> → predicted slope against `ln[(2L/π) sin(πℓ/L)]` is **`ln2/12 ≈ 0.05776`**.

Cross-check: RM Eq. (22) gives `1/6 ln L` bits for two boundaries → `1/12 ln L` bits for
one → `(ln2/12) ln L` nats. Matches `(c̃/6)` with `c̃ = ln2/2`. ✓

**Power warning to be pre-registered, not discovered later.** Clean Ising is `c = 0.5`;
IRFP is `c̃ = 0.347`. The original acceptance interval `[0.35, 0.65]` straddles both and is
therefore useless as written. The two targets are only 0.153 apart; Refael–Moore is an
asymptotic `L → ∞` RG result; and this ensemble's disorder is moderate (`σ_lnh = 0.709`),
not infinite-randomness. At `L ≤ 12` the fitted value will plausibly land *between* the two
and may not separate them. The pre-registration must state the expected non-convergence and
declare in advance what "cannot distinguish" looks like, rather than treating an
intermediate value as support for either.

### (b) Is the training ensemble actually critical? — COMPUTED, author was correct: it is not

IRFP criticality requires `[ln J] = [ln h]`. The pipeline uses `J = 1.0` **uniform, with no
bond disorder at all** (`data.py`), so `[ln J] = 0` exactly and `Σ_i ln J_i ≡ 0` for every
realization. For `h ~ U(0.1, 2.0)`, computed exactly:

| Quantity | Value |
|---|---|
| `E[ln h]` | **−0.149183** |
| `Var[ln h]`, `σ_lnh` | 0.502803, **0.709086** |
| `[ln J] − [ln h]` | **+0.149183** → **ordered (ferromagnetic) side** |

Confirmed empirically against the real cached fields (`data/tfim_L8_N50k_seed{1,2,3}.pt`,
150,000 realizations): per-seed `E[ln h]` = −0.1498, −0.1490, −0.1480. The ensemble mean sits
**+0.595 σ on the ordered side** at L=8 (+0.665 at L=10, +0.729 at L=12 — it drifts further
from criticality as L grows, which is itself worth stating). Exact values go in
`RESULTS_STAGE0.md` per the author's instruction, recomputed there for all L and all seeds.

### (c) Realization-level tuning axis `δ_r` — FEASIBLE, well populated, no retraining

Adopt `δ_r = (Σ_i ln J_i − Σ_i ln h_i)/σ` as the H1 tuning axis in place of `h`. Since
`Σ_i ln J_i ≡ 0` here, this reduces to `δ_r = −(Σ_i ln h_i)/σ`.

**Normalization CONFIRMED by the author (2026-08-04): `σ = √L · σ_lnh`.**

**Why this is the principled choice, not a convenience (the ν = 2 reasoning).** At the
infinite-randomness fixed point of the RTFIM the correlation-length exponent is **ν = 2**
exactly (Fisher), not the clean-Ising ν = 1. The finite-size scaling variable is therefore
`δ · L^{1/ν} = δ · L^{1/2} = δ√L`. Writing the realization-level quantity out: for a typical
realization `Σ_i ln h_i = L·E[ln h] + √L·σ_lnh·ξ` with `ξ ~ N(0,1)`, so

> `δ_r = −(Σ_i ln h_i)/(√L σ_lnh) = √L·(−E[ln h]/σ_lnh) − ξ = δ_ens·L^{1/2} − ξ`

The `√L` normalization **is** `L^{1/ν}` with `ν = 2`. So `δ_r` is not an ad-hoc unit-variance
rescaling — it is exactly the IRFP finite-size scaling variable, with an O(1) realization-level
fluctuation on top. The rejected alternative (`σ = σ_lnh`) would carry a spurious extra factor
of `√L` and would not collapse.

**H1 UPGRADED to a finite-size-scaling collapse test.** Peak-drift is a weak, largely
qualitative claim. Since `δ_r` is already the scaling variable, the stronger and far more
falsifiable statement is:

> **H1 (revised):** `S_model(δ_r; L)` for L = 8, 10, 12 collapses onto a single scaling
> function `f(δ_r)` when plotted against `δ_r = δ·L^{1/2}`, matching the collapse of
> `S_exact(δ_r; L)` on the same realizations.
> **Falsifiers:** no collapse; collapse only under an exponent inconsistent with ν = 2;
> or collapse of `S_exact` without collapse of `S_model`.

Collapse quality to be scored by a pre-registered residual statistic with bootstrap CI, and
`ν` to be fitted as a free parameter in a stated secondary analysis so that "ν = 2 assumed"
and "ν = 2 recovered" are never conflated. This subsumes the old peak-drift H1, which is
retained as a reported-but-secondary descriptive.

Occupancy, measured on the real pooled train fields (L=8, 3 seeds, N=150,000):

| δ band | realizations | share | ≈ per seed |
|---|---|---|---|
| `\|δ_r\| < 0.05` | 5,503 | 3.67% | ~1,834 |
| `\|δ_r\| < 0.10` | 10,980 | 7.32% | ~3,660 |
| `\|δ_r\| < 0.25` | 27,154 | 18.10% | ~9,051 |
| `\|δ_r\| < 0.50` | 52,545 | 35.03% | ~17,515 |
| `δ_r < 0` (paramagnetic side) | 44,079 | 29.39% | — |

**The critical sub-ensemble is genuinely in-distribution and well populated** — the tightest
band still holds ~1,834 realizations per seed, and both phases are sampled (29.4% sit on the
paramagnetic side). Amendment (c) is viable exactly as the author specified, with no
retraining. Full stratification tables for L=8/10/12 × all seeds go in `RESULTS_STAGE0.md`.

Caveat to record: `δ_r` is a coarse, extensive summary of the realization. Two realizations
with equal `δ_r` can have very different low-energy structure at these sizes. `δ_r` orders
the ensemble; it does not make finite chains critical.

### (d) H2 primary becomes a paired per-realization test; `c_eff` demoted to secondary

New primary H2: for each disorder realization `r`, compare `S_model(ℓ; r)` against
`S_exact(ℓ; r)` **for that same realization**, paired. This is strictly stronger than a
fitted-constant comparison — exact ground truth is available per realization at these sizes
(Stage 0 gives it two independent ways), the pairing removes realization-to-realization
variance that would otherwise swamp the effect, and it does not depend on asymptotic RG
convergence, which §4A(a) says will not be reached at `L ≤ 12`. Test statistic, effect size,
and CI construction to be fixed in `PREREGISTRATION.md`; pairing is on realization identity
and must respect the realization-disjoint split of S1.4.

Report **both** disorder-averaged `[S]` and **typical** `S_typ` (median, equivalently
`exp[ln S]`) separately at every cut — they differ at an IRFP, and collapsing them hides the
distributional structure that is the signature of the fixed point. Note that Refael–Moore
argue entanglement is self-averaging as `N → ∞` (p. 4), so at `L ≤ 12` any mean/typical gap
is a finite-size statement and must be labeled as one, not as an IRFP confirmation.

`c_eff` remains reported, with bootstrap CI, as a **secondary** quantity against the
`(a)`-derived target `ln2/12 ≈ 0.05776` slope — carrying the power caveat above.

### Retained secondary: uniform-`h` OOD probe

The former option (3) stays in scope as an explicitly labeled **generalization probe under
distribution shift**. The uniform diagonal is a measure-zero corner of `U(0.1,2.0)^L`; any
result there describes what the model extrapolates to, not what it was trained on, and will
be captioned that way. It is never the primary evidence for H1 or H2.

---

## 4B. D3 OVERRIDE — new private repo, and the cross-repo pinning contract

Author-directed, 2026-08-03, overriding both the brief's own recommendation and mine. New
private repo. Part 1.4 verification applies in full from commit 1; the 38 legacy trailers
stay in `quantum-structure-sae` and never enter the new tree. The old repo's attribution
state is to be recorded in `DEVIATIONS.md` for the record.

### 4B.1 Mechanism: submodule + editable install on top (answering the author's question)

**Recommendation: git submodule, with `pip install -e` pointed at the submodule path.**
Reasoning, given requirement 5 wants a *test* that asserts the dependency SHA:

| Option | Pins an exact SHA? | Assertable offline? | Verdict |
|---|---|---|---|
| Editable install (`pip install -e ../quantum-structure-sae`) | **No** — you get whatever is checked out | no | **Fails requirement 1 outright.** This is precisely the drift being guarded against. |
| Pinned VCS install (`pip install "qsae @ git+https://…@SHA"`) | Yes, in the lockfile | needs metadata parsing; network to install | Workable, but the SHA lives in package metadata, not in the repo's own history |
| **Git submodule** | **Yes — recorded as a gitlink object in the new repo's tree** | **Yes — `git ls-tree HEAD submodules/qsae` returns the SHA directly** | **Recommended** |

The submodule makes the pinned SHA a first-class object in the new repo's own commit tree,
so `test_cross_repo_pin.py` can assert it with `git ls-tree` — no network, no package
metadata, no import side effects. Layering `pip install -e submodules/quantum-structure-sae`
on top gives working imports without weakening the pin. Bumping the pin becomes an explicit,
reviewable commit that changes one gitlink — which is exactly the audit trail requirement 1
is asking for.

### 4B.2 BLOCKER — the pin target is not on the remote

`HEAD` of the source repo is **`0c4e6e4`**. `origin/main` is **`a422a94`**. There are **5
unpushed commits**:

```
0c4e6e4 Task 5: claim -> source -> commit traceability table
ea4d539 Task 5: archive legacy run summaries into results/legacy/
3d99fbc Task 4: wire phase05/06/07 into the reproduction path
3a3d4ee Disambiguate 0.028 + flag superseded methodology in 8.1-8.2
479a6ed Task 3: reconcile tables, label sigma by measure, fix 8.3 rendering
```

A submodule pinned at `0c4e6e4` **cannot be cloned by anyone, including you on another
machine** — the SHA does not exist on the remote. Options: (i) push `quantum-structure-sae`
to origin and pin `0c4e6e4`; (ii) pin `a422a94` instead, which predates the traceability
table and the legacy-results archival that Stage 4 wants to vendor; (iii) submodule against
a local path, which is not reproducible and I do not recommend. Pushing the old repo is an
outward-facing action on a repo outside this arm's scope, so I will not do it unilaterally.

### 4B.3 BLOCKER — requirements 3 and 4 cannot be met by *any* git mechanism as things stand

The source repo's `.gitignore` excludes `runs/`, `data/`, and `*.pt`. Verified:

| Artifact | Size | Tracked in git? |
|---|---|---|
| `data/tfim_L8_N50k_seed{1..10}.pt` + others (15 files) | **38 MB** | **No — ignored** |
| `runs/ms_trained/seed{1..10}/best.pt` (10 checkpoints) | **18 MB** | **No — ignored** |
| `runs/ra08_scaling/scaling_results.json` (the +0.028 source) | small | **No — ignored** |
| `results/*.md`, `results/legacy/*.md`, `docs/*.md` | 60 KB | **Yes — tracked** |

So a submodule at any SHA delivers **code only**. The 150,000 realizations and all 10
checkpoints exist **solely as untracked local files on this one machine**, with no replica in
any git history. Requirements 3 and 4 therefore need a mechanism *outside* the submodule:

- `pins/ensemble.sha256` and `pins/checkpoints.sha256` — small text manifests, committed to
  the new repo, each line `<sha256>  <logical-name>`, alongside the source-repo SHA.
- Artifact root resolved from **`$QSAE_ARTIFACTS`** (per Brief 1.5: no absolute paths, no
  `/home/<user>/…` in tree), with a gitignored `.env.local` holding the machine-local value.
- A loader that hashes on read and **raises** on mismatch — never regenerates, never warns
  and continues. Requirement 3's failure mode (silently different `δ_r` values, silently
  broken realization-disjoint splits) is invisible without this, which is why it fails loud.
- `tests/test_cross_repo_pin.py` asserts submodule SHA + both manifests against what
  `PREREGISTRATION.md` declares, and runs in the Stage 0 gate (requirement 5).

Verified hashes to seed the manifests (SHA-256, spot sample):

```
10aacd0f50a4d16f8c3b1fc66bd8c4ddac6e5e8ac12c75b144306f16e5bec9f4  data/tfim_L8_N50k_seed1.pt
2bfc7dab9e5dc2db28385ed1e35db1464e7d28d0b2c3682f61c3df781cedf1c0  data/tfim_L8_N50k_seed2.pt
bf0bfa8597bee1decb18e9169015bcd92887d95ef131460b1c71922a9173a159  data/tfim_L8_N50k_seed3.pt
f1dcf0903f264b0aa4eab4addd02a6a5b30040ea24825ec7e01a7a555161e83b  runs/ms_trained/seed1/best.pt
76e5f8533a9cd9dee986f4dafa8421c8aa591f1a172e61174f29d6180630c6d8  runs/ms_trained/seed2/best.pt
```

**Risk I am escalating rather than absorbing:** 56 MB of artifacts on which the entire
cross-arm comparability claim rests are currently single-copy, untracked, and unreplicated.
Hashing them pins *identity*; it does not pin *existence*. If that disk is lost, requirement 4
becomes unsatisfiable permanently — the checkpoints the +0.028 was measured on cannot be
recreated, because `runs/ms_trained` was produced by training runs whose exact conditions are
only partially recorded. This should be backed up before Stage 0, not after. (Note: the
ensemble alone is probably regenerable — `data.py:266` uses `np.random.default_rng(seed)`,
whose stream NumPy's policy treats as stable — but the author has correctly forbidden
regeneration as a load path, and the checkpoints are not regenerable at all.)

### 4B.3a BLOCKER — `gh` is not installed on this machine

`gh repo create <name> --private` cannot run: `gh: command not found`, and there is no
`gh auth status` to inspect. Three ways forward, all yours to pick:
(i) install it yourself — `! sudo apt install gh && gh auth login` — the `!` prefix runs it
in this session so the output lands here;
(ii) create the private repo in the GitHub web UI and tell me the URL; I then
`git init` locally, `git remote add origin <url>`, and push when you say so;
(iii) I `git init` the new repo **locally only**, with no remote at all, and we defer repo
creation entirely. Given Brief 1.3 ("do not push anything until I say so") this is the
lowest-friction option and blocks nothing in Stages 0–1, which need no remote.

I recommend (iii) for now and (i) or (ii) whenever you actually want it on GitHub — no part
of Stage 0 or Stage 1 requires a remote to exist.

### 4B.3b Relocation of the two planning files

`PLAN.md` and `TENSOR_NETWORK_ARM_BRIEF.md` are currently untracked in the **old** repo's
working tree, which was correct when D3 pointed at a branch. Under the override they belong
to the new repo. On approval I will move both (`git mv` is not applicable — they are
untracked, so a plain move), leaving `quantum-structure-sae` exactly as I found it: `HEAD`
`0c4e6e4`, working tree carrying only its own pre-existing untracked
`scripts/exp_ra12_ablations.py`. I have made **no** modification to any tracked file in the
old repo and will not.

### 4B.4 Vendoring scope (requirement 2)

Vendorable today, tracked and small: `results/legacy/ra08_scaling.md` (the +0.028 gain),
`results/legacy/ra09_mixedfield.md` (the null), `results/phase06_multiseed_trained.md`,
`results/phase05_input_control.md`, `results/TRACEABILITY.md`, `results/legacy/ra04_sae_grid.md`.
Each copied under `vendor/qsae-summaries/` with its own SHA-256 and the source-repo commit SHA
in a sidecar.

**Gap:** the **per-layer SAE feature-gain table** named in requirement 2 **does not exist in
the source repo** — `analysis/extract.py` hooks only `layers[-1]`, so every SAE number there is
final-layer. There is nothing to vendor. That table must be *produced by this arm* (via S0.4's
8-point extraction) before H4 can be tested, and it will then be a new-repo artifact, not a
vendored one. This does not change requirement 2; it changes what requirement 2 can cover today.

---

## 5. Blocking decisions I need from you before the first commit

**D1 — RESOLVED (2026-08-03).** The author is Miheer Kulkarni. Existing identity
`Miheer Kulkarni <222050236+miheer-smk@users.noreply.github.com>` is correct and stays; the
"not Miheer" clause in Brief 1.2 means *never add a co-author trailer for anyone*, which is
already the configured behaviour. No `git config` identity change is needed.

**D2 — Existing attribution cannot be cleaned without violating Part 1.3.** The history
already contains **38 AI co-author trailers** (33 × `Claude Opus 4.8`, 5 ×
`Claude Sonnet 4.6`), all on commits already pushed to `origin/main`. Removing them means
rewriting pushed history, which Part 1.3 forbids outright. Your `~/.claude/settings.json`
already has `attribution.commit`/`.pr` set to `""`, so **new** commits are clean — the Part
1.4 verification will show `CLEAN` only if it is scoped to the new branch's commits, and
will show 38 hits if run over full history. Tell me which you want: leave history as-is and
scope the check, or rewrite (against the brief's own rule).

**D3 — RESOLVED by author override (2026-08-03): new private repo.** See §4B for the
cross-repo pinning contract, the recommended mechanism (submodule + editable install), and
the two blockers found while specifying it. **Still needed from you: the repo name** — the
instruction was literally `gh repo create <name> --private`.

**D4 — RESOLVED by the author (2026-08-04): NO amplitude head.** Construction B only —
layer-resolved **linear** readout. The studied model is not modified.

**Runtime Schmidt-rank assertion (author-directed).** A linear readout can only produce a
`ψ̂` whose Schmidt rank at cut ℓ is bounded by the readout's rank, so the decoded entropy
carries a ceiling of `ln(rank)` — the Brief Part 8 item 1 bond-dimension trap arriving through
the probe instead of through a truncation. Assert at runtime, per cut:

> `schmidt_rank ≤ min(feature_dim, 2^min(ℓ, L−ℓ))`, and **flag `censored=True`** whenever the
> first term binds.

Per-token feature dim is `d_model = 64`; flattened it is `L·d_model` (512 at L=8). Taking the
author's stated `min(d_model, 2^(L/2))` as the conservative per-token form: L=8 → `min(64,16)
= 16 = 2^4`, full rank, **not binding**; L=12 → `min(64,64) = 64 = 2^6`, exactly full,
**not binding**; L=16 → `min(64,256) = 64 < 256`, **binding — entropy censored at ln 64 ≈
4.16 nats**. So across the planned L ≤ 12 the readout does not censor, which is worth stating
explicitly in results rather than left implicit; at L = 16 it would, and the assertion will
say so loudly rather than silently manufacturing an area law.

**(superseded)** Attaching a
`d_model → 2^L` head and fitting it would make the strongest version of the experiment
possible, but it changes the model, and Part 4 says that decision is yours. Note it is a
*new* model, not the studied one — so it answers a different question ("can this
architecture represent the wavefunction") than Construction B ("is the wavefunction
decodable from the energy-trained model").

**D5 — RESOLVED by D1.** `LICENSE` (MIT, "Copyright (c) 2026 Miheer Kulkarni") and
`CITATION.cff` already name the sole author correctly. Nothing to change; I will not touch
either. If you later want to drop to all-rights-reserved pre-acceptance per Brief 1.5, say
so and I will, but I am not raising it again.

---

## 6. The scientific risk I think is largest, stated up front

`h` fully determines `ψ_h`. Any readout that faithfully encodes the input can in principle
reconstruct the ground state, so a high `S_decoded` at layer k may reflect nothing more
than "layer k still carries `h`." This is the same trap the project already walked into
once: **Phase 0.5 found that entropy specifically does *not* beat the random-init null
under the full degree-2 input control**, even though the non-local order headline survived
it (`docs/phase05_input_control.md`). That is a direct, in-repo, prior warning about H3.

So Construction B is not scientifically meaningful without the input control wired in from
the start — `S_decoded(layer k)` must be reported against (i) a random-init model of
identical architecture (C1) and (ii) a readout fit on the poly-2 features of raw `h`. The
inherited `src/qsae/analysis/input_control.py` does exactly this and will be a dependency
of the Stage 2 driver, not an afterthought. I am stating this now so that a weak Stage 2
result is read as the informative null it would be, rather than as something to tune.

---

## 7. What I will not do without asking

Create a repo; push; add a remote; force-push; rewrite history; run any experiment;
train any model; add an amplitude head; change `LICENSE` or `CITATION.cff`; commit
`PREREGISTRATION.md` before K1 is resolved; proceed past a stage gate before you have read
that stage's results file.

---

## 8. Immediate next step on approval

**All decisions are now resolved.** D1, D2, D3, D4, D5, K1 (+ four amendments), repo name,
pin target, `δ_r` normalization, and the Stage 1.5 restructure. K2/K3 remain deferred to the
Stage 1 gate. Execution order, as directed:

1. **Artifact backup — DONE.** Two copies + round-trip verification (§9).
2. **Push `quantum-structure-sae`, pin `0c4e6e4`.** Awaiting the backup gate, then executed.
3. `git init quantum-structure-entanglement` locally, **no remote**. Configure identity
   (unchanged), `attribution` (already set), `commit-msg` hook, and `CLAUDE.md` attribution
   rule **before commit 1**. Add submodule at `0c4e6e4`; `pip install -e` on it.
4. `pins/ensemble.sha256`, `pins/checkpoints.sha256`, `pins/README.md`; `$QSAE_ARTIFACTS`
   root outside both repos; fail-loud hashing loader; `tests/test_cross_repo_pin.py`.
5. `DEVIATIONS.md` recording the old repo's 38-trailer state and the new repo's clean start.
6. Commit 1, then Part 1.4 verification — must print `CLEAN` over **full** history.
7. Stage 0 → gate. Stage 1 → gate. **Stage 1.5 → gate.** Only then `PREREGISTRATION.md`.

**One item needs your decision before §3.5 can run** — see §3.5.0: the reproduction gate as
specified compares an SAE-derived number to a probe-derived number, and the +0.029 it names
was measured on checkpoints that no longer exist. My proposed substitute (R1 against
phase06's `0.0283 ± 0.0030` on the surviving `ms_trained` checkpoints, with the tolerance
pre-registered in §3.5.1) needs your sign-off, since you required the tolerance be fixed
before the number is seen.
3. Stage 0 build → gate → you read `RESULTS_STAGE0.md`.
4. Stage 1 build → gate → you read `RESULTS_STAGE1.md`. Stop.
