# quantum-structure-entanglement

**Entanglement structure of energy-predicting transformers on the disordered transverse-field Ising chain.**

Status: **Stage 1 complete.** Apparatus and exact-physics reference measurements only —
**no transformer results exist yet.** Stage 2 (the first measurement on trained models) has
not been run. 57 tests passing.

---

## What this is

A companion experimental arm to [`quantum-structure-sae`](https://github.com/miheer-smk/quantum-structure-sae),
which asks whether a transformer trained only to predict ground-state *energies* of the
disordered transverse-field Ising model (TFIM) encodes quantum observables it was never
trained on.

This arm asks a sharper, more falsifiable version of that question:

> Does a specific, measurable invariant of the model's represented state — its **entanglement
> entropy** — behave the way the physics predicts, as a function of a criticality parameter?

Entanglement entropy is a good instrument for this because it is *native to the physics*
rather than borrowed from machine-learning interpretability, and because the TFIM is exactly
solvable, so every model-side number has an exact per-realization ground truth to be compared
against.

### The system

    H = - Σ_j J_j Z_j Z_{j+1}  -  Σ_j h_j X_j        (open boundaries)

with `J ≡ 1` and per-site disorder `h_j ~ U(0.1, 2.0)` i.i.d. The studied model maps
`h ∈ R^L` → a **single scalar energy**; it has no amplitude head and never sees a spin
configuration. That fact constrains the whole design (see *Method* below).

---

## Pre-registered hypotheses

Confirmatory and exploratory hypotheses are separated **in advance**, not in a later
limitations section. The full text lives in `PLAN.md` §3.55 and §3.6.

### Confirmatory — the spine

| | |
|---|---|
| **H1** | `S_model` collapses as a function of the realization-level criticality parameter `δ_r = δ·L^{1/ν}` with `ν = 2`, matching the collapse of `S_exact` on the same realizations. |
| **H2** | **Primary:** paired per-realization comparison of `S_model(ℓ; r)` against `S_exact(ℓ; r)`. **Secondary:** `c_eff`, reported descriptively only (see *Honest limitations*). |

Both are powered by realization count, not by network depth.

### Exploratory — stated as such up front

**H3** (depth profile) and **H4** (layer coincidence with probe gain) are **exploratory**, and
the pre-registration says why: the model has **3 blocks**, giving 7 residual-stream hook points
of which only 3 are block outputs, and they are not mutually independent. The depth axis is not
resolvable, so no result along it can be confirmatory. Making them confirmatory would require
deeper models — a separate arm, which would also break checkpoint reuse for H4.

**H5** concerns null concordance in the mixed-field regime, where the published probe gain
collapses.

---

## Method: which construction is available

The natural approach — read amplitudes `ψ_θ(σ)` off the model and compute entanglement
directly — is **unavailable**: the model emits one scalar and takes no configuration input.
So:

| Construction | Status |
|---|---|
| **A** — amplitude-space entanglement | **Not available.** No amplitude head; none was added, by design. |
| **B** — layer-resolved decodable entanglement | **Primary.** Linear readout `residual_k(h) → ψ_h`, with `ψ_h` from exact diagonalization, then entanglement of the decoded state. |
| **C** — `d_model` reshape MPS | Secondary; reported only against permutation and basis-rotation nulls. |
| **Token-axis covariance** | **Promoted to primary basis-independent measure.** The residual stream is `(N, L, d_model)` and **token i ↔ site i**, so cutting the token axis is a physically canonical bipartition — the locality that Construction C lacks. |

Construction B carries a runtime **Schmidt-rank assertion**
`rank ≤ min(feature_dim, 2^min(ℓ, L−ℓ))`, flagging `censored=True` when the readout rather
than the physics limits the entropy. Non-binding at L = 8 and 12; binding at L = 16.

---

## Results so far (exact physics only — no model numbers)

Everything below is a property of exact ground truth or of the estimators, established
**before** any model was touched.

### Ground truth validated two independent ways

Exact diagonalization vs the free-fermion (Peschel/Majorana) solver agree to
**1.648 × 10⁻¹¹** across 14 cases — L = 8/10/12 at uniform `h ∈ {0.5, 1, 2}`, plus five
disordered realizations chosen to span `δ_r` from clearly ordered to clearly paramagnetic.

### The uniform-field tests are degenerate — proven, not assumed

A deliberately **site-blind** solver (reads `h[0]`, applies it everywhere) passes *every*
uniform-field test at 10⁻¹¹–10⁻¹⁴ and fails every disordered one. With a uniform field, "uses
per-site `h_j`" and "uses `h[0]`" are the same computation, so the uniform gate cannot
separate them *in principle*. The disordered cases are the load-bearing ones.

### `c_eff` finite-size bias — measured, and it decides a hypothesis

| | L = 8 | L = 10 | L = 12 |
|---|---|---|---|
| Clean critical chain (true `c = 0.5`) | +0.0881 | +0.0845 | +0.0809 |
| Disordered, disorder-averaged (target `ln2/2 = 0.347`) | **+0.1953** | **+0.2091** | **+0.1763** |
| Disordered, typical (median) | +0.2954 | +0.3013 | +0.2559 |

The disordered bias **exceeds the entire 0.153 gap** separating clean Ising (`c = 1/2`) from
the infinite-randomness fixed point (`c̃ = ln2/2`). Even-odd oscillation terms remove none of
it. Bootstrapped clean-vs-disordered gaps are **+0.046 / +0.029 / +0.058**, and **at L = 10 the
95% CI spans zero**.

**Pre-registered consequence:** at these system sizes `c_eff` **cannot distinguish the two
universality classes**, and no such claim will be made from it. A bias-corrected estimator was
considered and rejected — the correction is larger than the effect and differs between the two
classes, so applying it would amount to fitting the answer. **H2's primary test is unaffected**,
because the paired per-realization comparison uses no `c_eff` fit at all.

### H1's machinery is testable on exact ground truth

Collapse quality `Q` (0 = perfect), 12,000 realizations per L, 400 bootstrap resamples:

| collapse variable | Q | 95% CI |
|---|---|---|
| **ν = 2** (`δ·L^{1/2}`, pre-registered) | **0.0091** | [0.0095, 0.0152] |
| ν = 1 (wrong exponent — control) | 0.0376 | [0.0413, 0.0710] |

Exact ground truth collapses, and the pre-registered exponent beats the control ~4× with
non-overlapping CIs. The control matters: a small `Q` alone could just mean coarse binning.

### The extraction stack is commensurable with the published pipeline

The predecessor's published probe gain was measured on a hook at `encoder.layers[-1]`. Because
`nn.TransformerEncoder` is built with no `norm=` argument, that tensor is the post-residual-add,
**pre-`final_norm`** residual stream. This arm's hook `k=6` reproduces it to **8.94 × 10⁻⁷**
(float32 machine precision). `post_final_norm` differs by 2.39 and is **excluded** from the
layer axis — it is a different normalisation, and mixing it in would place unlike tensors on
the axis a rank correlation runs along.

---

## Reproducibility

### Two-repo pin contract

Physics and data-generation code are **never copied or forked** into this repo — they are
pinned as a submodule at an exact commit. If the two generators could drift, comparability
with the SAE arm would be lost.

```bash
git ls-tree HEAD submodules/quantum-structure-sae
# 160000 commit 0c4e6e4a8a0fb68aec6820eea8c7eed49d05a539
```

Binary artifacts (disorder ensembles, model checkpoints) are **gitignored in the upstream
repo** and so cannot be pinned by any git mechanism. They are instead referenced by content
hash under `$QSAE_ARTIFACTS` (`pins/ensemble.sha256`, `pins/checkpoints.sha256`). The loader
**hashes on read and raises** on mismatch — it never regenerates and never warns-and-continues,
because a regenerated ensemble carries different `δ_r` values and would silently break every
realization-disjoint split.

### Setup

```bash
git clone --recurse-submodules <this repo>
cd quantum-structure-entanglement
pip install -e . -e submodules/quantum-structure-sae

cp .env.local.example .env.local     # then set QSAE_ARTIFACTS (outside both repos)
OMP_NUM_THREADS=4 pytest -q          # 57 tests
```

> The suite does dense `eigvalsh` at L = 12 (4096×4096). Leave BLAS threads bounded and do
> not run it concurrently with itself, or it will appear to hang.

---

## Repository layout

```
src/qsent/
  exact.py          exact diagonalization; the binding site/bit convention
  free_fermions.py  Peschel/Majorana solver (independent ground truth)
  extraction.py     7-point residual-stream extraction; k=6 == the published hook
  disorder.py       delta_r criticality parameter and stratification
  splits.py         realization- and field-value-disjointness, asserted at runtime
  pins.py           hash-verified artifact loading; published constants read, not typed
tests/              57 tests, incl. falsifiability checks (see below)
pins/               content hashes + the cross-repo contract
PLAN.md             staged plan, pre-registration text, power analyses
DEVIATIONS.md       every departure from the brief, dated, with reasons
RESULTS_STAGE*.md   per-stage results
AUTHOR_HANDOFF.md   what co-authors need to know
```

---

## Methodological standard

**A check is not a gate until it is shown able to fail** on the specific error it targets
(`CLAUDE.md`). This rule was adopted after three "looked right, unverified source" errors
reached committed code, each passing every test that existed at the time:

| Error | Slipped past | Caught by |
|---|---|---|
| Fabricated `h` vectors labelled as pinned realizations | the ED/free-fermion gate, which holds for *any* `h` | out-of-band provenance comparison |
| A regex reading `0.560` (partial correlation) as `0.0283` (incremental R²) | would have passed if the value were hardcoded | deriving the constant instead of typing it |
| A mirrored entropy profile | the `c_eff` fit — identical `c_eff` to 6.7 × 10⁻¹⁶ and identical residuals, since Calabrese–Cardy is symmetric under `ℓ → L−ℓ` | an explicit asymmetric-realization orientation test |

The common shape: **an assertion that holds for any input cannot detect wrong input.** Several
tests exist solely to satisfy this rule rather than to test physics, including an AST lint that
fails on any new inlined numeric array in `tests/` lacking pinned provenance.

Other standing commitments: stage gates are binding; nulls are written up as nulls at full
precision and never tuned away; every deviation is dated in `DEVIATIONS.md`; and claims are
restricted to what was measured. The claim is and remains narrow: *a specific, measurable
invariant of the represented state behaves in a specific, predicted way.* No broader assertion
about what the network is doing follows from it, and the pre-registration lists the stronger
framings that are ruled out of all outputs.

---

## Honest limitations

- **No model results yet.** Stage 2 has not run. Everything above is apparatus and exact physics.
- **`c_eff` cannot identify a universality class at L ≤ 12** (measured above). Reported descriptively only.
- **Depth is 3 blocks.** H3/H4 are exploratory by construction. At n = 7 hook points a Spearman test needs `ρ ≥ 0.786` for p < 0.05; at n = 3 or 4, significance is unreachable at any `ρ`. A null H4 at this n is **uninformative, not evidence of dissociation**.
- **Only L = 8 has a pinned ensemble**, and no L = 10/12 checkpoints survive upstream. A claim-bearing cross-L result needs new pinned ensembles *and* new training.
- **Two upstream published numbers have no recoverable artifacts** — their checkpoints were never saved. See `AUTHOR_HANDOFF.md`.

---

## Citation

See `CITATION.cff`. Unpublished work in progress; please do not cite results from this
repository without contacting the author.

## License

**MIT** — see `LICENSE`. Resolved in `PLAN.md` §5 D5; this section previously read "none yet,
all rights reserved pending publication", which contradicted the `LICENSE` file shipped in the
same tree. The `LICENSE` file is authoritative and the drift is recorded in `DEVIATIONS.md`.

The MIT grant covers the **code**. It is not an invitation to cite unpublished results: see
the Citation section above, and contact the author before citing anything from this repository.
