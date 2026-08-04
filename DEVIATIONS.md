# DEVIATIONS.md

Every departure from `TENSOR_NETWORK_ARM_BRIEF.md` or `PREREGISTRATION.md`, with a reason and
a date. Append-only. A deviation that is not recorded here did not happen legitimately.

---

## 2026-08-04 — Repository split (Brief 1.3)

**Deviation.** The brief's own recommendation, and the initial plan, was a branch
(`tensor-network-arm`) on `quantum-structure-sae`. The author overrode this in favour of a
new private repository, `quantum-structure-entanglement`.

**Reason.** Author's decision, to keep this arm's history clean from commit 1 (see next
entry) and to give the two arms a legible parallel relationship in a reference list. The
cost — Stage 4 must cross repos — is paid by the pinning contract in `pins/README.md`.

## 2026-08-04 — Attribution state of the predecessor repository (Brief 1.1, 1.4)

**For the record, not a deviation in this repo.** `quantum-structure-sae` carries **38
AI-attribution trailers** in its history (33 × `Co-Authored-By: Claude Opus 4.8`, 5 ×
`Co-Authored-By: Claude Sonnet 4.6`), on commits already pushed to its origin. Brief 1.3
forbids rewriting pushed history, so they remain.

**Consequence for this repo: none.** Those commits live in the predecessor repository and are
reachable here only through the submodule, which is a separate history. `quantum-structure-
entanglement` starts clean at commit 1, so the Brief 1.4 verification applies over **full**
history with no scoping. It is expected to print `CLEAN` permanently; if it ever does not,
that is a defect, not a legacy artifact.

## 2026-08-04 — Predecessor repository pushed before pinning

**Action taken.** `quantum-structure-sae` had 5 unpushed commits (`479a6ed`…`0c4e6e4`) on a
single machine. A submodule pinned at `0c4e6e4` would have been unclonable. On the author's
instruction the repository was pushed (`a422a94..0c4e6e4`, `main`), then pinned at `0c4e6e4`.

**Alternative rejected.** Pinning the already-pushed `a422a94` would have avoided the push but
lost `results/TRACEABILITY.md` and the `results/legacy/` archival that Stage 4 needs to vendor.
The author judged that the wrong trade.

## 2026-08-04 — Construction ranking departs from Brief Part 4

**Deviation.** The brief ranks the `d_model`-reshape MPS (Construction C) above the
ensemble-covariance variant, which it calls a "cheap sanity companion". This arm promotes the
covariance construction to the **primary basis-independent measure**.

**Reason.** The brief's objection to C — that the residual stream has no canonical
tensor-product structure — does not apply to the token axis. The residual stream is
`(N, L, d_model)` and **token index i corresponds to physical site i**, so a cut of the token
axis at ℓ inherits exactly the locality that makes the spin-chain tensor product canonical.
C is still implemented and still reported, with its C4/C5 nulls, but not as the headline.

## 2026-08-04 — Construction A unavailable; no amplitude head added

**Not a deviation — a decision.** The studied model maps `h ∈ R^L` to a **scalar** energy and
has no amplitude head, so Construction A is unavailable. The author declined to attach one
(Brief Part 4 requires asking). Construction B uses a layer-resolved **linear** readout to
`ψ_h` from exact diagonalization, with a runtime Schmidt-rank assertion
(`rank ≤ min(feature_dim, 2^min(ℓ, L−ℓ))`, `censored=True` when the first term binds) so a
probe-induced entropy ceiling can never be mistaken for a physical area law.

## 2026-08-04 — H1/H2 re-pre-registered for disorder; H1 upgraded to an FSS collapse

**Deviation.** The brief pre-registers `c_eff ∈ [0.35, 0.65]` against the clean Ising
`c = 1/2`. Every trained model is disordered (`h_i ~ U(0.1, 2.0)` i.i.d. per site), so the
relevant fixed point is the infinite-randomness one, where `c̃ = (ln 2)/2 ≈ 0.347`
(Refael & Moore, PRL 93, 260602 (2004), Eq. 22). The brief's interval straddles both values
and cannot discriminate.

**Also corrected:** an earlier statement in planning gave the RTFIM constant as `ln 2 ≈ 0.693`.
That is the random-singlet (Heisenberg) value, Eq. 18 of the same paper, and was wrong for
this model. Caught by the author.

**Convention pinned:** Refael & Moore work in **bits** with a **two-boundary segment**
(`c̃/3`); this codebase works in **nats** with a **one-boundary open cut** (`c̃/6`). The two
differences compound to `2 × ln2`. In repo units the prediction is a slope of `ln2/12 ≈ 0.0578`
against `ln[(2L/π) sin(πℓ/L)]`.

**Upgrade.** H1's tuning axis becomes the realization-level `δ_r = −(Σ_i ln h_i)/(√L σ_lnh)`,
and H1 becomes a finite-size-scaling **collapse** test rather than a peak-drift claim. The
`√L` normalization is `L^{1/ν}` with `ν = 2`, the IRFP correlation-length exponent, so `δ_r`
*is* the scaling variable rather than an arbitrary rescaling.

**Recorded limitation.** The training ensemble is **not** critical: `J ≡ 1` exactly (no bond
disorder) and `E[ln h] = −0.149183`, so `[ln J] − [ln h] = +0.149` — the ordered side — and the
offset grows with L (`+0.595σ`, `+0.665σ`, `+0.729σ` at L = 8, 10, 12). H1 is therefore tested
on a δ-stratified sub-ensemble, which is in-distribution and needs no retraining.

## 2026-08-04 — Uniform-field evaluation demoted to a secondary probe

**Deviation.** Evaluating on uniform `h` was considered as a route to clean criticality. It is
retained only as an explicitly labeled **generalization probe under distribution shift** — the
uniform diagonal is a measure-zero corner of `U(0.1, 2.0)^L`. It is never primary evidence for
H1 or H2. No clean-field models are trained.

## 2026-08-04 — Per-layer SAE work moved to its own Stage 1.5

**Deviation.** Producing the per-layer SAE feature-gain table was initially folded into Stage 0
as an inventory/extraction step. It requires training 8 SAEs per seed — an experiment, not an
inventory step — so it is now **Stage 1.5**, with its own gate, run **before**
`PREREGISTRATION.md` is committed.

**Substantive finding that forced a restatement.** The brief's H4 references "the layer of
maximum SAE feature gain". No such quantity exists in the predecessor repo:
`src/qsae/analysis/extract.py` hooks only `model.encoder.layers[-1]`, and
`results/legacy/ra04_sae_grid.md` — the only SAE result — reports matched-cosine universality,
dead fraction, and reconstruction MSE, and **no gain quantity at all**. The `+0.028`/`+0.029`
figures are **probe** gains on raw activations, not SAE quantities. H4's x-axis must therefore
be produced by this arm and reported as new work, and the reproduction gate applies to the
probe measure (R1), not to the SAE measure (R2), which has no published counterpart.

## 2026-08-04 — H4 primary axis changed to per-layer probe gain; H5 restated

**Deviation.** H4's pre-registered x-axis was "the layer of maximum SAE feature gain". No such
quantity exists (see the 2026-08-04 Stage 1.5 entry). H4's **primary, confirmatory** axis
becomes the per-layer `long_range_zz` incremental-R²-beyond-poly2 **probe** gain, extended to
all 8 hook points — which *extends* the published quantity instead of inventing one. The
per-layer **SAE** gain is retained as a **secondary, EXPLORATORY** axis; a new quantity with
no published counterpart cannot carry a confirmatory hypothesis.

**H5 audited, not assumed.** `runs/ra09_mixedfield/scaling_results.json` carries the keys
`probe_r2_{trained,untrained,raw_h,mean_h}` and `learned_gain`; a recursive case-insensitive
search for "sae" across the mixed-field artifacts returns nothing. The mixed-field null is a
**probe** result — the same defect, second instance. H5 is restated to concern the probe
measure throughout, with `learned_gain` = −0.0175 (L=8), −0.0070 (L=10), `g=0.5`.

**Also found:** `ra09` ran at `n_train=15000, epochs=100, seed=0` — same as `ra08` — so its
checkpoints were **also never saved**. Two published numbers, not one, have no recoverable
artifact. Recorded in `AUTHOR_HANDOFF.md`. Separately,
`results/legacy/ra09_mixedfield.md` is **mislabeled** in the predecessor repo (titled "RA-08",
carrying RA-08's caption, with RA-09's numbers). Left uncorrected in the pinned submodule so
the pin reflects what was published.

## 2026-08-04 — Cluster and hardware identifiers redacted (Brief 1.5)

**Deviation.** The brief's Part 9 names a specific national HPC cluster. Replaced with a
generic "HPC cluster" in all committed text, including inside the archived copy of the brief,
because the cluster name narrows the institution. Machine specifics live in gitignored local
config. This is the one place the archived brief departs from the author's verbatim text.

## 2026-08-04 — Stage 0: site-ordering convention fixed explicitly

**Finding.** `qsae.observables._reduce_density_matrix` computes `state.reshape(dim_A, dim_B)`,
making subsystem A the **high** bits, i.e. sites `[n - n_A, n)`; its docstring says
"qubits 0..n_A-1 (left block)". The predecessor's Hamiltonian uses `bit i = site i`, so site 0
is the **low** bit. The two are reflections of each other.

**Impact on published results: none** — all inherited results use the half cut, where
`S(left) = S(right)` exactly for a pure state. Not harmless for this arm, which needs the full
profile on asymmetric disordered chains: the free-fermion profile matched ED *reversed* to
4.86e-12 while differing unreversed by 4.2e-01.

**Resolution.** This arm fixes `site j <-> bit j`, block A = sites `[0, cut)`, in
`qsent/exact.py`, and does not use the inherited entropy function for profiles. The inherited
function is left untouched in the pinned submodule.

## 2026-08-04 — Stage 0: hook family is 7 points, not 8

**Deviation.** The plan proposed 8 hook points including `post_final_norm`. That tensor has
passed through `final_norm` and is a different normalisation from the published hook
(measured difference 2.39 vs the published tensor). Including it would place unlike tensors on
the same layer axis that H4's rank correlation runs along. The family is therefore **7**
post-residual-add, pre-`final_norm` points; `post_final_norm` remains extractable but is
excluded from the layer axis. `k=6` is verified identical to
`qsae.analysis.extract.last_layer_pooled` to 8.94e-07 (float32 machine precision).

## 2026-08-04 — Stage 0: provenance test added after a fabrication caught in draft

**Recorded because it is the kind of error that must not be quiet.** The first draft of
`tests/test_exact_entropy.py` inlined four of five "spanning realizations" as
plausible-looking numbers rather than the actual pinned vectors. Every physics assertion
still passed, because the ED/free-fermion gate holds for any `h`; the `delta_r` labels would
have been false while the suite stayed green. Caught by an out-of-band provenance comparison
against the pinned ensemble. `test_spanning_realizations_match_pinned_ensemble` and
`test_spanning_realizations_actually_span` are now permanent parts of the gate.
