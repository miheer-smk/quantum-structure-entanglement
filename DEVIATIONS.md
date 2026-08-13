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

## 2026-08-04 — Paper spine: H1/H2 confirmatory, H3/H4 exploratory

**Deviation.** The brief treats H1–H5 as a flat set. This arm designates **H1 and H2 as
confirmatory** and **H3 and H4 as exploratory**, in the pre-registration rather than in a
limitations section. Reason, stated in advance: the model has 3 blocks, so the depth axis
H3/H4 require is not resolvable (7 hook points, only 3 of them block outputs, and not
mutually independent). H1/H2 need no depth axis and are powered by realization count.
Making H3/H4 confirmatory would require deeper models — a separate arm, which would also
break checkpoint reuse for H4. Noted, not done.

## 2026-08-04 — Standing rule added: a check is not a gate until shown able to fail

Three "looked right, unverified source" errors reached committed code during Stage 0, each
passing every test then in existence: fabricated `h` vectors (invisible to a gate that holds
for any `h`), a regex returning the partial-correlation `0.560` in place of the
incremental-R² `0.0283`, and a mirrored entropy profile (invisible to the `c_eff` fit, which
returns identical `c_eff` and identical residuals under `l -> L-l`). `CLAUDE.md` now requires
every validation check to be demonstrated capable of failing on the error it targets.

## 2026-08-04 — c_eff demoted further: no universality-class claim at L <= 12

**Deviation.** The brief's H2 pre-registers `c_eff` against an acceptance interval. Measured
finite-size bias at the actual system sizes makes that untestable: on the disordered critical
sub-ensemble the bias is +0.176 to +0.209 (disorder-averaged) and +0.256 to +0.301 (typical),
against a clean-vs-IRFP gap of only 0.153. Even-odd oscillation terms remove none of it, and
the observed clean-vs-disordered separation (+0.046, +0.029, +0.058 at L = 8, 10, 12) is
non-monotonic, so nothing can be extrapolated.

**Bias-corrected estimator considered and rejected**: the correction exceeds the effect,
differs between the clean and disordered universality classes, and is non-monotonic in L.
Applying it would amount to fitting the answer.

**Pre-registered instead:** `c_eff` is reported descriptively with a bootstrap CI plus the
measured bias table, and **no universality-class identification is claimed from it** at these
sizes. H2's PRIMARY (paired per-realization `S_model` vs `S_exact`) is unaffected — it uses no
`c_eff` fit — and the write-up must say so adjacent to the caveat.

**Reference data note.** L = 10 and L = 12 disordered chains used for this measurement were
generated reference-only under a documented separate RNG. They are explicitly NOT a pinned
ensemble and are never used for any model claim; only L = 8 has a pinned ensemble.

## 2026-08-11 — Remote repository exists, is PUBLIC, and has diverged; licensing is contradictory

**Recorded on discovery, before any corrective action.** Nothing below has been changed. No
`LICENSE` was written, no `README.md` line was edited, and the local branch has **not** been
synced to the remote. The only mutation performed was `git fetch origin`, which advances the
read-only `origin/main` tracking ref and touches neither the working tree nor `main`.

### (1) A remote exists, contrary to the plan of record

`PLAN.md` §4B.3a recommended option (iii) — `git init` locally, **no remote** — and §8 records
that as the execution order. §7 lists "create a repo; push; add a remote" among the actions not
to be taken without the author's instruction. A remote nevertheless exists:

| | |
|---|---|
| Remote | `git@github.com:miheer-smk/quantum-structure-entanglement.git` |
| Created | 2026-08-04T18:19:57Z |
| Last push | 2026-08-04T18:39:55Z |

The repository was created and pushed by the author directly. This is not a deviation committed
by the assistant; it is recorded here because the plan of record says no remote exists, and a
reader of `PLAN.md` alone would be wrong about the project's outward-facing state.

### (2) The repository is PUBLIC, where D3 and §4B specified private

`PLAN.md` §4B and Decision **D3** both specify a **new private repo**
(`gh repo create <name> --private`). The GitHub API reports otherwise, and an
**unauthenticated** request succeeds (HTTP 200), which is itself proof of public readability:

```
private:    False
visibility: public
```

**Consequences that follow from public visibility, stated plainly:**

- The unpublished pre-registration, the two Stage results files, and every measured number in
  them are world-readable now. Pre-registration is *supposed* to be public, but normally after
  it is frozen, not while it is still being written.
- `AUTHOR_HANDOFF.md` publicly documents defects in the predecessor work — two published
  numbers with no recoverable artifacts, and a mislabeled file in `quantum-structure-sae`.
  That is honest and belongs in the paper, but it is currently disclosed with no surrounding
  context and before any co-author has seen it.
- The anonymizability rule in `CLAUDE.md` exists so the repo can be shared as an anonymized
  snapshot for double-blind review. A public repo under the author's own account, already
  indexed, forecloses anonymous submission to any venue that requires it.

**No action taken.** Whether the repository should be public is the author's decision and is
being escalated, not resolved here.

### (3) Local has diverged from the remote by one commit

| Ref | SHA | Contents |
|---|---|---|
| local `main` | `2ee2833` | "Add README and citation metadata" |
| `origin/main` | `58d1b79` | local `main` + one commit |

The extra commit is:

```
58d1b7952fac28dd6dcee25a05c0bc8aa7f9df8c
subject:   Add MIT License to the project
author:    Miheer Satish Kulkarni <miheer.smk@gmail.com>
committer: GitHub <noreply@github.com>
date:      Wed Aug 5 00:09:55 2026 +0530
files:     LICENSE | 21 +++++++++++++++++++++
```

The `GitHub <noreply@github.com>` committer identity shows this was made through the GitHub web
UI, not from this machine — which is why the file never existed locally.

**Attribution is clean.** A trailer scan over the full history *including* the remote-only
commit returns **0** matches for `Co-Authored-By` / `Generated with`. The Brief 1.4 verification
still prints CLEAN over full history.

### (4) The licensing state is self-contradictory, in three places at once

All three are live on the public remote simultaneously:

| Artifact | What it says |
|---|---|
| `LICENSE` (remote only, `58d1b79`) | **MIT** — grants use, modification, sublicense, and sale to anyone |
| `README.md` §License (lines 240–242) | "None yet — **all rights reserved** pending publication" |
| `CITATION.cff` | `license-url` points at that README anchor; message reads "Unpublished work in progress. Please contact the author before citing any result" |

MIT and all-rights-reserved are not reconcilable: one grants redistribution rights
unconditionally, the other reserves them. `CITATION.cff` compounds it by pointing its
`license-url` at the README's *disclaimer of* a license. `PLAN.md` D5 also records
"`LICENSE` (MIT …) already name the sole author correctly" as resolved — but no `LICENSE`
existed in the tracked tree at that time, so D5 was resolved against a file that was not there.

**Two lesser inconsistencies in the same material:**

- **Copyright name.** `LICENSE` reads "Copyright (c) 2026 Miheer Satish Kulkarni"; `CITATION.cff`
  gives `family-names: Kulkarni, given-names: Miheer`; `PLAN.md` D5 quotes
  "Copyright (c) 2026 Miheer Kulkarni". Three spellings of the copyright holder.
- **Author email.** Every commit on local `main` uses
  `222050236+miheer-smk@users.noreply.github.com`, per D1. The remote-only commit uses a
  personal address, `miheer.smk@gmail.com`. On a public repo that is a deanonymization vector
  against the `CLAUDE.md` anonymizability rule. It is already pushed, and Brief 1.3 forbids
  rewriting pushed history, so it stands — recorded rather than corrected.

**Nothing here has been changed pending the author's decision.** Per `CLAUDE.md`, changing
`LICENSE` or `CITATION.cff` requires the author's instruction, and the README/LICENSE conflict
cannot be resolved by picking one without knowing which posture is intended.

### ADJUDICATED by the author, 2026-08-11

**(a) `LICENSE` stands; the README was the drifted artifact.** `PLAN.md` §5 D5 had already
resolved licensing as MIT and directed that the file not be touched. The `LICENSE` file is
authoritative. `README.md` §License — which read "None yet — **all rights reserved** pending
publication. A license will be selected at acceptance" — contradicted a `LICENSE` file shipping
in the same tree, and has been corrected to state MIT and point at `LICENSE`.

The drift arose because D5 was recorded as resolved on 2026-08-03 against a `LICENSE` that did
not exist in the tracked tree until `58d1b79` on 2026-08-05, while the README was written on
2026-08-04 in between. Nobody was wrong at the time they wrote; the tree passed through a state
where the two disagreed and no check compared them. The README's "do not cite results without
contacting the author" line is retained — the MIT grant covers the **code**, not permission to
cite unpublished results.

**(b) The repository stays PUBLIC.** Author's decision, with the reasoning recorded so the
choice is legible later: a public, timestamped, frozen-before-results pre-registration is a
**credibility asset** for the eventual paper and establishes priority on the design. That is
judged to outweigh the confidentiality, which is largely notional given the repository is
already indexed.

This is therefore a **documented, accepted deviation from `PLAN.md` §4B and Decision D3**,
which specified a private repository, and from §4B.3a/§8, which recorded that no remote would
exist. The remote `git@github.com:miheer-smk/quantum-structure-entanglement.git` (created
2026-08-04T18:19:57Z) is an accepted part of the project's outward-facing state from this date.
`PLAN.md` is left unedited: it is the plan of record as it stood, and this file is where the
departures from it live.

**Cost accepted with it, stated once so it is not rediscovered as a surprise:** public
visibility forecloses anonymous submission to venues requiring double-blind review, and
`AUTHOR_HANDOFF.md`'s documentation of defects in the predecessor work is world-readable. The
author has weighed both.

**Not legal advice.** The author has been advised separately that if the MIT grant on
unpublished work matters for institutional or funder reasons, that is a question for counsel,
not for this file.

## 2026-08-11 — Third instance of stated-source ≠ actual-source (RESULTS_STAGE0.md §2)

**This is the third occurrence of one error class, which makes it a pattern rather than bad
luck, and it is being retired structurally rather than patched again.** The prior two, both
already recorded above and in `CLAUDE.md`:

1. Fabricated `h` vectors labelled as pinned realizations — invisible to a gate that holds for
   *any* `h`; caught only by an out-of-band provenance comparison.
2. An unanchored regex returning the partial correlation `0.560` in place of the incremental-R²
   `0.0283` — caught only because the tolerance was derived rather than typed.

**Third instance.** `RESULTS_STAGE0.md` §2 stated the hook-equality measurement was taken on
"512 pinned **test** realizations". Regenerating on `h_test[:512]` gives `1.0133e-06`, not the
committed `8.94e-07`. Regenerating on `h_val[:512]` **and** `h_train[:512]` reproduces
`8.9407e-07` exactly. The committed number was therefore correct and reproducible, but was not
measured on the array the results file names.

Same shape as the prior two: **a value whose stated source is not its actual source, where
every check in existence passed because no check compared the two.** The scientific conclusion
is unaffected — both values are float32 machine precision and `k=6` is still the published
tensor — but R1's entire premise rests on that sentence, so the sentence has to be right.

**Correction applied, per the author's ruling of 2026-08-11**, going further than a label fix:

- The equality is re-measured on the arrays **R1 actually consumes** —
  `data/ra03_states_L8_N800_s{42,43,44}.pt` — rather than on any training-ensemble split. R1's
  premise is asserted on R1's own data.
- It is reported as a **bound**, not a 4-significant-figure value. Measured across five
  different arrays the value ranges over `8.94e-07 … 1.01e-06`; the honest and stronger claim
  is that it is below a stated bound at float32 machine precision on **every** array tested,
  because that claim does not depend on which array anyone happens to pick.
- The extraction gate asserts the bound on a stated split, with a failure demonstration.

## 2026-08-11 — The environment behind the Stage 0/1 tables was never recorded; this is a RECONSTRUCTION

**Stated as plainly as it can be: the software environment that produced every number in
`RESULTS_STAGE0.md` and `RESULTS_STAGE1.md` was never recorded, and cannot be recovered.**

At the point this entry was written, no Python environment capable of running the suite
existed on the machine holding the repository — no `pytest`, `numpy`, `scipy`, `torch` or
`scikit-learn` anywhere on the filesystem, no virtualenv, no conda, no lockfile.
`pyproject.toml` declared `["numpy", "scipy", "torch", "scikit-learn", "pyyaml"]` with **no
version constraints of any kind**. The committed tables therefore rested on an environment
that is not described anywhere in the repository's history.

This is the same defect class as the fabricated `h` vectors and the unanchored regex, in a
third set of clothes: **a result whose provenance nobody checked.** A repository that
hash-pins 35 artifacts and a submodule SHA, while leaving the software that reads them
entirely unpinned, pins identity of the inputs and not identity of the computation.

### What was built

| Layer | Pin | File |
|---|---|---|
| Base image | `nvcr.io/nvidia/pytorch@sha256:43c018d6a129…d210e1` — **digest, not tag** | `env/base-image.digest`, `env/Dockerfile` |
| Added packages | exact `==` pins incl. full transitive closure | `env/requirements.lock` |
| Invocation | read-only artifact mount, `OMP_NUM_THREADS=4` | `env/run.sh` |

The tag `nvcr.io/nvidia/pytorch:26.06-py3` is deliberately **not** the pin — tags are mutable.
Base versions: python 3.12.3, numpy 2.1.0, scipy 1.17.1, torch 2.13.0a0+8145d630e8.nv26.06,
PyYAML 6.0.1. None are upgraded; upgrading numpy or scipy would silently change `eigvalsh`
and the RNG streams every committed number depends on.

`pennylane==0.45.1` is installed for one reason, recorded so it is not mistaken for scope
creep: the pinned checkpoints are pickles referencing `qsae.reverse_arrow.transformer`, and
`qsae/__init__.py` eagerly imports `.qnn` → pennylane, so `torch.load` on a pinned checkpoint
cannot succeed without it. **This arm never evaluates a quantum circuit.** Making that import
lazy would mean editing the pinned submodule, which the pin contract forbids.

### This is a reconstruction, not the original, and is validated rather than assumed

**It is not claimed that this environment is the one that produced the committed tables.** It
demonstrably is not the same by construction — the original is unknown. Two observations that
prove non-identity rather than hide it:

1. The suite runs in **~34 s** here against the **219 s** recorded in `RESULTS_STAGE1.md`.
   Different hardware, and the arm64 GB10 machine this now runs on may not even be the machine
   the tables were produced on.
2. `pytest` here is 9.1.1; the original version is unknown and unknowable.

The reconstruction is therefore **validated empirically, by regenerating every committed
number from pinned artifacts and diffing against what is in the results files** — not by
assertion that the versions are right. That regeneration, its committed scripts under
`scripts/`, and the value-by-value diff table are the acceptance test. Author-directed
(2026-08-11): any mismatch beyond stated precision stops all other work immediately.

**First run of the reconstructed environment: 57 passed.** The submodule pin asserts, and all
35 pinned artifact hashes verify. This establishes only that the suite is green here; it says
nothing yet about whether the *numbers* match, which is what the acceptance test measures.

## 2026-08-13 — `scripts/diff_stage0.py` deleted, superseded by the provenance gate

**It was committed and non-functional.** The tool parsed the committed values out of
`RESULTS_STAGE0.md` with free-text regexes — deliberately, so that no number was ever typed
twice — and compared them against the regenerated ones. The 2026-08-11 corrections rewrote
the §1 and §2 prose those regexes matched: `**8.94e-07**` became `< 2e-06`, and the header's
`cross-validation cases: **1.648e-11**` became `**< 2e-11**`. The very first pattern then
matched zero times and the tool aborted with
`RuntimeError: expected exactly one match for worst, got 0`. It had been dead since the commit
that corrected the results file, and nothing noticed, because nothing runs it.

**The lesson, stated so it is not relearned:** *the value-checker was coupled to prose
formatting.* Its correctness depended on the wording and punctuation of a Markdown sentence,
so an honest edit to that sentence — one that made the results file *more* accurate — silently
broke the checker meant to police it. Any harness that recovers structured facts by pattern-
matching unstructured prose has this failure mode, and it fails in the dangerous direction:
a regex that stops matching raises, but a regex that matches the *wrong* number does not, which
is exactly instance two of the stated-source error class (`0.560` read as `0.0283`).

**Superseded by `scripts/check_provenance.py`**, which does the same job through the structured
`<!--prov …-->` tags: the tag carries `md=<the literal as written>`, the gate asserts that
literal appears in the markdown beneath it, and compares it to the independently registered
claim at the literal's own significant figures. Prose may be rewritten freely; only the tagged
literal is load-bearing, and a stale literal is a gate failure rather than a silent pass. The
tag also carries the array, seed and artifact hash, so it checks provenance as well as value —
strictly more than the deleted tool did.

Nothing is lost by the deletion: every quantity `diff_stage0.py` compared is now a registered
claim with a tag, and the gate runs in the suite (`tests/test_provenance_gate.py`) rather than
by hand.

## 2026-08-13 — The precision rule was wrong, and the author corrected it

**A wrong adjudication by the author, recorded here because that is what this file is for.**
The ruling of 2026-08-11 directed that near-machine-precision values be restated as bounds,
with the operative test being *does the value move under BLAS thread count*. The author
withdrew that rule on 2026-08-13 as too crude and issued the corrected one below. The
assistant implemented the original rule as given; the defect is in the rule, not in its
execution, and the correction is the author's.

### What the withdrawn rule got wrong

*Moves at all → BOUND* is a binary test applied to a continuous quantity, so it returned the
same verdict for two situations that are not alike:

| Quantity | Spread across thread counts | What the withdrawn rule said | What is true |
|---|---|---|---|
| `ed_vs_ff_worst` | 2.0e-15 abs, **1.2e-04** rel | BOUND — no quotable digits | identical in its first **4** figures in every configuration; wobbles in the 5th |
| `ceff_mirror_dc` | 1.1e-15 abs, **8.3e-01** rel | BOUND — no quotable digits | does not reproduce even its **first** figure |

Collapsing those into one word discarded honestly measured precision from the figure that
carries the **entire two-solver ground-truth claim**. Under-claiming is not automatically the
safe direction: a bound where a value exists throws away evidence, and it makes the repository
look less reproducible than it measurably is.

### The corrected rule (author, 2026-08-13)

> Report to the number of significant figures that are **stable across configurations**, and
> state the measured spread alongside.

Implemented in `scripts/audit_precision.py` as a `stable_sigfigs` column: each configuration's
value is rounded to *k* significant figures and *k* is increased until the configurations
disagree. Below **two** stable figures the quantity is still reported as a bound, because one
stable digit is an order-of-magnitude statement and an inequality says that more honestly.
Claim kinds are **derived** from that output by `regen_stage0.audited_kind()`, never typed.

### What the corrected rule changes, measured over 33 quantities at 1/2/4/8 threads

- **`ed_vs_ff_worst` is restored to `1.648e-11`** (spread 2.0e-15), 4 stable figures, in the
  `RESULTS_STAGE0.md` header and as a `kind=value` claim. It was `< 2e-11`.
- The uniform site-blind rows split rather than sharing one verdict: `h=0.5` → `1.648e-11`
  (4 s.f.), `h=2.0` → `4.9e-13` (2 s.f.), `h=1.0` stays a bound at 1 stable figure.
- 25 of 33 quantities are values, 8 are bounds. Under the withdrawn rule, 14 of the 14
  ED-vs-free-fermion agreements were bounds; now 6 of them are.
- Nothing about §2 changes. `hook_k6_vs_published_eval_s*` is bit-stable across thread counts
  yet stays a bound, because its binding axis is **which array is chosen** (14/17/15 float32
  ULPs), an axis the thread audit does not sweep. That distinction is now carried in the audit
  output as `other_axis_caveat`, so `verdict: value` there cannot be misread as clearance to
  quote digits the quantity does not have.

### Two things measured on the way that were not known before

1. **The uniform site-blind rows are bit-identical to the ED-vs-free-fermion cross-validation
   rows** at the same `(L = 8, h)`. §1 argued on paper that the site-blind solver performs the
   identical computation on a uniform field; the audit now measures it, through the two
   separate code paths. The degeneracy finding is stronger for it.
2. **A figure's reproducibility and its meaning are independent.** `site_blind_uniform_h0.5_L8`
   reproduces to 4 significant figures and still detects nothing whatsoever about
   site-blindness. The corrected rule governs how many digits may be printed; it says nothing
   about whether the quantity is evidence for anything, and `RESULTS_STAGE0.md` §1 now states
   that adjacent to the table so a reader cannot borrow precision as significance.

### Limitation, stated so it is not overread

BLAS thread count is **one** reconfiguration axis — the one that can be varied on this machine.
`stable_sigfigs` is therefore an **upper** bound on the digits worth quoting, not a claim of
bitwise portability across BLAS implementations, CPU architectures or compilers. Recorded in
`scripts/audit_precision.py` alongside the rule.

**Standing-rule compliance.** `tests/test_precision_bounds.py` asserts the rule in **both**
directions: the headline IS stable at 4 figures, it is NOT stable at 5 (so the results file
cannot quietly grow a digit), a noise-dominated quantity has fewer than 2 stable figures (so
"bound" cannot become a blanket excuse), and a genuinely physical quantity is stable far beyond
any quoted precision (the control).

## 2026-08-13 — Resolved: the "41%" was a cross-environment delta, and it is evidence

**Where the number came from.** A 41% figure was carried forward from the Stage 0 regeneration
as though it were a thread-count spread. It is not, and it does not appear anywhere in
`scripts/out/precision_audit.json`. It is the **committed-vs-regenerated** delta for the
`uniform h = 1.0, L = 8` site-blind row from the original regeneration diff: committed
`3.225e-14`, regenerated `4.546e-14`, i.e. +41.0%. The committed value is recoverable from
history (`git show 209b6d2^:RESULTS_STAGE0.md`), so this resolution is derived, not recalled.

**The part that matters, and the reason this is not just bookkeeping.** `3.225e-14` lies
**outside** the range this quantity occupies across every BLAS thread configuration on the
reconstructed environment:

| Uniform row | committed (pre-2026-08-11) | thread-sweep range here (1/2/4/8) | verdict |
|---|---|---|---|
| `h = 0.5, L = 8` | `1.648e-11` | `1.647860e-11 … 1.648059e-11` | **inside** — agrees to 4 s.f. |
| `h = 1.0, L = 8` | `3.225e-14` | `4.501954e-14 … 4.835021e-14` | **OUTSIDE**, 33% below the range |
| `h = 2.0, L = 8` | `5.045e-13` | `4.882900e-13 … 4.899692e-13` | **OUTSIDE**, 3.0% above the range |

**Thread count cannot explain either mismatch.** Both sit outside the interval that thread
count can produce, in opposite directions, so the difference is a property of the *environment*
and not of the parallelism. The original values came from the **unrecorded, pre-lockfile
environment** — a different BLAS, LAPACK, numpy build or CPU architecture, none of which can be
identified after the fact because none was recorded. That is the whole reason
`env/requirements.lock` and the digest-pinned base image now exist.

**Scientific impact: none.** Both values are three to four orders of magnitude inside the
`< 1e-10` agreement gate, and both are quantities that section 1 shows detect nothing anyway
(on a uniform field the site-blind solver performs the identical computation). The `h = 0.5`
row — the largest of the three and the one that sets the headline — reproduces to all four of
the significant figures it possesses.

**Why it is recorded rather than dismissed.** The 2026-08-11 entry stated that the
reconstruction "demonstrably is not the same by construction" and offered two *indirect*
arguments: wall-clock (34 s vs 219 s) and an unknowable original `pytest` version. This is the
first **direct, numerical** evidence that the two environments differ — a committed number that
the reconstructed environment cannot produce under any thread configuration. It converts
"presumably different" into "measurably different, by this much, on these quantities", and it
is exactly what the acceptance test was built to surface.

## 2026-08-13 — Author name: three spellings, and PLAN.md D5 misquotes LICENSE

**Recorded now, corrected on the accuracy pass, per the author's instruction. `LICENSE` is not
to be touched** (PLAN §5 D5).

| Artifact | Name as written |
|---|---|
| `LICENSE` (authoritative) | `Copyright (c) 2026 **Miheer Satish Kulkarni**` |
| `CITATION.cff` | `family-names: Kulkarni`, `given-names: Miheer` → "**Miheer Kulkarni**" |
| git commit identity (D1) | `Miheer Kulkarni <222050236+miheer-smk@users.noreply.github.com>` |
| `PLAN.md` §5 D5 | quotes LICENSE as `"Copyright (c) 2026 **Miheer Kulkarni**"` |

Two distinct defects, not one:

1. **A metadata mismatch on a public repository.** `CITATION.cff` is the machine-readable
   record a citation manager reads, and it disagrees with the `LICENSE` copyright holder.
2. **`PLAN.md` D5 misquotes the file it cites.** It presents a quotation of `LICENSE` that is
   not what `LICENSE` says — and D5 was resolved on 2026-08-03 against a `LICENSE` that did not
   exist in the tracked tree until `58d1b79` on 2026-08-05 (see the 2026-08-11 entry). The
   quoted text was therefore written from expectation rather than from the file, which is the
   stated-source error class in its mildest form: cosmetic here, and the same shape as the
   three instances that were not.

**Resolution directed by the author:** `LICENSE` stands unmodified and is authoritative;
`CITATION.cff` and the `PLAN.md` D5 quotation are to be brought into exact agreement with it
on the accuracy pass. Not done in this pass, deliberately, to keep the extraction gate first.

## 2026-08-13 — Fourth instance: docstrings naming files that do not exist

**Same class, fourth occurrence: a stated source that is not an actual source.** The prior
three, all recorded above and in `CLAUDE.md`: fabricated `h` vectors labelled as pinned
realizations; an unanchored regex returning `0.560` where `0.0283` was claimed; and
`RESULTS_STAGE0.md` §2 attributing the hook-equality measurement to an array it was not
measured on.

**Fourth instance.** Two module docstrings asserted the existence of code that was not in the
tree, in the present tense, for the whole of Stage 0 and Stage 1:

| Docstring | Claimed | Reality until 2026-08-13 |
|---|---|---|
| `src/qsent/extraction.py` | "`tests/test_extraction.py` asserts `pooled(k=6) == last_layer_pooled(...)` to machine precision" | the file did not exist; nothing asserted it anywhere |
| `src/qsent/free_fermions.py` | "See `convention.py`" (for the `2 × ln2` conversion) | `qsent/convention.py` did not exist |

The shape is the one this repository keeps meeting: **a claim about provenance that no check
compared against the thing it names.** A reader — or an author, months later — takes "asserted
in tests/test_extraction.py" as evidence the assertion exists. It is weaker than the other
three instances in consequence (no number was wrong) and identical in mechanism. It is
recorded at the same weight deliberately: the three that mattered were each preceded by ones
that did not, and the difference was luck, not kind.

**Both are now true rather than deleted.** `qsent/convention.py` was written and its audit
executed as a test (`tests/test_convention.py`, 2026-08-13); `tests/test_extraction.py` was
written and now carries the extraction gate, and the `extraction.py` docstring was rewritten to
describe what that file actually asserts — a bound rather than "machine precision", every hook
point rather than only `k=6`, and four failure demonstrations.

**Not yet retired structurally.** Per the author's ruling, a lint asserting that every
repo-relative path named in a docstring or comment resolves — shipped with a failure
demonstration — is outstanding. Until it exists, this instance is patched, not gated, and the
class remains alive for paths nobody has happened to check.

## 2026-08-13 — Stage 1 regenerated: one precision defect corrected, three populations found unreproducible

**The exercise that found three defects in Stage 0 was run on Stage 1, which had never had it.**
`scripts/regen_stage1.py` regenerates every Stage 1 quantity the repository contains enough
information to recompute. Two findings, of unequal weight.

### (1) `|dSSR| = 0.00e+00` — corrected to a bound

`RESULTS_STAGE1.md` §3 reported the mirrored-vs-forward difference in the Calabrese–Cardy fit
residual as **exactly zero**. It regenerates as `5.551115e-17`, and across BLAS thread counts
here the quantity occupies `[5.551115e-17, 1.665335e-16]` — **it is never zero in any
configuration on this machine**. The committed value lies outside the achievable range: the
same signature as the two Stage 0 site-blind rows, and the **third instance** of a committed
number that this environment cannot produce, attributable to the unrecorded pre-lockfile
environment.

It is also a precision overstatement independent of the environment: `stable_sigfigs = 0`, so
under the corrected rule (author, 2026-08-13) the quantity cannot be quoted as a value at all.
`0.00e+00` asserts an *exact equality* that a least-squares residual computed two ways does not
have. Restated as `|dSSR| < 1e-15`.

**`|dc_eff| = 6.66e-16` restated the same way**, to `< 1e-14`. This one is **not** a mismatch —
the committed value sits inside its achievable range `[2.22e-16, 1.33e-15]` — but it too has
zero stable significant figures and was quoted to three.

**The scientific claim is untouched.** §3 asserts that the `c_eff` fit is blind to mirroring,
and that rests on the ratio between **0.569541 nats** of real asymmetry and a fit difference at
the `1e-16` noise floor — about fifteen orders of magnitude. Quoting the noise floor to three
significant figures added nothing to the argument and asserted precision that does not exist.

### (2) Three populations cannot be regenerated at all — the larger finding

| population | quantities | the exact missing fact |
|---|---|---|
| §9 collapse quality | `Q` = 0.0091 and 0.0376, both bootstrap CIs | bin edges, bin count, minimum occupancy per bin |
| §8 / §10, L = 10 and 12 | 4 bias figures, 2 `c_eff` pairs, 2 gaps, 2 CIs | pool size drawn before filtering to `\|δ_r\| < 0.05` |
| §2 site-ordering | 15 difference figures | which disordered chains were measured |

**Deliberately not reconstructed, per the author's ruling of 2026-08-13.** Each missing fact is
a **free parameter**. Searching parameter space for values that reproduce `0.0091` is fitting
the answer, and — the decisive argument — *a recipe found that way would be indistinguishable
from the original whether or not it was the original*, so it could never support the claim it
appeared to verify. Marking them unreproducible is the stronger artifact: it is true, it is
checkable, and it tells a reader exactly what would be needed.

This is a **worse** defect class than Stage 0's, which was fully recomputable once the
environment was rebuilt. Here the generating procedure itself was never recorded, and no
environment work recovers it. It is recorded at full weight in `RESULTS_STAGE1.md` (in place,
per population), in `README.md` (which quotes several of these numbers publicly, now marked),
and in `AUTHOR_HANDOFF.md`.

**Not a licence to delete.** These are real measurements and deleting them would be its own
dishonesty; they stand, marked. Re-measuring them under a *documented* procedure would produce
new numbers, honestly labelled as new — that is legitimate and is not reconstruction.

### (3) Coverage wording corrected everywhere it appeared

Reporting "N claims across M gated files" reads as coverage of those files, and it is not: the
gate guarantees that every **registered claim** is tagged and correct, and is blind to any
number nobody registered. `README.md` carried an explicit overstatement ("every numeric claim …
carries a machine-checked provenance tag") which was false, and 43 measurement-shaped literals
in that file alone carry no tag. `scripts/check_provenance.py` now prints the untagged literal
count per file beside the claim count, so the distinction cannot be lost again.
