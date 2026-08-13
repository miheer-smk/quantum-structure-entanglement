# quantum-structure-entanglement

### The question

> A transformer is trained only to predict the ground-state **energy** of a disordered
> transverse-field Ising chain. Does the state it represents internally carry the right
> **entanglement entropy** — the physics-native invariant — as a function of criticality?

### Status in five seconds

| | |
|---|---|
| **Asked** | does model-side entanglement track exact ground truth across the critical point |
| **Established** | exact-physics ground truth (two independent solvers), a bitwise-verified extraction instrument, a passed reproduction gate against the published pipeline, and a frozen public pre-registration |
| **Deliberately not done** | **the entanglement measurement itself.** Stage 2 has not run; H1–H5 are untested; `PREREGISTRATION.md` Part II is pending by design |

**Stage 1.5 complete.** Everything in this repository is apparatus, exact physics, and
instrument verification. Nothing here is a result about entanglement in a transformer.

---

## What is verifiable here

Claims a reader can check beat claims a reader must trust. Each line names the artifact that
settles it.

<!--prov id=ed_vs_ff_worst script=scripts/regen_stage0.py array=data/tfim_L8_N50k_seed1.pt seed=none sha256=10aacd0f50a4 kind=value md=1.648e-11 -->
<!--prov id=ed_vs_ff_worst_spread_abs script=scripts/regen_stage0.py array=data/tfim_L8_N50k_seed1.pt seed=none sha256=10aacd0f50a4 kind=value md=2.0e-15 -->

| Claim | Check it |
|---|---|
| Ground truth is computed **two independent ways** and they agree to `1.648e-11` (spread `2.0e-15`) against a `< 1e-10` gate | `OMP_NUM_THREADS=4 pytest -q` — `tests/test_exact_entropy.py` |
| The extraction instrument reproduces the model's real forward pass **bitwise at all 7 hook points**, and **rejects 4 deliberately wrong hooks** | `tests/test_extraction.py` (the rejections print their own failure messages) |
| **R1 passed**, and its tolerance was fixed **before the number existed** | `git show 5f58a7b:RESULTS_STAGE1_5.md` — that tree holds the pre-commitment and **zero** occurrences of a verdict; `git show 5f58a7b --stat` shows no results artifact |
| The pre-registration is **byte-identical** to the text frozen in `PLAN.md` on 2026-08-04 | `tests/test_preregistration.py` — asserts the lift character-for-character and that a single softened word is caught |
| Every registered numeric claim names the script, array, seed and artifact hash it came from | `env/run.sh python scripts/check_provenance.py` |

---

## Where this stands

<!--prov id=ed_vs_ff_worst script=scripts/regen_stage0.py array=data/tfim_L8_N50k_seed1.pt seed=none sha256=10aacd0f50a4 kind=value md=1.648e-11 -->
<!--prov id=ed_vs_ff_worst_spread_abs script=scripts/regen_stage0.py array=data/tfim_L8_N50k_seed1.pt seed=none sha256=10aacd0f50a4 kind=value md=2.0e-15 -->

1. **Exact-physics ground truth is validated by two independent solvers.** Exact
   diagonalization and a free-fermion (Peschel/Majorana) solver agree to **1.648e-11**
   (spread **2.0e-15** across BLAS thread counts) over 14 cross-validation cases, against a
   pre-set gate of `< 1e-10`.

2. **The extraction instrument is verified bitwise.** The hand-written Pre-LN reconstruction
   reproduces the model's actual forward output **exactly — a difference of 0.0 — at all seven
   hook points**, checked against tensors captured from a real forward pass, and ships with
   four demonstrated failure modes (a normalised hook, an off-by-one block, an off-by-one
   point within a block, and the wrong end of the stack) that the same assertions reject.

<!--prov id=r1_mean script=scripts/run_r1.py array=runs/ms_trained/seed1/best.pt,runs/ms_trained/seed2/best.pt,runs/ms_trained/seed3/best.pt,runs/ms_trained/seed4/best.pt,runs/ms_trained/seed5/best.pt,runs/ms_trained/seed6/best.pt,runs/ms_trained/seed7/best.pt,runs/ms_trained/seed8/best.pt,runs/ms_trained/seed9/best.pt,runs/ms_trained/seed10/best.pt,data/ra03_states_L8_N800_s42.pt,data/ra03_states_L8_N800_s43.pt,data/ra03_states_L8_N800_s44.pt seed=42 sha256=f1dcf0903f26,76e5f8533a9c,cbef74267f22,75a5f498ac63,0f0e3ef98eea,bcfdd3db0295,98d10b25bb0c,60510def1f28,4ff920bf3164,d65635e48f4e,b605c43da217,47a0e6afacae,cc7d8ba56e25 kind=value md=0.028338425896742396 -->
<!--prov id=r1_n_seeds_within_tolerance script=scripts/run_r1.py array=runs/ms_trained/seed1/best.pt,runs/ms_trained/seed2/best.pt,runs/ms_trained/seed3/best.pt,runs/ms_trained/seed4/best.pt,runs/ms_trained/seed5/best.pt,runs/ms_trained/seed6/best.pt,runs/ms_trained/seed7/best.pt,runs/ms_trained/seed8/best.pt,runs/ms_trained/seed9/best.pt,runs/ms_trained/seed10/best.pt,data/ra03_states_L8_N800_s42.pt,data/ra03_states_L8_N800_s43.pt,data/ra03_states_L8_N800_s44.pt seed=42 sha256=f1dcf0903f26,76e5f8533a9c,cbef74267f22,75a5f498ac63,0f0e3ef98eea,bcfdd3db0295,98d10b25bb0c,60510def1f28,4ff920bf3164,d65635e48f4e,b605c43da217,47a0e6afacae,cc7d8ba56e25 kind=value md=10 -->

3. **The R1 reproduction gate PASSED on pinned checkpoints.** Substituting this arm's
   extraction stack for the published one and changing nothing else reproduces the published
   `long_range_zz` incremental-R² result: 10-seed mean **0.028338425896742396** inside the
   pre-registered `[0.0223, 0.0343]`, with all **10** of 10 seeds inside the per-seed
   tolerance.
   > **What R1 validates.** R1 validates **this arm's extraction stack** against a published
   > **probe-gain** number on pinned checkpoints. **It does not validate anything about SAEs.**
   > It says **nothing about entanglement**. (`PLAN.md` §3.6 A1, verbatim.)

4. **The pre-registration is frozen and publicly timestamped.** `PREREGISTRATION.md` Part I —
   the confirmatory/exploratory split, the measured `c_eff` bias decision, the H1 feasibility
   result, the power analysis, and the null plan — is committed and public. Part II (Stage 2
   endpoint definitions) is marked PENDING and will be fixed before Stage 2 runs.

5. > ### THE ENTANGLEMENT MEASUREMENT HAS NOT BEEN MADE.
   > **Stage 2 has not been run.** No model-side entanglement number exists anywhere in this
   > repository. Everything above is apparatus, exact physics, and an instrument check. The
   > question this arm exists to answer — whether the represented state's entanglement behaves
   > as the physics predicts — is **open and untested**.

198 tests passing.

**What the provenance gate guarantees, stated precisely:** every *registered claim* — 108 of
them — is tagged and matches the script, input array, seed and artifact hash it came from. That
is the meaningful guarantee, and it is the one worth reading.

It does **not** mean every number in every file is checked: a number nobody registered and
nobody tagged is invisible to the gate. The gate reports that gap rather than hiding it — 50
measurement-shaped literals in this README, 430 across all six gated files. **That figure is a
deliberately crude upper bound and should not be read as a coverage ratio:** it is dominated by
definitional constants (`1e-10` tolerances, `L = 8`, `U(0.1, 2.0)`), section and version
numbers, and figures quoted in prose from the pinned predecessor work — none of which are
measurements this repository produced or wants provenance for. The measurements that genuinely
are unchecked are marked individually where they appear, below.

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

> ### Status: H1, H2, H3, H4, H5 — **NOT YET TESTED.**
> **Stage 2 has not been run.** Every hypothesis below is stated as pre-registered, not as
> examined. No result — positive, negative, or partial — exists for any of them. They are
> listed in full detail because pre-registration requires it, and the detail must not be
> mistaken for evidence.

Confirmatory and exploratory hypotheses are separated **in advance**, not in a later
limitations section. The full text lives in `PLAN.md` §3.55 and §3.6, and verbatim in
`PREREGISTRATION.md`.

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

<!--prov id=ed_vs_ff_worst script=scripts/regen_stage0.py array=data/tfim_L8_N50k_seed1.pt seed=none sha256=10aacd0f50a4 kind=value md=1.648e-11 -->
<!--prov id=ed_vs_ff_worst_spread_abs script=scripts/regen_stage0.py array=data/tfim_L8_N50k_seed1.pt seed=none sha256=10aacd0f50a4 kind=value md=2.0e-15 -->

Exact diagonalization vs the free-fermion (Peschel/Majorana) solver agree to **1.648e-11**,
spread **2.0e-15** across BLAS thread counts, over 14 cases — L = 8/10/12 at uniform
`h ∈ {0.5, 1, 2}`, plus five disordered realizations chosen to span `δ_r` from clearly ordered
to clearly paramagnetic.

Quoted to four significant figures because four is what survives every thread configuration;
the fifth does not. Values whose digits move are reported as bounds instead
(`scripts/audit_precision.py`, `DEVIATIONS.md` 2026-08-13).

### The uniform-field tests are degenerate — proven, not assumed

<!--prov id=site_blind_uniform_h0.5_L8 script=scripts/regen_stage0.py array=none seed=none sha256=none kind=value md=1.648e-11 -->
<!--prov id=site_blind_uniform_h1.0_L8 script=scripts/regen_stage0.py array=none seed=none sha256=none kind=bound md=1e-13 -->
<!--prov id=site_blind_disordered_r2_critical script=scripts/regen_stage0.py array=data/tfim_L8_N50k_seed1.pt seed=none sha256=10aacd0f50a4 kind=value md=0.46357 -->

A deliberately **site-blind** solver (reads `h[0]`, applies it everywhere) passes *every*
uniform-field test — **1.648e-11** at `h = 0.5`, below **1e-13** at `h = 1.0` — and fails
every disordered one, by as much as **0.46357**. With a uniform field, "uses
per-site `h_j`" and "uses `h[0]`" are the same computation, so the uniform gate cannot
separate them *in principle*. The disordered cases are the load-bearing ones.

### `c_eff` finite-size bias — measured, and it decides a hypothesis

| | L = 8 | L = 10 ⚠ | L = 12 ⚠ |
|---|---|---|---|
| Clean critical chain (true `c = 0.5`) | +0.0881 | +0.0845 | +0.0809 |
| Disordered, disorder-averaged (target `ln2/2 = 0.347`) | **+0.1953** | **+0.2091** ⚠ | **+0.1763** ⚠ |
| Disordered, typical (median) | +0.2954 | +0.3013 ⚠ | +0.2559 ⚠ |

> ⚠ **The disordered L = 10 and L = 12 entries are NOT currently reproducible from committed
> code.** They are real measurements, retained rather than deleted, but nobody can recompute
> them from this repository: `PLAN.md` §A0b records the reference chains as
> `default_rng(20260804 + L)` and the sub-ensemble as N = 2000, and **does not record the pool
> size drawn before filtering to `|δ_r| < 0.05`** — which selects a different 2000 chains and
> a different fit. Deliberately not reconstructed: the pool size is a free parameter, and
> choosing one that reproduces `+0.2091` would be fitting the answer. See
> `RESULTS_STAGE1.md` §8. The clean row and every L = 8 entry **do** regenerate and are
> provenance-tagged.

The disordered bias **exceeds the entire 0.153 gap** separating clean Ising (`c = 1/2`) from
the infinite-randomness fixed point (`c̃ = ln2/2`). Even-odd oscillation terms remove none of
it. Bootstrapped clean-vs-disordered gaps are **+0.046 / +0.029 ⚠ / +0.058 ⚠**, and **at L = 10
the 95% CI spans zero**. (The L = 8 gap regenerates exactly, CI included; the L = 10 and L = 12
gaps inherit the unreproducibility above.)

**Pre-registered consequence:** at these system sizes `c_eff` **cannot distinguish the two
universality classes**, and no such claim will be made from it. A bias-corrected estimator was
considered and rejected — the correction is larger than the effect and differs between the two
classes, so applying it would amount to fitting the answer. **H2's primary test is unaffected**,
because the paired per-realization comparison uses no `c_eff` fit at all.

### H1's machinery is testable on exact ground truth

Collapse quality `Q` (0 = perfect), 12,000 realizations per L, 400 bootstrap resamples:

| collapse variable | Q ⚠ | 95% CI ⚠ |
|---|---|---|
| **ν = 2** (`δ·L^{1/2}`, pre-registered) | **0.0091** | [0.0095, 0.0152] |
| ν = 1 (wrong exponent — control) | 0.0376 | [0.0413, 0.0710] |

> ⚠ **This entire table is NOT currently reproducible from committed code.** The recipe in
> `PLAN.md` §A0b fixes the *shape* of `Q` but not its *value*: the **bin edges, the bin count,
> and any minimum occupancy per bin** are recorded nowhere, and each changes the number. With
> three free parameters, a search that lands on `0.0091` would demonstrate only that a search
> was run, so no reconstruction was attempted. The numbers are real measurements and are
> retained; they are simply not checkable today. See `RESULTS_STAGE1.md` §9.

Exact ground truth collapses, and the pre-registered exponent beats the control ~4× with
non-overlapping CIs. The control matters: a small `Q` alone could just mean coarse binning.
That *comparison* is more robust than either value **if both sides used the same binning** —
the natural reading, but undocumented, so assumed rather than known (`RESULTS_STAGE1.md` §9).

### The extraction stack is commensurable with the published pipeline

The predecessor's published probe gain was measured on a hook at `encoder.layers[-1]`. Because
`nn.TransformerEncoder` is built with no `norm=` argument, that tensor is the post-residual-add,
**pre-`final_norm`** residual stream. This arm's hook `k=6` reproduces it to within a bound of

<!--prov id=hook_k6_vs_published_eval_s42 script=scripts/regen_stage0.py array=data/ra03_states_L8_N800_s42.pt seed=none sha256=b605c43da217 kind=bound md=2e-06 -->
<!--prov id=hook_k6_vs_published_eval_s43 script=scripts/regen_stage0.py array=data/ra03_states_L8_N800_s43.pt seed=none sha256=47a0e6afacae kind=bound md=2e-06 -->
<!--prov id=hook_k6_vs_published_eval_s44 script=scripts/regen_stage0.py array=data/ra03_states_L8_N800_s44.pt seed=none sha256=cc7d8ba56e25 kind=bound md=2e-06 -->

**< 2e-06** on every pinned eval array R1 consumes. Reported as a bound deliberately: the
per-array values are exact integer multiples of `2⁻²⁴` — 14, 17 and 15 float32 ULPs — so any
quoted figure reports *which array was picked* rather than a property of the extraction.

<!--prov id=hook_postnorm_vs_published_eval_s42 script=scripts/regen_stage0.py array=data/ra03_states_L8_N800_s42.pt seed=none sha256=b605c43da217 kind=bound md=3.0 -->
<!--prov id=hook_postnorm_vs_published_eval_s43 script=scripts/regen_stage0.py array=data/ra03_states_L8_N800_s43.pt seed=none sha256=47a0e6afacae kind=bound md=3.0 -->
<!--prov id=hook_postnorm_vs_published_eval_s44 script=scripts/regen_stage0.py array=data/ra03_states_L8_N800_s44.pt seed=none sha256=cc7d8ba56e25 kind=bound md=3.0 -->

`post_final_norm`, by contrast, differs from the published tensor by **more than 2.4 and
less than 3.0** on every one of those arrays (measured 2.4008, 2.4058 and 2.4351) — six orders
of magnitude above the agreement bound. It is **excluded** from the layer axis: it is a
different normalisation, and mixing it in would place unlike tensors on the axis a rank
correlation runs along.

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
OMP_NUM_THREADS=4 pytest -q          # 198 tests
```

> The suite does dense `eigvalsh` at L = 12 (4096×4096). Leave BLAS threads bounded and do
> not run it concurrently with itself, or it will appear to hang.

---

## Where to start reading

**`CLAUDE.md`** carries the methodological rules this repository is built on — chiefly that a
check is not a gate until it has been demonstrated capable of failing on the error it targets.
**`PLAN.md`** is the staged design and the source of the frozen pre-registration text;
**`PREREGISTRATION.md`** is what was committed in advance of any measurement, byte-verified
against it. **`DEVIATIONS.md`** records every departure from the plan, dated and with reasons,
including the errors this project made and how they were caught — read it if you want to know
what went wrong rather than what went right.

## Repository layout

```
src/qsent/
  exact.py          exact diagonalization; the binding site/bit convention
  free_fermions.py  Peschel/Majorana solver (independent ground truth)
  extraction.py     7-point residual-stream extraction; k=6 == the published hook
  disorder.py       delta_r criticality parameter and stratification
  splits.py         realization- and field-value-disjointness, asserted at runtime
  pins.py           hash-verified artifact loading; published constants read, not typed
tests/              198 tests, incl. falsifiability checks (see below)
pins/               content hashes + the cross-repo contract
scripts/            regeneration, precision audit, provenance gate, R1 runner
env/                digest-pinned image + lockfile for the analysis environment
PREREGISTRATION.md  Part I, frozen and public; Part II pending before Stage 2
PLAN.md             staged plan, pre-registration text, power analyses
DEVIATIONS.md       every departure from the brief, dated, with reasons
RESULTS_STAGE*.md   per-stage results, incl. RESULTS_STAGE1_5.md (R1)
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

- **No entanglement result exists.** Stage 2 has not run. The one model-side number in the repository is R1's, which measures the *instrument* and not the physics.
- **`c_eff` cannot identify a universality class at L ≤ 12** (measured above). Reported descriptively only.
- **Depth is 3 blocks.** H3/H4 are exploratory by construction. At n = 7 hook points a Spearman test needs `ρ ≥ 0.786` for p < 0.05; at n = 3 or 4, significance is unreachable at any `ρ`. A null H4 at this n is **uninformative, not evidence of dissociation**.
- **Only L = 8 has a pinned ensemble**, and no L = 10/12 checkpoints survive upstream. A claim-bearing cross-L result needs new pinned ensembles *and* new training.
- **Three Stage 1 populations cannot be regenerated at all** — the collapse-quality table, the
  L = 10/12 disordered `c_eff` rows, and the §2 site-ordering diffs. Their generating recipes
  were never recorded. Marked in place above and in `RESULTS_STAGE1.md`; not reconstructed,
  because the missing parameters are free and fitting them would be fitting the answer.
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
