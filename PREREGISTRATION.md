# PREREGISTRATION.md — Part I

**Status: PART I ONLY.** Part II (Stage 2 endpoint definitions) is **PENDING** — see the final
section. This is staged pre-registration, and the staging is declared here rather than
inferred from the absence of a section.

Committed after the author read the Stage 1.5 reproduction result, as PLAN.md §3.6 requires:
`PREREGISTRATION.md` "is **not** created or committed until the author has read the Stage 1.5
reproduction result."

**How to read this document.** The two sections below marked *lifted verbatim* are extracted
from `PLAN.md` mechanically by `scripts/build_preregistration.py` and asserted byte-identical to their source by
`tests/test_preregistration.py`. They were frozen on 2026-08-04, before any model number
existed, and nothing in them has been rewritten in the light of a result. Everything after
them is new pre-registered text, dated where it appears.

Provenance: submodule pin `0c4e6e4`; artifact hashes in `pins/`; every published
number quoted below is parsed from the pinned publication at build time and carries a
machine-readable provenance tag.

---

# Lifted verbatim from PLAN.md §3.55 (frozen 2026-08-04)

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

# Lifted verbatim from PLAN.md §3.6 — A0, A0b, A0c, A1, A2, A2a, A3 (frozen 2026-08-04)

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

# New pre-registered text (2026-08-13)

## The phase06 prior for Stage 2's primary endpoint family

**New pre-registered text**, added 2026-08-13. Every number below is parsed from the pinned
`results/phase06_multiseed_trained.md` at build time by `scripts/build_preregistration.py`
via `qsent.pins.published_constant`; none is typed into this document. Submodule pin
`0c4e6e4`.

### The prior, stated in full — both halves

**Half one: on the designated primary endpoint family, entropy is the strongest observable in
the published data.** On **incremental R² beyond poly-2**, scalar half-chain entropy separates
from the random-init distribution by

<!--prov id=phase06_entropy_incr_r2_sep_sd script=scripts/build_preregistration.py array=none seed=none sha256=none kind=value md=+4.57 -->
<!--prov id=phase06_entropy_incr_r2_trained_mean script=scripts/build_preregistration.py array=none seed=none sha256=none kind=value md=0.0305 -->
<!--prov id=phase06_entropy_incr_r2_trained_sd script=scripts/build_preregistration.py array=none seed=none sha256=none kind=value md=0.0023 -->
<!--prov id=phase06_entropy_incr_r2_random_mean script=scripts/build_preregistration.py array=none seed=none sha256=none kind=value md=0.0185 -->
<!--prov id=phase06_entropy_incr_r2_random_sd script=scripts/build_preregistration.py array=none seed=none sha256=none kind=value md=0.0026 -->
<!--prov id=phase06_lrzz_incr_r2_sep_sd script=scripts/build_preregistration.py array=none seed=none sha256=none kind=value md=+4.20 -->

> **+4.57** σ — trained **0.0305** ± **0.0023** against random-init **0.0185** ± **0.0026**.

That is the **largest separation of any observable** in that table, and larger than
`long_range_zz`'s **+4.20** σ — the observable this arm's own R1 gate was built around.

**Half two: on partial correlation, the same observable fails at 10 seeds.**

<!--prov id=phase06_entropy_partial_r_sep_sd script=scripts/build_preregistration.py array=none seed=none sha256=none kind=value md=+1.79 -->
<!--prov id=phase06_entropy_partial_r_trained_mean script=scripts/build_preregistration.py array=none seed=none sha256=none kind=value md=0.565 -->
<!--prov id=phase06_entropy_partial_r_trained_sd script=scripts/build_preregistration.py array=none seed=none sha256=none kind=value md=0.040 -->

> **+1.79** σ — trained **0.565** ± **0.040**. The weakest separation of any non-degenerate
> observable in that table.

**Both halves are the prior.** Entropy is simultaneously the best-separating observable under
one statistic and the worst under the other, on the same models, the same seeds and the same
eval arrays. A prior quoted as "+4.57σ, the strongest observable" without its second half
would be a selective read of a table this repository has pinned in full.

### CAVEAT, recorded at equal weight

> **The published quantity is scalar half-chain entropy used as a PROBE TARGET. It is not a
> Construction-B decoded entanglement profile.** phase06 asks whether a ridge probe on the
> pooled residual stream can predict one number — `S(L/2)` — beyond what poly-2 features of
> `h` already predict. Stage 2 asks something structurally different: whether a linear readout
> to `ψ_h` produces a decoded state whose **entanglement profile across every cut** tracks
> `S_exact(ℓ; r)` per realization. Different target (a profile, not a scalar), different
> estimator (a decoded state, not a regression prediction), different failure modes.
>
> **The prior therefore constrains without determining.** It says the residual stream carries
> entropy-related information beyond the input controls, which makes Stage 2 worth running and
> would make a total null surprising. It does **not** predict that ΔS_incremental will clear
> its own control, because that quantity has never been measured on anything.

### The NULL PLAN, unchanged by the prior

**Pre-registered, and deliberately identical to what it would have been had the prior been
unfavourable:**

> If `ΔS_incremental` does not clear the input control, **it is written up as a null, at full
> precision, in the abstract and the results section** — not softened into a "trend", not
> relegated to an appendix, not re-cut against a different control until it clears one. The
> apparatus contribution stands on its own: the pinned artifact chain, the reconstructed and
> validated environment, the provenance gate, the extraction gate, the measured `c_eff`
> finite-size bias (§A0), and R1's demonstration of commensurability are results whether or
> not the entanglement signal is there.

**A favourable prior is not a reason to weaken pre-committed null handling.** It is a reason
to expect a signal, which is precisely when a pre-committed null plan is load-bearing: the
temptation to rescue a null is largest exactly when a prior said the signal should be there.
Recorded here so that temptation meets text written before the number existed.

### R1's outcome, as context

R1 **PASSED** on 2026-08-13 (`RESULTS_STAGE1_5.md`): 10-seed mean `0.028338425896742396`
inside the pre-registered `[0.0223, 0.0343]`, with `|Δ_s| ≤ 0.010` on 10 of 10 seeds and a
largest paired difference of `3.89e-04` — itself below the ±0.0005 rounding envelope of the
published series, so the substitution's effect is smaller than that comparison can resolve.

Consequences for this pre-registration, and no others:

- **PLAN.md §3.5.2's FAIL branch does not apply.** H4 is **not** demoted to the single
  published final-layer value, and the layer-coincidence claim is **not** dropped.
- **H4 retains per-layer probe gain as its primary axis** (§A2), and remains **subject to the
  §A2a power limitation in full**. R1 says nothing about power: it establishes that this arm's
  extraction stack is commensurable with the published one, not that a 7-point depth axis can
  resolve a rank correlation. `ρ ≥ 0.7857` is still required for p < 0.05 at n = 7, and at the
  conservative effective n significance remains unreachable at any ρ.
- **R1 licenses nothing else.** Per §A1, lifted verbatim above, it validates this arm's
  extraction stack and says nothing about SAEs or about entanglement.

---

## PART II — Stage 2 endpoint definitions — **PENDING**

**Deliberately not written today. This is staged pre-registration, declared as such rather
than discovered later.**

Part II fixes:

- the **H2 test statistic** — the exact paired per-realization comparison of `S_model(ℓ; r)`
  against `S_exact(ℓ; r)`, its aggregation across cuts and realizations, and its acceptance
  criterion;
- **`ΔS_incremental`** — the definition of the entanglement endpoint and of the input control
  it must clear, in the form the null plan above refers to;
- the **multiplicity plan** — the family of tests, the correction, and which results are
  confirmatory within it;
- the **stopping rules** — what ends Stage 2, and what constitutes a completed measurement
  rather than one to extend.

**Why they are not being written tonight, stated plainly:** these are design decisions whose
cost of being wrong is **permanent**. A test statistic chosen carelessly cannot be repaired
after the data are seen — changing it afterwards converts a confirmatory test into an
exploratory one, and pre-registering the wrong statistic is worse than pre-registering none,
because it lends the appearance of rigour to a choice that was never examined. They will be
fixed **before Stage 2 runs**, under their own review, not under deadline.

**Binding constraint on Part II:** it is committed **before** any Stage 2 measurement is taken,
and Part I is not reopened when it is written. If writing Part II reveals that something in
Part I needs to change, that change is recorded in `DEVIATIONS.md` with its reason and date,
in the open, and both versions stand in the record.
