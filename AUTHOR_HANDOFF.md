# AUTHOR_HANDOFF.md

Final deliverable, accumulated as the arm progresses: what was run, what was found, what the
honest numbers are, what to tell co-authors, and what remains open. Sections fill in at each
stage gate.

**Where this stands (2026-08-13).** Stages 0, 1 and 1.5 are complete and their gates are green.
One measurement on trained models has been made — **R1, the reproduction gate**, which passed.
**No entanglement measurement exists**: Stage 2 has not been run, and H1–H5 are all untested.

---

## Things co-authors need to know now (not at the end)

### Two published numbers have no recoverable artifact

Both figures below were produced by runs whose **model checkpoints were never saved**. They
are cited in the existing write-up, but they cannot be re-measured, re-probed, or pinned by
hash — the weights no longer exist anywhere, on any machine or backup.

<!--prov id=ra08_learned_gain_L8 script=scripts/public_claims.py array=none seed=none sha256=none kind=value md=+0.029 -->
<!--prov id=ra08_learned_gain_L10 script=scripts/public_claims.py array=none seed=none sha256=none kind=value md=+0.028 -->
<!--prov id=ra08_learned_gain_L12 script=scripts/public_claims.py array=none seed=none sha256=none kind=value md=+0.027 -->
<!--prov id=ra09_learned_gain_L8 script=scripts/public_claims.py array=none seed=none sha256=none kind=value md=-0.018 -->
<!--prov id=ra09_learned_gain_L10 script=scripts/public_claims.py array=none seed=none sha256=none kind=value md=-0.007 -->

| Published number | Where cited | Run config | Artifact status |
|---|---|---|---|
| Probe gain **+0.029 / +0.028 / +0.027** (L = 8/10/12) | `results/legacy/ra08_scaling.md` | `n_train=15000, epochs=100, seed=0` | **checkpoints never written** |
| Mixed-field null **-0.018 / -0.007** (L = 8/10) | `results/legacy/ra09_mixedfield.md` | `n_train=15000, epochs=100, seed=0, g=0.5` | **checkpoints never written** |

Both rows are parsed from the pinned submodule at build time, so the *quotations* are
verifiable even though the artifacts behind them are not. One precision note, recorded rather
than smoothed: `PLAN.md` §A3 gives the mixed-field null as **−0.0175 / −0.0070**, read from
`runs/ra09_mixedfield/scaling_results.json`. That file exists on this machine but is **not
hash-pinned** in `pins/`, so the numbers quoted above are the pinned markdown's rounded
`-0.018 / -0.007` instead. The extra digits are not wrong; they are simply not pinned, and
this file quotes only what a reader could check.

Consequences to state plainly to co-authors:

1. Neither number can be reproduced on its original models. Any re-measurement is a
   re-training, and therefore a different measurement.
2. Both were produced at a **different training configuration** from the surviving
   `ms_trained` checkpoints (`n_train=50000, epochs=200, seeds 1–10`). Comparisons across
   them are comparisons across training budgets and must be labeled as such.
3. The number that *is* anchored to surviving, hash-pinned checkpoints is `phase06`'s
   `long_range_zz` incremental R² beyond poly2-h:

<!--prov id=phase06_lrzz_incr_r2_mean script=scripts/public_claims.py array=none seed=none sha256=none kind=value md=0.0283 -->
<!--prov id=phase06_lrzz_incr_r2_sd script=scripts/public_claims.py array=none seed=none sha256=none kind=value md=0.0030 -->
<!--prov id=phase06_lrzz_incr_r2_min script=scripts/public_claims.py array=none seed=none sha256=none kind=value md=0.0231 -->
<!--prov id=phase06_lrzz_incr_r2_max script=scripts/public_claims.py array=none seed=none sha256=none kind=value md=0.0320 -->

   **0.0283 ± 0.0030 [0.0231, 0.0320]**, 10 seeds. This arm's reproduction gate (R1) is fixed
   against that, not against +0.029 — and R1 has now run against it and **passed** (below).
4. The surviving artifacts are now backed up in three verified places (see `pins/README.md`).
   That protects everything from here forward; it does not recover what was already lost.

### The "SAE cross-reference" was premised on data that does not exist

`+0.028` and the mixed-field null are **probe** gains on raw activations, not SAE quantities.
The predecessor repo's only SAE result (`results/legacy/ra04_sae_grid.md`) reports matched-
cosine universality, dead fraction, and reconstruction MSE — **no gain quantity at all**, at
any layer. Per-layer SAE gain had to be reconceived rather than reused; see `PLAN.md` §3.6.

### A mislabeled file in the predecessor repo

`results/legacy/ra09_mixedfield.md` is titled "RA-08 — L-scaling of the ⟨Z₀Z_{L-1}⟩ signal"
and carries RA-08's caption, although its numbers are genuinely RA-09's. A reviewer reading
that file directly will be misled about which experiment it describes. Left uncorrected in the
pinned submodule deliberately — the pin must reflect what was actually published.

---

## Stage results

### Stage 0 — complete; environment reconstructed and validated by regeneration

The software environment that produced the original Stage 0/1 tables was never recorded and
cannot be recovered. It has been **reconstructed** (digest-pinned base image, exact lockfile,
read-only artifact mount) and then **validated empirically rather than assumed**: every
committed Stage 0 number was regenerated from the pinned artifacts and compared against what
the results file states. That acceptance test is the basis on which the environment was
declared usable.

**Outcome: the substantive numbers reproduce; two do not, and both are explained.** The
`uniform h = 1.0` and `uniform h = 2.0` site-blind rows land outside the range those
quantities occupy across *any* BLAS thread configuration here, in opposite directions — so the
difference is a property of the unrecorded pre-lockfile environment, not of parallelism. Both
sit three to four orders of magnitude inside the `< 1e-10` gate, and both are quantities that
§1 shows detect nothing anyway. This is the first direct numerical evidence that the
reconstructed environment differs measurably from the original, and it is exactly what the
lockfile exists to make visible. Full detail: `DEVIATIONS.md`, 2026-08-13.

Stage 0's 46 numeric claims are now regenerated from pinned artifacts on demand and each
carries a machine-checked provenance tag naming the script, input array, seed and artifact
hash that produced it.

### Stage 1 — complete

Toy-case validation, Construction-B pipeline validation, split disjointness, the estimator-bias
harness, the measured `c_eff` finite-size bias, and the H1 collapse-feasibility result. See
`RESULTS_STAGE1.md`. Its numbers are **not yet under the provenance gate** — Stage 1
regeneration is outstanding and is the next scheduled piece of work.

### Stage 1.5 — complete; R1 PASSED

<!--prov id=r1_max_abs_difference script=scripts/run_r1.py array=runs/ms_trained/seed1/best.pt,runs/ms_trained/seed2/best.pt,runs/ms_trained/seed3/best.pt,runs/ms_trained/seed4/best.pt,runs/ms_trained/seed5/best.pt,runs/ms_trained/seed6/best.pt,runs/ms_trained/seed7/best.pt,runs/ms_trained/seed8/best.pt,runs/ms_trained/seed9/best.pt,runs/ms_trained/seed10/best.pt,data/ra03_states_L8_N800_s42.pt,data/ra03_states_L8_N800_s43.pt,data/ra03_states_L8_N800_s44.pt seed=42 sha256=f1dcf0903f26,76e5f8533a9c,cbef74267f22,75a5f498ac63,0f0e3ef98eea,bcfdd3db0295,98d10b25bb0c,60510def1f28,4ff920bf3164,d65635e48f4e,b605c43da217,47a0e6afacae,cc7d8ba56e25 kind=value md=0.00038875036249028105 -->

R1 re-ran the published `phase06` protocol on the pinned `ms_trained` seeds 1–10, substituting
**one line** — this arm's extraction stack in place of `last_layer_pooled` — and changing
nothing else. Ridge alpha, folds, fold seed, eval arrays and seeds were read from the pinned
config at runtime.

**Verdict: PASS.** 10-seed mean **0.028338425896742396** inside the pre-registered
`[0.0223, 0.0343]`; paired per-seed difference within `0.010` on **10 of 10** seeds against a
requirement of 8; largest paired difference **0.00038875036249028105**. That largest difference
is *below* the ±0.0005 rounding envelope of the published series, so the substitution's effect
is smaller than the comparison can resolve — the individual per-seed differences are not
measured discrepancies. The tolerance was committed, with its pre-commitment block, in a commit
that contains no result.

> **What R1 validates.** R1 validates **this arm's extraction stack** against a published
> **probe-gain** number on pinned checkpoints. **It does not validate anything about SAEs.**
> The SAE line of the predecessor work is **not** reproduced here, and no claim about it should
> be inferred from R1 passing. It says **nothing about entanglement**.
> (`PLAN.md` §3.6 A1, verbatim.)

**Tell co-authors plainly:** R1 passing means the instrument is commensurable with the
published pipeline. It is not a physics result, and it is not evidence for any hypothesis in
this arm.

### Stages 2–4 — not started

**No entanglement measurement has been made.** H1–H5 are untested.

## Open questions

- K2: H4 is underpowered by construction at 3 blocks / 7 hook points. At n = 7 a Spearman test
  needs `ρ ≥ 0.7857` for p < 0.05; at the conservative effective n, significance is unreachable
  at any `ρ`. A null H4 is uninformative, not evidence of dissociation.
- K3: no L = 10 or L = 12 checkpoints survive; scaling claims need new training, and a
  claim-bearing cross-L H1 needs new *pinned ensembles* as well.
- Stage 1's numbers are not yet regenerated or provenance-tagged.
- `PREREGISTRATION.md` Part II (Stage 2 endpoint definitions) is pending and must be committed
  before Stage 2 runs.

**Resolved since this file was written:** the repository owner question. The repository exists
at `github.com/miheer-smk/quantum-structure-entanglement`, is **public** by the author's
documented decision, and the local branch is synced to it. See `DEVIATIONS.md`, 2026-08-11.
