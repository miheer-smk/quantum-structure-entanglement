# AUTHOR_HANDOFF.md

Final deliverable, accumulated as the arm progresses: what was run, what was found, what the
honest numbers are, what to tell co-authors, and what remains open. Sections fill in at each
stage gate. **Nothing here is a result yet — no experiment has been run.**

---

## Things co-authors need to know now (not at the end)

### Two published numbers have no recoverable artifact

Both figures below were produced by runs whose **model checkpoints were never saved**. They
are cited in the existing write-up, but they cannot be re-measured, re-probed, or pinned by
hash — the weights no longer exist anywhere, on any machine or backup.

| Published number | Where cited | Run config | Artifact status |
|---|---|---|---|
| Probe gain **+0.029 / +0.028 / +0.027** (L = 8/10/12) | `results/legacy/ra08_scaling.md` | `n_train=15000, epochs=100, seed=0` | **checkpoints never written** |
| Mixed-field null **−0.0175 / −0.0070** (L = 8/10) | `results/legacy/ra09_mixedfield.md` | `n_train=15000, epochs=100, seed=0, g=0.5` | **checkpoints never written** |

Consequences to state plainly to co-authors:

1. Neither number can be reproduced on its original models. Any re-measurement is a
   re-training, and therefore a different measurement.
2. Both were produced at a **different training configuration** from the surviving
   `ms_trained` checkpoints (`n_train=50000, epochs=200, seeds 1–10`). Comparisons across
   them are comparisons across training budgets and must be labeled as such.
3. The number that *is* anchored to surviving, hash-pinned checkpoints is `phase06`'s
   `long_range_zz` incremental R² beyond poly2-h: **0.0283 ± 0.0030 [0.0231, 0.0320]**,
   10 seeds. This arm's reproduction gate (R1) is fixed against that, not against +0.029.
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

_Stage 0 — pending._
_Stage 1 — pending._
_Stage 1.5 — pending._
_Stages 2–4 — not started._

## Open questions

- Which GitHub account owns the new repository (no remote created yet).
- K2: H4 is underpowered by construction at 3 blocks / 8 hook points.
- K3: no L = 10 or L = 12 checkpoints survive; scaling claims need new training.
