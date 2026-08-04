> **DRAFT — NOT SENT.** For the author to review, edit, and decide whether and when to send.
> Recipient line left generic deliberately; see the note at the bottom of this file.

---

Subject: Two artifact-provenance issues in quantum-structure-sae, found while building the tensor-network arm

Hi both,

While setting up the entanglement arm I audited the artifacts behind a few of the published
numbers and found two things worth recording. Neither affects the validity of any published
claim; both affect what we can say about reproducibility if asked.

**1. Two published numbers have no recoverable checkpoints.**

RA-08 (the +0.029 / +0.028 / +0.027 long-range-ZZ probe gain at L = 8/10/12) and RA-09 (the
mixed-field null, −0.018 at L = 8 and −0.007 at L = 10) were both produced at
`n_train=15000, epochs=100, seed=0`. Neither run saved its model weights — only the summary
JSON survives. The trained models behind both numbers no longer exist.

Two consequences:

- Neither number can be re-measured on its original models. Any new measurement is a
  retraining, so it is a different measurement.
- Both configurations differ from the `ms_trained` checkpoints we still have
  (`n_train=50000, epochs=200, seeds 1–10`). Comparisons between them are comparisons across
  training budgets.

The number that *is* backed by surviving weights is the Phase 0.6 `long_range_zz`
incremental R² beyond the poly-2 control: 0.0283 ± 0.0030 [0.0231, 0.0320] over 10 seeds. The
entanglement arm anchors its reproduction check to that one rather than to +0.029, for this
reason.

**2. `results/legacy/ra09_mixedfield.md` is mislabeled.**

That file carries RA-08's title ("RA-08 — L-scaling of the ⟨Z₀Z_{L-1}⟩ signal") and RA-08's
closing caption ("the scaling prediction is that this grows with L"), over RA-09's numbers.
The numbers themselves are correct and match `runs/ra09_mixedfield/scaling_results.json`; the
surrounding text describes the wrong experiment, and the caption does not make sense for a
null result. Anyone reading that file on its own would misidentify which experiment it
reports.

**3. A latent bug in `qsae/observables.py` — no impact on published results, verified.**

`_reduce_density_matrix` computes `state.reshape(dim_A, dim_B)`, which makes subsystem A the
**high** bits of the state index, i.e. sites `[n - n_A, n)`. Its docstring says
"qubits 0..n_A-1 (left block)" — the opposite block. The two are reflections of each other.

I checked every call site rather than assuming. No call anywhere in the repo passes `cut=`;
all use the default `cut = n//2`, and every L in the repo is even (6, 8, 10, 12 in
configs/experiments; 4, 6, 8 in tests). For a pure state `S(A) = S(complement)`, and the
complement of `[0, n_A)` equals `[n - n_A, n)` only when `n_A = n/2` — exactly the even-L
half cut. Confirmed numerically on disordered chains: agreement to 1e-15 at L = 4, 6, 8, 10,
12. So **no published number is affected**.

It would bite anything else. At odd L the two differ by up to 8.7e-02; at asymmetric cuts on
even L, by up to 2.6e-01, with the profile coming out exactly mirrored (cut 2 ↔ cut 6,
cut 3 ↔ cut 5, agreeing only at the midpoint). A reflected entropy profile looks completely
plausible, which is what makes it worth fixing before anyone computes one.

My suggestion is an upstream fix as its own commit with a dated correction note in
`docs/CODE_MAP.md` recording that published results use the even-L half cut and are
unaffected — rather than a silent edit. A silent change would make the pinned commit
disagree with a later checkout in a way that looks like data drift, and would remove the
evidence that earlier numbers were fine. The entanglement arm pins the current commit and
works around it locally, so there is no time pressure.

**What regeneration would involve.**

The configs are recorded and the data pipeline uses `np.random.default_rng(seed)`
(`data.py:266`), whose stream NumPy treats as stable, so retraining at the original config is
possible. But the resulting checkpoints would be new artifacts, not the originals. If we ever
publish a regenerated value it should be reported as a re-run, not as a recovery of the
original — the distinction matters if a reviewer asks whether the number was reproduced or
re-derived.

**Status.** Flagging early rather than at write-up. The surviving artifacts (ensembles and
the 10 `ms_trained` checkpoints) are now hash-pinned and backed up in three verified
locations, so this does not extend forward. Happy to fix the RA-09 file labeling, or to leave
it and note the discrepancy in the paper — whichever you prefer.

---

> **Note for the author before sending.** The recipient line reads "Hi both" because the
> brief refers to "my guide and Miheer" while the confirmed git identity for this work is
> Miheer Kulkarni. Set the salutation and recipients before sending — I did not want to guess
> who is being addressed.
