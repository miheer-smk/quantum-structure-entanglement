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
