# RESULTS_STAGE1_5.md — R1, the commensurability reproduction gate

**Status at the time this section was committed: R1 has NOT been run. No R1 number exists.**
Everything below the pre-commitment block was fixed before the number was seen, and the
commit that adds it contains no R1 result.

---

## What R1 validates — PLAN.md §3.6 A1, lifted verbatim

> **What R1 validates.** R1 validates **this arm's extraction stack** against a published
> **probe-gain** number on pinned checkpoints. **It does not validate anything about SAEs.**
> The SAE line of the predecessor work is **not** reproduced here, and no claim about it
> should be inferred from R1 passing.

Stated once more in this file's own words, because a reader must find it here rather than
infer it: R1 says nothing about entanglement, nothing about SAE features, and nothing about
whether the transformer represents anything in particular. It asks one question — *does this
arm's extraction stack, substituted for the published one, reproduce a published number on the
same pinned checkpoints* — and answers only that.

---

## PRE-COMMITMENT — what may and may not change if R1 FAILS

Fixed before the number exists. A FAIL is a legitimate, reportable outcome (PLAN.md §3.5.2)
and **is not retried**.

### PERMITTED — correctness of the substitution only

1. A demonstrable bug in **which tensor is extracted, or how it is pooled**.
2. A demonstrable bug in the **split, pairing, or seed mapping**.
3. A demonstrable bug in **parsing the published per-seed values**.

**Each requires a FAILING TEST that demonstrates the bug independently of R1's verdict.** The
test must fail on the bug and pass once it is fixed, and it must be written so that it would
have failed had R1 never been run at all. *"The number moved into range"* is **not** evidence
of a bug — it is the thing under test, and treating it as diagnostic is how a reproduction
gate becomes a fitting procedure.

### FORBIDDEN — protocol

- `ridge alpha`, `n_folds`, `fold_seed`, scaler placement
- which eval arrays, which seeds, which aggregation
- the tolerance, **in either direction**
- dropping or substituting a seed, **for any reason**

### If anything at all is changed after the number is first seen

This file reports **both numbers**, the change made, and the failing test that justified it.
No silent re-runs. The first number stands in the record next to the second, permanently.

---

## The protocol, fixed in advance

**One substitution, everything else byte-identical.** The published driver
(`experiments/phase06_multiseed_trained.py` in the pinned submodule, function `control_point`)
computes the representation as:

```python
R = last_layer_pooled(model, h)
```

R1 replaces exactly that line with this arm's extraction stack at the published hook:

```python
R = mean_pool(extract_residual_stream(model, h)["block2_mlp"])      # k=6
```

Everything downstream is the pinned submodule's own code, **called rather than
reimplemented** — `build_input_controls`, `incremental_r2`, `oof_ridge_predict`. Ridge alpha,
fold count, fold seed, eval arrays, trained seeds, observable and control are read from the
pinned `configs/phase06_multiseed_trained.yaml` at runtime and are never typed into this
repository. `scripts/run_r1.py` is the whole of R1.

`k=6` is the published tensor: `tests/test_extraction.py` asserts `mean_pool(k=6)` agrees with
`last_layer_pooled` to `< 2e-06` on these very eval arrays, and that the hand-written Pre-LN
reconstruction reproduces the model's actual forward output bitwise at all seven hook points.
R1 asks whether that identity survives an entire analysis pipeline — whether a difference of
14–17 float32 ULPs in the representation moves a published incremental R² by more than seed
noise.

### Pre-registered tolerance (PLAN.md §3.5.1), derived and not typed

Published: mean `0.0283`, sd `0.0030`, both parsed from the pinned
`results/phase06_multiseed_trained.md` through anchored extraction with a single-match
assertion. Per-seed values are parsed from the `**incremental R²** per-seed:` line of the
same file, under the same discipline — the section also carries a *partial-correlation*
per-seed line, and reading it instead would be the `0.560`-for-`0.0283` error committed
directly into R1's pairing. `tests/test_r1_gate.py` demonstrates that dropping the line anchor
silently returns the partial-correlation series rather than raising.

> **PASS** iff *both*:
> **(i)** the new 10-seed mean lies within `0.0283 ± 0.0060` (± 2 published sd), i.e.
> **[0.0223, 0.0343]**; and
> **(ii)** the paired per-seed difference satisfies **|Δ_s| ≤ 0.010 for at least 8 of 10
> seeds**, pairing on seed identity.
> **FAIL** otherwise. No third outcome, no post-hoc widening.

The verdict is **computed by the committed rule**, not read off by eye:
`tests/test_r1_gate.py::test_recorded_verdict_matches_the_pre_registered_rule` recomputes both
limbs from the recorded numbers and fails if the reported verdict disagrees with them. That
test deliberately does **not** assert that the verdict is PASS — a suite that goes red on a
legitimate FAIL would create pressure to retry, which is precisely what §3.5.2 forbids.

---

## R1 result

_Not yet run. This section is filled in by the run that follows this commit._
