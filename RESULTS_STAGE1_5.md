# RESULTS_STAGE1_5.md — R1, the commensurability reproduction gate

**The pre-commitment below was committed before R1 was run, and the record shows it.** Commit
`5f58a7b` ("Pre-commit R1's tolerance, protocol and scope before the number exists") contains
this file's scope statement, pre-commitment block, protocol, tolerance and gate tests, and
**no** R1 result — `scripts/out/r1_reproduction.json` did not exist in that tree. The result
section was added by the commit that followed the run. Anyone can check that ordering with
`git log --follow RESULTS_STAGE1_5.md` and `git show 5f58a7b:RESULTS_STAGE1_5.md`.

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

## R1 result — **PASS**

**Run once, on 2026-08-13, against the pre-commitment above. Nothing was changed after the
number was seen: there is one number, from one run, and no second column to report.**

### Per-seed, at full precision

Published values carry three decimals because that is the precision
`results/phase06_multiseed_trained.md` states them to — they are quoted as parsed, not
padded. Pairing is on seed identity.

<!--prov id=r1_incr_r2_seed1 script=scripts/run_r1.py array=runs/ms_trained/seed1/best.pt,data/ra03_states_L8_N800_s42.pt,data/ra03_states_L8_N800_s43.pt,data/ra03_states_L8_N800_s44.pt seed=42 sha256=f1dcf0903f26,b605c43da217,47a0e6afacae,cc7d8ba56e25 kind=value md=0.02314741102926006 -->
<!--prov id=r1_incr_r2_seed2 script=scripts/run_r1.py array=runs/ms_trained/seed2/best.pt,data/ra03_states_L8_N800_s42.pt,data/ra03_states_L8_N800_s43.pt,data/ra03_states_L8_N800_s44.pt seed=42 sha256=76e5f8533a9c,b605c43da217,47a0e6afacae,cc7d8ba56e25 kind=value md=0.02889582355339008 -->
<!--prov id=r1_incr_r2_seed3 script=scripts/run_r1.py array=runs/ms_trained/seed3/best.pt,data/ra03_states_L8_N800_s42.pt,data/ra03_states_L8_N800_s43.pt,data/ra03_states_L8_N800_s44.pt seed=42 sha256=cbef74267f22,b605c43da217,47a0e6afacae,cc7d8ba56e25 kind=value md=0.02518820292223086 -->
<!--prov id=r1_incr_r2_seed4 script=scripts/run_r1.py array=runs/ms_trained/seed4/best.pt,data/ra03_states_L8_N800_s42.pt,data/ra03_states_L8_N800_s43.pt,data/ra03_states_L8_N800_s44.pt seed=42 sha256=75a5f498ac63,b605c43da217,47a0e6afacae,cc7d8ba56e25 kind=value md=0.03204326820852443 -->
<!--prov id=r1_incr_r2_seed5 script=scripts/run_r1.py array=runs/ms_trained/seed5/best.pt,data/ra03_states_L8_N800_s42.pt,data/ra03_states_L8_N800_s43.pt,data/ra03_states_L8_N800_s44.pt seed=42 sha256=0f0e3ef98eea,b605c43da217,47a0e6afacae,cc7d8ba56e25 kind=value md=0.02664731330765250 -->
<!--prov id=r1_incr_r2_seed6 script=scripts/run_r1.py array=runs/ms_trained/seed6/best.pt,data/ra03_states_L8_N800_s42.pt,data/ra03_states_L8_N800_s43.pt,data/ra03_states_L8_N800_s44.pt seed=42 sha256=bcfdd3db0295,b605c43da217,47a0e6afacae,cc7d8ba56e25 kind=value md=0.03126193098138722 -->
<!--prov id=r1_incr_r2_seed7 script=scripts/run_r1.py array=runs/ms_trained/seed7/best.pt,data/ra03_states_L8_N800_s42.pt,data/ra03_states_L8_N800_s43.pt,data/ra03_states_L8_N800_s44.pt seed=42 sha256=98d10b25bb0c,b605c43da217,47a0e6afacae,cc7d8ba56e25 kind=value md=0.02596236055994890 -->
<!--prov id=r1_incr_r2_seed8 script=scripts/run_r1.py array=runs/ms_trained/seed8/best.pt,data/ra03_states_L8_N800_s42.pt,data/ra03_states_L8_N800_s43.pt,data/ra03_states_L8_N800_s44.pt seed=42 sha256=60510def1f28,b605c43da217,47a0e6afacae,cc7d8ba56e25 kind=value md=0.03161124963750972 -->
<!--prov id=r1_incr_r2_seed9 script=scripts/run_r1.py array=runs/ms_trained/seed9/best.pt,data/ra03_states_L8_N800_s42.pt,data/ra03_states_L8_N800_s43.pt,data/ra03_states_L8_N800_s44.pt seed=42 sha256=4ff920bf3164,b605c43da217,47a0e6afacae,cc7d8ba56e25 kind=value md=0.02879765270191192 -->
<!--prov id=r1_incr_r2_seed10 script=scripts/run_r1.py array=runs/ms_trained/seed10/best.pt,data/ra03_states_L8_N800_s42.pt,data/ra03_states_L8_N800_s43.pt,data/ra03_states_L8_N800_s44.pt seed=42 sha256=d65635e48f4e,b605c43da217,47a0e6afacae,cc7d8ba56e25 kind=value md=0.02982904606560825 -->

| trained seed | R1 (this arm's stack, k=6) | published | Δ_s = R1 − published | \|Δ_s\| ≤ 0.010 |
|---|---|---|---|---|
| s1 | 0.02314741102926006 | 0.023 | +0.000147 | ✅ |
| s2 | 0.02889582355339008 | 0.029 | −0.000104 | ✅ |
| s3 | 0.02518820292223086 | 0.025 | +0.000188 | ✅ |
| s4 | 0.03204326820852443 | 0.032 | +0.000043 | ✅ |
| s5 | 0.02664731330765250 | 0.027 | −0.000353 | ✅ |
| s6 | 0.03126193098138722 | 0.031 | +0.000262 | ✅ |
| s7 | 0.02596236055994890 | 0.026 | −0.000038 | ✅ |
| s8 | 0.03161124963750972 | 0.032 | −0.000389 | ✅ |
| s9 | 0.02879765270191192 | 0.029 | −0.000202 | ✅ |
| s10 | 0.02982904606560825 | 0.030 | −0.000171 | ✅ |

### Verdict, computed by the committed rule

<!--prov id=r1_mean script=scripts/run_r1.py array=runs/ms_trained/seed1/best.pt,runs/ms_trained/seed2/best.pt,runs/ms_trained/seed3/best.pt,runs/ms_trained/seed4/best.pt,runs/ms_trained/seed5/best.pt,runs/ms_trained/seed6/best.pt,runs/ms_trained/seed7/best.pt,runs/ms_trained/seed8/best.pt,runs/ms_trained/seed9/best.pt,runs/ms_trained/seed10/best.pt,data/ra03_states_L8_N800_s42.pt,data/ra03_states_L8_N800_s43.pt,data/ra03_states_L8_N800_s44.pt seed=42 sha256=f1dcf0903f26,76e5f8533a9c,cbef74267f22,75a5f498ac63,0f0e3ef98eea,bcfdd3db0295,98d10b25bb0c,60510def1f28,4ff920bf3164,d65635e48f4e,b605c43da217,47a0e6afacae,cc7d8ba56e25 kind=value md=0.028338425896742396 -->
<!--prov id=r1_max_abs_difference script=scripts/run_r1.py array=runs/ms_trained/seed1/best.pt,runs/ms_trained/seed2/best.pt,runs/ms_trained/seed3/best.pt,runs/ms_trained/seed4/best.pt,runs/ms_trained/seed5/best.pt,runs/ms_trained/seed6/best.pt,runs/ms_trained/seed7/best.pt,runs/ms_trained/seed8/best.pt,runs/ms_trained/seed9/best.pt,runs/ms_trained/seed10/best.pt,data/ra03_states_L8_N800_s42.pt,data/ra03_states_L8_N800_s43.pt,data/ra03_states_L8_N800_s44.pt seed=42 sha256=f1dcf0903f26,76e5f8533a9c,cbef74267f22,75a5f498ac63,0f0e3ef98eea,bcfdd3db0295,98d10b25bb0c,60510def1f28,4ff920bf3164,d65635e48f4e,b605c43da217,47a0e6afacae,cc7d8ba56e25 kind=value md=0.00038875036249028105 -->
<!--prov id=r1_n_seeds_within_tolerance script=scripts/run_r1.py array=runs/ms_trained/seed1/best.pt,runs/ms_trained/seed2/best.pt,runs/ms_trained/seed3/best.pt,runs/ms_trained/seed4/best.pt,runs/ms_trained/seed5/best.pt,runs/ms_trained/seed6/best.pt,runs/ms_trained/seed7/best.pt,runs/ms_trained/seed8/best.pt,runs/ms_trained/seed9/best.pt,runs/ms_trained/seed10/best.pt,data/ra03_states_L8_N800_s42.pt,data/ra03_states_L8_N800_s43.pt,data/ra03_states_L8_N800_s44.pt seed=42 sha256=f1dcf0903f26,76e5f8533a9c,cbef74267f22,75a5f498ac63,0f0e3ef98eea,bcfdd3db0295,98d10b25bb0c,60510def1f28,4ff920bf3164,d65635e48f4e,b605c43da217,47a0e6afacae,cc7d8ba56e25 kind=value md=10 -->

| limb | requirement | measured | met |
|---|---|---|---|
| **(i)** | 10-seed mean within `[0.0223, 0.0343]` | **0.028338425896742396** | **yes** |
| **(ii)** | `\|Δ_s\| ≤ 0.010` for ≥ 8 of 10 seeds | **10 of 10**, max `\|Δ_s\|` = **0.00038875036249028105** | **yes** |

> ## R1 VERDICT: **PASS**

Both limbs hold with room to spare. The largest paired difference, `3.89e-04`, is **26 times
smaller** than the per-seed tolerance and about **an eighth of one published sd** (`0.0030`);
the mean lands `0.000038` from the published `0.0283`, roughly `1/80` of a published sd.
Every one of the ten seeds moved in the same direction as its own noise rather than
systematically: five differences positive, five negative.

### What this does and does not license

The substitution changes the representation by 14–17 float32 ULPs
(`RESULTS_STAGE0.md` §2). That perturbation propagates through a standardiser, a 5-fold ridge
probe at `alpha = 1.0`, a poly2-h control, and an average over three eval arrays, and emerges
as a shift of `< 4e-04` in incremental R² on every seed. **This arm's extraction stack is
commensurable with the published one**, so a number produced here may be reported next to a
number produced there.

That is the entire content of the result. Re-read the scope statement above: nothing here
concerns SAEs, entanglement, or what the transformer represents. In particular, PASS does
**not** mean the predecessor's SAE line was reproduced — it was not attempted, and per
PLAN.md §3.5.0 it could not be, because `+0.028` never came from an SAE.

### Provenance

- Substitution: `last_layer_pooled` → `mean_pool(extract_residual_stream(...)["block2_mlp"])`,
  one line, in `scripts/run_r1.py`.
- Protocol read at runtime from the pinned `configs/phase06_multiseed_trained.yaml`:
  `ridge_alpha = 1.0`, `n_folds = 5`, `fold_seed = 42`, eval arrays `s42/s43/s44`,
  train seeds 1–10, observable `long_range_zz`, control `poly2_h`. Nothing typed here.
- All 10 checkpoints and all 3 eval arrays hash-verified against `pins/` on load.
- Full output, including per-eval-array values for every seed: `scripts/out/r1_reproduction.json`.
- Verdict recomputed from the recorded numbers by
  `tests/test_r1_gate.py::test_recorded_verdict_matches_the_pre_registered_rule`.

### Consequences

- PLAN.md §3.5.2's FAIL branch does **not** apply. H4 keeps the per-layer probe gain as its
  primary axis and the layer-coincidence claim is not dropped — subject to the power
  limitation pre-registered in §A2a, which is unaffected by R1 and still binds.
- R2 (per-layer SAE gain) remains a new quantity with no published counterpart, to be defined
  from scratch in `PREREGISTRATION.md`. R1 passing says nothing about it.
