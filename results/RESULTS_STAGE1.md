# RESULTS_STAGE1.md — Stage 1 (in progress)

**Status:** pre-Stage-1 audits complete (§1, §2). Toy-case validation not yet run.
**Tests: 29 passed.** Submodule pin `0c4e6e4` asserted; all 35 pinned artifact hashes verified.

---

## 1. Fabricated-constant audit

**Motivating failure (Stage 0):** four "pinned realizations" in `test_exact_entropy.py` were
plausible-looking numbers rather than the actual vectors. Every physics assertion passed,
because the ED/free-fermion gate holds for **any** `h`. The generalised principle:

> **An assertion that holds for any input cannot detect wrong input.** Correctness of the
> computation and identity of the input are independent properties and need independent checks.

Every numeric constant in `src/` and `tests/` that is supposed to originate from a pinned
artifact or a published result:

| Constant | Value | Source of truth | Load or assert | Status |
|---|---|---|---|---|
| `SPANNING` (5 × 8 `h` values) | see test file | `$QSAE_ARTIFACTS/data/tfim_L8_N50k_seed1.pt` | **assert** vs pinned, tol 1e-12 | ✅ `test_spanning_realizations_match_pinned_ensemble` |
| `SPANNING_INDICES` | 29086, 40316, 49390, 28980, 11303 | same | **assert** (indexes the pinned tensor) | ✅ same test |
| `SPANNING_DELTA` | +2.000, +1.000, 0.000, −1.000, −2.001 | recomputed from pinned `h` | **assert**, tol 1e-5 | ✅ same test |
| δ_r spanning property | ordered / critical / paramagnetic | recomputed | **assert** | ✅ `test_spanning_realizations_actually_span` |
| `h_min`, `h_max`, `J` | 0.1, 2.0, 1.0 | artifact `meta` dict | **load** via `ensemble_meta()` | ✅ `test_disorder_parameters_come_from_the_artifact_not_a_literal` |
| `E[ln h]` | −0.149183 | closed form **and** pinned sample | **assert** both, tol 1e-6 / 5e-3 | ✅ same test |
| `σ_lnh` | 0.709086 | closed form from loaded `h_min/h_max` | **assert**, tol 1e-6 | ✅ same test |
| R1 mean | 0.0283 | pinned `results/phase06_multiseed_trained.md` | **load** via `published_constant()` | ✅ `test_r1_tolerance_is_derived_from_the_pinned_publication` |
| R1 sd | 0.0030 | same | **load** | ✅ same test |
| R1 window | [0.0223, 0.0343] | **derived** as mean ± 2sd | **computed**, never typed | ✅ same test |
| Submodule SHA | `0c4e6e4a8a…` | this repo's own gitlink | **assert** vs `git ls-tree` | ✅ `test_submodule_sha_matches_declared_pin` |
| 15 ensemble hashes | `pins/ensemble.sha256` | `$QSAE_ARTIFACTS` | **assert** SHA-256 | ✅ `test_every_pinned_artifact_hash_matches` |
| 20 checkpoint/config hashes | `pins/checkpoints.sha256` | `$QSAE_ARTIFACTS` | **assert** SHA-256 | ✅ same test |
| `GATE = 1e-10` | — | policy, not an artifact | n/a | ⬜ out of scope |
| `_EPS` (1e-12, 1e-14) | — | numerical, not an artifact | n/a | ⬜ out of scope |

**No exceptions were granted for "obviously right" values.** `h_min=0.1`/`h_max=2.0` were
the most tempting — they appear in every config — and are now read from the artifact's own
`meta`, because they determine `σ_lnh` and therefore every `δ_r`.

### The audit immediately caught a second error

Deriving the R1 tolerance instead of hardcoding it exposed a live bug **in this audit's own
code**. `phase06` reports `long_range_zz` in two tables:

```
## Partial correlation | poly2-h …
| long_range_zz | 0.560±0.046 [0.503,0.626] | …
## Incremental R² beyond poly2-h …
| long_range_zz | 0.0283±0.0030 [0.0231,0.0320] | …
```

The first row regex was unanchored and silently returned **0.560** — the wrong quantity, off
by a factor of 20. Had the tolerance been typed in as `0.0283`, the test would have passed
and the wrong provenance would have persisted invisibly. `published_constant()` now requires
a **section anchor** and asserts **exactly one** match, failing otherwise.

This is the same failure mode as the fabricated realizations, in different clothing: a value
that looked right, from a source nobody checked.

### Lint

`test_no_unverified_inlined_float_arrays_in_tests` walks the AST of every file in `tests/`
and fails on any assignment containing a numeric list/tuple of length ≥ 4 whose name is not
in `ALLOWED_INLINE`. Each `ALLOWED_INLINE` entry must name a verifying test, and
`test_allowed_inline_entries_have_a_live_verifier` asserts that test exists — so the
allowlist cannot degrade into a parking space for unverified constants.

---

## 2. Site-ordering bug: impact verified, not asserted

### Every call site in the pinned submodule

`grep -rn "_reduce_density_matrix\|half_chain_entanglement_entropy\|entanglement_spectrum"`
over `submodules/quantum-structure-sae` (excluding `.venv`, `__pycache__`):

| Location | Call | `cut=` passed? | L | Half cut? |
|---|---|---|---|---|
| `src/qsae/analysis/family_data.py:65` | `half_chain_entanglement_entropy(psi, L)` | no → default `L//2` | 8, 12 | ✅ |
| `src/qsae/observables.py:351` | `half_chain_entanglement_entropy(psi, n)` | no → default | 8, 12 | ✅ |
| `src/qsae/observables.py:436` | `half_chain_entanglement_entropy(psi, n)` | no → default | 8, 12 | ✅ |
| `tests/test_observables.py:101,108,115,126,141,142` | `(psi, n)` | no → default | 4, 6, 8 | ✅ |
| `tests/test_observables.py:161,169` | `entanglement_spectrum(psi, n)` | no → default | 4, 6 | ✅ |
| `src/qsae/__init__.py:24,25,53` | re-export only | n/a | n/a | n/a |

**No call site anywhere passes `cut=`.** Every one uses the default `cut = n//2`, and every
`L` in the repo is **even** (configs and experiments contain only 6, 8, 12; the scaling run
used 8, 10, 12; tests use 4, 6, 8).

### Why even L makes it harmless — and the empirical confirmation

The function returns `S` of the **high** `n_A` bits, i.e. sites `[L−n_A, L)`. The docstring
claims sites `[0, n_A)`. For a pure state `S(A) = S(complement of A)`, and the complement of
`[0, n_A)` is `[n_A, L)`, which equals `[L−n_A, L)` **only when `n_A = L/2`** — i.e. only for
even `L` at the half cut. Measured, on disordered chains:

```
EVEN L, half cut (every call site in the repo):
  L=4  diff 3.33e-16     L=6  diff 3.16e-15     L=8  diff 5.55e-15
  L=10 diff 1.11e-15     L=12 diff 3.00e-15

ODD L (never used in the repo):
  L=5  diff 2.04e-02     L=7  diff 8.73e-02     L=9  diff 6.04e-02

ASYMMETRIC CUTS at even L (any future profile work):
  cut=1 5.40e-03   cut=2 2.55e-01   cut=3 3.21e-02   cut=4 2.00e-15
  cut=5 3.21e-02   cut=6 2.55e-01   cut=7 5.40e-03
```

**Verdict: no published number is affected.** The claim is now backed by an exhaustive call-site
enumeration plus numerical confirmation at every `L` the repo uses, not by argument alone.

The asymmetric-cut row also shows the failure's exact shape: the profile comes out **mirrored**
(`cut=2 ↔ cut=6`, `cut=3 ↔ cut=5`), agreeing only at `cut=4`. Any future profile work that
called this function would get a reflected curve that looks entirely plausible.

### Recommendation

**Fix upstream with a dated correction note, not a silent edit**, and **keep the pin at
`0c4e6e4`**. Suggested upstream change, in the predecessor repo, as its own commit:

1. Correct `_reduce_density_matrix` to match its docstring (or correct the docstring to match
   the code — either is defensible, but they must agree), and add an asymmetric-cut regression
   test on a disordered chain, which is the only thing that would have caught it.
2. Add a dated note to `docs/CODE_MAP.md` recording that all published results use the even-L
   half cut and are therefore unaffected — so a future reader who finds the fix in the history
   does not have to re-derive whether earlier numbers were wrong.

A silent edit would make the pinned SHA disagree with a later checkout in a way that looks
like data drift, and would destroy the evidence that published results were unaffected.

---

## 3. Orientation gate

### Does this arm inherit the mirror?

**No.** `grep -rn "qsae" src/ tests/ --include=*.py` returns **four docstring mentions and
zero imports**. This arm calls neither `qsae.observables._reduce_density_matrix` nor
`half_chain_entanglement_entropy` at any cut, half or otherwise. It uses
`qsent.exact.reduced_density_matrix`, whose orientation is documented and pinned:

> `site j <-> bit j` (site 0 = least significant bit); cut `l` -> block A = sites `[0, l)`,
> the **LEFT** block; profile index `i` (0-based) corresponds to cut `l = i + 1`.

### The c_eff fit cannot serve as the orientation check — confirmed

Calabrese–Cardy goes as `sin(pi*l/L)`, symmetric under `l -> L-l`. Measured on the
max-asymmetry realization:

```
profile          c_eff = 1.5617669761   SSR = 2.894919e-01
MIRRORED profile c_eff = 1.5617669761   SSR = 2.894919e-01
|dc_eff| = 6.66e-16    |dSSR| = 0.00e+00
true asymmetry present in the data: max|S(l)-S(L-l)| = 0.569541 nats
```

**Identical to machine precision, against 0.57 nats of real asymmetry.** The Stage 1 `c_eff`
recovery check would have passed green on a fully mirrored profile. Frozen as
`test_ceff_fit_is_blind_to_mirroring`.

(Sanity, separately: the fit does recover the clean-critical value — `c_eff` = 0.5570,
0.5414, 0.5289 at L = 32, 64, 128 on a clean critical open chain, converging to 0.5 from
above as finite-size corrections shrink.)

### `test_profile_orientation` — realization and power

Chosen by maximising `max |S(l) - S(L-l)|` over 4,000 sampled members of the pinned seed-1
ensemble:

| | |
|---|---|
| index | **46009** |
| δ_r | +1.716380 |
| **asymmetry margin** | **0.569541 nats** |
| profile | 0.104188, 0.217091, 0.569087, 0.662536, 0.670177, 0.675944, 0.673729 |

The margin is the test's power: an undetected mirror surfaces as a 0.57-nat error, ~5×10⁹
times the 1e-10 agreement gate. The test asserts each provider matches ED **unreversed** to
< 1e-10 **and** differs from the reversed reference by > 0.1.

**Result: PASS** for both `exact_ed` and `free_fermion`.

### Convention agreement across providers

`PROFILE_PROVIDERS` registers every entropy-profile producer, and
`test_every_profile_provider_is_registered` walks `qsent` by introspection and fails if any
`entropy_profile_*` function is not registered. **Construction B cannot bypass the gate**:
when it lands as `entropy_profile_decoded` the suite fails until it is registered and passes
the orientation assertions.

This matters for the reason stated: a mirror applied consistently to *both* the exact
reference and the model-side path would cancel in the H2 paired test and remain wrong in
every figure and every per-cut number. Orientation is therefore asserted against ED directly,
per provider, not pairwise between them.

### The gate is shown able to fail

`test_orientation_gate_catches_a_mirrored_provider` runs a deliberately mirrored provider
through the same assertions and confirms rejection (forward error 0.57 > 0.1 threshold;
reversed error < 1e-10). Per the standing rule now in `CLAUDE.md`.

## 4. Still to run in Stage 1

Toy closed-form cases (product → 0; Bell → ln 2; GHZ → ln 2 at every internal cut;
W → `H₂(ℓ/L)`; TFIM L=8 vs Stage 0 golden values), the untruncated-spectrum assertion, the
Construction-B perfect/rank-1 readout validation, split-disjointness tests, the
estimator-bias harness, and the `c_eff` recovery fit validating the `2 × ln2` convention
conversion on a clean critical chain.
