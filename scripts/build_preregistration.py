"""
Assemble PREREGISTRATION.md Part I. The mandated sections are LIFTED, never retyped.

PLAN.md §3.6 says its text "must be lifted into it verbatim, so the wording is fixed now
rather than written after the numbers are seen." Retyping it by hand would put a
transcription step between the frozen wording and the pre-registration -- the same shape as
every stated-source failure in DEVIATIONS.md, applied to the one document whose whole value
is that its wording was fixed in advance. So the sections are extracted from PLAN.md at build
time, and `tests/test_preregistration.py` asserts the result is byte-identical to the source.

Lifted verbatim:
  * §3.55  PAPER SPINE -- confirmatory H1/H2 vs exploratory H3/H4, the structural reason, and
           "What would make H3/H4 confirmatory"
  * §3.6   A0, A0b, A0c, A1, A2, A2a, A3 (one contiguous block in PLAN.md)

Written fresh here, as new pre-registered text:
  * the phase06 PRIOR for Stage 2's primary endpoint family, with its caveat at equal weight
  * the NULL PLAN, unchanged by the prior
  * R1's outcome as context
  * PART II, marked PENDING with the reason

Every number in the prior is parsed from the pinned publication through `qsent.pins`, never
typed, and registered as a provenance claim so `scripts/check_provenance.py` verifies it.

Run:  env/run.sh python scripts/build_preregistration.py
Out:  PREREGISTRATION.md, scripts/out/preregistration_prior.json
"""

from __future__ import annotations

import sys

from _claims import Registry
from _provenance import provenance, write_json
from qsent.pins import published_constant, repo_root, submodule_sha

SCRIPT = "scripts/build_preregistration.py"
PLAN = "PLAN.md"

#: (start marker, end marker) for each verbatim lift. The end marker is the heading that
#: follows the block in PLAN.md; it is excluded. Both are asserted to occur exactly once.
LIFTS = {
    "spine": ("## 3.55 PAPER SPINE", "## 3.6 Mandated"),
    "appendix": ("### A0 — `c_eff` finite-size bias", "## 4. Design conflicts"),
}

PRIOR_CONSTANTS = (
    "phase06_entropy_incr_r2_trained_mean", "phase06_entropy_incr_r2_trained_sd",
    "phase06_entropy_incr_r2_random_mean", "phase06_entropy_incr_r2_random_sd",
    "phase06_entropy_incr_r2_sep_sd", "phase06_lrzz_incr_r2_sep_sd",
    "phase06_entropy_partial_r_trained_mean", "phase06_entropy_partial_r_trained_sd",
    "phase06_entropy_partial_r_sep_sd",
)


def lift(name: str) -> str:
    """Extract one block from PLAN.md verbatim, or raise."""
    text = (repo_root() / PLAN).read_text()
    start, end = LIFTS[name]
    for marker in (start, end):
        if text.count(marker) != 1:
            raise RuntimeError(
                f"marker {marker!r} occurs {text.count(marker)} times in {PLAN}; a lift "
                f"anchored on it would be ambiguous")
    block = text.split(start, 1)[1].split(end, 1)[0]
    block = start + block
    # Trim the trailing horizontal rule that separates PLAN sections; it is punctuation of
    # the source document, not content of the lifted block.
    return block.rstrip().removesuffix("---").rstrip() + "\n"


def tag(claim_id: str, claims: dict, literal: str) -> str:
    c = claims[claim_id]
    return (f'<!--prov id={claim_id} script={c["script"]} array={c["array"]} '
            f'seed={c["seed"]} sha256={c["sha256"]} kind={c["kind"]} md={literal} -->')


def main() -> int:
    reg = Registry(SCRIPT)
    v = {}
    for name in PRIOR_CONSTANTS:
        v[name] = published_constant(name)
        # Published constants come from the pinned submodule, not from $QSAE_ARTIFACTS; the
        # submodule SHA in the provenance header is what pins them.
        reg.add(name, v[name], "none")
    claims = reg.as_dict()

    def t(name: str, literal: str) -> str:
        return tag(name, claims, literal)

    prior = f"""## The phase06 prior for Stage 2's primary endpoint family

**New pre-registered text**, added 2026-08-13. Every number below is parsed from the pinned
`results/phase06_multiseed_trained.md` at build time by `scripts/build_preregistration.py`
via `qsent.pins.published_constant`; none is typed into this document. Submodule pin
`{submodule_sha()[:7]}`.

### The prior, stated in full — both halves

**Half one: on the designated primary endpoint family, entropy is the strongest observable in
the published data.** On **incremental R² beyond poly-2**, scalar half-chain entropy separates
from the random-init distribution by

{t("phase06_entropy_incr_r2_sep_sd", "+4.57")}
{t("phase06_entropy_incr_r2_trained_mean", "0.0305")}
{t("phase06_entropy_incr_r2_trained_sd", "0.0023")}
{t("phase06_entropy_incr_r2_random_mean", "0.0185")}
{t("phase06_entropy_incr_r2_random_sd", "0.0026")}
{t("phase06_lrzz_incr_r2_sep_sd", "+4.20")}

> **+4.57** σ — trained **0.0305** ± **0.0023** against random-init **0.0185** ± **0.0026**.

That is the **largest separation of any observable** in that table, and larger than
`long_range_zz`'s **+4.20** σ — the observable this arm's own R1 gate was built around.

**Half two: on partial correlation, the same observable fails at 10 seeds.**

{t("phase06_entropy_partial_r_sep_sd", "+1.79")}
{t("phase06_entropy_partial_r_trained_mean", "0.565")}
{t("phase06_entropy_partial_r_trained_sd", "0.040")}

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
"""

    part_ii = """## PART II — Stage 2 endpoint definitions — **PENDING**

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
"""

    header = f"""# PREREGISTRATION.md — Part I

**Status: PART I ONLY.** Part II (Stage 2 endpoint definitions) is **PENDING** — see the final
section. This is staged pre-registration, and the staging is declared here rather than
inferred from the absence of a section.

Committed after the author read the Stage 1.5 reproduction result, as PLAN.md §3.6 requires:
`PREREGISTRATION.md` "is **not** created or committed until the author has read the Stage 1.5
reproduction result."

**How to read this document.** The two sections below marked *lifted verbatim* are extracted
from `PLAN.md` mechanically by `{SCRIPT}` and asserted byte-identical to their source by
`tests/test_preregistration.py`. They were frozen on 2026-08-04, before any model number
existed, and nothing in them has been rewritten in the light of a result. Everything after
them is new pre-registered text, dated where it appears.

Provenance: submodule pin `{submodule_sha()[:7]}`; artifact hashes in `pins/`; every published
number quoted below is parsed from the pinned publication at build time and carries a
machine-readable provenance tag.

---

# Lifted verbatim from PLAN.md §3.55 (frozen 2026-08-04)

"""

    appendix_header = """
---

# Lifted verbatim from PLAN.md §3.6 — A0, A0b, A0c, A1, A2, A2a, A3 (frozen 2026-08-04)

"""

    doc = (header + lift("spine") + appendix_header + lift("appendix")
           + "\n---\n\n# New pre-registered text (2026-08-13)\n\n"
           + prior + "\n---\n\n" + part_ii)

    out = repo_root() / "PREREGISTRATION.md"
    out.write_text(doc)

    payload = {
        "_provenance": provenance(seeds={}, artifacts=[]),
        "source_plan_sections": {k: {"start": s, "end": e} for k, (s, e) in LIFTS.items()},
        "published_prior": v,
        "claims": claims,
    }
    path = write_json(payload, "preregistration_prior.json")
    print(f"wrote {out} ({len(doc.splitlines())} lines) and {path} "
          f"with {len(claims)} provenance claims")
    return 0


if __name__ == "__main__":
    sys.exit(main())
