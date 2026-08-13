"""
R1's tolerance, its parsing, and its verdict -- asserted, never eyeballed.

Three things are checked here, and none of them is "did R1 pass":

1. **The tolerance is derived from the pinned publication**, not typed. `[0.0223, 0.0343]`
   must fall out of the published mean and sd read from the submodule.
2. **The published per-seed series is parsed from the right line.** The "Per-seed
   transparency" section contains a partial-correlation series AND an incremental-R² series.
   Reading the wrong one is the `0.560`-for-`0.0283` error, in the numbers R1 pairs against.
   The anchor is shown to be load-bearing by a failure demonstration.
3. **The recorded verdict is the one the pre-registered rule computes** from the recorded
   numbers. If the results file ever said PASS while the numbers said FAIL, this fails.

Deliberately NOT asserted: that the verdict is PASS. A FAIL is a legitimate, reportable
outcome under PLAN.md §3.5.2 and must not turn the suite red -- the suite's job is to
guarantee the verdict is computed correctly and reported honestly, not to guarantee its value.
"""

from __future__ import annotations

import json
import os
import re

import pytest

from qsent.pins import (
    _PUBLISHED_SERIES, _pinned_text, _section_of, published_constant, published_per_seed,
    repo_root,
)

needs_artifacts = pytest.mark.skipif(
    not os.environ.get("QSAE_ARTIFACTS"), reason="QSAE_ARTIFACTS not configured")

RESULTS_JSON = repo_root() / "scripts" / "out" / "r1_reproduction.json"
SEEDS = tuple(range(1, 11))


@pytest.fixture(scope="module")
def r1() -> dict:
    if not RESULTS_JSON.exists():
        pytest.skip("R1 has not been run yet (scripts/out/r1_reproduction.json absent)")
    return json.loads(RESULTS_JSON.read_text())


# ---------------------------------------------------------------------------------------
# 1. The tolerance is derived, not typed
# ---------------------------------------------------------------------------------------

def test_tolerance_window_is_derived_from_the_pinned_publication():
    mean = published_constant("phase06_lrzz_incr_r2_mean")
    sd = published_constant("phase06_lrzz_incr_r2_sd")
    lo, hi = mean - 2 * sd, mean + 2 * sd
    assert abs(lo - 0.0223) < 1e-9 and abs(hi - 0.0343) < 1e-9, (
        f"the pre-registered window [0.0223, 0.0343] no longer follows from the published "
        f"mean {mean} and sd {sd}; the publication or the parser changed")


# ---------------------------------------------------------------------------------------
# 2. The per-seed series comes from the right line
# ---------------------------------------------------------------------------------------

def test_published_per_seed_parses_ten_distinct_seeds():
    pub = published_per_seed("phase06_lrzz_incr_r2_per_seed")
    assert sorted(pub) == list(SEEDS), f"expected seeds 1..10, got {sorted(pub)}"
    assert all(0.0 < v < 0.1 for v in pub.values()), f"implausible incremental R2 values: {pub}"


def test_published_per_seed_is_consistent_with_the_published_mean():
    """The parsed series must average to the separately parsed headline mean.

    Two independent reads of the same publication that disagree would mean one of them is
    reading the wrong thing -- which is exactly how the partial-correlation series would
    announce itself, since it averages to ~0.56 rather than ~0.028.
    """
    pub = published_per_seed("phase06_lrzz_incr_r2_per_seed")
    mean = published_constant("phase06_lrzz_incr_r2_mean")
    got = sum(pub.values()) / len(pub)
    assert abs(got - mean) < 5e-4, (
        f"per-seed series averages to {got:.6f} but the headline mean is {mean:.6f}; one of "
        f"the two reads is pointed at the wrong table")


def test_dropping_the_line_anchor_returns_the_partial_correlation_series():
    """FAILURE DEMONSTRATION: the anchor is load-bearing, and this is what it prevents.

    The section carries TWO per-seed series. Without the `**incremental R²** per-seed:` line
    anchor the entry pattern matches BOTH, in document order, and the partial-correlation
    series comes first. Any reader that takes the first ten entries -- or the first match, as
    the `_one`-style helpers in this repository do -- silently gets `0.512` where `0.023` was
    meant. That is the `0.560`-for-`0.0283` error, in R1's pairing.

    Note the trap inside the trap: collecting the 20 matches into a dict makes the SECOND
    series overwrite the first, so a careless demonstration concludes the anchor is
    unnecessary. The first draft of this test did exactly that and passed the wrong way round.
    """
    rel, heading, _line_pat, entry_pat, expected_n = \
        _PUBLISHED_SERIES["phase06_lrzz_incr_r2_per_seed"]
    section = _section_of(_pinned_text(rel), heading, rel)
    correct = published_per_seed("phase06_lrzz_incr_r2_per_seed")

    entries = re.findall(entry_pat, section)
    assert len(entries) == 2 * expected_n, (
        f"expected the unanchored pattern to match both per-seed series ({2 * expected_n} "
        f"entries), got {len(entries)}; the publication's layout changed and this "
        f"demonstration no longer demonstrates anything")

    first_ten = {int(s): float(v) for s, v in entries[:expected_n]}
    assert sorted(first_ten) == list(SEEDS), "the wrong series still looks well-formed"
    assert first_ten != correct, "the unanchored read returned the right series by accident"
    assert min(first_ten.values()) > 0.4, (
        f"expected the unanchored read to return partial correlations (~0.5), got {first_ten}")

    # Defence in depth: the anchored parser would ALSO have caught this via its count check,
    # because 20 != 10. Both guards are kept -- the anchor states the intent, the count
    # notices when the file changes shape underneath it.
    assert len(entries) != expected_n


# ---------------------------------------------------------------------------------------
# 3. The recorded verdict is the computed verdict
# ---------------------------------------------------------------------------------------

@needs_artifacts
def test_recorded_verdict_matches_the_pre_registered_rule(r1):
    """Recompute both limbs from the recorded numbers and compare to the recorded verdict."""
    new = {int(k): v for k, v in r1["r1_per_seed"].items()}
    pub = {int(k): v for k, v in r1["published_per_seed"].items()}
    tol = r1["tolerance"]
    lo, hi = tol["window"]

    mean_new = sum(new.values()) / len(new)
    n_within = sum(1 for s in new if abs(new[s] - pub[s]) <= tol["per_seed_delta_max"])
    limb_i = lo <= mean_new <= hi
    limb_ii = n_within >= tol["min_seeds_within"]
    expected = "PASS" if (limb_i and limb_ii) else "FAIL"

    assert r1["result"]["verdict"] == expected, (
        f"recorded verdict {r1['result']['verdict']} but the rule computes {expected}: "
        f"mean {mean_new:.6f} in [{lo}, {hi}] = {limb_i}, {n_within}/10 seeds within "
        f"{tol['per_seed_delta_max']} = {limb_ii}")
    assert abs(r1["result"]["mean_new"] - mean_new) < 1e-12
    assert r1["result"]["n_seeds_within_tolerance"] == n_within


@needs_artifacts
def test_r1_used_the_published_protocol_unmodified(r1):
    """The forbidden list, asserted: alpha, folds, fold seed, arrays, seeds, aggregation.

    Read from the pinned config rather than typed, so this test compares R1's run against the
    publication's own parameters and not against a second copy of them.
    """
    import yaml
    cfg = yaml.safe_load(
        (repo_root() / "submodules" / "quantum-structure-sae"
         / "configs" / "phase06_multiseed_trained.yaml").read_text())
    p = r1["protocol"]
    assert p["ridge_alpha"] == cfg["probe"]["ridge_alpha"]
    assert p["n_folds"] == cfg["probe"]["n_folds"]
    assert p["fold_seed"] == cfg["probe"]["fold_seed"]
    assert p["eval_seeds"] == cfg["eval"]["seeds"]
    assert p["train_seeds"] == cfg["train_seeds"]
    assert p["observable"] == "long_range_zz" and p["control"] == "poly2_h"
    assert p["hook"] == "block2_mlp", "R1 must run at k=6, the published hook"


@needs_artifacts
def test_no_seed_was_dropped_or_substituted(r1):
    """All ten trained seeds present on both sides, paired on seed identity."""
    assert sorted(int(k) for k in r1["r1_per_seed"]) == list(SEEDS)
    assert sorted(int(k) for k in r1["published_per_seed"]) == list(SEEDS)
    assert sorted(int(k) for k in r1["result"]["paired_differences"]) == list(SEEDS)


@needs_artifacts
def test_published_side_matches_the_pinned_publication(r1):
    """The recorded published series must equal a fresh parse of the pinned file.

    Guards against the recorded JSON drifting from the publication it claims to quote -- the
    stated-source class applied to R1's own output.
    """
    fresh = published_per_seed("phase06_lrzz_incr_r2_per_seed")
    recorded = {int(k): v for k, v in r1["published_per_seed"].items()}
    assert recorded == fresh, f"recorded {recorded} != freshly parsed {fresh}"
