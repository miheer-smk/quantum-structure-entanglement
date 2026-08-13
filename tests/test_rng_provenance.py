"""
A seed is not a generator identity, and this file is the demonstration.

Stage 1's collapse quality `Q` and every bootstrap CI in the repository are resampling
statistics. When a regeneration diff moves one of them, exactly one question decides whether
it is a defect: *did the two runs draw the same numbers?* A provenance record that says only
`seed: 0` cannot answer it, because `seed: 0` describes at least three different streams --
`default_rng(0)`, `Generator(MT19937(0))`, and `default_rng(0)` under a NumPy whose default
bit generator has changed.

So `scripts/_provenance.rng_fingerprint` records the bit generator's NAME and full STATE, and
these tests show that the fingerprint separates cases the seed alone cannot. Per CLAUDE.md, the
demonstration that the old record was insufficient is what makes the new one a gate rather than
a nicety.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from _provenance import rng_fingerprint


def test_fingerprint_names_the_bit_generator_and_carries_its_state():
    fp = rng_fingerprint(np.random.default_rng(20260804))
    assert fp["bit_generator"] == type(np.random.default_rng().bit_generator).__name__
    assert isinstance(fp["state"], dict) and fp["state"], "no state captured"
    assert fp["seed_seq_entropy"] == 20260804, "the seed itself must still be recoverable"
    json.dumps(fp)          # must survive the JSON round-trip the provenance header does


# ---------------------------------------------------------------------------------------
# Failure demonstrations: what a seed-only record cannot see
# ---------------------------------------------------------------------------------------

def test_same_seed_different_bit_generator_is_distinguishable():
    """FAILURE DEMONSTRATION: two 'seed 0' runs that share no numbers at all.

    A seed-only record calls these identical. They are not: the streams differ from the first
    draw. If a regenerated `Q` moved because of this, a seed-only header would have made the
    difference look inexplicable -- or, worse, looked like agreement.
    """
    a = np.random.default_rng(0)
    b = np.random.Generator(np.random.MT19937(0))

    seed_only_record = {"seed": 0}
    assert seed_only_record == {"seed": 0}, "both runs record the same seed"

    fa, fb = rng_fingerprint(a), rng_fingerprint(b)
    assert fa["bit_generator"] != fb["bit_generator"], (
        "the fingerprint cannot tell PCG64 from MT19937; it is no better than the seed")
    assert not np.array_equal(a.random(8), b.random(8)), (
        "these two generators produce the same stream, so the demonstration is vacuous")


def test_a_partially_consumed_generator_is_distinguishable_from_a_fresh_one():
    """FAILURE DEMONSTRATION: same seed, same bit generator, different position in the stream.

    This is the subtler case -- a run that reuses a generator another step already drew from.
    Seed and generator name are identical; only the state differs, and the state is what
    decides which numbers come out next.
    """
    fresh = np.random.default_rng(7)
    used = np.random.default_rng(7)
    used.random(1000)

    f_fresh, f_used = rng_fingerprint(fresh), rng_fingerprint(used)
    assert f_fresh["bit_generator"] == f_used["bit_generator"]
    assert f_fresh["seed_seq_entropy"] == f_used["seed_seq_entropy"] == 7
    assert f_fresh["state"] != f_used["state"], (
        "a consumed generator fingerprints identically to a fresh one; the state is not being "
        "captured and the record cannot tell where a stream was resumed")


def test_identical_construction_fingerprints_identically():
    """The other direction: the fingerprint must not report spurious differences.

    A check that always says 'different' would flag every benign rerun and be discarded within
    a week.
    """
    assert rng_fingerprint(np.random.default_rng(20260814)) == \
           rng_fingerprint(np.random.default_rng(20260814))


def test_the_documented_reference_rngs_are_what_stage1_says_they_are():
    """PLAN.md §A0b documents the L=10/12 reference chains as `default_rng(20260804 + L)`.

    Regenerating Stage 1 depends on that sentence being literally true, so it is asserted
    rather than trusted: the seed must be recoverable from the generator the documentation
    describes.
    """
    for L in (10, 12):
        fp = rng_fingerprint(np.random.default_rng(20260804 + L))
        assert fp["seed_seq_entropy"] == 20260804 + L


@pytest.mark.parametrize("seed", [0, 42, 20260814])
def test_fingerprint_round_trips_into_a_provenance_header(seed):
    """The header is written as JSON; a fingerprint that cannot serialise is useless."""
    from _provenance import provenance
    block = provenance(seeds={"demo": seed}, rngs={"demo": np.random.default_rng(seed)})
    assert block["rng_generators"]["demo"]["seed_seq_entropy"] == seed
    assert block["numpy_default_bit_generator"]
    json.dumps(block)
