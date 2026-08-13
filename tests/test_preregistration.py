"""
The pre-registration says what PLAN.md said, and says which parts are not written yet.

PLAN.md §3.6 requires its text to be lifted into `PREREGISTRATION.md` **verbatim**, "so the
wording is fixed now rather than written after the numbers are seen". A pre-registration whose
mandated text has been quietly paraphrased -- softened, tightened, or reordered -- is worth
less than none, because its value is entirely in having been fixed in advance.

So the lift is mechanical (`scripts/build_preregistration.py` extracts it) and this file is
the gate on it: byte-identity against the source, plus a demonstration that a single altered
word is caught. Per CLAUDE.md, a check that has never been shown to fail on the error it
targets is not a gate.

Also asserted: Part II is present and marked PENDING. A staged pre-registration must declare
its staging in the document; a reader must not have to notice an absence.
"""

from __future__ import annotations

import pytest

from build_preregistration import LIFTS, PRIOR_CONSTANTS, lift
from qsent.pins import published_constant, repo_root

PREREG = repo_root() / "PREREGISTRATION.md"


@pytest.fixture(scope="module")
def prereg() -> str:
    if not PREREG.exists():
        pytest.skip("PREREGISTRATION.md has not been built yet")
    return PREREG.read_text()


@pytest.fixture(scope="module")
def flat(prereg) -> str:
    """The document with markdown wrapping flattened: no line breaks, no blockquote markers.

    Phrase assertions must survive the fact that a sentence in this document is hard-wrapped
    at 96 characters and may carry a `> ` prefix on each line. Asserting against the raw text
    would make the tests fail on rewrapping -- a check coupled to formatting, which is the
    defect that killed scripts/diff_stage0.py (DEVIATIONS.md, 2026-08-13).
    """
    import re
    return re.sub(r"\s+", " ", prereg.replace("\n> ", " ").replace("**", ""))


# ---------------------------------------------------------------------------------------
# The verbatim lift
# ---------------------------------------------------------------------------------------

@pytest.mark.parametrize("name", sorted(LIFTS))
def test_lifted_block_is_byte_identical_to_plan(prereg, name):
    """Every character of the mandated text, as PLAN.md has it."""
    block = lift(name)
    assert block in prereg, (
        f"the {name!r} block in PREREGISTRATION.md is not byte-identical to PLAN.md. It was "
        f"lifted verbatim by scripts/build_preregistration.py and must not be edited in "
        f"place; edit PLAN.md and rebuild, or record a deviation.")


@pytest.mark.parametrize("name", sorted(LIFTS))
def test_a_single_altered_word_is_caught(prereg, name):
    """FAILURE DEMONSTRATION: paraphrase is what this gate exists to reject.

    Changes one word of the lifted block and asserts the altered text is NOT what the file
    contains. Without this, `in` could be passing for a reason unrelated to fidelity.
    """
    import re as _re
    block = lift(name)
    for original, softened in (("cannot", "may not"), ("MUST", "should"),
                               ("NOT", "not necessarily"), ("must", "should"),
                               ("not", "not necessarily")):
        if _re.search(rf"\b{original}\b", block):
            altered = _re.sub(rf"\b{original}\b", softened, block, count=1)
            assert altered != block, "the mutation did not change the block"
            assert altered not in prereg, (
                f"a softened variant of the {name!r} block is present in "
                f"PREREGISTRATION.md ({original!r} -> {softened!r})")
            return
    pytest.fail(f"no modal verb found to soften in the {name!r} block; "
                f"the demonstration cannot demonstrate anything")


def test_the_mandated_sections_are_all_present(prereg):
    """§3.55 and every A-section the author named."""
    for marker in ("## 3.55 PAPER SPINE", "### A0 — ", "### A0b — ", "### A0c — ",
                   "### A1 — ", "### A2 — ", "### A2a — ", "### A3 — "):
        assert marker in prereg, f"mandated section {marker!r} is missing"


# ---------------------------------------------------------------------------------------
# The prior is parsed, not typed
# ---------------------------------------------------------------------------------------

@pytest.mark.parametrize("name", PRIOR_CONSTANTS)
def test_every_prior_number_is_readable_from_the_pinned_publication(name):
    value = published_constant(name)
    assert isinstance(value, float)


def test_the_prior_reports_both_halves(prereg, flat):
    """The prior is only honest with its second half present.

    Entropy is the strongest observable on incremental R2 (+4.57) and the weakest
    non-degenerate one on partial-r (+1.79). Quoting the first without the second would be a
    selective read of a table this repository has pinned in full.
    """
    incr = published_constant("phase06_entropy_incr_r2_sep_sd")
    partial = published_constant("phase06_entropy_partial_r_sep_sd")
    lrzz = published_constant("phase06_lrzz_incr_r2_sep_sd")
    assert incr > lrzz > partial, (
        f"the prior's structure changed in the publication: incr {incr}, lrzz {lrzz}, "
        f"partial {partial}")
    assert f"+{incr:.2f}" in prereg, "the incremental-R2 separation is not quoted"
    assert f"+{partial:.2f}" in prereg, "the partial-r separation (the second half) is missing"
    assert f"+{lrzz:.2f}" in prereg, "the long_range_zz comparison is missing"


def test_the_probe_target_caveat_is_present(prereg, flat):
    """The caveat carries the same weight as the prior, so it must be in the document."""
    assert "PROBE TARGET" in prereg
    assert "constrains without determining" in flat.lower()


def test_the_null_plan_is_unconditional(flat):
    """The null plan must not be phrased as contingent on the prior."""
    assert "written up as a null, at full precision" in flat
    assert "A favourable prior is not a reason to weaken pre-committed null handling" in flat


# ---------------------------------------------------------------------------------------
# Staging is declared, not implied
# ---------------------------------------------------------------------------------------

def test_part_two_is_present_and_marked_pending(prereg, flat):
    assert "## PART II" in prereg and "PENDING" in prereg
    for item in ("H2 test statistic", "ΔS_incremental", "multiplicity plan", "stopping rules"):
        assert item in prereg, f"Part II does not name {item!r} as pending"
    assert "committed before any Stage 2 measurement" in flat, (
        "Part II must state that it is fixed before Stage 2 runs")


def test_r1_context_does_not_overclaim(flat):
    """R1 passing must not be recorded as licensing more than §A1 allows."""
    assert "§3.5.2's FAIL branch does not apply" in flat
    assert "subject to the §A2a power limitation" in flat
    assert "R1 licenses nothing else" in flat
