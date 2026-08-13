"""
The public files stay accurate: staleness is a defect, and this is the check for it.

`README.md` is the only file most readers will ever open, and by 2026-08-13 it was the least
accurate file in the repository -- it claimed 57 tests when there were 134, said "no
transformer results exist yet" after R1 had run and passed, gave the stage as "Stage 1
complete" when Stage 1.5 was complete, quoted `1.648e-11` bare after the precision rule
required a spread beside it, quoted `8.94e-07` as a value after it was established to be an
array-dependent ULP count, and reported `2.39` where the R1 arrays measure 2.4008-2.4351.

None of those were caught by anything, because nothing checked the README. Prose drifts in one
direction -- optimistically -- and the drift is invisible from inside the file. So the facts
that go stale fastest are asserted here against their sources:

  * the test count, against a real collection of the suite;
  * the author's name, against `LICENSE` (the authoritative file, not to be edited);
  * the phrases that were false, asserted absent so they cannot come back;
  * the honesty markers that must be present -- hypothesis status, "Stage 2 not run", and the
    statement that the entanglement measurement has not been made.
"""

from __future__ import annotations

import re
import subprocess
import sys

import pytest

from qsent.pins import repo_root

README = repo_root() / "README.md"
HANDOFF = repo_root() / "AUTHOR_HANDOFF.md"
LICENSE = repo_root() / "LICENSE"
CITATION = repo_root() / "CITATION.cff"


@pytest.fixture(scope="module")
def readme() -> str:
    return README.read_text()


@pytest.fixture(scope="module")
def handoff() -> str:
    return HANDOFF.read_text()


def _flat(text: str) -> str:
    return re.sub(r"\s+", " ", text.replace("\n> ", " ").replace("**", ""))


# ---------------------------------------------------------------------------------------
# The test count, against a real collection
# ---------------------------------------------------------------------------------------

def collected_test_count() -> int:
    """Collect the suite in a subprocess and return the number of tests.

    A subprocess rather than this session's own item count, so the answer does not depend on
    how the suite was invoked -- running a single file must not make the README look wrong.
    """
    r = subprocess.run(
        [sys.executable, "-m", "pytest", "--collect-only", "-q", "-p", "no:cacheprovider"],
        cwd=repo_root(), capture_output=True, text=True)
    m = re.search(r"(\d+) tests? collected", r.stdout)
    if not m:
        pytest.skip(f"could not parse a collection count from pytest output:\n{r.stdout[-400:]}")
    return int(m.group(1))


def test_readme_test_count_is_current(readme):
    """The specific staleness that survived five commits: '57 tests' with 134 in the tree."""
    n = collected_test_count()
    stated = set(int(x) for x in re.findall(r"(\d+)\s+tests", readme))
    assert stated, "README no longer states a test count; it should, and it should be right"
    assert stated == {n}, (
        f"README states test counts {sorted(stated)} but the suite collects {n}. Update every "
        f"occurrence -- this is exactly the drift that left '57 tests' in a public file "
        f"through five commits.")


# ---------------------------------------------------------------------------------------
# Names: LICENSE is authoritative and is not edited
# ---------------------------------------------------------------------------------------

def license_copyright_holder() -> str:
    m = re.search(r"Copyright \(c\) \d{4} (.+)", LICENSE.read_text())
    assert m, "LICENSE has no parseable copyright line"
    return m.group(1).strip()


def test_citation_names_the_author_exactly_as_license_does():
    """Three spellings existed across LICENSE, CITATION.cff and PLAN.md D5 (DEVIATIONS.md).

    LICENSE is authoritative per PLAN.md §5 D5 and is not to be touched, so the others move.
    """
    holder = license_copyright_holder()
    cff = CITATION.read_text()
    given = re.search(r"given-names:\s*(.+)", cff).group(1).strip()
    family = re.search(r"family-names:\s*(.+)", cff).group(1).strip()
    assert f"{given} {family}" == holder, (
        f"CITATION.cff names '{given} {family}', LICENSE names '{holder}'")


def test_plan_d5_quotes_license_accurately():
    """D5 quoted LICENSE as saying something LICENSE does not say."""
    holder = license_copyright_holder()
    plan = (repo_root() / "PLAN.md").read_text()
    quoted = re.findall(r'`LICENSE` \(MIT, "Copyright \(c\) \d{4} ([^"]+)"\)', plan)
    assert quoted, "PLAN.md D5 no longer quotes LICENSE; the test needs updating with it"
    for q in quoted:
        assert q.strip() == holder, (
            f"PLAN.md D5 quotes LICENSE as naming '{q}', LICENSE names '{holder}'")


def test_readme_license_section_matches_license(readme):
    """The section must state MIT, and any 'all rights reserved' must be historical only."""
    assert "MIT" in LICENSE.read_text()
    section = _flat(readme.split("## License", 1)[1]).lower()
    assert "mit" in section and "`license`" in section
    if "all rights reserved" in section:
        assert section.index("previously read") < section.index("all rights reserved"), (
            "the README states all-rights-reserved as current, contradicting LICENSE")


# ---------------------------------------------------------------------------------------
# Statements that were false, asserted absent
# ---------------------------------------------------------------------------------------

FALSE_CLAIMS = [
    ("no transformer results exist yet", "R1 has run and passed"),
    ("Status: **Stage 1 complete.**", "Stage 1.5 is complete"),
    ("57 tests", "the suite is larger than that"),
]


@pytest.mark.parametrize("phrase,why", FALSE_CLAIMS)
def test_known_false_statements_are_gone_from_the_readme(readme, phrase, why):
    assert phrase not in readme, f"README still says {phrase!r}, which is false: {why}"


def test_the_hook_agreement_is_not_quoted_as_a_value(readme):
    """8.94e-07 is one array's ULP count, not a measurement of the extraction."""
    flat = _flat(readme)
    assert "8.94" not in flat, (
        "README quotes 8.94e-07 as a value; it is 15 x 2^-24 on one array and the claim that "
        "survives every array is the bound < 2e-06")
    assert "2e-06" in flat, "README should state the agreement as a bound"


def test_the_headline_agreement_carries_its_spread(readme):
    """The corrected precision rule: the spread travels with the value."""
    flat = _flat(readme)
    assert "1.648" in flat, "README should quote the headline agreement"
    i = flat.index("1.648")
    assert "2.0e-15" in flat[i:i + 400] or "2.0 × 10⁻¹⁵" in flat[i:i + 400], (
        "1.648e-11 is quoted without its measured spread")


def test_post_final_norm_difference_is_stated_from_the_r1_arrays(readme):
    """2.39 was measured on a training split; the R1 arrays give 2.4008-2.4351."""
    assert "2.39" not in _flat(readme), "README still quotes the superseded 2.39"


# ---------------------------------------------------------------------------------------
# Honesty markers that must be present
# ---------------------------------------------------------------------------------------

def test_hypotheses_are_marked_not_yet_tested(readme):
    """A skimmer reading H1-H5 in full detail must not conclude they were tested."""
    flat = _flat(readme)
    assert "NOT YET TESTED" in readme, (
        "the hypothesis section must carry an explicit status; listing H1-H5 in detail with "
        "no status reads as tested")
    for h in ("H1", "H2", "H3", "H4", "H5"):
        assert h in flat


def test_the_status_block_says_the_measurement_has_not_been_made(readme):
    flat = _flat(readme)
    assert "THE ENTANGLEMENT MEASUREMENT HAS NOT BEEN MADE" in readme, (
        "the central negative result must be prominent, not buried")
    assert "Stage 2" in flat and "not been run" in flat.lower()


def test_the_status_block_carries_r1_scope(readme):
    """R1's scope sentence must travel with R1's result wherever it is reported."""
    flat = _flat(readme)
    assert "R1" in flat
    assert "does not validate anything about SAEs" in flat, (
        "A1's scope sentence is missing from the README")
    assert "nothing about entanglement" in flat


def test_handoff_no_longer_claims_nothing_has_been_run(handoff):
    flat = _flat(handoff)
    assert "no experiment has been run" not in flat, "AUTHOR_HANDOFF.md line 5 is false"
    assert "no remote created yet" not in flat, "the repository is public and has a remote"
    assert "R1" in flat and "PASS" in handoff, "the R1 verdict belongs in the handoff"
    assert "does not validate anything about SAEs" in flat, (
        "R1's scope sentence must accompany its verdict in the handoff too")
    assert "nothing about entanglement" in flat


# ---------------------------------------------------------------------------------------
# Coverage claims must match what the gate actually guarantees
# ---------------------------------------------------------------------------------------

def test_readme_claim_count_matches_the_gate(readme):
    """The README states a registered-claim count; it must be the gate's, not a memory of it.

    This is the same staleness that left "57 tests" in a public file, applied to a number that
    is easier to overstate: a claim count reads as coverage, so it must at least be current.
    """
    from check_provenance import load_claims
    n = len(load_claims())
    stated = [int(x) for x in re.findall(r"(\d+) of them", readme)]
    assert stated, "README no longer states the registered-claim count"
    assert stated == [n], f"README states {stated} registered claims, the gate has {n}"


def test_readme_does_not_claim_universal_coverage(readme):
    """The overstatement that was live in this file: 'every numeric claim ... carries a tag'.

    The gate is blind to any number nobody registered, so a sentence implying full coverage is
    false regardless of how many claims exist.
    """
    flat = _flat(readme)
    assert "every numeric claim in the results files, the pre-registration, this README" \
        not in flat, "the universal-coverage sentence is back"
    assert "does not mean every number in these files is checked" in flat, (
        "the README must say what the gate does NOT guarantee, not only what it does")


def test_unreproducible_populations_are_marked_in_both_public_files(readme, handoff):
    """Numbers that cannot be regenerated must say so where they are quoted."""
    for phrase in ("NOT currently reproducible from committed code",):
        assert phrase in readme, f"README does not mark the unreproducible populations"
    for missing_fact in ("bin edges", "pool size"):
        assert missing_fact in readme, f"README does not name the missing fact: {missing_fact}"
    flat = _flat(handoff)
    assert "cannot be regenerated" in flat, (
        "AUTHOR_HANDOFF must state the stronger truth, not 'not yet tagged'")
    assert "not yet under the provenance gate" not in flat, (
        "the old, weaker wording is still present")


def test_readme_untagged_count_matches_the_gate(readme):
    """The README states how many literals the gate does NOT check; it must be measured.

    Stating a coverage gap and then letting the figure drift would be a worse failure than not
    stating one, so the number is asserted against the gate's own count rather than trusted.
    """
    from check_provenance import untagged_literals
    n = len(untagged_literals(README))
    stated = [int(x) for x in re.findall(r"(\d+) measurement-shaped literals", readme)]
    assert stated, "README no longer states its untagged-literal count"
    assert stated == [n], f"README states {stated} untagged literals, the gate counts {n}"
