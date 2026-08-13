"""
The provenance gate, and the demonstrations that it can FAIL.

Per the standing rule in `CLAUDE.md`, a validation check counts as a gate only once it has
been shown capable of failing on the specific error it targets. The error this one targets is
**stated source != actual source**, which has occurred three times in this repository:

1. fabricated `h` vectors labelled as pinned realizations;
2. an unanchored regex returning `0.560` where `0.0283` was claimed;
3. `RESULTS_STAGE0.md` section 2 attributing a measurement to "test realizations" when the
   value came from another array.

`test_gate_catches_a_mislabelled_source` reproduces defect (3) exactly -- a tag naming an
array the value did not come from -- and asserts the gate rejects it. That test is the point
of this file; without it the gate is merely a check.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from check_provenance import check, load_claims

ARRAY_A = "data/tfim_L8_N50k_seed1.pt"
ARRAY_B = "data/tfim_L8_N50k_seed2.pt"
SHA_A = "10aacd0f50a4"
SHA_B = "2bfc7dab9e5d"


def _claim(**over) -> dict[str, dict]:
    base = {"id": "demo", "value": 1.25, "script": "scripts/regen_stage0.py",
            "array": ARRAY_A, "kind": "value", "seed": "none", "sha256": SHA_A + "0" * 52}
    base.update(over)
    return {base["id"]: base}


def _doc(tmp_path: Path, tag_fields: str, body: str) -> list[Path]:
    p = tmp_path / "RESULTS_STAGEX.md"
    p.write_text(f"<!--prov {tag_fields} -->\n{body}\n")
    return [p]


GOOD_TAG = (f"id=demo script=scripts/regen_stage0.py array={ARRAY_A} "
            f"seed=none sha256={SHA_A} kind=value md=1.25")


def test_gate_passes_on_a_correct_tag(tmp_path):
    assert check(_doc(tmp_path, GOOD_TAG, "the value is 1.25 nats"), _claim()) == []


# ---------------------------------------------------------------------------------------
# Failure demonstrations -- each targets one way provenance can be wrong
# ---------------------------------------------------------------------------------------

def test_gate_catches_a_mislabelled_source(tmp_path):
    """THE defect: the tag names an array the number was not computed from.

    This is `RESULTS_STAGE0.md` section 2 reproduced in miniature. The value is correct, the
    hash is a genuine pinned artifact's hash, and every other field agrees -- only the source
    is wrong. Every check that existed before this gate passed on exactly this situation.
    """
    tag = GOOD_TAG.replace(f"array={ARRAY_A}", f"array={ARRAY_B}").replace(
        f"sha256={SHA_A}", f"sha256={SHA_B}")
    failures = check(_doc(tmp_path, tag, "the value is 1.25 nats"), _claim())
    assert failures, "gate did NOT catch a mislabelled source"
    assert any("MISLABELLED SOURCE" in f for f in failures), failures


def test_gate_catches_a_missing_tag(tmp_path):
    """A number with no provenance at all must not pass silently."""
    p = tmp_path / "RESULTS_STAGEX.md"
    p.write_text("the value is 1.25 nats\n")
    failures = check([p], _claim())
    assert any("UNTAGGED" in f for f in failures), failures


def test_gate_catches_a_drifted_value(tmp_path):
    """The prose says one thing, the regenerated claim says another."""
    tag = GOOD_TAG.replace("md=1.25", "md=1.31")
    failures = check(_doc(tmp_path, tag, "the value is 1.31 nats"), _claim())
    assert any("value disagrees" in f for f in failures), failures


def test_gate_catches_prose_that_lost_its_number(tmp_path):
    """The tag claims a literal the markdown no longer contains -- a stale edit."""
    failures = check(_doc(tmp_path, GOOD_TAG, "the value is unchanged"), _claim())
    assert any("does not appear under its tag" in f for f in failures), failures


def test_gate_catches_a_wrong_hash(tmp_path):
    """Right array name, wrong hash: the artifact is not the one that was read."""
    tag = GOOD_TAG.replace(f"sha256={SHA_A}", "sha256=deadbeef1234")
    failures = check(_doc(tmp_path, tag, "the value is 1.25 nats"), _claim())
    assert any("not a prefix list" in f for f in failures), failures


def test_gate_catches_a_nonexistent_array(tmp_path):
    """A tag may not name an artifact that is not under $QSAE_ARTIFACTS."""
    missing = "data/not-a-real-artifact.pt"
    tag = GOOD_TAG.replace(f"array={ARRAY_A}", f"array={missing}")
    failures = check(_doc(tmp_path, tag, "the value is 1.25 nats"),
                     _claim(array=missing, sha256=SHA_A + "0" * 52))
    assert any("is not in $QSAE_ARTIFACTS" in f for f in failures), failures


def test_gate_catches_an_unknown_claim_id(tmp_path):
    """A tag may not invent a claim no script registered."""
    tag = GOOD_TAG.replace("id=demo", "id=never_registered")
    failures = check(_doc(tmp_path, tag, "the value is 1.25 nats"), _claim())
    assert any("unknown claim id" in f for f in failures), failures


def test_gate_catches_a_bound_that_is_violated(tmp_path):
    """kind=bound asserts the inequality, so a value above the bound must fail."""
    tag = GOOD_TAG.replace("kind=value", "kind=bound").replace("md=1.25", "md=1.0")
    failures = check(_doc(tmp_path, tag, "below 1.0 nats"), _claim(kind="bound", value=1.25))
    assert any("bound violated" in f for f in failures), failures


def test_documentation_of_the_tag_format_is_not_parsed_as_a_tag(tmp_path):
    """Prose describing the mechanism must not be mistaken for a use of it."""
    p = tmp_path / "RESULTS_STAGEX.md"
    p.write_text("<!--prov …-->\nexplanatory prose about the tag format\n")
    assert [f for f in check([p], _claim()) if "unknown claim id" in f] == []


# ---------------------------------------------------------------------------------------
# The real tree
# ---------------------------------------------------------------------------------------

@pytest.mark.skipif(not __import__("os").environ.get("QSAE_ARTIFACTS"),
                    reason="QSAE_ARTIFACTS not configured")
def test_committed_results_files_pass_the_gate():
    """Every tagged number in every gated file matches the source it names."""
    try:
        load_claims()
    except RuntimeError as e:
        pytest.skip(str(e))
    assert check() == []


# ---------------------------------------------------------------------------------------
# The public face is under the same gate -- demonstrated, not asserted
# ---------------------------------------------------------------------------------------

@pytest.mark.parametrize("filename", ["README.md", "AUTHOR_HANDOFF.md"])
def test_gate_catches_a_mislabelled_source_in_the_public_files(tmp_path, filename):
    """The section-2 defect, planted in the two files most readers will ever open.

    Extending a gate to new files is worth nothing until it is shown to reject a wrong tag
    IN those files, so this repeats the mislabelled-source demonstration with the document
    named README.md and AUTHOR_HANDOFF.md rather than RESULTS_STAGEX.md.
    """
    p = tmp_path / filename
    tag = GOOD_TAG.replace(f"array={ARRAY_A}", f"array={ARRAY_B}").replace(
        f"sha256={SHA_A}", f"sha256={SHA_B}")
    p.write_text(f"<!--prov {tag} -->\nthe value is 1.25 nats\n")

    failures = check([p], _claim())
    assert failures, f"gate did NOT catch a mislabelled source in {filename}"
    assert any("MISLABELLED SOURCE" in f for f in failures), failures
    assert any(filename in f for f in failures), (
        f"the failure does not name {filename}, so a reader could not locate it: {failures}")


def test_the_public_files_are_actually_in_the_gate_globs():
    """Coverage must not regress silently.

    The demonstration above proves the gate rejects a bad tag in a file NAMED README.md; this
    proves the real README.md is among the files the gate walks. Both are needed: the first
    without the second would be a gate aimed at nothing.
    """
    from check_provenance import RESULTS_GLOBS
    from qsent.pins import repo_root

    walked = {p.name for g in RESULTS_GLOBS for p in repo_root().glob(g)}
    for required in ("README.md", "AUTHOR_HANDOFF.md", "PREREGISTRATION.md",
                     "RESULTS_STAGE0.md", "RESULTS_STAGE1_5.md"):
        assert required in walked, f"{required} is not covered by the provenance gate"
