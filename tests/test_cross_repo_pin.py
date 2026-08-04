"""Cross-repo pin assertions + the fabricated-constant lint. Runs in every stage gate."""

from __future__ import annotations

import ast
import os
from pathlib import Path

import numpy as np
import pytest

from qsent.disorder import sigma_lnh_uniform
from qsent.pins import (
    artifact_root, ensemble_meta, published_constant, repo_root,
    sha256_file, submodule_sha, verify_pin, _manifest,
)

PINNED_SUBMODULE_SHA = "0c4e6e4a8a0fb68aec6820eea8c7eed49d05a539"
needs_artifacts = pytest.mark.skipif(
    not os.environ.get("QSAE_ARTIFACTS"), reason="QSAE_ARTIFACTS not configured")


def test_submodule_sha_matches_declared_pin():
    assert submodule_sha() == PINNED_SUBMODULE_SHA


@needs_artifacts
def test_every_pinned_artifact_hash_matches():
    for manifest in ("ensemble.sha256", "checkpoints.sha256"):
        for rel in _manifest(manifest):
            verify_pin(rel, manifest)          # raises on mismatch


@needs_artifacts
def test_hash_mismatch_raises_rather_than_warning():
    """The loader must fail loudly. A warn-and-continue path would be worse than useless."""
    manifest = _manifest("ensemble.sha256")
    rel = next(iter(manifest))
    real = sha256_file(artifact_root() / rel)
    assert len(real) == 64 and real == manifest[rel]
    with pytest.raises(KeyError):
        verify_pin("data/not-a-pinned-file.pt", "ensemble.sha256")


@needs_artifacts
def test_disorder_parameters_come_from_the_artifact_not_a_literal():
    """h_min/h_max drive sigma_lnh and therefore every delta_r. They must be read, not assumed."""
    meta = ensemble_meta(1)
    assert meta["h_min"] == 0.1 and meta["h_max"] == 2.0 and meta["J"] == 1.0
    m1, sd = sigma_lnh_uniform(meta["h_min"], meta["h_max"])
    assert abs(m1 - (-0.149183)) < 1e-6
    assert abs(sd - 0.709086) < 1e-6
    # and the closed form must match the actual pinned sample
    from qsent.pins import load_ensemble
    lnh = np.log(np.asarray(load_ensemble(1)["h_train"], dtype=np.float64))
    assert abs(float(lnh.mean()) - m1) < 5e-3


def test_r1_tolerance_is_derived_from_the_pinned_publication():
    """R1's frozen window must be computed from phase06's published numbers, not typed in."""
    mean = published_constant("phase06_lrzz_incr_r2_mean")
    sd = published_constant("phase06_lrzz_incr_r2_sd")
    assert abs(mean - 0.0283) < 1e-9, f"published mean drifted: {mean}"
    assert abs(sd - 0.0030) < 1e-9, f"published sd drifted: {sd}"
    lo, hi = mean - 2 * sd, mean + 2 * sd
    assert abs(lo - 0.0223) < 1e-9 and abs(hi - 0.0343) < 1e-9


# --------------------------------------------------------------------------------------
# Lint: no new inlined float arrays in tests/
# --------------------------------------------------------------------------------------

ALLOWED_INLINE = {
    # name -> the test that asserts it against the pinned artifact
    "SPANNING": "test_spanning_realizations_match_pinned_ensemble",
    "SPANNING_DELTA": "test_spanning_realizations_match_pinned_ensemble",
    "SPANNING_INDICES": "test_spanning_realizations_match_pinned_ensemble",
}
MIN_FLAGGED_LEN = 4


def _inlined_numeric_collections(path: Path):
    tree = ast.parse(path.read_text())
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign):
            continue
        names = [t.id for t in node.targets if isinstance(t, ast.Name)]
        for value in ast.walk(node.value):
            if isinstance(value, (ast.List, ast.Tuple)) and len(value.elts) >= MIN_FLAGGED_LEN:
                if all(isinstance(e, ast.Constant) and isinstance(e.value, (int, float))
                       for e in value.elts):
                    yield names, len(value.elts)
                    break


def test_no_unverified_inlined_float_arrays_in_tests():
    """Fails on any new inlined numeric array in tests/ that nothing checks against a pin.

    Stage 0 produced fabricated 'pinned realizations' that passed every physics assertion.
    An inlined array is data, and data needs provenance, so each one must either be removed
    or added to ALLOWED_INLINE alongside the test that verifies it.
    """
    offenders = []
    for path in sorted((repo_root() / "tests").glob("*.py")):
        for names, n in _inlined_numeric_collections(path):
            for name in names:
                if name not in ALLOWED_INLINE:
                    offenders.append(f"{path.name}::{name} ({n} numeric literals)")
    assert not offenders, (
        "Inlined numeric arrays without pinned provenance:\n  " + "\n  ".join(offenders)
        + "\nEither load them via qsent.pins, or add them to ALLOWED_INLINE together with "
          "a test asserting them against the pinned artifact.")


def test_allowed_inline_entries_have_a_live_verifier():
    """ALLOWED_INLINE must not become a place to park unverified constants."""
    src = (repo_root() / "tests" / "test_exact_entropy.py").read_text()
    for name, verifier in ALLOWED_INLINE.items():
        assert f"def {verifier}(" in src, f"{name} claims {verifier}, which does not exist"
