"""
Hash-verified access to pinned artifacts and published constants.

Every number that is *supposed* to come from a pinned artifact must be obtained through
this module -- either loaded at runtime or asserted against the artifact. Nothing is
inlined on the grounds of being obviously right.

The rule this enforces, learned the hard way in Stage 0: **assertions that hold for any
input cannot detect wrong input.** The ED/free-fermion gate passes for any `h` vector, so
fabricated "pinned realizations" sailed through every physics test. Identity of the input
has to be checked separately from correctness of the computation.

Loaders raise on hash mismatch. They never regenerate and never warn-and-continue: a
regenerated ensemble carries different delta_r values and silently breaks every
realization-disjoint split and delta-bin count.
"""

from __future__ import annotations

import hashlib
import os
import re
from pathlib import Path

import numpy as np

__all__ = [
    "artifact_root", "repo_root", "sha256_file", "verify_pin",
    "load_ensemble", "ensemble_meta", "load_checkpoint",
    "published_constant", "published_per_seed", "submodule_sha",
]


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def artifact_root() -> Path:
    """$QSAE_ARTIFACTS, or the value in .env.local. Raises if unset or missing."""
    root = os.environ.get("QSAE_ARTIFACTS")
    if not root:
        env = repo_root() / ".env.local"
        if env.exists():
            for line in env.read_text().splitlines():
                if line.startswith("QSAE_ARTIFACTS="):
                    root = line.split("=", 1)[1].strip()
                    break
    if not root:
        raise RuntimeError(
            "QSAE_ARTIFACTS is not set. Copy .env.local.example to .env.local and point it "
            "at the pinned artifact root (see pins/README.md)."
        )
    p = Path(root)
    if not p.is_dir():
        raise RuntimeError(f"QSAE_ARTIFACTS points at a non-directory: {p}")
    return p


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _manifest(name: str) -> dict[str, str]:
    out: dict[str, str] = {}
    for line in (repo_root() / "pins" / name).read_text().splitlines():
        if line.strip():
            digest, rel = line.split(None, 1)
            out[rel.strip()] = digest
    return out


def verify_pin(rel_path: str, manifest: str) -> Path:
    """Resolve `rel_path` under $QSAE_ARTIFACTS and verify its SHA-256. Raises on mismatch."""
    expected = _manifest(manifest).get(rel_path)
    if expected is None:
        raise KeyError(f"{rel_path} is not listed in pins/{manifest}")
    path = artifact_root() / rel_path
    if not path.exists():
        raise FileNotFoundError(f"pinned artifact missing: {path}")
    actual = sha256_file(path)
    if actual != expected:
        raise RuntimeError(
            f"PINNED ARTIFACT MISMATCH for {rel_path}\n"
            f"  expected {expected}\n  actual   {actual}\n"
            "Refusing to continue. Do not regenerate -- restore from backup (pins/README.md)."
        )
    return path


def load_ensemble(seed: int) -> dict:
    """Load a pinned disorder ensemble after verifying its hash."""
    import torch
    rel = f"data/tfim_L8_N50k_seed{seed}.pt"
    return torch.load(verify_pin(rel, "ensemble.sha256"), map_location="cpu", weights_only=False)


def ensemble_meta(seed: int) -> dict:
    """The generating parameters (L, J, h_min, h_max, seed) as recorded in the artifact.

    These are the authority for h_min/h_max -- they must never be hardcoded, because
    sigma_lnh and therefore every delta_r depends on them.
    """
    return load_ensemble(seed)["meta"]


def load_checkpoint(seed: int) -> dict:
    import torch
    rel = f"runs/ms_trained/seed{seed}/best.pt"
    return torch.load(verify_pin(rel, "checkpoints.sha256"), map_location="cpu", weights_only=False)


def submodule_sha() -> str:
    """The pinned quantum-structure-sae SHA, read from this repo's own tree."""
    import subprocess
    out = subprocess.run(
        ["git", "ls-tree", "HEAD", "submodules/quantum-structure-sae"],
        cwd=repo_root(), capture_output=True, text=True, check=True).stdout
    return out.split()[2]


# name -> (file, section heading it must appear under, row regex)
#
# The section anchor is not decorative. phase06 reports `long_range_zz` in BOTH a partial
# correlation table (0.560±0.046) and an incremental-R2 table (0.0283±0.0030). An
# unanchored row regex silently returns 0.560 -- which is what the first version of this
# code did, and which only surfaced because the tolerance is derived here rather than
# hardcoded.
_PUBLISHED = {
    "phase06_lrzz_incr_r2_mean": (
        "results/phase06_multiseed_trained.md",
        "Incremental R² beyond poly2-h",
        r"\|\s*long_range_zz\s*\|\s*(0\.\d+)±"),
    "phase06_lrzz_incr_r2_sd": (
        "results/phase06_multiseed_trained.md",
        "Incremental R² beyond poly2-h",
        r"\|\s*long_range_zz\s*\|\s*0\.\d+±(0\.\d+)\s*\["),

    # --- the phase06 PRIOR for Stage 2's primary endpoint family ------------------------
    # `entropy` appears as a row in BOTH the partial-correlation table and the
    # incremental-R2 table, with different values. The section heading is the only thing
    # separating them, which is the same anchor the 0.560/0.0283 error taught.
    "phase06_entropy_incr_r2_trained_mean": (
        "results/phase06_multiseed_trained.md", "Incremental R² beyond poly2-h",
        r"\|\s*entropy\s*\|\s*(0\.\d+)±"),
    "phase06_entropy_incr_r2_trained_sd": (
        "results/phase06_multiseed_trained.md", "Incremental R² beyond poly2-h",
        r"\|\s*entropy\s*\|\s*0\.\d+±(0\.\d+)\s*\["),
    "phase06_entropy_incr_r2_random_mean": (
        "results/phase06_multiseed_trained.md", "Incremental R² beyond poly2-h",
        r"\|\s*entropy\s*\|[^|]*\|\s*(0\.\d+)±"),
    "phase06_entropy_incr_r2_random_sd": (
        "results/phase06_multiseed_trained.md", "Incremental R² beyond poly2-h",
        r"\|\s*entropy\s*\|[^|]*\|\s*0\.\d+±(0\.\d+)\s*\("),
    "phase06_entropy_incr_r2_sep_sd": (
        "results/phase06_multiseed_trained.md", "Incremental R² beyond poly2-h",
        r"\|\s*entropy\s*\|[^|]*\|[^|]*\|\s*([+-]\d+\.\d+)\s*\|"),
    "phase06_lrzz_incr_r2_sep_sd": (
        "results/phase06_multiseed_trained.md", "Incremental R² beyond poly2-h",
        r"\|\s*long_range_zz\s*\|[^|]*\|[^|]*\|\s*([+-]\d+\.\d+)\s*\|"),
    "phase06_entropy_partial_r_trained_mean": (
        "results/phase06_multiseed_trained.md", "Partial correlation | poly2-h",
        r"\|\s*entropy\s*\|\s*(0\.\d+)±"),
    "phase06_entropy_partial_r_trained_sd": (
        "results/phase06_multiseed_trained.md", "Partial correlation | poly2-h",
        r"\|\s*entropy\s*\|\s*0\.\d+±(0\.\d+)\s*\["),
    "phase06_entropy_partial_r_sep_sd": (
        "results/phase06_multiseed_trained.md", "Partial correlation | poly2-h",
        r"\|\s*entropy\s*\|[^|]*\|[^|]*\|\s*([+-]\d+\.\d+)\s*\|"),
}


# name -> (file, section heading, line anchor, per-entry pattern, expected count)
#
# The SAME trap as _PUBLISHED, one level down. The "Per-seed transparency" section carries
# TWO per-seed lines: partial-r (s1:0.512, ...) and incremental R² (s1:0.023, ...). A pattern
# matching `s(\d+):(0\.\d+)` without the line anchor returns the PARTIAL CORRELATION series --
# the 0.560-for-0.0283 error reproduced exactly, in the numbers R1 is paired against. The
# line anchor is therefore load-bearing, and tests/test_r1_gate.py demonstrates that removing
# it returns the wrong series rather than raising.
_PUBLISHED_SERIES = {
    "phase06_lrzz_incr_r2_per_seed": (
        "results/phase06_multiseed_trained.md",
        "## Per-seed transparency",
        r"\*\*incremental R²\*\* per-seed:\s*(.+)",
        r"s(\d+):\s*(0\.\d+)",
        10),
}


def _pinned_text(rel: str) -> str:
    return (repo_root() / "submodules" / "quantum-structure-sae" / rel).read_text()


def _section_of(text: str, heading: str, rel: str) -> str:
    if heading not in text:
        raise RuntimeError(f"section {heading!r} not found in pinned {rel}")
    return text.split(heading, 1)[1].split("\n## ", 1)[0]


def published_per_seed(name: str) -> dict[int, float]:
    """Read a published PER-SEED series out of the pinned submodule. Nothing is typed.

    Raises unless the line anchor matches exactly once and yields exactly the expected number
    of entries with distinct seed ids. R1 pairs on seed identity, so a series that silently
    came from the wrong line, or lost an entry, would corrupt the pairing rather than fail.
    """
    rel, heading, line_pat, entry_pat, expected = _PUBLISHED_SERIES[name]
    section = _section_of(_pinned_text(rel), heading, rel)

    lines = re.findall(line_pat, section)
    if len(lines) != 1:
        raise RuntimeError(
            f"expected exactly one line matching {line_pat!r} under {heading!r} in {rel}, "
            f"got {len(lines)}")

    entries = re.findall(entry_pat, lines[0])
    if len(entries) != expected:
        raise RuntimeError(
            f"expected {expected} per-seed entries for {name}, got {len(entries)}: {entries}")

    out = {int(s): float(v) for s, v in entries}
    if len(out) != expected:
        raise RuntimeError(f"duplicate seed ids in {name}: {entries}")
    return out


def published_constant(name: str) -> float:
    """Read a published constant out of the pinned submodule rather than inlining it.

    R1's tolerance is derived from phase06's published mean and sd. Hardcoding them would
    reintroduce exactly the failure mode this module exists to prevent.
    """
    rel, heading, pattern = _PUBLISHED[name]
    text = (repo_root() / "submodules" / "quantum-structure-sae" / rel).read_text()
    if heading not in text:
        raise RuntimeError(f"section {heading!r} not found in pinned {rel}")
    section = text.split(heading, 1)[1].split("\n## ", 1)[0]
    hits = re.findall(pattern, section)
    if len(hits) != 1:
        raise RuntimeError(
            f"expected exactly one match for {name} under {heading!r} in {rel}, got {len(hits)}")
    return float(hits[0])
