"""
Provenance header stamped onto every generated artifact.

Records what produced a number: the git SHA of this repo, the pinned submodule SHA, the
SHA-256 of every artifact actually read, the RNG seeds used, a hostname *class* (never a
hostname -- see the anonymizability rule in CLAUDE.md), wall-clock, and the SLURM job id.

`slurm_job_id` is read from the environment and is `None` when there is none. It is never
invented: a fabricated job id would be exactly the class of unverifiable provenance this
repository exists to prevent.
"""

from __future__ import annotations

import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from qsent.pins import repo_root, sha256_file, submodule_sha, artifact_root


def _git_sha() -> str:
    out = subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo_root(),
                         capture_output=True, text=True, check=True)
    return out.stdout.strip()


def _git_dirty() -> bool:
    out = subprocess.run(["git", "status", "--porcelain"], cwd=repo_root(),
                         capture_output=True, text=True, check=True)
    return bool(out.stdout.strip())


def _hostname_class() -> str:
    """A CLASS of machine, never the hostname. `platform.node()` is deliberately not used."""
    return f"{platform.system().lower()}-{platform.machine()}-{os.cpu_count()}core"


def provenance(seeds: dict[str, int] | None = None,
               artifacts: list[str] | None = None) -> dict[str, Any]:
    """Build the provenance block. `artifacts` are paths relative to $QSAE_ARTIFACTS."""
    hashes = {}
    for rel in sorted(artifacts or []):
        p = artifact_root() / rel
        hashes[rel] = sha256_file(p) if p.exists() else "MISSING"
    return {
        "repo_git_sha": _git_sha(),
        "repo_working_tree_dirty": _git_dirty(),
        "submodule_sha": submodule_sha(),
        "artifact_sha256": hashes,
        "rng_seeds": seeds or {},
        "hostname_class": _hostname_class(),
        "python": sys.version.split()[0],
        "numpy": __import__("numpy").__version__,
        "torch": __import__("torch").__version__,
        "base_image_digest": (repo_root() / "env" / "base-image.digest").read_text().strip(),
        "wall_clock_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),   # None when there is none
    }


def write_json(payload: dict[str, Any], out_name: str) -> Path:
    out_dir = repo_root() / "scripts" / "out"
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / out_name
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return path
