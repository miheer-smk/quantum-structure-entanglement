"""
Provenance header stamped onto every generated artifact.

Records what produced a number: the git SHA of this repo, whether anything that could affect
the computation differed from it, the pinned submodule SHA, the SHA-256 of every artifact
actually read, the RNG seeds used, a hostname *class* (never a hostname -- see the
anonymizability rule in CLAUDE.md), wall-clock, and the SLURM job id.

`slurm_job_id` is read from the environment and is `None` when there is none. It is never
invented: a fabricated job id would be exactly the class of unverifiable provenance this
repository exists to prevent.

ONE THING THIS HEADER CANNOT DO, STATED SO IT IS NOT MISREAD
------------------------------------------------------------
`repo_git_sha` names the commit the generating CODE was at, which is necessarily the commit
BEFORE the one that carries this file: a file recording a hash cannot contain the hash of a
tree containing itself. The honest reading of a header is "produced by the tree at <sha>,
with no uncommitted inputs", and the commit that adds the file should therefore change
generated outputs only. That is a property of how the file is committed, not something the
header can assert about itself.
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


#: Generated artifacts. Regenerating one of these does not change what produced the next one,
#: so they are excluded from the INPUT-dirtiness flag and reported separately.
OUTPUT_PREFIX = "scripts/out/"


def _porcelain() -> list[str]:
    out = subprocess.run(["git", "status", "--porcelain"], cwd=repo_root(),
                         capture_output=True, text=True, check=True)
    return [ln for ln in out.stdout.splitlines() if ln.strip()]


def _paths(lines: list[str]) -> list[str]:
    """Paths from `git status --porcelain`, taking the destination half of any rename."""
    return [ln[3:].split(" -> ")[-1].strip().strip('"') for ln in lines]


def _git_dirty() -> tuple[bool, bool]:
    """(inputs differ from the named commit, generated outputs differ from it).

    Split because generated outputs are not inputs, and conflating them made the flag
    unusable in exactly the case it exists for. `regen_stage0.py` READS
    `scripts/out/precision_audit.json`, so regenerating the pair in dependency order
    guaranteed the second file recorded `dirty: true` -- reporting a tree that had not been
    edited by anyone as modified, and doing so on the artifact whose entire job is to say
    what produced it. The provenance artifacts were themselves mis-provenanced.

    `repo_working_tree_dirty` therefore now means: **does any file that could have changed
    this computation differ from the commit named above.** The raw fact is not hidden --
    output dirtiness is reported alongside it, so nothing is concealed by the split.
    """
    paths = _paths(_porcelain())
    return (any(not p.startswith(OUTPUT_PREFIX) for p in paths),
            any(p.startswith(OUTPUT_PREFIX) for p in paths))


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
    inputs_dirty, outputs_dirty = _git_dirty()
    return {
        "repo_git_sha": _git_sha(),
        "repo_working_tree_dirty": inputs_dirty,
        "generated_outputs_dirty": outputs_dirty,
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
