#!/usr/bin/env bash
# Run a command inside the pinned analysis environment.
#
#   env/run.sh pytest -q
#   env/run.sh python scripts/regen_stage0.py
#
# Machine-local paths come from the gitignored .env.local (see .env.local.example); none
# appear in this file, per the anonymizability rule in CLAUDE.md.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
IMAGE="${QSENT_IMAGE:-qsent-env:0.1.0}"

if [[ ! -f "$REPO/.env.local" ]]; then
  echo "error: .env.local not found. Copy .env.local.example and set QSAE_ARTIFACTS." >&2
  exit 1
fi
set -a; . "$REPO/.env.local"; set +a

if [[ -z "${QSAE_ARTIFACTS:-}" || ! -d "${QSAE_ARTIFACTS}" ]]; then
  echo "error: QSAE_ARTIFACTS is unset or not a directory." >&2
  exit 1
fi

# Build on first use. The base is digest-pinned in env/Dockerfile, so this is deterministic
# apart from the wheel downloads, which are themselves version-pinned.
if ! docker image inspect "$IMAGE" >/dev/null 2>&1; then
  echo "building $IMAGE from env/Dockerfile ..." >&2
  docker build -q -t "$IMAGE" "$REPO/env" >&2
fi

# The artifact root is mounted READ-ONLY. The pins contract says artifacts are loaded and
# never regenerated; a read-only mount makes that a property of the sandbox rather than a
# promise in a document.
exec docker run --rm \
  -v "$REPO":/work \
  -v "$QSAE_ARTIFACTS":/artifacts:ro \
  -e QSAE_ARTIFACTS=/artifacts \
  -e PYTHONPATH=/work/src:/work/tests:/work/submodules/quantum-structure-sae/src \
  -e HOME=/tmp \
  -u "$(id -u):$(id -g)" \
  -w /work \
  "$IMAGE" "$@"
