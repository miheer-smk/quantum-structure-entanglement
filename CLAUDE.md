# CLAUDE.md — quantum-structure-entanglement

## Attribution (non-negotiable)

**Never add a `Co-Authored-By` trailer or any AI-attribution line to commit messages or PR
bodies.** No `🤖 Generated with …`, no `Generated with [Claude Code]`, no bot attribution of
any kind. This repository has exactly one author and one contributor.

Enforced in three places, deliberately redundant:
1. `attribution.commit` / `attribution.pr` set to `""` in user-scope `~/.claude/settings.json`.
2. A repo-local `.git/hooks/commit-msg` that strips the trailers regardless of tool version.
3. This file.

Never pass `--author` or `--committer`. Never add a co-author for anyone unless the author
explicitly asks.

## Anonymizability

No name, email, institution, supervisor, cluster hostname, HPC username, or `/home/<user>/…`
path in code, comments, docstrings, configs, SLURM scripts, figure metadata, or filenames.
Use environment variables (`$QSAE_ARTIFACTS`, `$SCRATCH`, `$HPC_USER`) and a gitignored
`.env.local`. The repo must be shareable as an anonymized snapshot without rewriting history.

## Scientific conduct

- Stage gates are stop signs. Do not begin a stage before its predecessor's gate is green and
  the author has read that stage's results file.
- Never silently retune. Any change to a hyperparameter, split, or estimator that moves a
  result goes in `DEVIATIONS.md` with a reason and a date.
- Nulls are written up as nulls, at full precision, never tuned away.
- Banned phrasings, anywhere including commit messages and figure captions: "the transformer
  is a quantum computer", "quantum advantage", "emergent quantum behaviour", "the network
  learns quantum mechanics".

## Cross-repo contract

The physics and data-generation code is **not** copied into this repo. It is pinned as a
submodule at an exact commit SHA (see `pins/README.md`). Artifacts are referenced by
content hash under `$QSAE_ARTIFACTS` and are **never regenerated** — regeneration silently
changes every `δ_r` value and breaks realization-disjoint splits.
