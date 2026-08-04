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

## Standing rule: a check is not a gate until it is shown able to fail

**Any validation check must be demonstrated capable of FAILING on the specific error it
targets, before it counts as a gate.** Write the falsifying case, run it, and keep it in the
suite next to the check it justifies.

This rule exists because three "looked right, unverified source" errors reached committed
code in Stage 0 alone, each passing every test that existed at the time:

| Error | What it slipped past | What actually caught it |
|---|---|---|
| Fabricated `h` vectors labelled as pinned realizations | the ED/free-fermion gate, which holds for *any* `h` | an out-of-band provenance comparison |
| Regex reading `0.560` (partial correlation) as the R1 mean `0.0283` (incremental R²) | would have passed had the value been hardcoded | deriving the constant instead of typing it |
| Mirrored entropy profile (site-ordering) | the `c_eff` fit — identical `c_eff` to 6.7e-16 and identical residuals, because Calabrese–Cardy is symmetric under `l -> L-l` | an explicit asymmetric-realization orientation test |

The common shape: **an assertion that holds for any input cannot detect wrong input.**
Correctness of a computation and identity of its inputs are independent properties requiring
independent checks. A green suite is evidence only about the errors its checks can express.

Concretely, in this repo: `test_uniform_cases_are_degenerate`,
`test_orientation_gate_catches_a_mirrored_provider`, `test_ceff_fit_is_blind_to_mirroring`,
and `test_no_unverified_inlined_float_arrays_in_tests` all exist to satisfy this rule rather
than to test physics.

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
