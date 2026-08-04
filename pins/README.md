# `pins/` — the cross-repo reproducibility contract

This arm depends on a second repository (`quantum-structure-sae`) and on binary artifacts
that live in **neither** repo. Three things are pinned. All three are asserted by
`tests/test_cross_repo_pin.py`, which runs in the Stage 0 gate.

> **Anonymizability note.** Per `CLAUDE.md`, no absolute paths, usernames, or hostnames
> appear in this file. Concrete machine-local values live in the gitignored `.env.local`
> (template: `.env.local.example`). Where this document must refer to a location, it names
> an environment variable, not a path.

---

## 1. Code — pinned by git submodule

`submodules/quantum-structure-sae` @ **`0c4e6e4a8a0fb68aec6820eea8c7eed49d05a539`**

The SHA is recorded as a gitlink object in this repo's own tree, so it is verifiable with
no network and no package metadata:

```bash
git ls-tree HEAD submodules/quantum-structure-sae
# 160000 commit 0c4e6e4a8a0fb68aec6820eea8c7eed49d05a539	submodules/quantum-structure-sae
```

Imports are made to work with `pip install -e submodules/quantum-structure-sae`, which does
**not** weaken the pin — the SHA is still owned by git, not by pip. Bumping the pin is an
explicit commit that changes one gitlink and is reviewable as such.

**The data-generation code is never copied or forked into this repo.** If the two generators
could drift, the comparability claim to the SAE arm is dead.

---

## 2. Ensemble — `ensemble.sha256` (15 entries)

SHA-256 of every cached disorder-realization file, paths relative to **`$QSAE_ARTIFACTS`**.
Covers `data/tfim_L8_N50k*.pt` (the 50,000-realization training ensembles, seeds 1–10 plus
the base and `hcrit` variants) and `data/ra03_states_L8_N800_s4{2,3,4}.pt`.

**These are loaded, never regenerated.** The ensemble is *nominally* reproducible — the
generator uses `np.random.default_rng(seed)` (`src/qsae/reverse_arrow/data.py:266`), whose
bit-generator stream NumPy's policy treats as stable — but **regeneration is forbidden as a
load path**. This is recorded as provenance, not as a fallback. A regenerated ensemble would
carry different `δ_r` values and would silently invalidate every realization-disjoint split
and every δ-bin count, with no error raised. The loader hashes on read and **raises** on
mismatch; it never warns and continues.

---

## 3. Checkpoints — `checkpoints.sha256` (20 entries)

SHA-256 of `runs/ms_trained/seed{1..10}/best.pt` and their `config.json` siblings (the
configs are pinned too, so a checkpoint can never be silently reinterpreted under different
training metadata). Paths relative to **`$QSAE_ARTIFACTS`**.

These are the 10 checkpoints on which `phase06`'s `long_range_zz` incremental-R² beyond
poly2-h — **0.0283 ± 0.0030 [0.0231, 0.0320]** — was measured. They are **not reproducible**;
no backup or hashing scheme changes that, so they are treated as irreplaceable.

### Known limitation, recorded deliberately

The `+0.029` figure in `results/legacy/ra08_scaling.md` was measured on **different**
models — `n_train=15000, epochs=100, seed=0`, versus `ms_trained`'s `n_train=50000,
epochs=200, seeds 1–10` — and **those checkpoints were never saved**. They cannot be pinned,
because they no longer exist. Any comparison against `+0.029` is therefore a comparison
across training configurations and must be labeled as one. The number anchored to artifacts
that still exist is `phase06`'s `0.0283 ± 0.0030`. See `PLAN.md` §3.5.0.

---

## 4. Where the artifacts live

| Copy | Location | Verified |
|---|---|---|
| Working root | `$QSAE_ARTIFACTS` (outside both repos) | `sha256sum -c`, 161/161 OK |
| Local archive | `$QSAE_BACKUPS` (timestamped `.tar.gz` + manifest + restore readme) | extracted and `sha256sum -c`, 161/161 OK |
| Off-machine | rclone remote, `quantum-structure-backup/` | downloaded, extracted, `sha256sum -c`, 161/161 OK |

The off-machine copy is the only one that survives loss of the primary disk; the working
root and the local archive share it.

## 5. Verifying by hand

```bash
set -a; . ./.env.local; set +a
cd "$QSAE_ARTIFACTS" && sha256sum -c "$OLDPWD/pins/ensemble.sha256" \
                     && sha256sum -c "$OLDPWD/pins/checkpoints.sha256"
git ls-tree HEAD submodules/quantum-structure-sae
```
