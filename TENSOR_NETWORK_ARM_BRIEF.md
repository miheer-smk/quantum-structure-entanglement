# Tensor Network Arm — Project Brief

> Spec of record for the tensor-network / entanglement-structure experimental arm of
> `quantum-structure-sae`. Saved verbatim as provided by the author. Downstream results
> files must record this file's commit SHA alongside `PREREGISTRATION.md`'s.

---

## Part 1 — Authorship, repo, and git hygiene (non-negotiable, do this first)

### 1.1 Attribution must be off before the first commit

Claude Code appends attribution to commits and PRs by default. Turn it off with the
attribution setting (this is the current key; it replaced the older `includeCoAuthoredBy`
— set one or the other, never both):

```json
// ~/.claude/settings.json  (user scope, applies to all projects)
{
  "attribution": {
    "commit": "",
    "pr": ""
  }
}
```

Then belt-and-braces it with a repo-local hook, so the rule holds regardless of tool
version or who runs what:

```bash
cat > .git/hooks/commit-msg << 'EOF'
#!/bin/sh
sed -i.bak -e '/^Co-Authored-By:/d' -e '/Generated with \[Claude Code\]/d' -e '/🤖 Generated with/d' "$1"
rm -f "$1.bak"
EOF
chmod +x .git/hooks/commit-msg
```

Add the same instruction to `CLAUDE.md` in the repo root: "Never add a Co-Authored-By
trailer or any AI-attribution line to commit messages or PR bodies."

### 1.2 Identity must resolve to one person

```bash
git config user.name  "<my name exactly as on GitHub>"
git config user.email "<the email verified on my GitHub account>"
```

The GitHub contributors panel is built from commit author and committer emails. If the
email isn't the one attached to my GitHub account, the commits show up as an unlinked
ghost contributor — which is the same problem in a different costume. Never pass
`--author` or `--committer`. Never add a co-author trailer for anyone — not Miheer, not
my guide — unless I explicitly ask.

### 1.3 Repo creation rules

- Ask me first whether this is a new repo or a branch. My recommendation is a branch
  (`tensor-network-arm`) on the existing `quantum-structure-sae` repo, because Stage 4
  requires the SAE results in-tree and a split repo makes the cross-reference painful.
  Do not decide this for me.
- If a new repo is warranted: `gh repo create <name> --private` only. Never `--public`,
  never `--internal`.
- No collaborators, no teams, no org transfer, no GitHub Apps, no Actions workflows that
  commit or comment as a bot.
- Do not push anything until I say so. Work on a branch. Never force-push to `main`.
  Never rewrite history that has been pushed.

### 1.4 Verify, then show me

After the first commit, run and show me the output of:

```bash
git log --format='%an <%ae> | committer: %cn <%ce>' | sort -u
git log --format='%B' | grep -iE 'co-authored|claude|generated with' || echo "CLEAN"
```

The first command must show exactly one identity. The second must print `CLEAN`.

### 1.5 Keep the repo anonymizable and secret-free

- No name, email, institution, supervisor, cluster hostname, HPC username, or
  `/home/<user>/...` path in code, comments, docstrings, configs, SLURM scripts, figure
  metadata, or filenames. Use environment variables (`$SCRATCH`, `$HPC_USER`) and a
  gitignored `.env.local`. This is not paranoia — it means I can share an anonymized
  snapshot for review without rewriting history.
- Add a pre-commit secret scan (gitleaks or detect-secrets) and a check for the strings
  above.
- `LICENSE` and `CITATION.cff`: me as sole author. Do not pick a license unilaterally —
  flag it and ask. For unpublished research, "all rights reserved" (i.e. no license file)
  is often correct until acceptance.

---

## Part 2 — Context you inherit (read the code; do not re-derive any of this)

This is the next experimental arm of an existing project, `quantum-structure-sae`, not a
new codebase. That project trains transformers on transverse-field Ising model (TFIM)
ground-state data and fits sparse autoencoders (SAEs) to the residual stream, asking
whether classical networks develop quantum-structured internal representations.

Established results to build on, not re-litigate:

- **L-scaling robustness:** SAE reconstruction gain of ≈ +0.028, stable across
  L = 8, 10, 12.
- **Mixed-field null:** the SAE advantage collapsed once the observable became trivial.
  This was reported as an honest negative result. The same standard binds this arm: a
  clean null is a publishable finding, and you will not tune toward a positive.
- A **disordered-longitudinal-field** experiment is planned in the SAE line. This arm
  must either run in parallel with it or reuse the identical data-generation pipeline, so
  the two arms are comparable on the same underlying physical systems. Prefer reuse; if
  you must fork the generator, document exactly why in `DEVIATIONS.md`.
- Closest published reference: **Qi & Earls, arXiv:2607.01336**, on sparse autoencoders
  for neural quantum states. Verify the citation before using it and do not assert
  anything about its contents you have not read. Positioning: this arm extends that line
  from sparse dictionary features to entanglement structure, using a measurement
  instrument native to the physics rather than borrowed from ML interpretability.

**Stage 0 deliverable, before anything else:** an inventory of what the existing repo
actually gives you — model architecture, what the model's output head predicts
(amplitudes? energies? observables? next token?), the exact data format, the trained
checkpoints available, their seeds, and the SAE result artifacts. Everything downstream
branches on the output-head question.

---

## Part 3 — The scientific claim, sharpened

The vision brief's framing ("does entanglement entropy track anything real?") is correct
but under-specified for a reviewer. Sharpen it into pre-registered, falsifiable
hypotheses. Write these into `PREREGISTRATION.md`, commit it before running Stage 2, and
record the commit SHA in every downstream results file.

**H1 — Criticality signature.** The half-chain entropy of the model's internal
representation, `S_model(ℓ = L/2; h)`, is maximized at a field value `h*(L)` that drifts
monotonically toward `h = 1` as L grows, matching the exact ground state's finite-size
pseudo-critical drift within bootstrap CI.
*Falsifier:* no peak, a peak at an L-independent wrong value, or a drift in the wrong
direction.

**H2 — Log-law vs area-law.** At `h = 1`, `S_model(ℓ)` fits the Calabrese–Cardy form with
an effective central charge `c_eff` consistent with the Ising value `c = 1/2`
(pre-register the acceptance interval, e.g. `c_eff ∈ [0.35, 0.65]` with the CI excluding
0). Away from criticality the profile saturates (area law) with a saturation scale that
grows as `|h − 1|^(−ν)`, `ν = 1`.
*Falsifier:* `c_eff` CI includes 0, or excludes 1/2 by a wide margin, or the off-critical
profile fails to saturate.

This is the single most valuable upgrade over "correlation X." A recovered central charge
is a quantitative, universal, hard-to-fake number. A correlation coefficient is not.

**H3 — Depth profile.** The linearly decodable entanglement `S_decoded(ℓ; layer k)`
increases with depth and approaches `S_exact` in late layers. There exists an identifiable
layer `k*` by which the physical entanglement structure is present.
*Falsifier:* flat across depth, or non-monotone in a way not explained by architecture.

**H4 — SAE cross-reference (the connective tissue).** The layer of maximum SAE feature
gain coincides with the layer of steepest entanglement acquisition `ΔS_decoded`.
Pre-specify the test: Spearman rank correlation across layers between SAE gain and
`ΔS_decoded`, with the layer count as the multiplicity family.
*Falsifier:* rank correlation CI spanning zero — which would be a genuinely interesting
dissociation between dictionary-learning features and physics-native entanglement, and
should be written up as such.

**H5 — Null concordance.** In the regimes where the SAE advantage collapsed (mixed field /
trivial observable), entanglement tracking should collapse too. If it does not collapse,
that dissociation is the most interesting result in the paper and must not be buried.

**Overclaim ban.** These phrases never appear in any output, code comment, figure caption,
or commit message: "the transformer is a quantum computer", "quantum advantage",
"emergent quantum behaviour", "the network learns quantum mechanics". The claim is and
remains: a specific, measurable invariant of the represented state behaves in a specific,
predicted way.

---

## Part 4 — Method: three constructions, ranked

The vision brief proposes reshaping a `d_model` activation vector into a tensor chain and
SVD-ing it. That construction has a real weakness a physicist reviewer will hit
immediately: the residual stream has no canonical tensor-product structure. The
factorization of `d_model` into local dimensions, and the coordinate basis itself, are
arbitrary — unlike a spin chain, where locality fixes the tensor product. Entropy computed
that way is a property of an arbitrary reshaping, not an invariant of the representation.
So implement it, but not as the headline, and never without its null distribution.

Rank the constructions like this and tell me at plan time which are actually available:

### Construction A (best, if available) — amplitude-space entanglement

If the model emits amplitudes `ψ_θ(σ)` over spin configurations `σ ∈ {0,1}^L`
(neural-quantum-state style), the model defines a wavefunction and its entanglement is
unambiguous.

- Enumerate all `2^L` basis states (trivial for L ≤ 16: 65,536 states), evaluate `ψ_θ`,
  normalize.
- Reshape to a `2^ℓ × 2^(L−ℓ)` matrix at cut ℓ, SVD, get the Schmidt spectrum `{λ_i}`, and
  `S(ℓ) = −Σ λ_i² log λ_i²`.
- Basis-fixed, physically meaningful, directly comparable to `S_exact(ℓ)` at the same cut.

If there is no amplitude head, check whether one can be attached and fit cheaply. **Ask me
before adding a head** — it changes the model, and I want that decision to be mine.

### Construction B (the workhorse) — layer-resolved decodable entanglement

This is the depth curve the vision brief wants, made principled.

- For each layer k, fit a readout (linear first; small MLP only as a stated ablation) from
  the layer-k residual stream to the amplitude / target.
- Reconstruct the implied wavefunction from that readout, then compute its entanglement
  exactly as in Construction A.
- Result: `S_decoded(ℓ, h; layer k)` — "how much of the true entanglement structure is
  linearly decodable at depth k" — plus the gap to `S_exact`.
- Fit the probe on a **field-value-disjoint split** (see Part 8, item 7). The probe must
  never see the h values it is evaluated on.

This is also the natural place for the SAE cross-reference: SAE feature gain at layer k
versus `ΔS_decoded` at layer k, same axis, same models, same seeds.

### Construction C (secondary, needs its null) — d_model reshape MPS

As specified in the vision brief: factor `d_model = ∏ dᵢ`, reshape the residual vector
into a chain, sequential left-to-right SVD, entropy at each bond.

Non-negotiable conditions on reporting C:

- Report it only relative to a null distribution over ≥ 200 random permutations of the
  `d_model` coordinates and ≥ 50 random orthogonal rotations of the residual basis. Report
  the null spread as an error bar on every C-derived number.
- State plainly in the write-up that C measures low-rank structure under a chosen
  factorization, not a basis-independent invariant.
- If C's signal sits inside its own basis-rotation null, say so and drop it to an appendix.
  That is a result, not a failure.

### Ensemble-covariance variant (cheap sanity companion)

Across a sample ensemble, take the cross-covariance between block-A and block-B token
activations at cut ℓ; normalize its singular values to `p_i` and compute `−Σ p_i log p_i`.
It's a Schmidt-spectrum analogue that needs no amplitude head. Two cautions: entropy
estimators are biased at finite sample count, so report plug-in and a bias-corrected
estimate (Miller–Madow or jackknife) plus a matched-Gaussian surrogate baseline; and use a
shrinkage covariance estimator when `N_samples` is not ≫ block dimension.

---

## Part 5 — Ground truth (compute it two independent ways)

The TFIM is exactly solvable, so there is no excuse for an approximate reference.

**Method 1 — exact diagonalization.** For L ≤ 16, build
`H = −J Σ σᶻᵢσᶻᵢ₊₁ − h Σ σˣᵢ`, get the ground state by sparse eigensolve, reshape, SVD at
each cut. Exact.

**Method 2 — free fermions (Peschel).** Jordan–Wigner to quadratic fermions; build the
Majorana correlation matrix restricted to block A; its eigenvalues come in pairs `±i νₖ`;
then

`S_A = Σₖ H₂((1 + νₖ)/2)`, where `H₂(x) = −x log x − (1−x) log(1−x)`.

This scales to large L and is the reference for finite-size scaling.

**Stage 0 acceptance gate:** the two methods must agree to `< 1e-10` for L = 8, 10, 12 at
`h ∈ {0.5, 1.0, 2.0}`, at every cut. Freeze those numbers as golden values in
`tests/test_exact_entropy.py`. Do not touch a transformer activation until this test is
green.

Scaling forms to fit against, with boundary conditions stated explicitly in the code and
the caption (the prefactor differs):

- Critical, periodic: `S(ℓ) = (c/3) log[(L/π) sin(πℓ/L)] + const`
- Critical, open: `S(ℓ) = (c/6) log[(2L/π) sin(πℓ/L)] + const`
- Ising open chains show even–odd oscillations in ℓ; fit them or exclude them
  deliberately, don't let them silently bias `c_eff`.
- Off-critical: saturation at `S ~ (c/6) log ξ` with correlation length `ξ ~ |h − 1|^(−1)`.

---

## Part 6 — Controls (a result without these is not a result)

Every headline number needs its null. Implement all of these as first-class experiments,
not afterthoughts:

| ID | Control | What it rules out |
|----|---------|-------------------|
| C1 | Randomly initialized model, identical architecture | Architecture alone producing the profile |
| C2 | Model trained on scrambled targets | Optimization dynamics alone producing it |
| C3 | Trivially entangled data (deep paramagnetic / product-state limit) | Should give `S ≈ 0`; a nonzero floor is an artifact |
| C4 | Random `d_model` permutation null (Construction C) | Factorization-choice artifact |
| C5 | Random orthogonal rotation of residual basis (Construction C) | Basis-dependence artifact |
| C6 | Matched-spectrum Gaussian surrogate | Second-order statistics alone explaining it |
| C7 | Token-order shuffle before cutting | Spatial locality actually mattering |
| C8 | With / without dataset-mean and outlier-dimension ablation | Massive activations dominating the spectrum |
| C9 | Plain PCA-spectrum entropy per layer | **The most important one.** If PCA entropy reproduces the whole story, the tensor-network machinery adds nothing and the paper must say so plainly. |

---

## Part 7 — Staged execution with binding gates

Do not start a stage before its predecessor's gate is green and I have read the stage
results file.

**Stage 0 — Inventory and exact-solver validation.**
Inventory the existing repo (Part 2). Implement both ground-truth methods; pass the
`< 1e-10` agreement test. Reproduce one known analytical value independently.
*Gate:* `pytest tests/` green; `RESULTS_STAGE0.md` written, including which construction
from Part 4 is available and why.

**Stage 1 — Toy-case pipeline validation.**
Run the full entanglement pipeline on states where the answer is known in closed form
(product state → `S = 0`; Bell-like → `S = log 2`; exact TFIM ground state at L = 8 →
matches Stage 0). Confirm the entropy is computed from the untruncated Schmidt spectrum,
or that truncation error is reported.
*Gate:* every toy case within `1e-10`; `RESULTS_STAGE1.md`.

**Stage 2 — Baseline layer-wise scan.**
Full depth profile on the existing trained checkpoints, L = 8, 10, 12, ≥ 5 seeds, with all
Part 6 controls.
*Gate:* every reported curve has its null overlaid; `RESULTS_STAGE2.md`.
`PREREGISTRATION.md` must be committed before this stage runs.

**Stage 3 — Criticality test.**
Scan h across the transition (dense near `h = 1`: e.g. `h ∈ [0.2, 2.0]`, step 0.05,
refined to 0.01 in `[0.85, 1.15]`). Test H1 and H2. Fit `c_eff` with bootstrap CI. Locate
the entropy peak by fitting, not argmax — argmax on a noisy curve is a bias generator —
and bootstrap the peak location.
*Gate:* `RESULTS_STAGE3.md` with `c_eff ± CI` and `h*(L) ± CI` for each L.

**Stage 4 — SAE cross-reference.**
Align against the +0.028 gain result, the mixed-field null, and the
disordered-longitudinal-field results when available. Test H4 and H5 with the
pre-registered Spearman test and multiplicity correction.
*Gate:* `RESULTS_STAGE4.md`, then `AUTHOR_HANDOFF.md`.

---

## Part 8 — Statistics and the ten ways this goes wrong

**Statistical protocol.** ≥ 5 seeds per configuration, matching the convention already
used in the TMLR paper (multi-seed distributions, not single draws). Report mean with
bootstrap 95% CI (10,000 resamples), never bare SEM. Holm–Bonferroni across layers and
cuts, with the family and its size declared up front. Effect sizes with CIs; no naked
p-values. Every deviation from `PREREGISTRATION.md` logged in `DEVIATIONS.md` with a
reason and a date.

Read this checklist before believing any number:

1. **Bond-dimension ceiling.** Entropy under truncation at bond dimension χ is capped at
   `log χ`. Truncating and then observing "saturation" manufactures a fake area law.
   Compute from the untruncated spectrum wherever the cut dimension allows (fine for
   L ≤ 16); otherwise mark the value censored and report truncation error.
2. **Basis dependence.** Construction C is not invariant. Never present it without C4/C5.
3. **Estimator bias.** Finite-sample entropy estimates are biased downward. Report
   bias-corrected estimates alongside plug-in.
4. **Normalization.** Activations must be normalized consistently (and the model's own
   LayerNorm/RMSNorm applied consistently) before being treated as amplitudes. Amplitudes
   must be L2-normalized; probabilities are `|ψ|²`, not `|ψ|`.
5. **Massive activations.** Transformer residual streams have outlier dimensions with
   enormous magnitude. They can single-handedly dominate a Schmidt spectrum. Run C8 with
   and without.
6. **Positional-encoding leakage.** The entropy profile may be measuring positional
   structure, not physics. Ablate positional encodings as a control.
7. **Leakage in the splits.** Same failure mode that bit the PRISM protocol, in different
   clothing: probes and readouts must be fit on field-value-disjoint (and, for the
   disordered arm, realization-disjoint) splits. No h value, and no disorder realization,
   may appear in both fit and evaluation. Write the split logic once, test it, and assert
   disjointness at runtime.
8. **Multiplicity.** Scanning layers × cuts × field values is a large family. Correct for
   it, and say what the family was.
9. **Peak-finding.** Fit and bootstrap; never argmax.
10. **Unmatched comparisons.** Comparing across L with different sample counts, different
    training budgets, or different seed counts is not a comparison. Match them or don't
    claim it.

---

## Part 9 — Engineering contract

- **Structure:** `src/` (importable package), `configs/` (one YAML per experiment, zero
  magic numbers in code), `scripts/` (thin entry points), `tests/`, `slurm/`, `results/`
  (gitignored raw, committed summaries), `figures/` (generated only), `notebooks/scratch/`
  (never load-bearing, outputs stripped).
- **Provenance header on every result file:** git SHA + dirty flag, config hash, seed,
  hostname, SLURM job ID, timestamp, exact package versions, dataset hash, and the
  `PREREGISTRATION.md` SHA. Mirror the provenance discipline already used in the PRISM
  major-revision branch.
- **Determinism:** seed python/numpy/torch, `torch.use_deterministic_algorithms(True)`,
  `CUBLAS_WORKSPACE_CONFIG=:4096:8`, and log any residual non-determinism rather than
  hiding it.
- **Environment:** pinned lockfile committed; `pip freeze` captured into each run's
  provenance.
- **SLURM (Param Rudra, A100):** sbatch scripts with module loads, `--array` for seed
  sweeps, logs to `logs/%x_%A_%a.out`, checkpoint/resume, sensible walltime and memory,
  and a `--dry-run` local mode so nothing is submitted blind. Note that most of this arm
  is small — exact diagonalization at L ≤ 16 and SVDs of ≤ 65536-dim vectors run fine on
  CPU. Do not request GPU nodes for work that doesn't need them. Scripts must be complete
  and directly submittable, not sketches.
- **Figures:** generated by scripts from result files only, never hand-edited. Every figure
  script writes a `.json` sidecar containing the exact numbers plotted — this is what makes
  the eventual table ↔ prose consistency pass tractable.
- **Commits:** small, atomic, imperative subject lines, one logical change each.

---

## Part 10 — How to work with me

- **Plan first.** `PLAN.md` for Stages 0–1, then stop. No experiment code before I approve.
- **Stage gates are stop signs, not suggestions.**
- **No scope creep.** New experiments outside the staged plan require asking first.
- **Never silently retune.** Changing a hyperparameter, split, or estimator to improve a
  result without logging it in `DEVIATIONS.md` is the worst thing you can do on this
  project.
- **Nulls get written up as nulls.** If Stage 3 shows no criticality signature, the
  deliverable is a precise characterization of the null — what was tested, at what power,
  with what CIs — not another round of tuning.
- **Errors:** when something breaks, show me the raw traceback and the exact command.
  Don't paraphrase it.
- **Final deliverable:** `AUTHOR_HANDOFF.md` in the same shape as the PRISM one — what was
  run, what was found, what the honest numbers are, what I should say to my co-authors,
  and what remains open.

---

## Appendix — What a good outcome reads like

> "Under an exact amplitude-space bipartition, the entanglement entropy of the model's
> represented state at layer k tracks the true TFIM ground-state profile across the
> transition, recovering an effective central charge `c_eff = 0.4x ± 0.0y` at `h = 1`
> (bootstrap 95% CI, 5 seeds, L = 8/10/12), against a randomly-initialized-model null of
> `c_eff = 0.0x ± 0.0y`. The layer at which entanglement is acquired most steeply coincides
> with the layer of maximum SAE feature gain (Spearman ρ = 0.xx, Holm-corrected
> p = 0.0xx across 12 layers)."

or, equally publishable:

> "Entanglement entropy of the model's represented state shows no criticality signature:
> the fitted peak location is L-independent and inconsistent with the exact pseudo-critical
> drift, and the fitted `c_eff` is indistinguishable from the random-initialization null.
> The measure does not track physical entanglement under the conditions tested, which we
> characterize precisely below."

Both are results. Only one of them is a discovery. Neither is a failure.
