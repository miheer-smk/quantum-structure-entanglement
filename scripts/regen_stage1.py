"""
Regenerate every recoverable number in RESULTS_STAGE1.md from pinned artifacts.

Same exercise as `regen_stage0.py`, on the population that has never had it. Stage 1's numbers
were produced by code that no longer exists, in an environment since shown to differ measurably
from this one (`DEVIATIONS.md`, 2026-08-13), and `README.md` quotes several of them publicly.

WHAT IS RECOVERABLE, AND WHAT IS NOT
------------------------------------
Stage 1 splits cleanly in two, and the split is itself a finding:

  DETERMINED -- computable from pinned artifacts plus a documented procedure, so a mismatch
  means something. Sections 3 (orientation), the clean-chain half of 8, the clean column of
  10, and the L=8 disordered rows, which use the PINNED seed-1 ensemble.

  UNDERDETERMINED -- the results file states a number but the repository does not contain
  enough information to recompute it. These are NOT regenerated here, because a guessed
  procedure that happens to reproduce a value would be worse than no check at all: it would
  license the number while testing something else. They are reported as gaps.

Underdetermined, with the specific missing fact:

  * L = 10/12 disordered rows (S8, S10) and the collapse (S9). `PLAN.md` A0b documents the
    reference RNG as `default_rng(20260804 + L)`, but not the POOL SIZE that was drawn before
    filtering to `|delta_r| < 0.05`, and the sub-ensemble N (2000) does not determine it.
  * Collapse quality `Q` (S9). The recipe -- bin on a common grid, remove a per-L offset,
    divide residual spread by dynamic range -- fixes the shape but not the BIN EDGES, the bin
    count, or any minimum occupancy. Different choices give different `Q`.
  * The section-2 site-ordering diffs. The results file says "measured, on disordered chains"
    without naming which chains.

Run:  env/run.sh python scripts/regen_stage1.py
Out:  scripts/out/stage1_regenerated.json
"""

from __future__ import annotations

import sys

import numpy as np

from _claims import Registry
from _provenance import provenance, rng_fingerprint, write_json
from qsent.convention import fit_c_eff
from qsent.disorder import delta_r, select_spanning_realizations, sigma_lnh_uniform
from qsent.exact import entropy_profile_ed
from qsent.free_fermions import entropy_profile_free_fermion
from qsent.pins import load_ensemble, repo_root

SCRIPT = "scripts/regen_stage1.py"
ENS1 = "data/tfim_L8_N50k_seed1.pt"
CLEAN_LS = (8, 10, 12)
CLEAN_LARGE_LS = (32, 64, 128)
DELTA_BAND = 0.05


def clean_c_eff(L: int) -> float:
    """c_eff of a clean critical open chain (J = h = 1), where the true value is 0.5."""
    return fit_c_eff(entropy_profile_free_fermion(np.ones(L - 1), np.ones(L)), L)[0]


def section3_orientation(reg: Registry) -> dict:
    """The max-asymmetry realization, its profile, and the fit's blindness to mirroring."""
    sys.path.insert(0, str(repo_root() / "tests"))
    from test_orientation import ASYM_H, ASYM_INDEX

    h = np.asarray(ASYM_H, dtype=np.float64)
    S = entropy_profile_ed(np.ones(len(h) - 1), h)
    c_f, ssr_f = fit_c_eff(S, len(h))
    c_r, ssr_r = fit_c_eff(S[::-1], len(h))

    h_train = np.asarray(load_ensemble(1)["h_train"], dtype=np.float64)
    d = float(delta_r(h_train[ASYM_INDEX : ASYM_INDEX + 1])[0])
    margin = float(np.max(np.abs(S - S[::-1])))

    reg.add("s3_asym_index", float(ASYM_INDEX), ENS1)
    reg.add("s3_asym_delta_r", d, ENS1)
    reg.add("s3_asym_margin", margin, ENS1)
    reg.add("s3_asym_c_eff", float(c_f), ENS1)
    reg.add("s3_asym_ssr", float(ssr_f), ENS1)
    reg.add("s3_mirror_dc_eff", abs(float(c_f - c_r)), ENS1, kind="bound")
    reg.add("s3_mirror_dssr", abs(float(ssr_f - ssr_r)), ENS1, kind="bound")
    for i, v in enumerate(S):
        reg.add(f"s3_asym_profile_cut{i + 1}", float(v), ENS1)
    for L in CLEAN_LARGE_LS:
        reg.add(f"s3_clean_c_eff_L{L}", clean_c_eff(L), "none")

    return {
        "asym_index": int(ASYM_INDEX), "asym_delta_r": d, "asym_margin": margin,
        "c_eff_forward": float(c_f), "ssr_forward": float(ssr_f),
        "c_eff_mirrored": float(c_r), "ssr_mirrored": float(ssr_r),
        "d_c_eff": abs(float(c_f - c_r)), "d_ssr": abs(float(ssr_f - ssr_r)),
        "profile": [float(x) for x in S],
        "clean_c_eff": {f"L{L}": clean_c_eff(L) for L in CLEAN_LARGE_LS},
    }


def section8_clean(reg: Registry) -> dict:
    """Clean-chain c_eff and its bias at the three system sizes Stage 1 reports."""
    out = {}
    for L in CLEAN_LS:
        c = clean_c_eff(L)
        out[f"L{L}"] = {"c_eff": c, "bias": c - 0.5}
        reg.add(f"s8_clean_c_eff_L{L}", c, "none")
        reg.add(f"s8_clean_bias_L{L}", c - 0.5, "none")
    return out


def section8_disordered_L8(reg: Registry) -> dict:
    """The L = 8 disordered rows -- the only ones on a PINNED ensemble.

    Sub-ensemble is |delta_r| < 0.05 of the pinned seed-1 training set. Both the
    disorder-averaged fit (fit the mean profile) and the typical fit (median profile) are
    computed, matching the two rows PLAN.md A0 reports.
    """
    h_train = np.asarray(load_ensemble(1)["h_train"], dtype=np.float64)
    d = delta_r(h_train)
    sel = np.abs(d) < DELTA_BAND
    H = h_train[sel]

    profiles = np.array([entropy_profile_ed(np.ones(7), h) for h in H])
    c_avg = fit_c_eff(profiles.mean(axis=0), 8)[0]
    c_typ = fit_c_eff(np.median(profiles, axis=0), 8)[0]
    target = float(np.log(2.0) / 2.0)

    reg.add("s8_disordered_n_L8", float(len(H)), ENS1)
    reg.add("s8_disordered_c_eff_avg_L8", c_avg, ENS1)
    reg.add("s8_disordered_c_eff_typ_L8", c_typ, ENS1)
    reg.add("s8_disordered_bias_avg_L8", c_avg - target, ENS1)
    reg.add("s8_disordered_bias_typ_L8", c_typ - target, ENS1)

    return {"n": int(len(H)), "c_eff_disorder_avg": c_avg, "c_eff_typical": c_typ,
            "bias_disorder_avg": c_avg - target, "bias_typical": c_typ - target,
            "target_ln2_over_2": target}


def section10_gap_L8(reg: Registry, n_boot: int = 4000, seed: int = 0) -> dict:
    """Clean-vs-disordered gap at L = 8 with a bootstrap CI over realizations.

    The clean value is deterministic (one chain), so the CI comes entirely from the disordered
    side, exactly as PLAN.md A0c states. The generator is fingerprinted at construction.
    """
    rng = np.random.default_rng(seed)
    fingerprint = rng_fingerprint(rng)

    h_train = np.asarray(load_ensemble(1)["h_train"], dtype=np.float64)
    sel = np.abs(delta_r(h_train)) < DELTA_BAND
    profiles = np.array([entropy_profile_ed(np.ones(7), h) for h in h_train[sel]])

    clean = clean_c_eff(8)
    dis = fit_c_eff(profiles.mean(axis=0), 8)[0]
    gap = clean - dis

    boot = np.empty(n_boot)
    n = len(profiles)
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        boot[b] = clean - fit_c_eff(profiles[idx].mean(axis=0), 8)[0]
    lo, hi = float(np.quantile(boot, 0.025)), float(np.quantile(boot, 0.975))

    reg.add("s10_clean_c_eff_L8", clean, "none")
    reg.add("s10_disordered_c_eff_L8", dis, ENS1)
    reg.add("s10_gap_L8", gap, ENS1)

    return {"clean": clean, "disordered": dis, "gap": gap, "ci95": [lo, hi],
            "spans_zero": bool(lo <= 0.0 <= hi), "n_bootstrap": n_boot,
            "rng": fingerprint}


def main() -> int:
    reg = Registry(SCRIPT)
    rng_seed = 0
    payload = {
        "_provenance": provenance(
            seeds={"bootstrap_seed": rng_seed},
            artifacts=[ENS1],
            rngs={"bootstrap": np.random.default_rng(rng_seed)}),
        "section3_orientation": section3_orientation(reg),
        "section8_clean": section8_clean(reg),
        "section8_disordered_L8": section8_disordered_L8(reg),
        "section10_gap_L8": section10_gap_L8(reg, seed=rng_seed),
        "not_regenerated": {
            "reason": "the repository does not contain enough information to recompute these; "
                      "a guessed procedure that reproduced a value would license it while "
                      "testing something else",
            "items": {
                "section2_site_ordering_diffs":
                    "results file says 'measured, on disordered chains' without naming which",
                "section8_and_10_L10_L12":
                    "reference RNG documented as default_rng(20260804 + L), but not the pool "
                    "size drawn before filtering to |delta_r| < 0.05",
                "section9_collapse_Q_and_CIs":
                    "bin edges, bin count and minimum occupancy are not documented; the "
                    "recipe fixes the shape of Q but not its value",
            },
        },
    }
    payload["claims"] = reg.as_dict()
    path = write_json(payload, "stage1_regenerated.json")
    print(f"wrote {path} with {len(payload['claims'])} provenance claims")
    return 0


if __name__ == "__main__":
    sys.exit(main())
