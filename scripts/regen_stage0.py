"""
Regenerate every number in RESULTS_STAGE0.md from pinned artifacts.

This script exists because those numbers were originally produced by code that was never
committed. A number in a results file whose generating code is absent is a number nobody
can check -- the same defect class as the fabricated `h` vectors, one level up.

Run:  env/run.sh python scripts/regen_stage0.py
Out:  scripts/out/stage0_regenerated.json

Units: entropy in nats throughout; open boundaries; J == 1 unless stated.
"""

from __future__ import annotations

import numpy as np
import torch

from _provenance import provenance, write_json
from qsent.disorder import delta_r, select_spanning_realizations, sigma_lnh_uniform
from qsent.exact import entropy_profile_ed
from qsent.extraction import HOOK_NAMES, extract_residual_stream, mean_pool
from qsent.free_fermions import entropy_profile_free_fermion
from qsent.pins import load_checkpoint, load_ensemble

UNIFORM_L = (8, 10, 12)
UNIFORM_H = (0.5, 1.0, 2.0)
N_HOOK_REALIZATIONS = 512          # RESULTS_STAGE0.md section 2: "512 pinned test realizations"
HOOK_CHECKPOINT_SEED = 1           # "with checkpoint ms_trained/seed1"


# ---------------------------------------------------------------------------------------
# Section 1 -- ED vs free-fermion agreement, and the site-blind degeneracy
# ---------------------------------------------------------------------------------------

def _site_blind_profile(J: np.ndarray, h: np.ndarray) -> np.ndarray:
    """A solver that reads h[0] and applies it everywhere -- the bug the gate must catch."""
    return entropy_profile_free_fermion(J, np.full(len(h), h[0]))


def cross_validation() -> dict:
    """The 14 cross-validation cases: 9 uniform (3 L x 3 h) + 5 spanning disordered."""
    h_train = np.asarray(load_ensemble(1)["h_train"], dtype=np.float64)
    idx, deltas = select_spanning_realizations(h_train)

    cases, site_blind = {}, {}

    for L in UNIFORM_L:
        for hv in UNIFORM_H:
            h, J = np.full(L, hv), np.ones(L - 1)
            d = float(np.max(np.abs(entropy_profile_ed(J, h) - entropy_profile_free_fermion(J, h))))
            cases[f"uniform_h{hv}_L{L}"] = d
            if L == 8:
                site_blind[f"uniform_h{hv}_L{L}"] = float(
                    np.max(np.abs(entropy_profile_ed(J, h) - _site_blind_profile(J, h))))

    labels = ["r0_ordered", "r1", "r2_critical", "r3", "r4_paramagnetic"]
    for label, i, d_r in zip(labels, idx, deltas):
        h, J = h_train[i], np.ones(7)
        cases[f"disordered_{label}"] = float(
            np.max(np.abs(entropy_profile_ed(J, h) - entropy_profile_free_fermion(J, h))))
        site_blind[f"disordered_{label}"] = float(
            np.max(np.abs(entropy_profile_ed(J, h) - _site_blind_profile(J, h))))

    return {
        "n_cases": len(cases),
        "per_case_ed_vs_free_fermion": cases,
        "worst_ed_vs_free_fermion": float(max(cases.values())),
        "site_blind_vs_ed": site_blind,
    }


# ---------------------------------------------------------------------------------------
# Section 2 -- hook family: is k=6 the published tensor?
# ---------------------------------------------------------------------------------------

def _build_model(seed: int):
    from qsae.reverse_arrow.transformer import TFIMTransformer
    ckpt = load_checkpoint(seed)
    model = TFIMTransformer(ckpt["cfg"])
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model


def hook_family() -> dict:
    """k=6 must equal `qsae.analysis.extract.last_layer_pooled`; post_final_norm must not."""
    from qsae.analysis.extract import last_layer_pooled

    h_test = np.asarray(load_ensemble(1)["h_test"], dtype=np.float64)[:N_HOOK_REALIZATIONS]
    model = _build_model(HOOK_CHECKPOINT_SEED)

    published = np.asarray(last_layer_pooled(model, h_test), dtype=np.float64)
    acts = extract_residual_stream(model, h_test, include_final_norm=True)

    return {
        "encoder_norm_is_none": model.encoder.norm is None,
        "n_realizations": int(len(h_test)),
        "max_abs_diff_k6_vs_published": float(
            np.max(np.abs(mean_pool(acts["block2_mlp"]) - published))),
        "max_abs_diff_post_final_norm_vs_published": float(
            np.max(np.abs(mean_pool(acts["post_final_norm"]) - published))),
        "rms_by_hook": {name: float(np.sqrt(np.mean(acts[name] ** 2))) for name in HOOK_NAMES},
    }


# ---------------------------------------------------------------------------------------
# Section 3 -- do the cross-validation realizations span delta_r?
# ---------------------------------------------------------------------------------------

def spanning() -> dict:
    h_train = np.asarray(load_ensemble(1)["h_train"], dtype=np.float64)
    idx, deltas = select_spanning_realizations(h_train)
    labels = ["r0_ordered", "r1", "r2_critical", "r3", "r4_paramagnetic"]
    return {
        "indices": {l: int(i) for l, i in zip(labels, idx)},
        "delta_r": {l: float(d) for l, d in zip(labels, deltas)},
    }


# ---------------------------------------------------------------------------------------
# Section 4 -- disorder-ensemble characterization
# ---------------------------------------------------------------------------------------

def ensemble_characterization() -> dict:
    meta = load_ensemble(1)["meta"]
    m1, sd = sigma_lnh_uniform(meta["h_min"], meta["h_max"])

    per_seed = {}
    pooled = []
    for s in (1, 2, 3):
        h = np.asarray(load_ensemble(s)["h_train"], dtype=np.float64)
        per_seed[f"seed{s}"] = float(np.log(h).mean())
        pooled.append(h)
    pooled = np.concatenate(pooled, axis=0)
    d = delta_r(pooled)

    # Mean delta_r at each L is analytic: delta_ens = sqrt(L) * (-E[ln h] / sigma_lnh).
    # Derived, never typed -- L=10/12 have no pinned ensemble, and this needs none.
    drift = {f"L{L}": float(np.sqrt(L) * (-m1 / sd)) for L in UNIFORM_L}

    n = len(d)
    bands = {}
    for edge in (0.05, 0.10, 0.25, 0.50):
        c = int((np.abs(d) < edge).sum())
        bands[f"abs_delta_lt_{edge:.2f}"] = {"count": c, "share_pct": 100.0 * c / n}
    c_neg = int((d < 0).sum())
    bands["delta_lt_0"] = {"count": c_neg, "share_pct": 100.0 * c_neg / n}

    return {
        "E_ln_h_closed_form": m1,
        "sd_ln_h_closed_form": sd,
        "lnJ_minus_lnh": -m1,
        "per_seed_E_ln_h": per_seed,
        "mean_delta_r_by_L": drift,
        "pooled_N": int(n),
        "occupancy": bands,
    }


def main() -> None:
    payload = {
        "_provenance": provenance(
            seeds={},                                  # no RNG used anywhere in Stage 0
            artifacts=[f"data/tfim_L8_N50k_seed{s}.pt" for s in (1, 2, 3)]
                      + [f"runs/ms_trained/seed{HOOK_CHECKPOINT_SEED}/best.pt"],
        ),
        "section1_cross_validation": cross_validation(),
        "section2_hook_family": hook_family(),
        "section3_spanning": spanning(),
        "section4_ensemble": ensemble_characterization(),
    }
    path = write_json(payload, "stage0_regenerated.json")
    print(f"wrote {path}")


if __name__ == "__main__":
    torch.manual_seed(0)
    main()
