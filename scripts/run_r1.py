"""
R1 — commensurability reproduction (PLAN.md §3.5.1). ONE substitution, nothing else changed.

R1 re-runs the phase06 `long_range_zz` incremental-R²-beyond-poly2-h protocol on the pinned
`ms_trained` seeds 1-10, replacing exactly one line of the published driver:

    published (experiments/phase06_multiseed_trained.py, control_point):
        R = last_layer_pooled(model, h)
    here:
        R = mean_pool(extract_residual_stream(model, h)["block2_mlp"])      # k=6

Everything downstream of that line is the pinned submodule's own code, called rather than
reimplemented: `build_input_controls`, `incremental_r2`, `oof_ridge_predict`. Ridge alpha,
fold count, fold seed, eval arrays, observable, control, and the averaging over eval seeds
are read from the pinned config `configs/phase06_multiseed_trained.yaml` and never typed
here. If this file computed the statistic itself, R1 would be testing this file's arithmetic
instead of this arm's extraction stack.

WHY THE SUBSTITUTION IS THE WHOLE POINT
---------------------------------------
`k=6` is the published tensor (tests/test_extraction.py asserts it to < 2e-06 on these very
arrays). R1 asks whether that identity survives an entire analysis pipeline: whether a
float32-vs-float64 difference of ~15 float32 ULPs in the representation moves a published
incremental R² by more than seed noise. It validates THIS ARM'S EXTRACTION STACK and nothing
else -- see the scope statement in RESULTS_STAGE1_5.md, lifted verbatim from PLAN.md §3.6 A1.

The verdict is computed by the pre-registered rule and written to JSON. The rule is NOT
re-implemented in the results file or evaluated by eye: `tests/test_r1_gate.py` recomputes it
from the recorded numbers and fails if the recorded verdict disagrees.

Run:  env/run.sh python scripts/run_r1.py
Out:  scripts/out/r1_reproduction.json
"""

from __future__ import annotations

import sys

import numpy as np
import torch
import yaml

from _claims import Registry
from _provenance import provenance, write_json
from qsent.extraction import HOOK_NAMES, PUBLISHED_HOOK_INDEX, extract_residual_stream, mean_pool
from qsent.pins import (
    artifact_root, load_checkpoint, published_constant, published_per_seed, repo_root,
    verify_pin,
)

SCRIPT = "scripts/run_r1.py"
SUBMODULE = repo_root() / "submodules" / "quantum-structure-sae"
CONFIG = "configs/phase06_multiseed_trained.yaml"
OBSERVABLE = "long_range_zz"
CONTROL = "poly2_h"
HOOK = HOOK_NAMES[PUBLISHED_HOOK_INDEX]          # block2_mlp == the published hook

#: Pre-registered tolerance (PLAN.md §3.5.1), fixed before the number was seen. Both limbs
#: must hold. The window is DERIVED from the published mean and sd, never typed.
N_SD_WINDOW = 2.0
PER_SEED_DELTA_MAX = 0.010
MIN_SEEDS_WITHIN = 8


def pinned_config() -> dict:
    """The published run's own config. Alpha, folds, fold seed and eval arrays come from here."""
    return yaml.safe_load((SUBMODULE / CONFIG).read_text())


def r1_per_seed(cfg: dict) -> tuple[dict[int, float], dict[int, dict[int, float]]]:
    """Incremental R² beyond poly2-h at k=6, per trained seed, averaged over eval arrays.

    Mirrors `control_point` in the published driver for the (long_range_zz, poly2_h) cell.
    """
    from qsae.analysis.extract import build_input_controls
    from qsae.analysis.input_control import incremental_r2
    from qsae.reverse_arrow.transformer import TFIMTransformer

    alpha = cfg["probe"]["ridge_alpha"]
    n_folds = cfg["probe"]["n_folds"]
    fold_seed = cfg["probe"]["fold_seed"]

    # Eval sets: the pinned N=800 arrays, hash-verified. Same arrays, same order as published.
    eval_sets = []
    for es in cfg["eval"]["seeds"]:
        rel = cfg["eval"]["cache_glob"].replace("{seed}", str(es))
        blob = torch.load(verify_pin(rel, "ensemble.sha256"), map_location="cpu",
                          weights_only=False)
        h = np.asarray(blob["h_fields"], dtype=np.float64)
        eval_sets.append((es, h, blob["obs"], build_input_controls(h)))

    per_seed: dict[int, float] = {}
    per_seed_per_eval: dict[int, dict[int, float]] = {}
    for s in cfg["train_seeds"]:
        ckpt = load_checkpoint(s)
        model = TFIMTransformer(ckpt["cfg"])
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()

        deltas = {}
        for es, h, obs, controls in eval_sets:
            # ---- THE SINGLE SUBSTITUTION -------------------------------------------------
            R = mean_pool(extract_residual_stream(model, h)[HOOK])
            # ------------------------------------------------------------------------------
            y = np.asarray(obs[OBSERVABLE], dtype=np.float64)
            deltas[es] = float(
                incremental_r2(R, controls[CONTROL], y, alpha, n_folds, fold_seed)["delta"])
        per_seed_per_eval[s] = deltas
        per_seed[s] = float(np.mean(list(deltas.values())))       # average over eval seeds
        print(f"  seed {s:2d}: incremental R2 = {per_seed[s]:.6f}   "
              f"per-eval {[f'{v:.6f}' for v in deltas.values()]}")
    return per_seed, per_seed_per_eval


def verdict(new: dict[int, float], published: dict[int, float],
            lo: float, hi: float) -> dict:
    """The pre-registered rule, applied. Both limbs required; no third outcome."""
    seeds = sorted(new)
    mean_new = float(np.mean([new[s] for s in seeds]))
    deltas = {s: new[s] - published[s] for s in seeds}
    within = [s for s in seeds if abs(deltas[s]) <= PER_SEED_DELTA_MAX]

    limb_i = bool(lo <= mean_new <= hi)
    limb_ii = bool(len(within) >= MIN_SEEDS_WITHIN)
    return {
        "mean_new": mean_new,
        "window": [lo, hi],
        "limb_i_mean_in_window": limb_i,
        "paired_differences": {str(s): deltas[s] for s in seeds},
        "max_abs_difference": float(max(abs(d) for d in deltas.values())),
        "n_seeds_within_tolerance": len(within),
        "seeds_within_tolerance": within,
        "seeds_outside_tolerance": [s for s in seeds if s not in within],
        "limb_ii_paired_seeds": limb_ii,
        "verdict": "PASS" if (limb_i and limb_ii) else "FAIL",
    }


def main() -> int:
    cfg = pinned_config()
    print(f"R1: {OBSERVABLE} incremental R2 beyond {CONTROL} at hook {HOOK}\n"
          f"    alpha={cfg['probe']['ridge_alpha']} n_folds={cfg['probe']['n_folds']} "
          f"fold_seed={cfg['probe']['fold_seed']} eval={cfg['eval']['seeds']} "
          f"train_seeds={cfg['train_seeds']}\n")

    published = published_per_seed("phase06_lrzz_incr_r2_per_seed")
    mean_pub = published_constant("phase06_lrzz_incr_r2_mean")
    sd_pub = published_constant("phase06_lrzz_incr_r2_sd")
    lo, hi = mean_pub - N_SD_WINDOW * sd_pub, mean_pub + N_SD_WINDOW * sd_pub

    new, per_eval = r1_per_seed(cfg)
    if sorted(new) != sorted(published):
        raise RuntimeError(f"seed mismatch: computed {sorted(new)}, published {sorted(published)}")

    v = verdict(new, published, lo, hi)

    reg = Registry(SCRIPT)
    evals = ",".join(cfg["eval"]["cache_glob"].replace("{seed}", str(es))
                     for es in cfg["eval"]["seeds"])
    for s in sorted(new):
        reg.add(f"r1_incr_r2_seed{s}", new[s],
                f"runs/ms_trained/seed{s}/best.pt,{evals}", seed=str(cfg["probe"]["fold_seed"]))
    all_ckpts = ",".join(f"runs/ms_trained/seed{s}/best.pt" for s in sorted(new))
    reg.add("r1_mean", v["mean_new"], f"{all_ckpts},{evals}",
            seed=str(cfg["probe"]["fold_seed"]))
    reg.add("r1_max_abs_difference", v["max_abs_difference"], f"{all_ckpts},{evals}",
            seed=str(cfg["probe"]["fold_seed"]))
    reg.add("r1_n_seeds_within_tolerance", float(v["n_seeds_within_tolerance"]),
            f"{all_ckpts},{evals}", seed=str(cfg["probe"]["fold_seed"]))

    payload = {
        "_provenance": provenance(
            seeds={"fold_seed": cfg["probe"]["fold_seed"], "analysis_seed": cfg["seed"]},
            artifacts=[f"runs/ms_trained/seed{s}/best.pt" for s in cfg["train_seeds"]]
                      + [cfg["eval"]["cache_glob"].replace("{seed}", str(es))
                         for es in cfg["eval"]["seeds"]]),
        "protocol": {
            "substitution": "last_layer_pooled -> mean_pool(extract_residual_stream(...)[k=6])",
            "hook": HOOK, "observable": OBSERVABLE, "control": CONTROL,
            "ridge_alpha": cfg["probe"]["ridge_alpha"], "n_folds": cfg["probe"]["n_folds"],
            "fold_seed": cfg["probe"]["fold_seed"], "eval_seeds": cfg["eval"]["seeds"],
            "train_seeds": cfg["train_seeds"], "config_source": CONFIG,
        },
        "tolerance": {
            "published_mean": mean_pub, "published_sd": sd_pub, "n_sd": N_SD_WINDOW,
            "window": [lo, hi], "per_seed_delta_max": PER_SEED_DELTA_MAX,
            "min_seeds_within": MIN_SEEDS_WITHIN,
        },
        "published_per_seed": {str(k): val for k, val in sorted(published.items())},
        "r1_per_seed": {str(k): val for k, val in sorted(new.items())},
        "r1_per_seed_per_eval_array": {str(k): {str(e): d for e, d in sorted(val.items())}
                                       for k, val in sorted(per_eval.items())},
        "result": v,
        "claims": reg.as_dict(),
    }
    path = write_json(payload, "r1_reproduction.json")

    print(f"\n  published mean {mean_pub:.4f} sd {sd_pub:.4f} -> window "
          f"[{lo:.4f}, {hi:.4f}]")
    print(f"  R1 mean        {v['mean_new']:.6f}   in window: {v['limb_i_mean_in_window']}")
    print(f"  paired |delta| <= {PER_SEED_DELTA_MAX}: {v['n_seeds_within_tolerance']}/10 "
          f"(need >= {MIN_SEEDS_WITHIN})   max |delta| = {v['max_abs_difference']:.6f}")
    print(f"\n  R1 VERDICT: {v['verdict']}")
    print(f"\nwrote {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
