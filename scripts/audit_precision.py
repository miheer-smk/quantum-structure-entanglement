"""
Audit which committed near-machine-precision values are STABLE VALUES and which are BOUNDS.

Motivation, author-directed 2026-08-11. Three of the repository's committed numbers sit at
or near the floating-point noise floor. A harness that only checks "does the number match"
cannot tell a genuinely stable quantity from one that matched by luck -- and the 1.648e-11
ED-vs-free-fermion agreement, which the entire two-solver ground-truth claim rests on, is in
that category.

Method: recompute each quantity under configurations that MUST NOT change a physical result
-- BLAS thread count -- and report the spread. A quantity whose value moves under thread
count carries no information in its trailing digits and must be reported as a bound. One
whose value does not move is a stable value and may be quoted.

BLAS thread count is fixed when the library loads, so each configuration runs in its own
subprocess.

Run:  env/run.sh python scripts/audit_precision.py
Out:  scripts/out/precision_audit.json
"""

from __future__ import annotations

import json
import os
import subprocess
import sys

import numpy as np

THREAD_CONFIGS = (1, 2, 4, 8)
UNIFORM_L = (8, 10, 12)
UNIFORM_H = (0.5, 1.0, 2.0)
EVAL_SEEDS = (42, 43, 44)


def _measure() -> dict[str, float]:
    """Every audited quantity, computed once in this process at the ambient thread count."""
    from qsent.exact import entropy_profile_ed
    from qsent.free_fermions import entropy_profile_free_fermion
    from qsent.extraction import extract_residual_stream, mean_pool
    from qsent.pins import load_checkpoint, load_ensemble, verify_pin
    from qsent.disorder import select_spanning_realizations

    out: dict[str, float] = {}

    # --- Stage 0 section 1: ED vs free-fermion, and the site-blind control -----------------
    for L in UNIFORM_L:
        for hv in UNIFORM_H:
            h, J = np.full(L, hv), np.ones(L - 1)
            out[f"ed_vs_ff_uniform_h{hv}_L{L}"] = float(
                np.max(np.abs(entropy_profile_ed(J, h) - entropy_profile_free_fermion(J, h))))

    h_train = np.asarray(load_ensemble(1)["h_train"], dtype=np.float64)
    idx, _ = select_spanning_realizations(h_train)
    labels = ["r0_ordered", "r1", "r2_critical", "r3", "r4_paramagnetic"]
    for label, i in zip(labels, idx):
        h, J = h_train[i], np.ones(7)
        out[f"ed_vs_ff_disordered_{label}"] = float(
            np.max(np.abs(entropy_profile_ed(J, h) - entropy_profile_free_fermion(J, h))))
    out["ed_vs_ff_worst"] = max(v for k, v in out.items() if k.startswith("ed_vs_ff_"))

    # --- Stage 0 section 2: hook equality, on the arrays R1 ACTUALLY CONSUMES --------------
    import torch
    from qsae.reverse_arrow.transformer import TFIMTransformer
    from qsae.analysis.extract import last_layer_pooled

    ck = load_checkpoint(1)
    model = TFIMTransformer(ck["cfg"])
    model.load_state_dict(ck["model_state_dict"])
    model.eval()

    for s in EVAL_SEEDS:
        p = verify_pin(f"data/ra03_states_L8_N800_s{s}.pt", "ensemble.sha256")
        h = np.asarray(torch.load(p, map_location="cpu", weights_only=False)["h_fields"],
                       dtype=np.float64)
        published = np.asarray(last_layer_pooled(model, h), dtype=np.float64)
        acts = extract_residual_stream(model, h, include_final_norm=True)
        out[f"hook_k6_vs_published_eval_s{s}"] = float(
            np.max(np.abs(mean_pool(acts["block2_mlp"]) - published)))
        out[f"hook_postnorm_vs_published_eval_s{s}"] = float(
            np.max(np.abs(mean_pool(acts["post_final_norm"]) - published)))

    # --- Stage 1 section 3: the c_eff fit's blindness to mirroring -------------------------
    # ASYM_H is the pinned max-asymmetry realization, verified against the ensemble by
    # tests/test_orientation.py::test_asym_realization_matches_pinned_ensemble.
    sys.path.insert(0, str(__import__("qsent.pins", fromlist=["repo_root"]).repo_root() / "tests"))
    from test_orientation import ASYM_H

    def ceff_fit(S: np.ndarray, L: int) -> tuple[float, float]:
        l = np.arange(1, L)
        x = np.log((2 * L / np.pi) * np.sin(np.pi * l / L))
        A = np.vstack([x, np.ones_like(x)]).T
        coef, *_ = np.linalg.lstsq(A, S, rcond=None)
        return 6 * coef[0], float(np.sum((S - A @ coef) ** 2))

    S = entropy_profile_ed(np.ones(7), np.array(ASYM_H))
    c_f, ssr_f = ceff_fit(S, 8)
    c_r, ssr_r = ceff_fit(S[::-1], 8)
    out["ceff_mirror_dc"] = abs(c_f - c_r)
    out["ceff_mirror_dssr"] = abs(ssr_f - ssr_r)
    out["ceff_asym_forward"] = float(c_f)
    out["asym_margin"] = float(np.max(np.abs(S - S[::-1])))

    return out


def main() -> None:
    if "--once" in sys.argv:
        print("JSONBEGIN" + json.dumps(_measure()) + "JSONEND")
        return

    from qsent.pins import repo_root
    from _provenance import provenance, write_json

    per_config: dict[str, dict[str, float]] = {}
    for t in THREAD_CONFIGS:
        env = dict(os.environ,
                   OMP_NUM_THREADS=str(t), MKL_NUM_THREADS=str(t), OPENBLAS_NUM_THREADS=str(t))
        r = subprocess.run([sys.executable, __file__, "--once"], env=env,
                           capture_output=True, text=True, check=True)
        blob = r.stdout.split("JSONBEGIN", 1)[1].split("JSONEND", 1)[0]
        per_config[f"threads_{t}"] = json.loads(blob)

    keys = sorted(next(iter(per_config.values())))
    summary = {}
    for k in keys:
        vals = [per_config[c][k] for c in per_config]
        lo, hi = min(vals), max(vals)
        spread_rel = (hi - lo) / abs(hi) if hi != 0 else 0.0
        # A quantity is a STABLE VALUE only if configurations that cannot matter physically
        # leave it identical to well beyond the precision anyone would quote it at.
        summary[k] = {
            "min": lo, "max": hi, "spread_abs": hi - lo, "spread_rel": spread_rel,
            "verdict": "stable value" if spread_rel < 1e-12 else "BOUND",
        }

    payload = {
        "_provenance": provenance(
            seeds={},
            artifacts=[f"data/ra03_states_L8_N800_s{s}.pt" for s in EVAL_SEEDS]
                      + ["data/tfim_L8_N50k_seed1.pt", "runs/ms_trained/seed1/best.pt"]),
        "thread_configs": list(THREAD_CONFIGS),
        "per_config": per_config,
        "summary": summary,
    }
    path = write_json(payload, "precision_audit.json")

    w = max(len(k) for k in keys)
    print(f"{'quantity'.ljust(w)}  {'min':>14}  {'max':>14}  {'rel spread':>11}  verdict")
    print("-" * (w + 50))
    for k in keys:
        s = summary[k]
        print(f"{k.ljust(w)}  {s['min']:14.6e}  {s['max']:14.6e}  {s['spread_rel']:11.2e}  "
              f"{s['verdict']}")
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
