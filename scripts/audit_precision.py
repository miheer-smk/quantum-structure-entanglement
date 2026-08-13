"""
Audit how many significant figures of each near-precision value survive reconfiguration.

Motivation, author-directed 2026-08-11. Three of the repository's committed numbers sit at
or near the floating-point noise floor. A harness that only checks "does the number match"
cannot tell a genuinely stable quantity from one that matched by luck -- and the 1.648e-11
ED-vs-free-fermion agreement, which the entire two-solver ground-truth claim rests on, is in
that category.

Method: recompute each quantity under configurations that MUST NOT change a physical result
-- BLAS thread count -- and measure how far down the digits agree.

THE RULE, AND ITS CORRECTION (author, 2026-08-13)
------------------------------------------------
The rule this file first implemented was **"moves under thread count -> BOUND"**. The author
withdrew it as too crude, and it is easy to see why from its own output: it returned the same
verdict for `ed_vs_ff_worst`, whose four leading digits are identical across every
configuration and which wobbles only in the fifth, and for `ceff_mirror_dc`, which does not
reproduce even its first digit. Collapsing those into one word discards honestly measured
precision from the claim carrying the entire two-solver ground truth.

The rule now is: **report to the number of significant figures that are stable across
configurations, and state the measured spread alongside.** `stable_sigfigs` is computed by
rounding every configuration's value to k significant figures and increasing k until the
configurations disagree. A quantity is quotable at that many figures and no more:

  * `stable_sigfigs >= MIN_SIGFIGS_TO_QUOTE` -> a VALUE, quoted to exactly that many figures
    and never more, always accompanied by its spread;
  * `stable_sigfigs <  MIN_SIGFIGS_TO_QUOTE` -> a BOUND, because a single stable digit is an
    order-of-magnitude statement, which is what a bound already says.

WHAT THIS AUDIT DOES *NOT* ESTABLISH
------------------------------------
BLAS thread count is ONE reconfiguration axis. Stability under it is necessary, not
sufficient: a value stable here could still move under a different BLAS library, a different
CPU architecture, or a compiler change. `stable_sigfigs` is therefore an UPPER bound on the
digits worth quoting, established against the one axis that can be varied on this machine.
It is not a claim of bitwise portability, and nothing downstream should read it as one.

Separately: a quantity can be thread-stable and still not quotable, if it varies along an
axis this audit does not sweep. `hook_k6_vs_published_eval_s*` is exactly that case -- each
is bit-identical across thread counts, but the three arrays give 14, 17 and 15 float32 ULPs,
so the value is a property of the array chosen. Those stay bounds in the results file for a
reason this audit cannot see, and the `kind=bound` tag records it.

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

#: Fewer stable figures than this and the quantity is reported as a bound: one stable digit
#: is an order-of-magnitude statement, which an inequality already makes more honestly.
MIN_SIGFIGS_TO_QUOTE = 2
MAX_SIGFIGS = 17                     # float64 round-trips at 17 significant decimal digits

#: Quantities whose binding variation is along an axis this audit does not sweep. Recorded in
#: the output so a reader cannot take `verdict: value` -- which here means only "thread-stable"
#: -- as clearance to quote digits the quantity does not possess.
AXIS_CAVEATS = {
    "hook_k6_vs_published_eval_s": "bit-stable across threads, but varies ACROSS EVAL ARRAYS "
                                   "(14/17/15 x 2^-24); the array choice, not the thread "
                                   "count, is the binding axis. Reported as a bound.",
    "hook_postnorm_vs_published_eval_s": "bit-stable across threads, but varies across eval "
                                         "arrays; reported as an interval in the results file.",
}


def stable_sigfigs(values: list[float]) -> int:
    """How many leading significant figures are identical across every configuration.

    Rounds each value to k significant figures and increases k until the configurations
    disagree, so the answer is the number of digits it is honest to print -- not an estimate
    derived from the relative spread, which would blur across a rounding boundary.
    """
    if len(set(values)) == 1:
        return MAX_SIGFIGS
    n = 0
    for k in range(1, MAX_SIGFIGS + 1):
        if len({f"{v:.{k - 1}e}" for v in values}) != 1:
            break
        n = k
    return n


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

    # The site-blind control, audited through the site-blind path so that the quantity
    # measured here is the one the claim names. On a UNIFORM field this is by construction
    # the same expression as the ed_vs_ff row at the same (L=8, h) -- "uses h_j" and "uses
    # h[0] everywhere" coincide -- which is section 1's whole point; on DISORDERED chains it
    # is a real physical separation and its quoted digits need auditing like any other.
    def _blind(J, h):
        return entropy_profile_free_fermion(J, np.full(len(h), h[0]))

    for hv in UNIFORM_H:
        h, J = np.full(8, hv), np.ones(7)
        out[f"site_blind_uniform_h{hv}_L8"] = float(
            np.max(np.abs(entropy_profile_ed(J, h) - _blind(J, h))))
    for label, i in zip(labels, idx):
        h, J = h_train[i], np.ones(7)
        out[f"site_blind_disordered_{label}"] = float(
            np.max(np.abs(entropy_profile_ed(J, h) - _blind(J, h))))

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
        # Corrected rule (author, 2026-08-13): quote the figures that are stable, not a
        # binary moved/did-not-move. The spread travels with the value; it is not optional.
        sig = stable_sigfigs(vals)
        summary[k] = {
            "min": lo, "max": hi, "spread_abs": hi - lo, "spread_rel": spread_rel,
            "stable_sigfigs": sig,
            "quotable": f"{hi:.{min(sig, MAX_SIGFIGS) - 1}e}" if sig else None,
            "verdict": "value" if sig >= MIN_SIGFIGS_TO_QUOTE else "BOUND",
        }
        for prefix, caveat in AXIS_CAVEATS.items():
            if k.startswith(prefix):
                summary[k]["other_axis_caveat"] = caveat

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
    print(f"{'quantity'.ljust(w)}  {'max':>14}  {'spread_abs':>11}  {'spread_rel':>11}  "
          f"{'s.f.':>4}  {'quotable':>14}  verdict")
    print("-" * (w + 70))
    for k in sorted(keys, key=lambda k: -summary[k]["max"]):
        s = summary[k]
        sig = "full" if s["stable_sigfigs"] == MAX_SIGFIGS else str(s["stable_sigfigs"])
        print(f"{k.ljust(w)}  {s['max']:14.6e}  {s['spread_abs']:11.3e}  "
              f"{s['spread_rel']:11.3e}  {sig:>4}  {str(s['quotable']):>14}  {s['verdict']}")
    n_bound = sum(1 for k in keys if summary[k]["verdict"] == "BOUND")
    print(f"\n{len(keys)} quantities: {len(keys) - n_bound} value, {n_bound} bound "
          f"(rule: quote {MIN_SIGFIGS_TO_QUOTE}+ stable significant figures, else a bound)")
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
