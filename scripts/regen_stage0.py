"""
Regenerate every number in RESULTS_STAGE0.md from pinned artifacts, with provenance claims.

This script exists because those numbers were originally produced by code that was never
committed. A number in a results file whose generating code is absent is a number nobody can
check -- the same defect class as the fabricated `h` vectors, one level up.

Every value is registered as a `Claim` (see scripts/_claims.py) recording which array it came
from and that array's SHA-256, so `scripts/check_provenance.py` can verify that the source
stated in the markdown IS the source the number came from.

Run:  env/run.sh python scripts/regen_stage0.py
Out:  scripts/out/stage0_regenerated.json

Units: entropy in nats throughout; open boundaries; J == 1 unless stated.
"""

from __future__ import annotations

import json

import numpy as np
import torch

from _claims import Registry
from _provenance import provenance, write_json
from audit_precision import MIN_SIGFIGS_TO_QUOTE
from qsent.pins import repo_root
from qsent.disorder import delta_r, select_spanning_realizations, sigma_lnh_uniform
from qsent.exact import entropy_profile_ed
from qsent.extraction import HOOK_NAMES, extract_residual_stream, mean_pool
from qsent.free_fermions import entropy_profile_free_fermion
from qsent.pins import load_checkpoint, load_ensemble, verify_pin

SCRIPT = "scripts/regen_stage0.py"
UNIFORM_L = (8, 10, 12)
UNIFORM_H = (0.5, 1.0, 2.0)
EVAL_SEEDS = (42, 43, 44)           # the arrays R1 actually consumes
HOOK_CHECKPOINT_SEED = 1
ENS1 = "data/tfim_L8_N50k_seed1.pt"
CKPT1 = f"runs/ms_trained/seed{HOOK_CHECKPOINT_SEED}/best.pt"

LABELS = ["r0_ordered", "r1", "r2_critical", "r3", "r4_paramagnetic"]


def _site_blind_profile(J: np.ndarray, h: np.ndarray) -> np.ndarray:
    """A solver that reads h[0] and applies it everywhere -- the bug the gate must catch."""
    return entropy_profile_free_fermion(J, np.full(len(h), h[0]))


def audited_kind(name: str) -> str:
    """`value` or `bound` for an audited quantity, DERIVED from the measured stability.

    Read from `scripts/out/precision_audit.json` rather than typed here. Typing "this one is
    stable to 4 figures" would put a measured constant into source as a literal -- the shape of
    error this repository has already made three times. If the audit has never been run, the
    conservative answer is `bound`: a precision claim with no measurement behind it is exactly
    what the corrected rule forbids.
    """
    p = repo_root() / "scripts" / "out" / "precision_audit.json"
    if not p.exists():
        return "bound"
    summary = json.loads(p.read_text())["summary"]
    if name not in summary:
        return "bound"
    return "value" if summary[name]["stable_sigfigs"] >= MIN_SIGFIGS_TO_QUOTE else "bound"


def audited_spread(name: str) -> float:
    """The measured spread across BLAS thread counts, read from the audit rather than typed."""
    p = repo_root() / "scripts" / "out" / "precision_audit.json"
    if not p.exists():
        raise RuntimeError(
            "scripts/out/precision_audit.json is absent; run scripts/audit_precision.py first. "
            "A spread cannot be reported without having been measured.")
    return float(json.loads(p.read_text())["summary"][name]["spread_abs"])


def cross_validation(reg: Registry) -> dict:
    """The 14 cross-validation cases: 9 uniform (3 L x 3 h) + 5 spanning disordered.

    Precision policy, corrected by the author 2026-08-13 (see DEVIATIONS.md): each figure is
    registered at the number of significant figures `scripts/audit_precision.py` measures to
    be stable across BLAS thread counts, with the spread stated alongside it. The earlier rule
    -- "moves under thread count at all -> bound" -- discarded four honestly reproducible
    digits from `ed_vs_ff_worst`, the figure carrying the two-solver ground-truth claim.

    `ed_vs_ff_worst` is stable to 4 s.f. and is registered as a VALUE (1.648e-11, spread
    2.0e-15). The uniform site-blind rows split: h=0.5 is stable to 4 s.f., h=2.0 to 2, and
    h=1.0 to only 1, so the last stays a bound.
    """
    h_train = np.asarray(load_ensemble(1)["h_train"], dtype=np.float64)
    idx, deltas = select_spanning_realizations(h_train)

    cases, site_blind = {}, {}
    for L in UNIFORM_L:
        for hv in UNIFORM_H:
            h, J = np.full(L, hv), np.ones(L - 1)
            cases[f"uniform_h{hv}_L{L}"] = float(
                np.max(np.abs(entropy_profile_ed(J, h) - entropy_profile_free_fermion(J, h))))
            if L == 8:
                site_blind[f"uniform_h{hv}_L{L}"] = float(
                    np.max(np.abs(entropy_profile_ed(J, h) - _site_blind_profile(J, h))))

    for label, i in zip(LABELS, idx):
        h, J = h_train[i], np.ones(7)
        cases[f"disordered_{label}"] = float(
            np.max(np.abs(entropy_profile_ed(J, h) - entropy_profile_free_fermion(J, h))))
        site_blind[f"disordered_{label}"] = float(
            np.max(np.abs(entropy_profile_ed(J, h) - _site_blind_profile(J, h))))

    worst = float(max(cases.values()))
    # 4 significant figures survive every thread configuration; the fifth does not. Quoted at
    # exactly that, never more, with the spread reported next to it.
    reg.add("ed_vs_ff_worst", worst, ENS1, kind=audited_kind("ed_vs_ff_worst"))
    # Under the corrected precision rule the spread TRAVELS WITH the value -- every place that
    # quotes 1.648e-11 also quotes 2.0e-15 -- so it is a claim in its own right rather than
    # decoration, and is gated wherever the value is. Read from the audit, never typed.
    reg.add("ed_vs_ff_worst_spread_abs", audited_spread("ed_vs_ff_worst"), ENS1)
    for label in LABELS:
        # The site-blind separation on DISORDERED chains is a real physical separation,
        # ~1e10 times the noise floor, stable to 11-14 significant figures.
        reg.add(f"site_blind_disordered_{label}", site_blind[f"disordered_{label}"], ENS1)
    for hv in UNIFORM_H:
        # On UNIFORM chains the site-blind solver is the same computation, so this row is the
        # ED-vs-free-fermion residual re-measured through a second code path -- bit-identical
        # to the `uniform_h{hv}_L8` case above. It detects nothing about site-blindness, which
        # is section 1's finding. Its REPRODUCIBILITY is a separate question from its meaning:
        # h=0.5 and h=2.0 are stable to 4 and 2 figures, h=1.0 to only 1, so only the last is
        # a bound. A value that reproduces is not thereby a measurement of anything.
        name = f"site_blind_uniform_h{hv}_L8"
        reg.add(name, site_blind[f"uniform_h{hv}_L8"], "none", kind=audited_kind(name))

    return {"n_cases": len(cases), "per_case_ed_vs_free_fermion": cases,
            "worst_ed_vs_free_fermion": worst, "site_blind_vs_ed": site_blind}


def _build_model(seed: int):
    from qsae.reverse_arrow.transformer import TFIMTransformer
    ckpt = load_checkpoint(seed)
    model = TFIMTransformer(ckpt["cfg"])
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model


def hook_family(reg: Registry) -> dict:
    """Is k=6 the published tensor? Measured on the arrays R1 ACTUALLY CONSUMES.

    RESULTS_STAGE0.md originally attributed this to "512 pinned test realizations". It was
    not measured on `h_test`; see DEVIATIONS.md 2026-08-11. Re-measured here on
    `data/ra03_states_L8_N800_s{42,43,44}.pt`, which is what phase06 -- and therefore R1 --
    evaluates on, so R1's premise is asserted on R1's own data.

    Reported as a BOUND. Each per-array value is bit-stable (zero spread across BLAS thread
    counts) but differs BETWEEN arrays, because it is a count of float32 ULPs: the three
    values are exactly 14, 17 and 15 times 2^-24. Quoting any one of them to three
    significant figures reports the ULP count of one particular array.
    """
    from qsae.analysis.extract import last_layer_pooled

    model = _build_model(HOOK_CHECKPOINT_SEED)
    per_seed_k6, per_seed_norm, rms = {}, {}, {}

    for s in EVAL_SEEDS:
        rel = f"data/ra03_states_L8_N800_s{s}.pt"
        h = np.asarray(torch.load(verify_pin(rel, "ensemble.sha256"), map_location="cpu",
                                  weights_only=False)["h_fields"], dtype=np.float64)
        published = np.asarray(last_layer_pooled(model, h), dtype=np.float64)
        acts = extract_residual_stream(model, h, include_final_norm=True)

        per_seed_k6[f"s{s}"] = float(np.max(np.abs(mean_pool(acts["block2_mlp"]) - published)))
        per_seed_norm[f"s{s}"] = float(
            np.max(np.abs(mean_pool(acts["post_final_norm"]) - published)))
        reg.add(f"hook_k6_vs_published_eval_s{s}", per_seed_k6[f"s{s}"], rel, kind="bound")
        reg.add(f"hook_postnorm_vs_published_eval_s{s}", per_seed_norm[f"s{s}"], rel,
                kind="bound")
        if s == EVAL_SEEDS[0]:
            rms = {n: float(np.sqrt(np.mean(acts[n] ** 2))) for n in HOOK_NAMES}

    for n, v in rms.items():
        reg.add(f"rms_{n}", v, f"data/ra03_states_L8_N800_s{EVAL_SEEDS[0]}.pt")

    return {
        "encoder_norm_is_none": model.encoder.norm is None,
        "n_realizations_per_eval_seed": 800,
        "k6_vs_published_by_eval_seed": per_seed_k6,
        "k6_vs_published_max": max(per_seed_k6.values()),
        "k6_ulp_counts_float32": {k: round(v / 2.0 ** -24) for k, v in per_seed_k6.items()},
        "post_final_norm_vs_published_by_eval_seed": per_seed_norm,
        "post_final_norm_vs_published_min": min(per_seed_norm.values()),
        "rms_by_hook": rms,
    }


def spanning(reg: Registry) -> dict:
    h_train = np.asarray(load_ensemble(1)["h_train"], dtype=np.float64)
    idx, deltas = select_spanning_realizations(h_train)
    for label, i, d in zip(LABELS, idx, deltas):
        reg.add(f"index_{label}", float(i), ENS1)
        reg.add(f"delta_r_{label}", float(d), ENS1,
                kind="bound" if abs(d) < 1e-6 else "value")
    return {"indices": {l: int(i) for l, i in zip(LABELS, idx)},
            "delta_r": {l: float(d) for l, d in zip(LABELS, deltas)}}


def ensemble_characterization(reg: Registry) -> dict:
    # h_min/h_max come from the artifact's own `meta`, never a literal -- they determine
    # sigma_lnh and therefore every delta_r. So these are NOT analytic claims: they depend on
    # seed1, and the claim records that rather than "none".
    meta = load_ensemble(1)["meta"]
    m1, sd = sigma_lnh_uniform(meta["h_min"], meta["h_max"])
    reg.add("E_ln_h", m1, ENS1)
    reg.add("sd_ln_h", sd, ENS1)

    per_seed, pooled = {}, []
    for s in (1, 2, 3):
        h = np.asarray(load_ensemble(s)["h_train"], dtype=np.float64)
        per_seed[f"seed{s}"] = float(np.log(h).mean())
        reg.add(f"E_ln_h_seed{s}", per_seed[f"seed{s}"], f"data/tfim_L8_N50k_seed{s}.pt")
        pooled.append(h)
    d = delta_r(np.concatenate(pooled, axis=0))

    # Mean delta_r at each L is sqrt(L) * (-E[ln h] / sigma_lnh) -- derived, never typed. It
    # needs no ensemble at L=10/12, which matters because neither has a pinned one, but it
    # does inherit h_min/h_max from seed1's meta, so seed1 is the recorded source.
    drift = {}
    for L in UNIFORM_L:
        drift[f"L{L}"] = float(np.sqrt(L) * (-m1 / sd))
        reg.add(f"mean_delta_r_L{L}", drift[f"L{L}"], ENS1)

    # Pooled over THREE ensembles: all three are recorded, not just the first.
    pooled_arrays = ",".join(f"data/tfim_L8_N50k_seed{s}.pt" for s in (1, 2, 3))
    n, bands = len(d), {}
    for edge in (0.05, 0.10, 0.25, 0.50):
        c = int((np.abs(d) < edge).sum())
        bands[f"abs_delta_lt_{edge:.2f}"] = {"count": c, "share_pct": 100.0 * c / n}
        reg.add(f"occupancy_{edge:.2f}_count", float(c), pooled_arrays)
    c_neg = int((d < 0).sum())
    bands["delta_lt_0"] = {"count": c_neg, "share_pct": 100.0 * c_neg / n}
    reg.add("occupancy_neg_count", float(c_neg), pooled_arrays)

    return {"E_ln_h_closed_form": m1, "sd_ln_h_closed_form": sd, "lnJ_minus_lnh": -m1,
            "per_seed_E_ln_h": per_seed, "mean_delta_r_by_L": drift,
            "pooled_N": int(n), "occupancy": bands}


def main() -> None:
    reg = Registry(SCRIPT)
    payload = {
        "_provenance": provenance(
            seeds={},                                  # no RNG used anywhere in Stage 0
            artifacts=[f"data/tfim_L8_N50k_seed{s}.pt" for s in (1, 2, 3)]
                      + [f"data/ra03_states_L8_N800_s{s}.pt" for s in EVAL_SEEDS]
                      + [CKPT1],
        ),
        "section1_cross_validation": cross_validation(reg),
        "section2_hook_family": hook_family(reg),
        "section3_spanning": spanning(reg),
        "section4_ensemble": ensemble_characterization(reg),
    }
    payload["claims"] = reg.as_dict()
    path = write_json(payload, "stage0_regenerated.json")
    print(f"wrote {path} with {len(payload['claims'])} provenance claims")


if __name__ == "__main__":
    torch.manual_seed(0)
    main()
