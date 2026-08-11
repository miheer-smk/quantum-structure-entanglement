"""
Diff regenerated Stage 0 values against what RESULTS_STAGE0.md actually commits.

The committed values are PARSED out of the markdown, never typed here. Typing them would
reintroduce the exact failure this repository has already suffered twice: a constant that
looked right, from a source nobody checked. Every pattern is anchored to a section heading
and asserts exactly one match, the same discipline as `qsent.pins.published_constant`.

Run:  env/run.sh python scripts/diff_stage0.py
"""

from __future__ import annotations

import json
import re
import sys

from qsent.pins import repo_root


def _section(text: str, heading: str) -> str:
    if heading not in text:
        raise RuntimeError(f"section {heading!r} not found in RESULTS_STAGE0.md")
    return text.split(heading, 1)[1].split("\n## ", 1)[0]


def _one(section: str, pattern: str, label: str) -> str:
    hits = re.findall(pattern, section)
    if len(hits) != 1:
        raise RuntimeError(f"expected exactly one match for {label}, got {len(hits)}: {hits[:5]}")
    return hits[0]


def committed() -> dict[str, float]:
    text = (repo_root() / "RESULTS_STAGE0.md").read_text()
    out: dict[str, float] = {}

    head = text.split("---", 1)[0]
    out["worst_ed_vs_free_fermion"] = float(
        _one(head, r"cross-validation cases:\s*\*\*([0-9.]+e-[0-9]+)\*\*", "worst"))

    s1 = _section(text, "## 1. Did the uniform golden values pass")
    for hv in ("0.5", "1.0", "2.0"):
        out[f"site_blind_uniform_h{hv}_L8"] = float(
            _one(s1, rf"\|\s*uniform h = {re.escape(hv)}, L = 8\s*\|\s*([0-9.]+e-[0-9]+)\s*\|",
                 f"site_blind uniform {hv}"))
    for label, tag in [("r0", "r0"), ("r1", "r1"), ("r2", "r2"), ("r3", "r3"), ("r4", "r4")]:
        out[f"site_blind_disordered_{label}"] = float(
            _one(s1, rf"\|\s*disordered {tag} \(δ = [+−]?[0-9.]+\)\s*\|\s*([0-9.]+e-[0-9]+)\s*\|",
                 f"site_blind {tag}"))

    s2 = _section(text, "## 2. Are the 8 hook points")
    out["max_abs_diff_k6_vs_published"] = float(
        _one(s2, r"max \|mean_pool\(k=6\)\s*−\s*last_layer_pooled\(\.\.\.\)\|\s*=\s*([0-9.]+e-[0-9]+)",
             "k6 equality"))
    out["max_abs_diff_post_final_norm_vs_published"] = float(
        _one(s2, r"max \|mean_pool\(post_final_norm\).*?=\s*([0-9.]+e\+[0-9]+)", "post_final_norm"))
    rms = _one(s2, r"RMS magnitude grows monotonically through the stack \(([^)]+)\)", "rms")
    for i, v in enumerate(re.findall(r"[0-9.]+", rms)):
        out[f"rms_hook{i}"] = float(v)

    s3 = _section(text, "## 3. Do the 5 cross-validation realizations")
    for label, name in [("r0_ordered", "r0_ordered"), ("r1", "r1"), ("r2_critical", "r2_critical"),
                        ("r3", "r3"), ("r4_paramagnetic", "r4_paramagnetic")]:
        row = _one(s3, rf"\|\s*`{re.escape(name)}`\s*\|\s*(\d+)\s*\|", f"index {name}")
        out[f"index_{label}"] = float(row)

    s4 = _section(text, "## 4. Disorder-ensemble characterization")
    out["E_ln_h"] = float(_one(s4, r"\|\s*`E\[ln h\]`\s*\|\s*\*\*(−[0-9.]+)\*\*", "E[lnh]")
                          .replace("−", "-"))
    out["sd_ln_h"] = float(_one(s4, r"\|\s*`sd\[ln h\]`\s*\|\s*\*\*([0-9.]+)\*\*", "sd[lnh]"))
    for L in (8, 10, 12):
        out[f"mean_delta_r_L{L}"] = float(
            _one(s4, rf"\*\*\+([0-9.]+)\*\*\s*\(L={L}\)", f"drift L{L}"))
    for edge, key in [("0.05", "0.05"), ("0.10", "0.10"), ("0.25", "0.25"), ("0.50", "0.50")]:
        out[f"occupancy_{key}_count"] = float(
            _one(s4, rf"\\\|δ_r\\\| < {re.escape(edge)}\s*\|\s*([0-9,]+)\s*\|",
                 f"occ {edge}").replace(",", ""))
    out["occupancy_neg_count"] = float(
        _one(s4, r"δ_r < 0 \(paramagnetic\)\s*\|\s*([0-9,]+)\s*\|", "occ neg").replace(",", ""))
    return out


def regenerated() -> dict[str, float]:
    d = json.loads((repo_root() / "scripts" / "out" / "stage0_regenerated.json").read_text())
    s1, s2, s3, s4 = (d["section1_cross_validation"], d["section2_hook_family"],
                      d["section3_spanning"], d["section4_ensemble"])
    labels = ["r0_ordered", "r1", "r2_critical", "r3", "r4_paramagnetic"]
    out = {
        "worst_ed_vs_free_fermion": s1["worst_ed_vs_free_fermion"],
        "max_abs_diff_k6_vs_published": s2["max_abs_diff_k6_vs_published"],
        "max_abs_diff_post_final_norm_vs_published": s2["max_abs_diff_post_final_norm_vs_published"],
        "E_ln_h": s4["E_ln_h_closed_form"],
        "sd_ln_h": s4["sd_ln_h_closed_form"],
    }
    for hv in ("0.5", "1.0", "2.0"):
        out[f"site_blind_uniform_h{hv}_L8"] = s1["site_blind_vs_ed"][f"uniform_h{hv}_L8"]
    for short, full in zip(["r0", "r1", "r2", "r3", "r4"], labels):
        out[f"site_blind_disordered_{short}"] = s1["site_blind_vs_ed"][f"disordered_{full}"]
    from qsent.extraction import HOOK_NAMES
    for i, name in enumerate(HOOK_NAMES):
        out[f"rms_hook{i}"] = s2["rms_by_hook"][name]
    for l in labels:
        out[f"index_{l}"] = float(s3["indices"][l])
    for L in (8, 10, 12):
        out[f"mean_delta_r_L{L}"] = s4["mean_delta_r_by_L"][f"L{L}"]
    for edge in ("0.05", "0.10", "0.25", "0.50"):
        out[f"occupancy_{edge}_count"] = float(s4["occupancy"][f"abs_delta_lt_{edge}"]["count"])
    out["occupancy_neg_count"] = float(s4["occupancy"]["delta_lt_0"]["count"])
    return out


def main() -> int:
    c, r = committed(), regenerated()
    missing = set(c) - set(r)
    if missing:
        raise RuntimeError(f"regenerated is missing committed keys: {sorted(missing)}")

    rows, n_bad = [], 0
    for k in sorted(c):
        cv, rv = c[k], r[k]
        # Compare at the precision the results file actually states. Significant figures are
        # counted from the MANTISSA only: a first version of this counted the exponent digits
        # too, turning "1.648e-11" into 6 significant figures and manufacturing two mismatches
        # that were not real. Kept as a comment because a false MISMATCH is the same category
        # of error as a false match -- a number reported without checking its source.
        sig = len(f"{cv:e}".split("e")[0].lstrip("-").replace(".", "").rstrip("0")) or 1
        rel = abs(rv - cv) / abs(cv) if cv != 0 else abs(rv)
        ok = (cv == rv) or rel < 0.5 * 10.0 ** (-(sig - 1))
        n_bad += (not ok)
        rows.append((k, cv, rv, rv - cv, rel, sig, "match" if ok else "MISMATCH"))

    w = max(len(x[0]) for x in rows)
    print(f"{'quantity'.ljust(w)}  {'committed':>16}  {'regenerated':>16}  {'delta':>13}  "
          f"{'rel':>9}  {'sf':>2}  verdict")
    print("-" * (w + 74))
    for k, cv, rv, dv, rel, sig, verdict in rows:
        print(f"{k.ljust(w)}  {cv:16.6e}  {rv:16.6e}  {dv:13.3e}  {rel:9.2e}  {sig:2d}  {verdict}")
    print(f"\n{len(rows)} quantities compared, {n_bad} MISMATCH")
    return 1 if n_bad else 0


if __name__ == "__main__":
    sys.exit(main())
