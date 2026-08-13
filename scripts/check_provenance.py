"""
Provenance gate: every numeric claim in RESULTS_*.md must name the source it actually has.

This is the structural retirement of the stated-source != actual-source error class, which
has now occurred three times (see scripts/_claims.py and DEVIATIONS.md). Patching each
instance leaves the class alive; this compares the two sources mechanically.

Each numeric claim in a results file carries a tag on the line before it:

    <!--prov id=<claim id> script=<path> array=<rel path|none> seed=<seed|none>
             sha256=<12-hex prefix|none> kind=value|bound md=<the literal as written> -->

The gate fails if ANY of these does not hold:

  * the tag names a claim id the generating script did not register        (unknown claim)
  * a registered claim has no tag anywhere                                 (untagged number)
  * the tag's `array` differs from the array the value was computed from   (MISLABELLED -- the
                                                                            section-2 defect)
  * the tag's `sha256` prefix differs from that array's hash as read
  * the named array is absent from $QSAE_ARTIFACTS
  * the `md` literal does not appear in the markdown under the tag         (stale prose)
  * kind=value and the literal disagrees with the claim at its own precision
  * kind=bound and the claim's value does not satisfy the stated inequality

Run:  env/run.sh python scripts/check_provenance.py
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

from qsent.pins import artifact_root, repo_root

# `id=` is required in the pattern so that prose *describing* the tag format does not parse
# as a tag. Documentation about the mechanism must not be mistaken for a use of it.
TAG = re.compile(r"<!--\s*prov\s+(?P<body>id=\S+.*?)-->", re.DOTALL)
LOOKAHEAD_LINES = 14         # how many NON-TAG lines below a tag its literal may appear in
# PREREGISTRATION.md is gated too: it quotes published numbers, and a pre-registration
# that misquotes its own prior is the stated-source defect in the document least able to
# absorb it. README.md and AUTHOR_HANDOFF.md remain outstanding (author ruling, item 4).
RESULTS_GLOBS = ("RESULTS_STAGE*.md", "PREREGISTRATION.md")
CLAIM_SOURCES = ("stage0_regenerated.json", "stage1_regenerated.json",
                 "r1_reproduction.json", "preregistration_prior.json")


def load_claims() -> dict[str, dict]:
    claims: dict[str, dict] = {}
    for name in CLAIM_SOURCES:
        p = repo_root() / "scripts" / "out" / name
        if not p.exists():
            continue
        for cid, c in json.loads(p.read_text()).get("claims", {}).items():
            if cid in claims:
                raise RuntimeError(f"claim id {cid!r} registered by two scripts")
            claims[cid] = c
    if not claims:
        raise RuntimeError("no claims found; run scripts/regen_stage0.py first")
    return claims


def parse_tags(path: Path) -> list[dict]:
    text = path.read_text()
    lines = text.splitlines()
    tags = []
    for m in TAG.finditer(text):
        body = m.group("body")
        fields = dict(re.findall(r"(\w+)=(\S+)", body))
        line_no = text[: m.start()].count("\n")
        fields["_file"] = path.name
        fields["_line"] = line_no + 1
        # Tags are stacked above the table they annotate, so the window must skip over other
        # tag lines rather than counting them against the lookahead budget.
        window, i = [], line_no + 1
        while i < len(lines) and len(window) < LOOKAHEAD_LINES:
            if not lines[i].lstrip().startswith("<!--"):
                window.append(lines[i])
            i += 1
        fields["_below"] = "\n".join(window)
        tags.append(fields)
    return tags


def _sig_figs(literal: str) -> int:
    mant = literal.lower().split("e")[0].lstrip("+-")
    return len(mant.replace(".", "").lstrip("0").rstrip("0")) or 1


def check(results_files: list[Path] | None = None,
          claims: dict[str, dict] | None = None) -> list[str]:
    """Return a list of provenance failures; empty means the gate passes.

    Both arguments exist so `tests/test_provenance_gate.py` can point the gate at a
    deliberately corrupted copy of a results file. A gate that cannot be aimed at a known-bad
    input cannot be shown able to fail, and per CLAUDE.md would not count as a gate.
    """
    claims = load_claims() if claims is None else claims
    failures: list[str] = []
    seen: set[str] = set()

    for path in (results_files if results_files is not None
                 else sorted(q for g in RESULTS_GLOBS for q in repo_root().glob(g))):
        for t in parse_tags(path):
            where = f"{t['_file']}:{t['_line']}"
            cid = t.get("id")
            if cid not in claims:
                failures.append(f"{where}: tag names unknown claim id {cid!r}")
                continue
            seen.add(cid)
            c = claims[cid]

            if t.get("script") != c["script"]:
                failures.append(
                    f"{where}: tag says script={t.get('script')}, claim says {c['script']}")

            # The load-bearing check: is the array named in the prose the array used?
            if t.get("array") != c["array"]:
                failures.append(
                    f"{where}: MISLABELLED SOURCE for {cid} -- tag names array "
                    f"{t.get('array')!r}, value was computed from {c['array']!r}")
            elif c["array"] != "none":
                arrays = c["array"].split(",")
                hashes = c["sha256"].split(",")
                tag_hashes = t.get("sha256", "").split(",")
                for a in arrays:
                    if not (artifact_root() / a).exists():
                        failures.append(f"{where}: array {a} is not in $QSAE_ARTIFACTS")
                if len(tag_hashes) != len(hashes):
                    failures.append(
                        f"{where}: tag lists {len(tag_hashes)} hash(es) for {len(arrays)} array(s)")
                elif not all(h.startswith(p) and p for h, p in zip(hashes, tag_hashes)):
                    failures.append(
                        f"{where}: tag sha256={t.get('sha256')} is not a prefix list of the "
                        f"artifact hashes {[h[:12] for h in hashes]}")

            if t.get("seed") != c["seed"]:
                failures.append(f"{where}: tag seed={t.get('seed')}, claim seed={c['seed']}")
            if t.get("kind") != c["kind"]:
                failures.append(f"{where}: tag kind={t.get('kind')}, claim kind={c['kind']}")

            # The prose uses the typographic minus U+2212; tags carry ASCII hyphen. Normalise
            # so a rendering convention cannot masquerade as a provenance failure.
            md = t.get("md", "")
            literal = md.lstrip("<>~≈")
            if md and md not in t["_below"].replace("−", "-"):
                failures.append(f"{where}: literal {md!r} does not appear under its tag")
            try:
                num = float(literal)
            except ValueError:
                failures.append(f"{where}: md={md!r} is not numeric")
                continue

            if c["kind"] == "bound":
                if not abs(c["value"]) < abs(num):
                    failures.append(
                        f"{where}: bound violated for {cid} -- value {c['value']:.6e} "
                        f"is not below {num:.6e}")
            else:
                sig = _sig_figs(literal)
                tol = 0.5 * 10.0 ** (-(sig - 1))
                rel = abs(c["value"] - num) / abs(num) if num else abs(c["value"])
                if rel > tol:
                    failures.append(
                        f"{where}: value disagrees for {cid} -- markdown {num:.6e}, "
                        f"regenerated {c['value']:.6e} (rel {rel:.2e} > {tol:.2e})")

    for cid in sorted(set(claims) - seen):
        failures.append(f"UNTAGGED: claim {cid!r} is registered but appears in no results file")
    return failures


def main() -> int:
    failures = check()
    n = len(load_claims())
    if failures:
        print(f"PROVENANCE GATE FAILED -- {len(failures)} problem(s) over {n} claims:\n")
        for f in failures:
            print(f"  {f}")
        return 1
    print(f"provenance gate PASSED -- {n} claims, every one tagged and matching its source")
    return 0


if __name__ == "__main__":
    sys.exit(main())
