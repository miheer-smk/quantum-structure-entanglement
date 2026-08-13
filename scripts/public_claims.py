"""
Claims for the published numbers quoted in the PUBLIC-facing files.

`README.md` and `AUTHOR_HANDOFF.md` are the two documents most readers will ever open, and
until 2026-08-13 they were the only markdown in the repository outside the provenance gate.
That is backwards: the files with the widest audience carried the least-checked numbers, and
the README had gone stale on six of them at once (test count, "no transformer results exist
yet", a bare `1.648e-11`, `8.94e-07` quoted as a value, `2.39`, and the stage line).

Everything here is a PUBLISHED constant read out of the pinned submodule -- the predecessor's
own results files -- through `qsent.pins.published_constant`, which anchors on a section
heading and asserts a single match. Nothing is typed. `array=none` because these come from
the pinned submodule rather than from `$QSAE_ARTIFACTS`; the submodule SHA in the provenance
header is what pins them.

Run:  env/run.sh python scripts/public_claims.py
Out:  scripts/out/public_claims.json
"""

from __future__ import annotations

import sys

from _claims import Registry
from _provenance import provenance, write_json
from qsent.pins import published_constant, submodule_sha

SCRIPT = "scripts/public_claims.py"

#: Quoted in AUTHOR_HANDOFF.md. The phase06 numbers are the anchor R1 was gated against; the
#: ra08/ra09 numbers are the two published figures whose checkpoints were never saved -- the
#: quotations are verifiable from the pinned submodule even though the artifacts behind them
#: are gone, which is exactly the distinction AUTHOR_HANDOFF.md exists to draw.
PUBLIC_CONSTANTS = (
    "phase06_lrzz_incr_r2_mean", "phase06_lrzz_incr_r2_sd",
    "phase06_lrzz_incr_r2_min", "phase06_lrzz_incr_r2_max",
    "ra08_learned_gain_L8", "ra08_learned_gain_L10", "ra08_learned_gain_L12",
    "ra09_learned_gain_L8", "ra09_learned_gain_L10",
)


def main() -> int:
    reg = Registry(SCRIPT)
    values = {}
    for name in PUBLIC_CONSTANTS:
        values[name] = published_constant(name)
        reg.add(name, values[name], "none")

    payload = {
        "_provenance": provenance(seeds={}, artifacts=[]),
        "submodule_sha": submodule_sha(),
        "published_constants": values,
        "claims": reg.as_dict(),
    }
    path = write_json(payload, "public_claims.json")
    print(f"wrote {path} with {len(values)} published-constant claims")
    for k, v in values.items():
        print(f"  {k:32s} {v}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
