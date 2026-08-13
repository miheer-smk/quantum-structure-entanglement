"""
Machine-readable provenance claims: the structural fix for stated-source != actual-source.

THE PROBLEM THIS RETIRES
------------------------
Three times now a committed number has had a source different from the one stated next to it:

1. Fabricated `h` vectors labelled as pinned realizations.
2. An unanchored regex returning `0.560` (partial correlation) where `0.0283` (incremental
   R^2) was claimed.
3. `RESULTS_STAGE0.md` section 2 attributing the hook-equality measurement to "test
   realizations" when the value came from another array.

Each passed every check in existence, because **no check compared the stated source with the
actual source**. Patching the third instance would leave the class alive.

THE MECHANISM
-------------
A generating script registers a `Claim` for every number it produces, recording the value AND
its provenance: which script, which input array, which seed, and the SHA-256 of that array as
read. Each numeric claim in a `RESULTS_*.md` file carries a matching tag:

    <!--prov id=... script=... array=... seed=... sha256=... kind=... -->

`scripts/check_provenance.py` fails if the value disagrees, if the tag is missing, if the tag
names an array that does not exist, if the recorded hash does not match the artifact on disk,
or -- the case that matters -- **if the array named in the tag is not the array the value was
actually computed from**. A mislabelled tag is caught because the claim carries the truth
independently of the prose.

KINDS
-----
`value` -- quoted to the number of significant figures `scripts/audit_precision.py` measures
           to be STABLE across configurations, with the spread stated alongside. The harness
           compares at exactly the precision the markdown states, so quoting fewer figures is
           always safe and quoting more is a gate failure waiting to happen.
`bound`  -- fewer than two stable significant figures, so the digits are an order-of-magnitude
            statement at best. The markdown states an inequality and the harness asserts the
            inequality, never the digits.

The rule that assigns these was corrected by the author on 2026-08-13; the earlier version
("moves at all under reconfiguration -> bound") is recorded in DEVIATIONS.md along with what
it cost. Kinds are DERIVED from the audit output by `regen_stage0.audited_kind`, never typed:
a hand-typed "this one is stable to 4 figures" would be a measured constant inlined as a
literal, which is the error class this whole mechanism exists to retire.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Literal

from qsent.pins import artifact_root, sha256_file

Kind = Literal["value", "bound"]


@dataclass
class Claim:
    """One numeric claim, with the provenance that makes it checkable.

    `array` is a comma-separated list (no spaces) of paths relative to $QSAE_ARTIFACTS, or
    "none" for a purely analytic quantity. A list is required rather than convenient: the
    occupancy counts are pooled over three ensembles, and recording only the first would
    reintroduce -- inside the very harness built to catch it -- a claim whose stated source is
    narrower than its actual one. `sha256` is the matching comma-separated list of hashes.
    """
    id: str
    value: float
    script: str
    array: str
    kind: Kind = "value"
    seed: str = "none"              # RNG seed, or "none" when the computation is deterministic
    sha256: str = field(default="")

    def __post_init__(self) -> None:
        if " " in self.array:
            raise ValueError(f"claim {self.id}: array list must not contain spaces")
        if not self.sha256:
            self.sha256 = "none" if self.array == "none" else ",".join(
                sha256_file(artifact_root() / a) for a in self.array.split(","))


class Registry:
    """Collects claims and refuses duplicate ids."""

    def __init__(self, script: str) -> None:
        self.script = script
        self._claims: dict[str, Claim] = {}

    def add(self, id: str, value: float, array: str, *, kind: Kind = "value",
            seed: str = "none") -> None:
        if id in self._claims:
            raise ValueError(f"duplicate claim id: {id}")
        self._claims[id] = Claim(id=id, value=float(value), script=self.script,
                                 array=array, kind=kind, seed=seed)

    def as_dict(self) -> dict[str, dict]:
        return {k: asdict(v) for k, v in sorted(self._claims.items())}
