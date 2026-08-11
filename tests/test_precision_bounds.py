"""
Near-machine-precision quantities are BOUNDS, and this file is the evidence.

`RESULTS_STAGE0.md` originally quoted the ED vs free-fermion agreements to four significant
figures. They do not have four significant figures: changing only the BLAS thread count --
which cannot alter a physical result -- moves them. A results file that quotes such a value
asserts a precision the quantity does not possess, and a diff harness comparing against those
digits will report spurious mismatches forever.

So the committed claim is an inequality, and this file asserts the inequality across thread
configurations rather than asserting digits.

Per the standing rule in `CLAUDE.md`, the demonstration that the digits are NOT stable is
carried here as a test too (`test_digits_are_unstable_so_the_bound_is_necessary`). Without it,
"report it as a bound" is an assertion; with it, it is a measurement.

Thread count is fixed when the BLAS library loads, so each configuration runs in a subprocess.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys

import pytest

THREAD_CONFIGS = (1, 4)
GATE = 1e-10                 # the Stage 0 agreement gate the bounds must clear

_WORKER = r"""
import json
import numpy as np
from qsent.exact import entropy_profile_ed
from qsent.free_fermions import entropy_profile_free_fermion
from test_orientation import ASYM_H

out = {}
for hv in (0.5, 1.0, 2.0):
    h, J = np.full(8, hv), np.ones(7)
    out[f"uniform_h{hv}"] = float(np.max(np.abs(
        entropy_profile_ed(J, h) - entropy_profile_free_fermion(J, h))))
h, J = np.array(ASYM_H), np.ones(7)
out["asym_realization"] = float(np.max(np.abs(
    entropy_profile_ed(J, h) - entropy_profile_free_fermion(J, h))))
out["asym_margin"] = float(np.max(np.abs(
    entropy_profile_ed(J, h) - entropy_profile_ed(J, h)[::-1])))
print("JSON" + json.dumps(out))
"""


def _measure(threads: int) -> dict[str, float]:
    env = dict(os.environ, OMP_NUM_THREADS=str(threads), MKL_NUM_THREADS=str(threads),
               OPENBLAS_NUM_THREADS=str(threads))
    r = subprocess.run([sys.executable, "-c", _WORKER], env=env,
                       capture_output=True, text=True, check=True)
    return json.loads(r.stdout.split("JSON", 1)[1])


@pytest.fixture(scope="module")
def across_threads() -> dict[int, dict[str, float]]:
    return {t: _measure(t) for t in THREAD_CONFIGS}


def test_agreement_bound_holds_across_thread_configurations(across_threads):
    """The claim that survives every configuration: agreement is below the Stage 0 gate."""
    for threads, vals in across_threads.items():
        for name, v in vals.items():
            if name == "asym_margin":
                continue
            assert v < GATE, f"{name} = {v:.3e} exceeds the gate at OMP_NUM_THREADS={threads}"


def test_digits_are_unstable_so_the_bound_is_necessary(across_threads):
    """Demonstration, not assertion: at least one agreement figure MOVES under thread count.

    This is what disqualifies the four-significant-figure form. If this test ever fails --
    i.e. every value becomes bit-stable across configurations -- the bounds could legitimately
    be tightened back to values, and this file should be revisited rather than deleted.
    """
    a, b = (across_threads[t] for t in THREAD_CONFIGS)
    moved = [k for k in a if k != "asym_margin" and a[k] != b[k]]
    assert moved, (
        "every agreement figure was bit-identical across thread counts; the justification "
        "for reporting them as bounds no longer holds and should be re-examined")


def test_a_physical_quantity_is_stable_across_thread_configurations(across_threads):
    """Control: the bounds are not an excuse. A real quantity IS reproducible.

    The orientation test's asymmetry margin is a physical entropy difference, not a noise
    floor, and it agrees to ~1e-15 relative across configurations. Without this control,
    "it moves under thread count" could be read as a blanket claim that nothing reproduces.
    """
    a, b = (across_threads[t] for t in THREAD_CONFIGS)
    rel = abs(a["asym_margin"] - b["asym_margin"]) / abs(b["asym_margin"])
    assert rel < 1e-12, f"asymmetry margin moved by {rel:.2e} across thread counts"
