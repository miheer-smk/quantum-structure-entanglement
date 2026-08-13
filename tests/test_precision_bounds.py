"""
Every quoted digit is a measured digit: this file is the evidence for the precision rule.

`RESULTS_STAGE0.md` originally quoted the ED vs free-fermion agreements to four significant
figures with nothing behind the choice of four. Changing only the BLAS thread count -- which
cannot alter a physical result -- moves them, so some of those digits were noise.

THE RULE, AND ITS CORRECTION (author, 2026-08-13)
------------------------------------------------
The first fix over-corrected: *moves at all under thread count -> report a bound*. That filed
the headline agreement, whose four leading digits are identical in every configuration, under
the same verdict as a quantity that does not reproduce its first digit, and so discarded
honestly measured precision from the number carrying the two-solver ground-truth claim.

The rule now is: **quote the significant figures that are stable across configurations, state
the spread, and drop to a bound only below two stable figures.** These tests assert both
directions of that rule, because a rule that can only confirm is not a rule:

  * the headline IS stable at the four figures it is quoted to      (it may be quoted)
  * the headline is NOT stable at five                              (it may not be quoted at 5)
  * a noise-dominated quantity has fewer than two stable figures    (it must stay a bound)
  * a genuinely physical quantity is stable far beyond quoting      (the control)

The middle two are failure demonstrations in the sense CLAUDE.md requires: they assert that
the rule's boundary is where it is claimed to be, so "1.648e-11" cannot quietly become
"1.64806e-11" and "this one is a bound" cannot become a blanket excuse.

Thread count is fixed when the BLAS library loads, so each configuration runs in a subprocess.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys

import pytest

from audit_precision import MAX_SIGFIGS, MIN_SIGFIGS_TO_QUOTE, stable_sigfigs

THREAD_CONFIGS = (1, 4)
GATE = 1e-10                 # the Stage 0 agreement gate the values must clear

#: What RESULTS_STAGE0.md quotes for the headline, and what it must not be able to quote.
HEADLINE_SIGFIGS = 4

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


def _values(across: dict[int, dict[str, float]], name: str) -> list[float]:
    return [across[t][name] for t in THREAD_CONFIGS]


# ---------------------------------------------------------------------------------------
# The gate itself
# ---------------------------------------------------------------------------------------

def test_agreement_clears_the_gate_in_every_thread_configuration(across_threads):
    for threads, vals in across_threads.items():
        for name, v in vals.items():
            if name == "asym_margin":
                continue
            assert v < GATE, f"{name} = {v:.3e} exceeds the gate at OMP_NUM_THREADS={threads}"


# ---------------------------------------------------------------------------------------
# The corrected rule, asserted in both directions
# ---------------------------------------------------------------------------------------

def test_headline_is_stable_at_the_figures_it_is_quoted_to(across_threads):
    """`uniform_h0.5` is the worst case, quoted as 1.648e-11 -- four figures it must possess."""
    vals = _values(across_threads, "uniform_h0.5")
    sig = stable_sigfigs(vals)
    assert sig >= HEADLINE_SIGFIGS, (
        f"the headline is quoted to {HEADLINE_SIGFIGS} s.f. but only {sig} are stable across "
        f"thread counts: {vals}. RESULTS_STAGE0.md must be reduced to {sig} figures.")
    assert {f"{v:.{HEADLINE_SIGFIGS - 1}e}" for v in vals} == {"1.648e-11"}


def test_the_fifth_figure_is_not_stable_so_it_is_not_quoted(across_threads):
    """FAILURE DEMONSTRATION: the boundary is where the results file claims it is.

    If this ever fails, the quantity became MORE reproducible and the file could legitimately
    quote a fifth figure. That is a reason to re-measure and re-quote, not to delete the test:
    the point is that the number of quoted digits is a measurement, not a habit.
    """
    vals = _values(across_threads, "uniform_h0.5")
    assert stable_sigfigs(vals) < MAX_SIGFIGS, f"expected thread sensitivity, got {vals}"
    assert len({f"{v:.4e}" for v in vals}) > 1, (
        f"a fifth significant figure is now stable across thread counts: {vals}")


def test_a_noise_dominated_quantity_stays_a_bound(across_threads):
    """The other side of the rule: `uniform_h1.0` has too few stable figures to quote.

    Without this, "report the stable figures" could be read as licence to quote everything to
    whatever precision one run happened to produce.
    """
    vals = _values(across_threads, "uniform_h1.0")
    sig = stable_sigfigs(vals)
    assert sig < MIN_SIGFIGS_TO_QUOTE, (
        f"uniform_h1.0 now has {sig} stable figures ({vals}); it is reported as a bound in "
        f"RESULTS_STAGE0.md and should be re-examined")


def test_a_physical_quantity_is_stable_across_thread_configurations(across_threads):
    """Control: the bounds are not an excuse. A real quantity IS reproducible.

    The orientation test's asymmetry margin is a physical entropy difference, not a noise
    floor, and it agrees to ~1e-15 relative across configurations. Without this control,
    "it moves under thread count" could be read as a blanket claim that nothing reproduces.
    """
    a, b = (across_threads[t] for t in THREAD_CONFIGS)
    rel = abs(a["asym_margin"] - b["asym_margin"]) / abs(b["asym_margin"])
    assert rel < 1e-12, f"asymmetry margin moved by {rel:.2e} across thread counts"
    assert stable_sigfigs(_values(across_threads, "asym_margin")) >= 10


# ---------------------------------------------------------------------------------------
# The digit counter itself, including the case that makes it non-trivial
# ---------------------------------------------------------------------------------------

def test_stable_sigfigs_counts_digits_not_spread():
    assert stable_sigfigs([1.0, 1.0]) == MAX_SIGFIGS
    assert stable_sigfigs([1.6478596e-11, 1.6480594e-11]) == 4
    assert stable_sigfigs([1.0e-14, 9.0e-14]) == 0
    # Values only 2e-5 apart in relative terms, yet they straddle a rounding boundary and
    # share NO significant figures. Deriving the digit count from the spread would call this
    # four stable figures; rounding, which is what quoting actually does, calls it zero.
    assert stable_sigfigs([1.499999e-11, 1.500001e-11]) == 0
