"""
The extraction gate. R1 depends on this file and on nothing else in the same way.

R1 compares this arm's per-layer probe gain against a published number measured through
`qsae.analysis.extract.last_layer_pooled`. That comparison is meaningful only if the tensor
this arm extracts at `k=6` IS the tensor the published pipeline read, and if the six other
hook points are the residual stream they claim to be rather than a hand-written approximation
of it. Both are asserted here, on pinned artifacts, before any model number is produced.

WHAT IS ASSERTED
----------------
(a) `mean_pool(k=6)` equals `last_layer_pooled(model, h)` at float32 precision, on the arrays
    R1 ACTUALLY CONSUMES -- `data/ra03_states_L8_N800_s{42,43,44}.pt`, all 800 realizations of
    each, with the pinned `ms_trained/seed1` checkpoint. Asserted as a BOUND, never as digits:
    the three per-array values are exactly 14, 17 and 15 times 2^-24, so a quoted figure
    reports which array was picked rather than a measurement. See RESULTS_STAGE0.md section 2
    and DEVIATIONS.md 2026-08-11.

(b) `_block_pieces` -- a HAND-WRITTEN reimplementation of a Pre-LN block's two residual
    writes -- reproduces the block's ACTUAL forward output, at every hook point k=0..6, not
    only at k=6. This is the assertion that matters most: a silent divergence inside that
    function would move every downstream number, including R1's, with nothing to show for it.
    The comparison is against tensors captured from a real `model(h)` forward pass:

        k=0        forward PRE-hook on `model.encoder`         -> the tensor entering block 0
        k=1,3,5    forward PRE-hook on `layer.norm2`           -> x_mid, the post-attention
                                                                  residual write, which is
                                                                  exactly what norm2 is fed
        k=2,4,6    forward hook on `model.encoder.layers[b]`   -> the block's own output

    The intermediate points are the reason for the `norm2` pre-hook: a Pre-LN block does not
    return x_mid, but it does hand it to `norm2`, so the real forward pass can be made to
    surrender it without reconstructing anything.

(c) FAILURE DEMONSTRATIONS. Per the standing rule in CLAUDE.md, a check that has never been
    shown to reject a wrong hook is not a gate. Three wrong hooks are put through the SAME
    assertions and must be rejected:

      * `post_final_norm` against the published tensor -- the exclusion in section 2, which
        differs by more than 2.4;
      * an off-by-one BLOCK index (block b's reconstruction against block b+1's output);
      * an off-by-one POINT within a block (the attention write against the MLP write).

    Each demonstration asserts that the rejection happens AND prints the rejection message, so
    the gate's teeth are visible in the test output rather than merely asserted to exist.
"""

from __future__ import annotations

import os

import numpy as np
import pytest
import torch

from qsent.extraction import (
    HOOK_NAMES, PUBLISHED_HOOK_INDEX, _block_pieces, extract_residual_stream, mean_pool,
)
from qsent.pins import load_checkpoint, verify_pin

needs_artifacts = pytest.mark.skipif(
    not os.environ.get("QSAE_ARTIFACTS"), reason="QSAE_ARTIFACTS not configured")

#: The arrays phase06 -- and therefore R1 -- evaluates on. Not a training split: see
#: DEVIATIONS.md 2026-08-11, where measuring this on the wrong array was the third instance
#: of the stated-source error class.
EVAL_SEEDS = (42, 43, 44)
CHECKPOINT_SEED = 1

#: float32 has a 24-bit significand. Agreement between the published float32 path and this
#: float64-casting one is a count of these, not a continuous quantity.
FLOAT32_ULP = 2.0 ** -24

#: The bound section 2 states. Deliberately looser than every measured value (14-17 ULPs
#: ~ 8.3e-07 to 1.1e-06) and far tighter than the 2.4 by which the excluded hook differs, so
#: it separates the two by six orders of magnitude.
PUBLISHED_AGREEMENT_BOUND = 2e-06

#: The reconstruction tolerance is ZERO because zero is what is measured: `_block_pieces`
#: reproduces the real forward pass BITWISE at all seven hook points. A tolerance chosen
#: above the measurement would let a real divergence hide underneath it. If this ever fails
#: with a small nonzero value, that is a fused/parallel kernel difference and a deliberate
#: decision to make -- not something to paper over by loosening the constant.
RECONSTRUCTION_TOL = 0.0


# ---------------------------------------------------------------------------------------
# Fixtures: the pinned model and the pinned eval arrays
# ---------------------------------------------------------------------------------------

@pytest.fixture(scope="module")
def model():
    from qsae.reverse_arrow.transformer import TFIMTransformer
    ckpt = load_checkpoint(CHECKPOINT_SEED)
    m = TFIMTransformer(ckpt["cfg"])
    m.load_state_dict(ckpt["model_state_dict"])
    m.eval()
    return m


def _eval_fields(seed: int) -> np.ndarray:
    path = verify_pin(f"data/ra03_states_L8_N800_s{seed}.pt", "ensemble.sha256")
    blob = torch.load(path, map_location="cpu", weights_only=False)
    return np.asarray(blob["h_fields"], dtype=np.float64)


@pytest.fixture(scope="module")
def published_agreement(model) -> dict[int, float]:
    """max |mean_pool(k=6) - last_layer_pooled| on all 800 realizations of each eval array."""
    from qsae.analysis.extract import last_layer_pooled

    out = {}
    for seed in EVAL_SEEDS:
        h = _eval_fields(seed)
        published = np.asarray(last_layer_pooled(model, h), dtype=np.float64)
        ours = mean_pool(extract_residual_stream(model, h)[HOOK_NAMES[PUBLISHED_HOOK_INDEX]])
        assert ours.shape == published.shape, f"shape mismatch on s{seed}"
        out[seed] = float(np.max(np.abs(ours - published)))
    return out


@pytest.fixture(scope="module")
def forward_capture(model) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """(captured from a real forward pass, produced by extract_residual_stream), keyed by hook.

    The capture side touches none of the extraction code: it registers hooks on the model's
    own modules and runs `model(h)`. That is what makes it an independent reference rather
    than a restatement of the thing under test.
    """
    h = _eval_fields(EVAL_SEEDS[0])[:256]
    hb = torch.as_tensor(h, dtype=torch.float32)

    captured: dict[str, torch.Tensor] = {}
    handles = [model.encoder.register_forward_pre_hook(
        lambda m, i: captured.__setitem__("embed", i[0].detach().clone()))]
    for b, layer in enumerate(model.encoder.layers):
        handles.append(layer.register_forward_hook(
            lambda m, i, o, b=b: captured.__setitem__(f"block{b}_mlp", o.detach().clone())))
        handles.append(layer.norm2.register_forward_pre_hook(
            lambda m, i, b=b: captured.__setitem__(f"block{b}_attn", i[0].detach().clone())))
    try:
        with torch.no_grad():
            model(hb)
    finally:
        for handle in handles:
            handle.remove()

    assert set(captured) == set(HOOK_NAMES), (
        f"the real forward pass did not surrender every hook point: missing "
        f"{sorted(set(HOOK_NAMES) - set(captured))}. A fused fast path may have bypassed the "
        f"submodules the pre-hooks are attached to, in which case this gate is not comparing "
        f"against the forward pass it believes it is.")

    actual = {k: v.double().numpy() for k, v in captured.items()}
    ours = extract_residual_stream(model, h)
    return actual, ours


# ---------------------------------------------------------------------------------------
# (a) k=6 IS the published tensor
# ---------------------------------------------------------------------------------------

def _assert_matches_published(diff: float, label: str) -> None:
    """The single assertion (a) and demonstration (c) both go through."""
    assert diff < PUBLISHED_AGREEMENT_BOUND, (
        f"{label}: max |mean_pool - last_layer_pooled| = {diff:.6e}, which is not below the "
        f"float32 agreement bound {PUBLISHED_AGREEMENT_BOUND:.0e}. This tensor is NOT the "
        f"published hook, so R1's premise does not hold on it "
        f"({diff / FLOAT32_ULP:.1f} float32 ULPs).")


@needs_artifacts
@pytest.mark.parametrize("seed", EVAL_SEEDS)
def test_pooled_k6_matches_the_published_hook(published_agreement, seed):
    """(a) On every array R1 consumes, k=6 agrees with the published tensor to < 2e-06."""
    _assert_matches_published(published_agreement[seed],
                              f"k={PUBLISHED_HOOK_INDEX} vs last_layer_pooled on s{seed}")


@needs_artifacts
def test_the_agreement_is_an_exact_count_of_float32_ulps(published_agreement):
    """Why (a) is a bound and not a number: the value is the array's ULP count.

    Each per-array agreement is an EXACT integer multiple of 2^-24 -- 14, 17 and 15. It is
    therefore a property of which array was chosen, not a measurement of the extraction, and
    quoting any one of them to three significant figures reports the choice. This is the
    finding that justifies the bound; asserting it keeps the justification alive in the suite
    rather than only in prose.
    """
    for seed, diff in published_agreement.items():
        ulps = diff / FLOAT32_ULP
        assert abs(ulps - round(ulps)) < 1e-9, (
            f"s{seed}: agreement {diff:.17e} is {ulps} ULPs, not an integer count. The "
            f"difference is no longer pure float32 representation error, which is the premise "
            f"for reporting it as a bound.")
        assert 1 <= round(ulps) <= 64, f"s{seed}: {round(ulps)} ULPs is outside the expected range"


@needs_artifacts
def test_the_published_agreement_differs_between_arrays(published_agreement):
    """The other half of the justification: the value is not array-independent.

    If all three arrays gave the same number, quoting it would be legitimate and the bound
    would be needless caution. They do not, which is why section 2 reports an inequality.
    """
    assert len(set(published_agreement.values())) > 1, (
        f"every eval array now gives the same agreement {published_agreement}; the reason "
        f"section 2 states a bound rather than a value no longer holds and should be revisited")


# ---------------------------------------------------------------------------------------
# (b) the hand-written Pre-LN reconstruction reproduces the real forward pass
# ---------------------------------------------------------------------------------------

def _assert_reconstructs(ours: np.ndarray, actual: np.ndarray, label: str) -> None:
    """The single assertion (b) and demonstration (c) both go through."""
    assert ours.shape == actual.shape, f"{label}: shape {ours.shape} vs {actual.shape}"
    diff = float(np.max(np.abs(ours - actual)))
    scale = float(np.max(np.abs(actual)))
    assert diff <= RECONSTRUCTION_TOL, (
        f"{label}: the manual Pre-LN reconstruction differs from the block's actual forward "
        f"output by {diff:.6e} (relative to a max magnitude of {scale:.4f}, i.e. "
        f"{diff / scale / FLOAT32_ULP:.1f} float32 ULPs). _block_pieces no longer reproduces "
        f"the model; every downstream number, R1 included, is measured on a tensor the model "
        f"does not compute.")


@needs_artifacts
@pytest.mark.parametrize("name", HOOK_NAMES)
def test_hook_point_reproduces_the_actual_forward_pass(forward_capture, name):
    """(b) Every hook point k=0..6, against the real forward pass, bitwise."""
    actual, ours = forward_capture
    _assert_reconstructs(ours[name], actual[name], f"hook {name}")


@needs_artifacts
def test_the_reconstruction_is_exact_not_merely_close(forward_capture):
    """Recorded as a fact, not just a tolerance: the agreement is bitwise at all 7 points.

    Dropout is 0.0 and the module is in eval() mode, so `_block_pieces` performs the same
    operations in the same order as the block itself. The measured difference is exactly zero
    at every point; this test states that as the claim, so that a future change producing
    "small but nonzero" is a visible event rather than a silent slide.
    """
    actual, ours = forward_capture
    worst = {n: float(np.max(np.abs(ours[n] - actual[n]))) for n in HOOK_NAMES}
    assert set(worst.values()) == {0.0}, f"reconstruction is no longer bitwise exact: {worst}"


@needs_artifacts
def test_every_hook_point_is_distinct(forward_capture):
    """Guards the axis itself: seven hook points that were secretly the same tensor would
    make the layer axis H3/H4 run along a fiction, and every pairwise assertion above would
    still pass."""
    actual, _ = forward_capture
    for i, a in enumerate(HOOK_NAMES):
        for b in HOOK_NAMES[i + 1:]:
            assert float(np.max(np.abs(actual[a] - actual[b]))) > 1e-6, (
                f"hook points {a} and {b} are the same tensor")


# ---------------------------------------------------------------------------------------
# (c) failure demonstrations -- the same assertions, given wrong hooks, must reject
# ---------------------------------------------------------------------------------------

def _rejected(fn, *args) -> str:
    """Run an assertion expected to FAIL; return its message, or fail if it passed."""
    try:
        fn(*args)
    except AssertionError as exc:
        return str(exc).splitlines()[0]
    raise AssertionError(f"{fn.__name__} ACCEPTED a deliberately wrong hook. The gate is "
                         f"not a gate: it cannot fail on the error it exists to catch.")


@needs_artifacts
def test_rejects_post_final_norm_as_the_published_hook(model, capsys):
    """(c1) The excluded hook, put through assertion (a). It differs by more than 2.4."""
    from qsae.analysis.extract import last_layer_pooled

    h = _eval_fields(EVAL_SEEDS[0])
    published = np.asarray(last_layer_pooled(model, h), dtype=np.float64)
    acts = extract_residual_stream(model, h, include_final_norm=True)
    diff = float(np.max(np.abs(mean_pool(acts["post_final_norm"]) - published)))

    msg = _rejected(_assert_matches_published, diff, "post_final_norm vs last_layer_pooled")
    with capsys.disabled():
        print(f"\n  REJECTED (c1) post_final_norm: {msg}")
    assert diff > 2.0, "post_final_norm should differ from the published tensor by O(1)"


@needs_artifacts
def test_rejects_an_off_by_one_block_index(forward_capture, capsys):
    """(c2) Block b's reconstruction checked against block b+1's actual output.

    This is the error a copy-paste in the extraction loop would produce, and it is invisible
    to any assertion that only checks shapes or only checks k=6.
    """
    actual, ours = forward_capture
    messages = []
    for b in (0, 1):
        messages.append(_rejected(
            _assert_reconstructs, ours[f"block{b}_mlp"], actual[f"block{b + 1}_mlp"],
            f"block{b}_mlp vs block{b + 1} actual output"))
    with capsys.disabled():
        for m in messages:
            print(f"\n  REJECTED (c2) off-by-one block: {m}")
    assert len(messages) == 2


@needs_artifacts
def test_rejects_an_off_by_one_point_within_a_block(forward_capture, capsys):
    """(c3) The attention write checked against the MLP write of the SAME block.

    Subtler than (c2): same block, same shape, adjacent tensors that differ only by the
    feed-forward residual. If the gate cannot separate these, it cannot certify the layer
    axis at all.
    """
    actual, ours = forward_capture
    msg = _rejected(_assert_reconstructs, ours["block2_attn"], actual["block2_mlp"],
                    "block2_attn vs block2_mlp actual output")
    with capsys.disabled():
        print(f"\n  REJECTED (c3) off-by-one point within a block: {msg}")


@needs_artifacts
def test_rejects_the_published_hook_check_when_given_the_wrong_tensor(forward_capture, capsys):
    """(c4) Assertion (a) applied to k=0 rather than k=6, at the pooled level.

    Demonstrates that (a) fails on a tensor from the wrong END of the stack, not merely on
    one that has been normalised.
    """
    actual, _ = forward_capture
    diff = float(np.max(np.abs(mean_pool(actual["embed"]) - mean_pool(actual["block2_mlp"]))))
    msg = _rejected(_assert_matches_published, diff, "k=0 (embed) vs the published tensor")
    with capsys.disabled():
        print(f"\n  REJECTED (c4) wrong hook index: {msg}")
