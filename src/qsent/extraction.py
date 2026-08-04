"""
Per-layer, per-token residual-stream extraction.

R1 tests THIS FILE. The frozen tolerance is meaningful only if the hook points here are
the same tensor family as the published `qsae.analysis.extract.last_layer_pooled`.

WHAT THE PUBLISHED HOOK CAPTURES
--------------------------------
`last_layer_pooled` registers a forward hook on `model.encoder.layers[-1]` and keeps the
block's forward **output** `o`, then mean-pools over sites. Two facts pin this down:

1. `TFIMTransformer` builds `nn.TransformerEncoder(encoder_layer, num_layers=3,
   enable_nested_tensor=False)` with **no `norm=` argument**, so `encoder.norm is None` and
   the encoder output is exactly the last block's output.
2. `model.final_norm` is applied in `TFIMTransformer.forward` *after* the encoder, so the
   published tensor is **pre-`final_norm`**.

With `norm_first=True` (Pre-LN), a block computes
    x = x + attn(norm1(x));   x = x + ff(norm2(x))
so the block output is a **post-residual-add, un-normalised** residual-stream state.

THE HOOK FAMILY USED HERE
-------------------------
Seven points, all of the same type -- post-residual-add, pre-`final_norm` residual stream,
shape (N, L, d_model):

    k=0  embedding + positional            (input to block 0)
    k=1  block 0, after attention add
    k=2  block 0, after MLP add            (== block 0 output)
    k=3  block 1, after attention add
    k=4  block 1, after MLP add            (== block 1 output)
    k=5  block 2, after attention add
    k=6  block 2, after MLP add            (== block 2 output == THE PUBLISHED HOOK)

`k=6` is byte-for-byte the tensor `last_layer_pooled` hooks; `tests/test_extraction.py`
asserts `pooled(k=6) == last_layer_pooled(...)` to machine precision.

`post_final_norm` is available separately and is **deliberately excluded from the family**:
it has passed through `final_norm`, so it is a different normalisation and mixing it into a
layer axis would compare unlike tensors.
"""

from __future__ import annotations

import numpy as np
import torch

__all__ = ["HOOK_NAMES", "PUBLISHED_HOOK_INDEX", "extract_residual_stream", "mean_pool"]

HOOK_NAMES = [
    "embed",
    "block0_attn", "block0_mlp",
    "block1_attn", "block1_mlp",
    "block2_attn", "block2_mlp",
]
PUBLISHED_HOOK_INDEX = 6            # block2_mlp == encoder.layers[-1] output


def _block_pieces(layer, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Recompute a Pre-LN block's two residual writes explicitly.

    Mirrors nn.TransformerEncoderLayer with norm_first=True. Dropout is 0.0 in these
    models and the module is in eval() mode, so this reconstruction is exact -- asserted
    against the block's own forward output in tests.
    """
    y = layer.norm1(x)
    attn_out, _ = layer.self_attn(y, y, y, need_weights=False)
    x_mid = x + attn_out
    z = layer.norm2(x_mid)
    ff = layer.linear2(layer.dropout(layer.activation(layer.linear1(z))))
    return x_mid, x_mid + ff


@torch.no_grad()
def extract_residual_stream(
    model, h_fields: np.ndarray, batch: int = 256, include_final_norm: bool = False
) -> dict[str, np.ndarray]:
    """Residual stream at every hook point. Values are (N, L, d_model) float64.

    No pooling: the token axis is retained because site i <-> token i is the physically
    canonical bipartition axis this arm depends on.
    """
    model.eval()
    h = torch.as_tensor(np.asarray(h_fields), dtype=torch.float32)
    acc: dict[str, list[torch.Tensor]] = {n: [] for n in HOOK_NAMES}
    if include_final_norm:
        acc["post_final_norm"] = []

    for start in range(0, len(h), batch):
        hb = h[start : start + batch]
        x = model.input_proj(hb.unsqueeze(-1)) + model.pos_emb.unsqueeze(0)
        acc["embed"].append(x.clone())
        for b, layer in enumerate(model.encoder.layers):
            x_mid, x = _block_pieces(layer, x)
            acc[f"block{b}_attn"].append(x_mid.clone())
            acc[f"block{b}_mlp"].append(x.clone())
        if include_final_norm:
            acc["post_final_norm"].append(model.final_norm(x).clone())

    return {k: torch.cat(v, 0).double().numpy() for k, v in acc.items()}


def mean_pool(acts: np.ndarray) -> np.ndarray:
    """Mean over the token axis, reproducing the published pooling."""
    return np.asarray(acts).mean(axis=1)
