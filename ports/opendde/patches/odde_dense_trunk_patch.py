"""Remove BOTH dense N_atom x N_atom scratch tensors from OpenDDE's local
(windowed) atom attention -- v2 of odde_dense_mask_patch.

Stock OpenDDE 1.1.1 lays atoms out in 32-query x 128-key windows for the atom
transformer, and in two sibling functions it builds the window structure by
materialising a DENSE N_atom^2 tensor, copying it through
`optimized_concat_split` (reshape -> permute -> reshape: the permute forces a
full copy) and then carving the windows out with `unfold`:

  1. rearrange_qk_to_dense_trunk(compute_mask=True)
       pad_mask = q[0].new_ones(n + q_pad, n + pad_left + pad_right)   # fp32 (ref_pos)
     -> the window MASK, once per prediction (update_input_feature_dict).
  2. rearrange_to_dense_trunk(attn_bias=None)
       attn_bias = q.new_zeros(n + q_pad, n + pad_left + pad_right)    # q's dtype: bf16 here
       attn_bias[pads] = -inf
     -> the window PADDING BIAS, on EVERY local-attention call: every atom
        transformer block, encoder and decoder, every diffusion step.

On Setonix (1) was the capacity wall for both the single-GCD and the sharded
line; with (1) patched, one GCD at 8,192 tokens then died on (2)
(`Tried to allocate 8.72 GiB` = 2 bytes x 68,400^2 atoms, in
optimized_concat_split, called from _local_attention). Both have the same
closed form. With w the window, qi the query and kk the key offset:

    inside[w, qi, kk] = (row < n) & (pad_left <= col < pad_left + n)
    mask_trunked      = inside                                   (bool)
    attn_bias_trunked = where(inside, 0, -inf)                   (q's dtype)

where (row, col) is the stock tensors' element that unfold+transpose places at
[w, qi, kk]. That decoding is done literally, from the stock arithmetic, so the
result has the same shape and the same bits.

MONKEYPATCH: the installed opendde package is untouched (stock/check_pins.py
passes). Armed by patches/hook2/sitecustomize.py through the kit's own
opt_core.autoload.patch_attr_at_import. torch is imported lazily: this module is
loaded at interpreter start, before the kit's allocator policy binds.
"""
from __future__ import annotations

import math
import os
import sys

_STATE = {"installed": [], "mask_calls": 0, "bias_calls": 0}


def _window_geometry(n: int, n_queries: int, n_keys: int):
    """The stock padding arithmetic, reproduced literally (incl. the int() in pad_right)."""
    n_trunks = int(math.ceil(n / n_queries))
    q_pad_length = n_trunks * n_queries - n
    pad_left = (n_keys - n_queries) // 2
    pad_right = int((n_trunks - 1 / 2) * n_queries + n_keys / 2 - n + 1 / 2)
    return n_trunks, q_pad_length, pad_left, pad_right


def _inside(q0, n: int, n_queries: int, n_keys: int):
    """bool [*lead, n_win, n_queries, n_keys]: True where the stock dense tensor is
    'inside' (mask 1 / bias 0), False on the pads. lead = (1,) * len(q0.shape[:-2])."""
    import torch
    n_trunks, q_pad_length, pad_left, pad_right = _window_geometry(n, n_queries, n_keys)
    D = n + q_pad_length                          # dense rows  (= n_trunks * n_queries)
    E = n + pad_left + pad_right                  # dense cols
    L = (D // n_queries) * E                      # last dim of optimized_concat_split's output
    step = E + n_queries                          # the stock unfold step
    n_win = (L - n_keys) // step + 1              # torch.unfold's window count
    dev = q0.device
    w = torch.arange(n_win, device=dev).view(-1, 1, 1)
    qi = torch.arange(n_queries, device=dev).view(1, -1, 1)
    kk = torch.arange(n_keys, device=dev).view(1, 1, -1)
    flat = w * step + kk                          # index into the concat axis
    row = (flat // E) * n_queries + qi            # dense row that element came from
    col = flat % E                                # dense column
    inside = (row < n) & (col >= pad_left) & (col < pad_left + n)
    return inside.reshape(*((1,) * len(q0.shape[:-2])), n_win, n_queries, n_keys)


def make_patched_qk(orig):
    """rearrange_qk_to_dense_trunk: the window MASK without the dense fp32 intermediate."""
    def rearrange_qk_to_dense_trunk(q, k, dim_q, dim_k, n_queries: int = 32, n_keys: int = 128,
                                    compute_mask: bool = True):
        q0 = (q if isinstance(q, list) else [q])[0]
        d0 = (dim_q if isinstance(dim_q, list) else [dim_q])[0]
        n = q0.size(d0)                           # read BEFORE orig: basic_checks mutates the dim lists
        qt, kt, info = orig(q, k, dim_q, dim_k, n_queries=n_queries, n_keys=n_keys, compute_mask=False)
        if compute_mask:
            info["mask_trunked"] = _inside(q0, n, n_queries, n_keys)
            _STATE["mask_calls"] += 1
        return qt, kt, info
    rearrange_qk_to_dense_trunk.__wrapped__ = orig
    return rearrange_qk_to_dense_trunk


def make_patched_dense(orig):
    """rearrange_to_dense_trunk: the window PADDING BIAS without the dense intermediate.
    A caller that passes its own dense attn_bias gets the stock path, unchanged."""
    def rearrange_to_dense_trunk(q, k, v, n_queries, n_keys, attn_bias=None, inf: float = 1e10):
        if attn_bias is not None:
            return orig(q, k, v, n_queries, n_keys, attn_bias=attn_bias, inf=inf)
        import torch
        assert n_keys >= n_queries
        assert n_queries & 0x01 == 0
        assert n_keys & 0x01 == 0
        n = q.shape[-2]
        rq = orig.__globals__["rearrange_qk_to_dense_trunk"]     # the module's current binding
        q_trunked, kv_trunked, info = rq(q=q, k=[k, v], dim_q=-2, dim_k=[-2, -2],
                                         n_queries=n_queries, n_keys=n_keys, compute_mask=False)
        inside = _inside(q, n, n_queries, n_keys)
        zero = torch.zeros((), dtype=q.dtype, device=q.device)
        neg = torch.full((), -inf, dtype=q.dtype, device=q.device)   # same rounding as stock's assignment
        bias = torch.where(inside, zero, neg)
        _STATE["bias_calls"] += 1
        return q_trunked, kv_trunked[0], kv_trunked[1], bias, info["q_pad"]
    rearrange_to_dense_trunk.__wrapped__ = orig
    return rearrange_to_dense_trunk


def _logged(maker, attr):
    def factory(orig):
        patched = maker(orig)
        _STATE["installed"].append(attr)
        sys.stderr.write("[odde-trunk-patch] %s: direct window construction installed "
                         "(pid %d, rank %s)\n" % (attr, os.getpid(), os.environ.get("ROWPAIR_RANK", "-")))
        sys.stderr.flush()
        return patched
    return factory


make_patched_qk_logged = _logged(make_patched_qk, "rearrange_qk_to_dense_trunk")
make_patched_dense_logged = _logged(make_patched_dense, "rearrange_to_dense_trunk")
