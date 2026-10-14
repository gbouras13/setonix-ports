"""Query sub-blocks for the stock triangle attention's oversized statements. Opt-in: ODDE_ATTN_QBLOCK=1
(patches/hook6; scripts/predict_patched6.sh sets it), with

    ODDE_ATTN_QBLOCK_MIN_GIB  only calls whose logits [*, H, Q, K] exceed this many GiB are blocked (default 4)
    ODDE_ATTN_QBLOCK_GIB      the logits budget of one query block, GiB (default 1; must not exceed MIN)

WHY. The structural refiner's pair stack attends through the kit's kernel=False route (opendde_opt.tp._triatt_fns):
the stock module's own mha, via TriangleAttention._chunk over row batches of the stock chunk, which is 1 at these
sizes. Every call ends in opendde.model.triangular.layers._attention: logits a = q @ k^T, [1, H, N_st, N_st]; a +=
biases; then softmax_no_cast(a), which allocates a second tensor of the same size. At 9,216 tokens (N_st 17,973, H 12,
bf16) that is 7.22 GiB twice, beside the 29.62 GiB structural shard and the 7.22 GiB gathered triangle bias. Job
50126928 died there on ranks 0-6: "Tried to allocate 7.22 GiB ... 6.09 GiB free ... 49.34 GiB allocated by
PyTorch, and 7.34 GiB is reserved by PyTorch but unallocated" (functional.py softmax). The stock has no low-memory
attention, and the kit's query-block knobs only reach its kernel=True route.

WHAT. Wrap layers._attention. When a call's logits would exceed MIN, the original runs over query sub-blocks sized to
the budget, and the outputs are assembled into [*, H, Q, D]. Each bias is sliced on its query axis, unless it
broadcasts there. Keys and values stay whole, and every query row's softmax is whole, so the per-query statement is
unchanged. GEMM launch shapes change (M only), so it is NOT claimed bitwise. Below MIN the original runs untouched:
the trunk's kit route launches 32 rows x 256 queries, 1.7 GiB of logits at 9,216 tokens, so it is not blocked. At
9,216 the refiner's 7.22 GiB call becomes 8 blocks of ~2,489 queries; the transient drops from 14.4 to ~2 GiB.
"""
from __future__ import annotations

import os
import sys

_STATE = {"blocked": 0, "installed": False}


def _gib(name: str, default: float) -> float:
    raw = os.environ.get(name) or str(default)
    v = float(raw)
    if not v > 0:
        raise ValueError("%s must be a positive number of GiB, got %r" % (name, raw))
    return v


def min_gib() -> float:
    return _gib("ODDE_ATTN_QBLOCK_MIN_GIB", 4.0)


def block_gib() -> float:
    b = _gib("ODDE_ATTN_QBLOCK_GIB", 1.0)
    if b > min_gib():
        raise ValueError("ODDE_ATTN_QBLOCK_GIB (%g) must not exceed ODDE_ATTN_QBLOCK_MIN_GIB (%g)" % (b, min_gib()))
    return b


def _rank() -> str:
    return os.environ.get("ROWPAIR_RANK", "-")


def _say(msg: str) -> None:
    sys.stderr.write("[odde-attn-qblock] %s\n" % msg)
    sys.stderr.flush()


def make_attention(orig):
    lo = min_gib() * 2 ** 30
    budget = block_gib() * 2 ** 30

    def _attention(query, key, value, biases):
        import torch
        Q, K = int(query.shape[-2]), int(key.shape[-2])
        lead = 1
        for d in torch.broadcast_shapes(tuple(query.shape[:-2]), tuple(key.shape[:-2])):
            lead *= int(d)
        per_q = lead * K * query.element_size()          # logits bytes of one query row, over every head and batch row
        if per_q * Q <= lo:
            return orig(query, key, value, biases)
        qb = max(1, int(budget // per_q))
        if qb >= Q:
            return orig(query, key, value, biases)
        out = None
        for j0 in range(0, Q, qb):
            j1 = min(Q, j0 + qb)
            ob = orig(query[..., j0:j1, :], key, value,
                      [b if (b.dim() < 2 or int(b.shape[-2]) == 1) else b[..., j0:j1, :] for b in biases])
            if out is None:
                out = ob.new_empty(tuple(ob.shape[:-2]) + (Q,) + tuple(ob.shape[-1:]))
            out[..., j0:j1, :] = ob
            del ob
        _STATE["blocked"] += 1
        if _STATE["blocked"] == 1:
            _say("logits %.2f GiB > %.2f: %d query blocks of %d (Q=%d K=%d lead=%d, rank %s)"
                 % (per_q * Q / 2 ** 30, lo / 2 ** 30, -(-Q // qb), qb, Q, K, lead, _rank()))
        return out
    _attention._odde_attn_qblock = True
    _attention.__wrapped__ = orig
    return _attention


def make_attention_logged(orig):
    w = make_attention(orig)
    _STATE["installed"] = True
    _say("opendde.model.triangular.layers._attention: query blocks above %.2f GiB of logits, %.2f GiB per block "
         "installed (pid %d, rank %s)" % (min_gib(), block_gib(), os.getpid(), _rank()))
    return w
