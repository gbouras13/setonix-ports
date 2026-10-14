"""Explicit row and query blocks for the kit's row-sharded triangle attention (line BIG_TP). Opt-in, and separate
from v2 and from the chunk floor: ODDE_TP_BLOCKS=1, with

    ODDE_TP_ATTN_ROWS      rows per triangle-attention / transition row batch (default 32 -- the kit's own default
                           when the stock passes no chunk; ROWPAIR_ATTN_ROWS)
    ODDE_TP_TRIATT_QBLOCK  query sub-block of the triangle-attention torch statement (default 256)
    ODDE_TP_PROGRESS       1 = rank 0 logs one line every ODDE_TP_PROGRESS_EVERY row-batch calls (default 96 = one
                           pairformer block at 12,288 tokens, P=8, 32 rows: 2 triangle attentions x 1,536/32 batches)

Both blocks must be positive multiples of 4 (opt_core.mem.rowpair.triatt.blocks4); anything else is refused here,
by name, before a rank starts.

WHY. The kit takes BOTH blocks from the stock's pairformer chunk whenever the stock passes one:

    tp._attn_rows(chunk_size, lay)          rows   = the chunk   (ROWPAIR_ATTN_ROWS only when the stock passes None)
    tp_kernels.attention_dispatch(t, c)     qblock = the chunk   (ROWPAIR_TRIATT_QBLOCK only when the stock passes None)

The stock budgets that chunk for its UNSHARDED statement, whose transient is chunk x H x N^2: 450e6 // N^2 gives 2
at 12,288 tokens (4 with the floor). On the kit's line the transient is rows x H x qblock x N instead, so the same
chunk means 4 x 4 = 16 query rows per launch: (1,536/4) x (12,288/4) = 1.18M core calls per triangle attention per
rank at 12,288 tokens, P=8. At 32 x 256 that is 48 x 48 = 2,304, and the logits transient is 32 x 4 x 256 x 12,288
bf16 = 0.8 GB. The stock chunk also feeds the input embedder, the structural refiner and later stock calls, so it
is NOT changed here -- only the two kit seams that turn it into triangle-attention blocks are.

WHAT. tp._attn_rows -> min(ODDE_TP_ATTN_ROWS, lay.Rmax), whatever the stock passes. tp_kernels.attention_dispatch(t,
chunk) -> the original with chunk = ODDE_TP_TRIATT_QBLOCK (its only use of chunk is the torch statement's query
block). Keys and values stay whole and every query row's softmax is whole, so the per-query statement is unchanged;
launch shapes change, so GEMM tiling may differ. NOT claimed bitwise.
"""
from __future__ import annotations

import os
import sys
import time

_STATE = {"rows": None, "qblock": None, "calls": 0, "t0": None, "installed": []}


def _block(name: str, default: int) -> int:
    raw = os.environ.get(name) or str(default)
    v = int(raw)
    if v <= 0 or v % 4:
        raise ValueError("%s must be a positive multiple of 4, got %r" % (name, raw))
    return v


def rows() -> int:
    return _block("ODDE_TP_ATTN_ROWS", 32)


def qblock() -> int:
    return _block("ODDE_TP_TRIATT_QBLOCK", 256)


def _rank() -> str:
    return os.environ.get("ROWPAIR_RANK", "-")


def _say(msg: str) -> None:
    sys.stderr.write("[odde-tp-blocks] %s\n" % msg)
    sys.stderr.flush()


def make_attn_rows(orig):
    r = rows()

    def _attn_rows(chunk_size, lay):
        out = min(r, int(lay.Rmax))
        if _STATE["rows"] != out:
            _STATE["rows"] = out
            _say("rows per triangle-attention/transition batch: %d (stock chunk %s; rank %s)" % (out, chunk_size, _rank()))
        return out
    _attn_rows._odde_tp_blocks = True
    _attn_rows.__wrapped__ = orig
    return _attn_rows


def make_attention_dispatch(orig):
    qb = qblock()
    every = int(os.environ.get("ODDE_TP_PROGRESS_EVERY") or 96)
    progress = os.environ.get("ODDE_TP_PROGRESS") == "1" and _rank() in ("0", "-")

    def attention_dispatch(t, chunk_size=None):
        if _STATE["qblock"] != qb:
            _STATE["qblock"] = qb
            _say("triangle-attention query block: %d (stock chunk %s; rank %s)" % (qb, chunk_size, _rank()))
        run = orig(t, qb)
        if not progress:
            return run

        def run_counted(x_rows, mask_bias, tb):
            if _STATE["t0"] is None:
                _STATE["t0"] = time.time()
            o = run(x_rows, mask_bias, tb)
            _STATE["calls"] += 1
            n = _STATE["calls"]
            if n % every == 0:
                mem = ""
                try:
                    import torch
                    if torch.cuda.is_available():
                        mem = " alloc_peak_gib=%.2f reserved_gib=%.2f" % (torch.cuda.max_memory_allocated() / 2**30,
                                                                        torch.cuda.memory_reserved() / 2**30)
                except Exception:
                    pass
                _say("progress rank=0 calls=%d rows=%d t=%.0fs%s" % (n, int(x_rows.shape[0]), time.time() - _STATE["t0"], mem))
            return o
        return run_counted
    attention_dispatch._odde_tp_blocks = True
    attention_dispatch.__wrapped__ = orig
    return attention_dispatch


def make_attn_rows_logged(orig):
    w = make_attn_rows(orig)
    _STATE["installed"].append("attn_rows")
    _say("opendde_opt.tp._attn_rows: explicit rows %d installed (pid %d, rank %s)" % (rows(), os.getpid(), _rank()))
    return w


def make_attention_dispatch_logged(orig):
    w = make_attention_dispatch(orig)
    _STATE["installed"].append("attention_dispatch")
    _say("opendde_opt.tp_kernels.attention_dispatch: explicit query block %d installed (pid %d, rank %s)"
         % (qblock(), os.getpid(), _rank()))
    return w
