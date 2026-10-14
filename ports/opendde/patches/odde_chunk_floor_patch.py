"""Floor OpenDDE's pairformer chunk at 4 -- the minimum the kit's row-sharded
triangle attention accepts. Opt-in (ODDE_CHUNK_FLOOR=1), separate from the
byte-identical dense-window patch (v2).

WHY. Stock OpenDDE bounds the pairformer's attention chunk by an N^2 budget
(OpenDDE._bound_pairformer_chunk_size):

    budget_chunk = max(1, 450_000_000 // (N * N));  chunk = min(table, 2**floor(log2(budget_chunk)))

which is 4 at N = 10,240 and 2 at N = 12,288. On the row-sharded line the kit
uses that chunk as its triangle-attention row block (tp._attn_rows: "the stock's
chunk size when it passes one"), and the core requires a multiple of 4
(opt_core.mem.rowpair.triatt.blocks4). So above ~10,600 tokens BIG_TP refuses by
name -- "row/query block size must be a positive multiple of 4, got 2" --
whatever memory is free. That is a stock/kit interaction, not a GPU limit, and
an H100 would hit it at the same N.

WHAT. Wrap OpenDDE._resolve_pairformer_chunk_size: when stock resolves a chunk
of 1-3, return 4. Nothing else changes. Cost: the chunked attention temporary
(~ chunk x N^2) is up to 2x stock's at those sizes. Triangle-attention rows are
independent, so a different row batch changes batch shapes, not the math --
but it is NOT claimed bitwise; this floor is a separate, labelled lever.

LIMIT. Only meaningful on BIG_TP, where the kit's chunk_lift lever is off. On a
line with chunk_lift on, the kit registers its own wrapper on this same site
later (opt_core.autoload: the last registration on a site wins) and this floor
is replaced -- which is why it logs every time it fires.
"""
from __future__ import annotations

import os
import sys

FLOOR = 4
_STATE = {"installed": False, "floored": {}}


def make_chunk_floor(orig):
    def _resolve_pairformer_chunk_size(self, n_token, chunk_size, *, dynamic_chunk_size):
        c = orig(self, n_token, chunk_size, dynamic_chunk_size=dynamic_chunk_size)
        if c is not None and 0 < int(c) < FLOOR:
            key = (int(n_token), int(c))
            if key not in _STATE["floored"]:
                sys.stderr.write("[odde-chunk-floor] N=%d: stock chunk %d -> %d (rank %s)\n"
                                 % (int(n_token), int(c), FLOOR, os.environ.get("ROWPAIR_RANK", "-")))
                sys.stderr.flush()
            _STATE["floored"][key] = _STATE["floored"].get(key, 0) + 1
            return FLOOR
        return c
    _resolve_pairformer_chunk_size._odde_chunk_floor = True
    _resolve_pairformer_chunk_size.__wrapped__ = orig
    return _resolve_pairformer_chunk_size


def make_chunk_floor_logged(orig):
    patched = make_chunk_floor(orig)
    _STATE["installed"] = True
    sys.stderr.write("[odde-chunk-floor] OpenDDE._resolve_pairformer_chunk_size: floor %d installed (pid %d, rank %s)\n"
                     % (FLOOR, os.getpid(), os.environ.get("ROWPAIR_RANK", "-")))
    sys.stderr.flush()
    return patched
