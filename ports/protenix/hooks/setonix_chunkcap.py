"""Cap stock Protenix's triangle-attention row chunk (diagnostic / capacity lever, env-gated).

Stock `_attention` (protenix/model/triangular/layers.py) materialises the full logits
`matmul(q, k^T)` -> [rows, H=8, N, N] plus its softmax, per call. `rows` is stock's dynamic
chunk (`Protenix._get_dynamic_chunk_size`: none <= 1024 tokens, 512/256/128 up to 2560, then 32),
and it applies identically on every rank of the sharded line (the kit's rowpair binding passes it
through; `ROWPAIR_TRIATT_ROWS` is ignored when a chunk is given). At N = 8192 a 32-row call is
~34 GB of bf16 logits: a P-independent transient that no amount of row sharding reduces.

PXS_TRIATT_CHUNK_CAP=<k> makes `_get_dynamic_chunk_size` return min(stock, k) (k when stock
returns None). Chunking is per-row exact in stock (chunk_layer), so only memory and speed
change. The kit's own knob for the sharded line only is PTX_TP_TRIATT_CHUNK; this one applies to
both lines through the same site, so the two are compared under the same transient bound. The
patch goes through the kit's own opt_core.autoload.patch_attr_at_import (no kit lever patches
this site).
"""
import os
import sys

_SEEN = set()


def arm():
    cap = int(os.environ["PXS_TRIATT_CHUNK_CAP"])
    from opt_core.autoload import patch_attr_at_import

    def make(orig):
        def _get_dynamic_chunk_size(self, N_token):
            c = orig(self, N_token)
            new = cap if c is None else min(int(c), cap)
            if N_token not in _SEEN:
                _SEEN.add(N_token)
                rank = os.environ.get("RANK", "0")
                print(f"[pxs-chunkcap r{rank}] triangle-attention chunk for N={N_token}: stock {c} -> {new} (PXS_TRIATT_CHUNK_CAP={cap})",
                      file=sys.stderr, flush=True)
            return new
        return _get_dynamic_chunk_size

    patch_attr_at_import("protenix.model.protenix", "Protenix._get_dynamic_chunk_size", make, tag="pxs-chunkcap", name="pxs:triatt_chunk_cap")
