"""Row blocks for the pair bias of the kit's attention-pair-bias (APB) statement. Opt-in: ODDE_APB_ROWS_PATCH=1
(patches/hook7; scripts/predict_patched7.sh sets it), with

    ODDE_APB_ROWS          rows per LayerNorm + projection block (default 256)
    ODDE_APB_ROWS_MIN_GIB  only a pair shard larger than this many GiB is blocked (default 8)

WHY. Each Pairformer block that carries the single track ends with s += APB(s, z). On the kit's sharded line
that is opendde_opt.tp._apb_rows_of(p, extra_attn_bias, lay) -> attn_fn, whose pair bias is

    bias = permute_final_dims(p.linear_nobias_z(p.layernorm_z(z_sh)), [2, 0, 1])     # [H, R, N]

with p.layernorm_z applied to the WHOLE shard at once. In the structural refiner at 9,216 tokens that is a 29.62
GiB LayerNorm'd copy of the [2304, 17,973, 384] bf16 shard. Job 50232013 died there on ranks 0-6, after both
triangle attentions and the transition of the first refiner block had passed: "Tried to allocate 29.62 GiB ...
28.75 GiB free ... 33.78 GiB allocated" (tp.py:453 attn_fn -> layers.py:256 LayerNorm). The kit lists
ROWPAIR_APB_ROWBLOCK in API.md, but no code in this kit reads it. opt_core's own pair_bias_rows does exactly this
per row block, but the adapter does not call it.

WHAT. Wrap _apb_rows_of so that attn_fn sees p through a proxy. The proxy's layernorm_z hands back a deferred
marker when the shard exceeds MIN, and its linear_nobias_z evaluates linear_nobias_z(layernorm_z(rows)) over row
blocks of ODDE_APB_ROWS into one [R, N, H] tensor, the same layout the original produced before its permute.
Everything else on p (layernorm_a, _align_bias_to_query, attention) is the real module's. Below MIN the real
layernorm_z runs, so the trunk's APB (a 7.6 GiB shard at 9,216) is untouched. LayerNorm is per position, so the
blocked LayerNorm is the same statement on fewer rows. The projection's GEMM changes only its M, so the result is
NOT claimed bitwise. A deferred marker reaching anything other than the proxy's linear_nobias_z raises; it never
silently skips a LayerNorm.

GUARD. The wrapper checks the kit's source of _apb_rows_of before installing. It must contain the bias statement
above exactly once, and no other call of layernorm_z or linear_nobias_z. Otherwise it refuses, and hook7 exits 3.
"""
from __future__ import annotations

import inspect
import os
import sys

_STATE = {"blocked": 0, "installed": False}
EXPECTED = "p.linear_nobias_z(p.layernorm_z(z_sh))"


def rows() -> int:
    raw = os.environ.get("ODDE_APB_ROWS") or "256"
    v = int(raw)
    if v <= 0:
        raise ValueError("ODDE_APB_ROWS must be a positive row count, got %r" % raw)
    return v


def min_gib() -> float:
    raw = os.environ.get("ODDE_APB_ROWS_MIN_GIB") or "8"
    v = float(raw)
    if not v > 0:
        raise ValueError("ODDE_APB_ROWS_MIN_GIB must be a positive number of GiB, got %r" % raw)
    return v


def _rank() -> str:
    return os.environ.get("ROWPAIR_RANK", "-")


def _say(msg: str) -> None:
    sys.stderr.write("[odde-apb-rows] %s\n" % msg)
    sys.stderr.flush()


def check_source(fn) -> None:
    src = inspect.getsource(fn)
    if src.count(EXPECTED) != 1 or src.count("layernorm_z(") != 1 or src.count("linear_nobias_z(") != 1:
        raise RuntimeError("opendde_opt.tp._apb_rows_of no longer has exactly one %r and no other layernorm_z / "
                           "linear_nobias_z call; refusing to wrap it" % EXPECTED)


class _Deferred(object):
    """layernorm_z(z) of a shard too large to normalise whole: only the proxy's linear_nobias_z may consume it."""
    __slots__ = ("z",)

    def __init__(self, z):
        self.z = z


class _BiasRowsProxy(object):
    def __init__(self, p, nrows: int, lo: float):
        self._p, self._rows, self._lo = p, int(nrows), float(lo)

    def __getattr__(self, name):                      # everything not defined here: the real module's
        return getattr(self._p, name)

    def layernorm_z(self, z):
        if z.dim() >= 3 and z.numel() * z.element_size() > self._lo and int(z.shape[-3]) > self._rows:
            return _Deferred(z)
        return self._p.layernorm_z(z)

    def linear_nobias_z(self, x):
        if not isinstance(x, _Deferred):
            return self._p.linear_nobias_z(x)
        z = x.z
        R = int(z.shape[-3])
        out = None
        for i0 in range(0, R, self._rows):
            n = min(self._rows, R - i0)
            y = self._p.linear_nobias_z(self._p.layernorm_z(z.narrow(-3, i0, n)))
            if out is None:
                out = y.new_empty(tuple(y.shape[:-3]) + (R,) + tuple(y.shape[-2:]))
            out.narrow(-3, i0, n).copy_(y)
            del y
        _STATE["blocked"] += 1
        if _STATE["blocked"] == 1:
            _say("pair bias of a %.2f GiB shard > %.2f: %d row blocks of %d (z %s, rank %s)"
                 % (z.numel() * z.element_size() / 2 ** 30, self._lo / 2 ** 30, -(-R // self._rows), self._rows,
                    tuple(z.shape), _rank()))
        return out


def make_apb_rows_of(orig):
    check_source(orig)
    nrows, lo = rows(), min_gib() * 2 ** 30

    def _apb_rows_of(p, extra_attn_bias, lay):
        return orig(_BiasRowsProxy(p, nrows, lo), extra_attn_bias, lay)
    _apb_rows_of._odde_apb_rows = True
    _apb_rows_of.__wrapped__ = orig
    return _apb_rows_of


def make_apb_rows_of_logged(orig):
    w = make_apb_rows_of(orig)
    _STATE["installed"] = True
    _say("opendde_opt.tp._apb_rows_of: pair bias in %d-row blocks above %.2f GiB shards installed (pid %d, rank %s)"
         % (rows(), min_gib(), os.getpid(), _rank()))
    return w
