"""A stricter whole-plane threshold for opt_core's triangle-multiplication A-block planner. Opt-in:
``ODDE_ABLOCK_WHOLE_FRAC=<fraction>`` (``patches/hook5``; ``scripts/predict_patched5.sh`` sets 0.60).

WHY. ``opt_core.mem.rowpair.trimul.ablock_auto_cap`` keeps an update's A plane ``u = Rmax*N*C*elt`` WHOLE (one
pass) when ``u <= ABLOCK_WHOLE_FRAC * free`` with ``ABLOCK_WHOLE_FRAC = 0.80``. Its own comment says the largest
ratio ever validated whole is ~0.72. At 8,192 tokens, P=8 (native allocator, job 49995079) the structural
refiner's width-384 update planned whole at u/free ~0.76, after the ~23.4 GiB structural pair shard. It then
died on every rank with "Tried to allocate 1.46 GiB ... 1.35 GiB free ... 59.96 GiB allocated by PyTorch". No
"ablock schedule" line was logged, and the kit logs one only for a multi-pass call.

WHAT. Wrap ``ablock_auto_cap``: when ``u`` exceeds ``ODDE_ABLOCK_WHOLE_FRAC * free`` the plane is row-blocked by the
kit's OWN rule (the largest multiple of 128 whose pass working set ``2*RA*N*C*elt`` fits ``ABLOCK_BLOCK_FRAC`` of
free, >= 128). Otherwise the original decides, unchanged. So the trunk's updates (u/free ~0.15 at these sizes)
stay whole, and only a plane near the edge goes multi-pass. The multi-pass schedule is the kit's validated one:
the A block and output row block per pass, plus a host row mirror of the shard (host bytes, ~23 GiB per rank at
8,192). Numerics: the same statements per tile. NOT claimed bitwise, because passes change launch shapes.
"""
from __future__ import annotations

import os
import sys

_STATE = {"forced": 0}


def frac() -> float:
    v = float(os.environ.get("ODDE_ABLOCK_WHOLE_FRAC") or "0.60")
    if not 0.0 < v <= 0.80:
        raise ValueError("ODDE_ABLOCK_WHOLE_FRAC must be in (0, 0.80], got %r" % v)
    return v


def make_auto_cap(orig, module=None):
    f = frac()

    def ablock_auto_cap(Rmax, N, C, elt_bytes, free_bytes):
        if free_bytes is not None:
            R, row = max(1, int(Rmax)), int(N) * int(C) * int(elt_bytes)
            if R * row > f * int(free_bytes):
                block = getattr(module, "ABLOCK_BLOCK_FRAC", 0.40) if module is not None else 0.40
                cap = max(128, (int(block * int(free_bytes)) // (2 * row)) // 128 * 128)
                if cap < R:
                    _STATE["forced"] += 1
                    if _STATE["forced"] == 1:
                        sys.stderr.write("[odde-ablock] u/free=%.2f > %.2f: A plane row-blocked, RA cap %d of Rmax %d (N=%d, rank %s)\n"
                                         % (R * row / max(1, int(free_bytes)), f, cap, R, int(N), os.environ.get("ROWPAIR_RANK", "-")))
                        sys.stderr.flush()
                    return cap, "auto"
        return orig(Rmax, N, C, elt_bytes, free_bytes)
    ablock_auto_cap._odde_ablock = True
    ablock_auto_cap.__wrapped__ = orig
    return ablock_auto_cap


def make_auto_cap_logged(orig):
    mod = sys.modules.get("opt_core.mem.rowpair.trimul")
    w = make_auto_cap(orig, mod)
    sys.stderr.write("[odde-ablock] opt_core.mem.rowpair.trimul.ablock_auto_cap: whole only when u/free <= %.2f installed (pid %d, rank %s)\n"
                     % (frac(), os.getpid(), os.environ.get("ROWPAIR_RANK", "-")))
    sys.stderr.flush()
    return w
