#!/usr/bin/env python3
"""The explicit triangle-attention blocks, checked on a CPU node before any GPU time is spent.

  1. The kit's own seams, unpatched (the diagnosis): tp._attn_rows(4, lay) is 4 -- the stock chunk -- and
     tp._attn_rows(None, lay) is the kit default of 32.
  2. The wrappers: _attn_rows -> min(ODDE_TP_ATTN_ROWS, Rmax) whatever the stock chunk; attention_dispatch hands the
     original ODDE_TP_TRIATT_QBLOCK as its chunk; a block that is not a positive multiple of 4 is refused.
  3. hook4 in fresh interpreters: with ODDE_TP_BLOCKS=1 ODDE_TRUNK_PATCH=1 ODDE_CHUNK_FLOOR=1, both kit seams are
     wrapped AND v2 + the floor are installed; with ODDE_TP_BLOCKS unset, the kit seams are untouched.
"""
import os
import subprocess
import sys

D = os.environ.get("ODDE_ROOT", "/scratch/pawsey1018/gbouras/opendde-setonix")
sys.path.insert(0, os.path.join(D, "patches"))
bad = 0


class Lay:
    Rmax = 1536


# 1. the diagnosis, from the kit's own function
import opendde_opt.tp as TP                                    # noqa: E402
print(f"opendde_opt.tp from {TP.__file__}")
for chunk, want in ((4, 4), (2, 2), (None, 32)):
    got = TP._attn_rows(chunk, Lay)
    print(f"kit tp._attn_rows(chunk={chunk}, Rmax=1536) = {got}   (expected {want})")
    bad += got != want

# 2. the wrappers
import odde_tp_blocks_patch as B                               # noqa: E402
w = B.make_attn_rows(TP._attn_rows)
for chunk in (2, 4, 64, None):
    got = w(chunk, Lay)
    if got != 32:
        bad += 1; print(f"ROWS WRONG: chunk {chunk} -> {got}, want 32")


class Small:
    Rmax = 16


if w(4, Small) != 16:
    bad += 1; print("ROWS WRONG: Rmax=16 not respected")
seen = []
d = B.make_attention_dispatch(lambda t, chunk_size=None: seen.append(chunk_size) or (lambda *a: "ran"))
run = d(object(), 4)
if seen != [256] or run() != "ran":
    bad += 1; print(f"QBLOCK WRONG: the original received {seen}, want [256]")
for name, val in (("ODDE_TP_ATTN_ROWS", "30"), ("ODDE_TP_TRIATT_QBLOCK", "0"), ("ODDE_TP_TRIATT_QBLOCK", "-8")):
    old = os.environ.get(name)
    os.environ[name] = val
    try:
        (B.rows if name == "ODDE_TP_ATTN_ROWS" else B.qblock)()
        bad += 1; print(f"NOT REFUSED: {name}={val}")
    except ValueError:
        pass
    finally:
        if old is None:
            os.environ.pop(name, None)
        else:
            os.environ[name] = old
print("wrappers: rows 32 whatever the chunk, Rmax respected, query block 256 handed down, bad sizes refused"
      if not bad else "wrappers: FAILED")

# 3. hook4, fresh interpreters
PROBE = r'''
import opendde.model.modules.primitives as Pm
import opendde.model.opendde as M
import opendde_opt.tp as TP
import opendde_opt.tp_kernels as TK
class L: Rmax = 1536
print("RESULT qk=%s dense=%s floor=%s rows=%s qblock=%s attn_rows(4)=%s" % (
    hasattr(Pm.rearrange_qk_to_dense_trunk, "__wrapped__"),
    hasattr(Pm.rearrange_to_dense_trunk, "__wrapped__"),
    getattr(M.OpenDDE._resolve_pairformer_chunk_size, "_odde_chunk_floor", False),
    getattr(TP._attn_rows, "_odde_tp_blocks", False),
    getattr(TK.attention_dispatch, "_odde_tp_blocks", False),
    TP._attn_rows(4, L)))
'''
ALL = {"ODDE_TRUNK_PATCH": "1", "ODDE_CHUNK_FLOOR": "1"}
for flags, expect in (({**ALL, "ODDE_TP_BLOCKS": "1"}, "qk=True dense=True floor=True rows=True qblock=True attn_rows(4)=32"),
                      (dict(ALL), "qk=True dense=True floor=True rows=False qblock=False attn_rows(4)=4")):
    env = {k: v for k, v in os.environ.items()
           if k not in ("ODDE_TRUNK_PATCH", "ODDE_CHUNK_FLOOR", "ODDE_MASK_PATCH", "ODDE_TP_BLOCKS")}
    env.update(flags)
    env["PYTHONPATH"] = os.path.join(D, "patches", "hook4") + os.pathsep + os.path.join(D, "scripts", "hooks")
    r = subprocess.run([os.path.join(D, "venv", "bin", "python"), "-c", PROBE], env=env,
                       capture_output=True, text=True, timeout=900)
    res = next((l for l in r.stdout.splitlines() if l.startswith("RESULT")), "NO RESULT: " + r.stderr[-600:])
    ok = r.returncode == 0 and expect in res
    bad += not ok
    print(f"[{' '.join(flags)}] rc={r.returncode} {res}  {'OK' if ok else 'WRONG, expected ' + expect}")
    for l in r.stderr.splitlines():
        if "odde-tp-blocks" in l:
            print("   " + l[:200])

print("TP_BLOCKS_RC=%d" % (1 if bad else 0))
sys.exit(1 if bad else 0)
