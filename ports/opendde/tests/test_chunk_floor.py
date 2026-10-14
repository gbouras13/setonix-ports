#!/usr/bin/env python3
"""The chunk floor, checked on a CPU node before any GPU time is spent.

  1. The diagnosis, from stock's OWN function: OpenDDE._bound_pairformer_chunk_size
     gives 2 at 12,288 tokens and 4 at 10,240 -- and where it first drops below 4.
  2. The floor wrapper changes only chunks of 1-3.
  3. hook3 in fresh interpreters: with ODDE_TRUNK_PATCH=1 ODDE_CHUNK_FLOOR=1, the two
     v2 patches AND the floor are installed; with the floor flag unset, only v2's two.
"""
import os
import subprocess
import sys

D = os.environ.get("ODDE_ROOT", "/scratch/pawsey1018/gbouras/opendde-setonix")
sys.path.insert(0, os.path.join(D, "patches"))
bad = 0

# 1. the diagnosis
from opendde.model.opendde import OpenDDE                      # noqa: E402
bound = OpenDDE._bound_pairformer_chunk_size
for n, want in ((10240, 4), (12288, 2)):
    got = bound(n, 32)
    print(f"stock _bound_pairformer_chunk_size(N={n}, table=32) = {got}   (expected {want})")
    bad += got != want
first_below = next(n for n in range(8000, 20000, 1) if (bound(n, 32) or n) < 4)
print(f"stock chunk first drops below the kit's minimum of 4 at N = {first_below}")

# 2. the wrapper
import odde_chunk_floor_patch as F                             # noqa: E402
for stock_c, want in ((1, 4), (2, 4), (3, 4), (4, 4), (8, 8), (32, 32), (None, None)):
    w = F.make_chunk_floor(lambda self, n, c, *, dynamic_chunk_size, _v=stock_c: _v)
    got = w(None, 12288, 32, dynamic_chunk_size=True)
    if got != want:
        bad += 1; print(f"FLOOR WRONG: stock {stock_c} -> {got}, want {want}")
print("floor wrapper: 1-3 -> 4, everything else unchanged" if not bad else "floor wrapper: FAILED")

# 3. hook3, fresh interpreters
PROBE = r'''
import sys
import opendde.model.modules.primitives as Pm
import opendde.model.opendde as M
print("RESULT qk=%s dense=%s floor=%s torch_early_ok=True" % (
    hasattr(Pm.rearrange_qk_to_dense_trunk, "__wrapped__"),
    hasattr(Pm.rearrange_to_dense_trunk, "__wrapped__"),
    getattr(M.OpenDDE._resolve_pairformer_chunk_size, "_odde_chunk_floor", False)))
'''
for flags, expect in (({"ODDE_TRUNK_PATCH": "1", "ODDE_CHUNK_FLOOR": "1"}, "qk=True dense=True floor=True"),
                      ({"ODDE_TRUNK_PATCH": "1"}, "qk=True dense=True floor=False")):
    env = {k: v for k, v in os.environ.items() if k not in ("ODDE_TRUNK_PATCH", "ODDE_CHUNK_FLOOR", "ODDE_MASK_PATCH")}
    env.update(flags)
    env["PYTHONPATH"] = os.path.join(D, "patches", "hook3") + os.pathsep + os.path.join(D, "scripts", "hooks")
    r = subprocess.run([os.path.join(D, "venv", "bin", "python"), "-c", PROBE], env=env,
                       capture_output=True, text=True, timeout=900)
    res = next((l for l in r.stdout.splitlines() if l.startswith("RESULT")), "NO RESULT: " + r.stderr[-400:])
    ok = r.returncode == 0 and expect in res
    bad += not ok
    print(f"[{' '.join(flags)}] rc={r.returncode} {res}  {'OK' if ok else 'WRONG, expected ' + expect}")

print("CHUNK_FLOOR_RC=%d" % (1 if bad else 0))
sys.exit(1 if bad else 0)
