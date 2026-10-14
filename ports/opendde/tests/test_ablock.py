#!/usr/bin/env python3
"""The A-block threshold patch, checked on a CPU node.

  1. The kit's planner, unpatched, at the 8,192-token structural refiner's numbers (Rmax 2048, N 15,976, C 384, bf16,
     ~30.7 GiB free): it keeps the plane WHOLE (u/free ~0.76 <= 0.80) -- the schedule that ran out of memory.
  2. The wrapper at 0.60: the same call is row-blocked by the kit's own rule (a 512-row cap, 4 passes); the trunk's
     numbers at 8,192 (Rmax 1024, N 8,192, ~40 GiB free) stay whole; free_bytes=None stays whole.
  3. hook5 in a fresh interpreter with ODDE_ABLOCK_WHOLE_FRAC=0.60 installs it; without the variable it does not.
"""
import os
import subprocess
import sys

D = os.environ.get("ODDE_ROOT", "/scratch/pawsey1018/gbouras/opendde-setonix")
sys.path.insert(0, os.path.join(D, "patches"))
bad = 0
G = 2 ** 30

import opt_core.mem.rowpair.trimul as T                        # noqa: E402
print(f"opt_core.mem.rowpair.trimul from {T.__file__}; ABLOCK_WHOLE_FRAC={T.ABLOCK_WHOLE_FRAC} ABLOCK_BLOCK_FRAC={T.ABLOCK_BLOCK_FRAC}")
STRUCT = (2048, 15976, 384, 2, int(30.7 * G))
TRUNK = (1024, 8192, 384, 2, int(40.0 * G))
got = T.ablock_auto_cap(*STRUCT)
print(f"unpatched, 8,192 structural refiner: {got}   (expected (2048, 'whole'): the schedule that OOMed)")
bad += got != (2048, "whole")

os.environ["ODDE_ABLOCK_WHOLE_FRAC"] = "0.60"
import odde_ablock_patch as A                                    # noqa: E402
w = A.make_auto_cap(T.ablock_auto_cap, T)
g1, g2, g3 = w(*STRUCT), w(*TRUNK), w(2048, 15976, 384, 2, None)
print(f"at 0.60: structural {g1} (expected (512, 'auto')), trunk {g2} (expected (1024, 'whole')), no CUDA {g3} (expected (2048, 'whole'))")
bad += g1 != (512, "auto")
bad += g2 != (1024, "whole")
bad += g3 != (2048, "whole")

PROBE = r'''
import opt_core.mem.rowpair.trimul as T
print("RESULT wrapped=%s" % getattr(T.ablock_auto_cap, "_odde_ablock", False))
'''
for flags, expect in (({"ODDE_ABLOCK_WHOLE_FRAC": "0.60"}, "wrapped=True"), ({}, "wrapped=False")):
    env = {k: v for k, v in os.environ.items() if k not in ("ODDE_ABLOCK_WHOLE_FRAC", "ODDE_NATIVE_ALLOC", "ODDE_TP_BLOCKS",
                                                           "ODDE_CHUNK_FLOOR", "ODDE_TRUNK_PATCH", "PYTORCH_CUDA_ALLOC_CONF")}
    env.update(flags)
    env["PYTHONPATH"] = os.path.join(D, "patches", "hook5") + os.pathsep + os.path.join(D, "scripts", "hooks")
    r = subprocess.run([os.path.join(D, "venv", "bin", "python"), "-c", PROBE], env=env, capture_output=True, text=True, timeout=900)
    res = next((l for l in r.stdout.splitlines() if l.startswith("RESULT")), "NO RESULT: " + r.stderr[-400:])
    ok = r.returncode == 0 and expect in res
    bad += not ok
    print(f"[{' '.join(flags) or 'no flag'}] rc={r.returncode} {res}  {'OK' if ok else 'WRONG, expected ' + expect}")

print("ABLOCK_RC=%d" % (1 if bad else 0))
sys.exit(1 if bad else 0)
