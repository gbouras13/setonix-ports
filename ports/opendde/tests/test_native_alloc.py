#!/usr/bin/env python3
"""The native-allocator patch, checked on a CPU node before any GPU time is spent.

  1. The kit's writer, unpatched: torch_alloc.conf_for("expandable") is expandable_segments:True, and export() writes it.
  2. hook5 in fresh interpreters with ODDE_NATIVE_ALLOC=1 (plus v2, floor and blocks): conf_for is wrapped, "expandable"
     yields expandable_segments:False, "capped" is unchanged, and export() writes False into an empty environment,
     accepts it when already present, and refuses a DIFFERENT value by name (the kit's refusal is kept, now against False).
  3. The same without ODDE_NATIVE_ALLOC: nothing changes.
"""
import os
import subprocess
import sys

D = os.environ.get("ODDE_ROOT", "/scratch/pawsey1018/gbouras/opendde-setonix")
bad = 0

from opt_core.mem import torch_alloc as TA                      # noqa: E402
print(f"opt_core.mem.torch_alloc from {TA.__file__}")
got = TA.conf_for("expandable")
print(f"unpatched conf_for('expandable') = {got!r}   (expected 'expandable_segments:True')")
bad += got != "expandable_segments:True"

PROBE = r'''
from opt_core.mem import torch_alloc as TA, MemLeverRefused
import opendde_opt.tp as TP
import opendde_opt.modes as MO
import os as _os
try:
    _res = MO.resolve("BIG_TP", _os.path.join(_os.environ["ODDE_ROOT_T"], "kit", "opendde"), {})
    res_conf = _res.exports.get("PYTORCH_CUDA_ALLOC_CONF")
except Exception as _e:
    res_conf = "ERROR:%s:%s" % (type(_e).__name__, str(_e)[:80])
env = {}
f = TA.export("expandable", environ=env, cuda_initialized=False)
again = TA.export("expandable", environ=dict(env), cuda_initialized=False)["alloc_export"]
try:
    TA.export("expandable", environ={"PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"}, cuda_initialized=False)
    refused = "no"
except MemLeverRefused:
    refused = "yes"
print("RESULT wrapped=%s resolve_wrapped=%s resolve_exports=%s expandable=%s capped_true=%s export_env=%s present=%s refuses_other=%s blocks=%s" % (
    getattr(TA.conf_for, "_odde_native_alloc", False), getattr(MO.resolve, "_odde_native_alloc", False), res_conf, TA.conf_for("expandable"),
    TA.conf_for("capped").startswith("expandable_segments:True"), env.get("PYTORCH_CUDA_ALLOC_CONF"), again, refused,
    getattr(TP._attn_rows, "_odde_tp_blocks", False)))
'''
BASE = {"ODDE_TRUNK_PATCH": "1", "ODDE_CHUNK_FLOOR": "1", "ODDE_TP_BLOCKS": "1"}
cases = (
    ({**BASE, "ODDE_NATIVE_ALLOC": "1"},
     "wrapped=True resolve_wrapped=True resolve_exports=expandable_segments:False expandable=expandable_segments:False capped_true=True export_env=expandable_segments:False present=present refuses_other=yes blocks=True"),
    (dict(BASE),
     "wrapped=False resolve_wrapped=False resolve_exports=expandable_segments:True expandable=expandable_segments:True capped_true=True export_env=expandable_segments:True present=present refuses_other=no blocks=True"),
)
for flags, expect in cases:
    env = {k: v for k, v in os.environ.items()
           if k not in ("ODDE_TRUNK_PATCH", "ODDE_CHUNK_FLOOR", "ODDE_TP_BLOCKS", "ODDE_NATIVE_ALLOC", "ODDE_MASK_PATCH",
                        "PYTORCH_CUDA_ALLOC_CONF", "PYTORCH_HIP_ALLOC_CONF")}
    env.update(flags)
    env["PYTHONPATH"] = os.path.join(D, "patches", "hook5") + os.pathsep + os.path.join(D, "scripts", "hooks")
    env["ODDE_ROOT_T"] = D
    r = subprocess.run([os.path.join(D, "venv", "bin", "python"), "-c", PROBE], env=env,
                       capture_output=True, text=True, timeout=900)
    res = next((l for l in r.stdout.splitlines() if l.startswith("RESULT")), "NO RESULT: " + r.stderr[-600:])
    ok = r.returncode == 0 and expect in res
    bad += not ok
    print(f"[{' '.join(sorted(flags))}] rc={r.returncode} {res}  {'OK' if ok else 'WRONG, expected ' + expect}")
    for l in r.stderr.splitlines():
        if "odde-native-alloc" in l:
            print("   " + l[:200])

print("NATIVE_ALLOC_RC=%d" % (1 if bad else 0))
sys.exit(1 if bad else 0)
