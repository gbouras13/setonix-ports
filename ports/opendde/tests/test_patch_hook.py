#!/usr/bin/env python3
"""Does the patched launch path install the direct mask, exactly once, without
importing torch at interpreter start? Runs the check in FRESH interpreters, so
the real sitecustomize / site machinery is exercised, not an in-process import.

  1. patches/hook on PYTHONPATH, ODDE_MASK_PATCH=1:
       - torch NOT in sys.modules before opendde is imported
       - after `import opendde.model.modules.transformer` (which pulls in
         primitives), BOTH module bindings are the patched function
       - the mask it returns equals the stock one (a small n)
       - exactly one install line
  2. same path, ODDE_MASK_PATCH unset: nothing patched (the hook is opt-in)
"""
import os
import subprocess
import sys

D = os.environ.get("ODDE_ROOT", "/scratch/pawsey1018/gbouras/opendde-setonix")
PY = os.path.join(D, "venv", "bin", "python")
V2 = os.environ.get("ODDE_HOOK_VERSION") == "2"
HOOK = os.path.join(D, "patches", "hook2" if V2 else "hook")
FLAG = "ODDE_TRUNK_PATCH" if V2 else "ODDE_MASK_PATCH"
INSTALL_LINE = "direct window construction installed" if V2 else "direct window mask installed"
EXPECT_INSTALLS = 2 if V2 else 1
BASEHOOK = os.path.join(D, "scripts", "hooks")

PROBE = r'''
import os, sys
V2 = os.environ.get("ODDE_HOOK_VERSION") == "2"
torch_early = "torch" in sys.modules
import opendde.model.modules.transformer as T
import opendde.model.modules.primitives as Pm
fp, ft = Pm.rearrange_qk_to_dense_trunk, T.rearrange_qk_to_dense_trunk
patched = hasattr(fp, "__wrapped__") and (not V2 or hasattr(Pm.rearrange_to_dense_trunk, "__wrapped__"))
same = fp is ft
import torch
n = 5011
r = torch.randn(n, 3); u = torch.randint(0, 7, (n,)).float()
kw = dict(n_queries=32, n_keys=128, compute_mask=True)
_, _, a = fp([r, u], [r, u], [-2, -1], [-2, -1], **kw)
if patched:
    _, _, b = fp.__wrapped__([r, u], [r, u], [-2, -1], [-2, -1], **kw)
    equal = bool(torch.equal(a["mask_trunked"], b["mask_trunked"]))
else:
    equal = None
print(f"RESULT torch_before_opendde={torch_early} patched={patched} both_bindings_same={same} mask_equal_stock={equal}")
'''

def run(env_extra):
    env = dict(os.environ)
    env.pop("ODDE_MASK_PATCH", None); env.pop("ODDE_TRUNK_PATCH", None)
    env.update(env_extra)
    env["PYTHONPATH"] = HOOK + os.pathsep + BASEHOOK
    r = subprocess.run([PY, "-c", PROBE], env=env, capture_output=True, text=True, timeout=900)
    res = [l for l in r.stdout.splitlines() if l.startswith("RESULT")]
    installs = r.stderr.count(INSTALL_LINE)
    return r.returncode, (res[0] if res else "NO RESULT: " + r.stderr[-600:]), installs

bad = 0
rc, res, n_inst = run({FLAG: "1"})
print(f"[patched] rc={rc} installs={n_inst} {res}")
if not (rc == 0 and "torch_before_opendde=False" in res and "patched=True" in res
        and "both_bindings_same=True" in res and "mask_equal_stock=True" in res and n_inst == EXPECT_INSTALLS):
    bad += 1
rc, res, n_inst = run({})
print(f"[unset]   rc={rc} installs={n_inst} {res}")
if not (rc == 0 and "patched=False" in res and n_inst == 0):
    bad += 1
print("HOOK_TEST_RC=%d" % (1 if bad else 0))
sys.exit(1 if bad else 0)
