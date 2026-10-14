#!/usr/bin/env python3
"""Row-blocked APB pair bias (patches/odde_apb_rows_patch.py), checked on a CPU node before any GPU time.

  1. The guard passes on the kit's own opendde_opt.tp._apb_rows_of (the statement it was written against is there,
     once), and refuses a function without it.
  2. The kit's real attn_fn, driven with stand-in modules (a LayerNorm, a no-bias Linear, a capturing attention):
     the pair bias handed to the attention is the same with and without the wrapper. Below MIN it is bitwise
     (the real layernorm_z runs). Above MIN, in row blocks of 3 (ragged last) and of 1, it is within 1e-6 in fp32.
     The bias's shape, dtype and layout are as before. A deferred marker used anywhere else raises.
  3. hook7 in fresh interpreters. With ODDE_APB_ROWS_PATCH=1 the kit's _apb_rows_of is wrapped AND hook6's attention
     wrapper and hook5's allocator and A-block patches still install. Without the flag it is untouched. A bad size
     exits 3.
"""
import os
import subprocess
import sys

D = os.environ.get("ODDE_ROOT", "/scratch/pawsey1018/gbouras/opendde-setonix")
sys.path.insert(0, os.path.join(D, "patches"))
bad = 0

import torch                                                    # noqa: E402
import opendde_opt.tp as TP                                     # noqa: E402
import odde_apb_rows_patch as A                                 # noqa: E402

orig = TP._apb_rows_of
print(f"opendde_opt.tp from {TP.__file__}; already wrapped: {hasattr(orig, '_odde_apb_rows')}")
bad += hasattr(orig, "_odde_apb_rows")

# 1. the guard
try:
    A.check_source(orig)
    print("guard: the kit's _apb_rows_of carries the expected statement once: OK")
except RuntimeError as e:
    bad += 1
    print(f"guard: REFUSED the kit's own function: {e}")


def _other(p, extra_attn_bias, lay):
    def attn_fn(_q, s_full, z_sh):
        return p.linear_nobias_z(z_sh)
    return attn_fn


try:
    A.check_source(_other)
    bad += 1
    print("guard: did NOT refuse a function without the statement")
except RuntimeError:
    print("guard: refuses a function without the statement: OK")


# 2. equivalence through the kit's real attn_fn
class Lay:
    def __init__(self, R):
        self.r0, self.r1 = 0, R


class P:
    """Stand-ins for AttentionPairBias's members; .attention captures the bias it is handed."""
    def __init__(self, C, H, cs, seed=0):
        torch.manual_seed(seed)
        self.layernorm_z = torch.nn.LayerNorm(C)
        with torch.no_grad():
            self.layernorm_z.weight.uniform_(0.5, 1.5)
            self.layernorm_z.bias.uniform_(-0.2, 0.2)
        self.linear_nobias_z = torch.nn.Linear(C, H, bias=False)
        self.layernorm_a = torch.nn.LayerNorm(cs)
        self.seen = []

    def _align_bias_to_query(self, bias, q, n_pair_dims=2):
        return bias

    def attention(self, q_x, kv_x, attn_bias):
        self.seen.append(attn_bias.detach().clone())
        return q_x


def bias_of(fn, p, R=11, N=13, C=8, cs=6):
    g = torch.Generator().manual_seed(1)
    z = torch.randn(R, N, C, generator=g)
    s = torch.randn(N, cs, generator=g)
    p.seen.clear()
    with torch.no_grad():
        fn(p, None, Lay(R))(None, s, z)
    return p.seen[-1]


p = P(8, 4, 6)
ref = bias_of(orig, p)
os.environ["ODDE_APB_ROWS"], os.environ["ODDE_APB_ROWS_MIN_GIB"] = "3", "8"
got = bias_of(A.make_apb_rows_of(orig), p)
same = torch.equal(ref, got) and A._STATE["blocked"] == 0
print(f"below MIN (8 GiB): bitwise equal, nothing blocked: {same}")
bad += not same
for nrows in ("3", "1"):
    os.environ["ODDE_APB_ROWS"], os.environ["ODDE_APB_ROWS_MIN_GIB"] = nrows, "1e-9"
    A._STATE["blocked"] = 0
    got = bias_of(A.make_apb_rows_of(orig), p)
    diff = (got - ref).abs().max().item()
    ok = (A._STATE["blocked"] == 1 and tuple(got.shape) == tuple(ref.shape) and got.dtype == ref.dtype
          and got.stride() == ref.stride() and diff <= 1e-6)
    bad += not ok
    print(f"above MIN, {nrows}-row blocks: blocked={A._STATE['blocked']} shape={tuple(got.shape)} stride={got.stride()} "
          f"(ref {ref.stride()}) max|diff|={diff:.3g}  {'OK' if ok else 'WRONG'}")

prox = A._BiasRowsProxy(p, 1, 0.0)
marker = prox.layernorm_z(torch.randn(4, 5, 8))
try:
    _ = marker + 1
    bad += 1
    print("deferred marker: used outside the proxy WITHOUT an error")
except TypeError:
    print("deferred marker: anything but the proxy's linear_nobias_z raises: OK")

for name, val in (("ODDE_APB_ROWS", "0"), ("ODDE_APB_ROWS", "x"), ("ODDE_APB_ROWS_MIN_GIB", "-1")):
    old = os.environ.get(name)
    os.environ[name] = val
    try:
        A.make_apb_rows_of(orig)
        bad += 1
        print(f"NOT REFUSED: {name}={val}")
    except ValueError:
        pass
    finally:
        os.environ[name] = old if old is not None else ""
os.environ.pop("ODDE_APB_ROWS", None)
os.environ.pop("ODDE_APB_ROWS_MIN_GIB", None)
print("wrapper: " + ("as specified" if not bad else "FAILED"))

# 3. hook7, fresh interpreters
PROBE = r'''
import opendde_opt.tp as TP
import opendde.model.triangular.layers as L
from opt_core.mem import torch_alloc as TA
import opt_core.mem.rowpair.trimul as T
print("RESULT apb=%s attnq=%s native=%s ablock=%s" % (getattr(TP._apb_rows_of, "_odde_apb_rows", False),
      getattr(L._attention, "_odde_attn_qblock", False), getattr(TA.conf_for, "_odde_native_alloc", False),
      getattr(T.ablock_auto_cap, "_odde_ablock", False)))
'''
BASE = {"ODDE_TRUNK_PATCH": "1", "ODDE_TP_BLOCKS": "1", "ODDE_NATIVE_ALLOC": "1", "ODDE_ABLOCK_WHOLE_FRAC": "0.60",
        "ODDE_ATTN_QBLOCK": "1"}
DROP = ("ODDE_TRUNK_PATCH", "ODDE_CHUNK_FLOOR", "ODDE_TP_BLOCKS", "ODDE_NATIVE_ALLOC", "ODDE_ABLOCK_WHOLE_FRAC",
        "ODDE_ATTN_QBLOCK", "ODDE_ATTN_QBLOCK_MIN_GIB", "ODDE_ATTN_QBLOCK_GIB", "ODDE_APB_ROWS_PATCH", "ODDE_APB_ROWS",
        "ODDE_APB_ROWS_MIN_GIB", "ODDE_MASK_PATCH", "PYTORCH_CUDA_ALLOC_CONF", "PYTORCH_HIP_ALLOC_CONF")
cases = (
    ({**BASE, "ODDE_APB_ROWS_PATCH": "1"}, 0, "apb=True attnq=True native=True ablock=True"),
    (dict(BASE), 0, "apb=False attnq=True native=True ablock=True"),
    ({**BASE, "ODDE_APB_ROWS_PATCH": "1", "ODDE_APB_ROWS": "0"}, 3, None),
)
for flags, want_rc, expect in cases:
    env = {k: v for k, v in os.environ.items() if k not in DROP}
    env.update(flags)
    env["PYTHONPATH"] = os.path.join(D, "patches", "hook7") + os.pathsep + os.path.join(D, "scripts", "hooks")
    r = subprocess.run([os.path.join(D, "venv", "bin", "python"), "-c", PROBE], env=env,
                       capture_output=True, text=True, timeout=900)
    res = next((l for l in r.stdout.splitlines() if l.startswith("RESULT")), "no RESULT")
    ok = r.returncode == want_rc and (expect is None or expect in res)
    bad += not ok
    tag = " ".join(f"{k}={v}" for k, v in flags.items() if k.startswith("ODDE_APB"))
    print(f"[{tag or 'no ODDE_APB flag'}] rc={r.returncode} (want {want_rc}) {res}  {'OK' if ok else 'WRONG'}")
    for l in r.stderr.splitlines():
        if "odde-apb-rows" in l:
            print("   " + l[:220])

print("APB_ROWS_RC=%d" % (1 if bad else 0))
sys.exit(1 if bad else 0)
