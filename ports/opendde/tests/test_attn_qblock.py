#!/usr/bin/env python3
"""The query-blocked stock attention (patches/odde_attn_qblock_patch.py), checked on a CPU node before any GPU time.

  1. The stock statement, unpatched: opendde.model.triangular.layers._attention (q @ k^T + biases, softmax, @ v).
  2. The wrapper against it, with the refiner's bias shapes (mask bias [rows, 1, 1, K], triangle bias [1, H, Q, K]).
     Below MIN it returns the original's own result, bitwise. Above MIN it runs query blocks, within 1e-5 in fp32
     and two bf16 steps in bf16 of the unblocked statement. Covered: a ragged last block, blocks of one query, a bias
     that broadcasts on its query axis, and unmodified biases. Bad sizes are refused.
  3. hook6 in fresh interpreters. With ODDE_ATTN_QBLOCK=1 the stock _attention is wrapped AND hook5's native
     allocator and A-block patches still install. Without the flag _attention is untouched. With a bad size the
     interpreter exits 3 (fail closed).
"""
import os
import subprocess
import sys

D = os.environ.get("ODDE_ROOT", "/scratch/pawsey1018/gbouras/opendde-setonix")
sys.path.insert(0, os.path.join(D, "patches"))
bad = 0

import torch                                                    # noqa: E402
import opendde.model.triangular.layers as L                     # noqa: E402
orig = L._attention
print(f"opendde.model.triangular.layers from {L.__file__}; already wrapped: {hasattr(orig, '_odde_attn_qblock')}")
bad += hasattr(orig, "_odde_attn_qblock")

import odde_attn_qblock_patch as A                               # noqa: E402


def setsizes(min_gib, gib):
    os.environ["ODDE_ATTN_QBLOCK_MIN_GIB"] = repr(min_gib)
    os.environ["ODDE_ATTN_QBLOCK_GIB"] = repr(gib)


def case(dtype, rows=2, H=4, Q=301, K=257, Dh=16, seed=0):
    g = torch.Generator().manual_seed(seed)
    q = torch.randn(rows, H, Q, Dh, generator=g).to(dtype)
    k = torch.randn(rows, H, K, Dh, generator=g).to(dtype)
    v = torch.randn(rows, H, K, Dh, generator=g).to(dtype)
    keep = (torch.rand(rows, K, generator=g) > 0.1).to(dtype)
    mask_bias = (1e9 * (keep - 1))[:, None, None, :]             # [rows, 1, 1, K]: broadcasts on the query axis
    tb = torch.randn(1, H, Q, K, generator=g).to(dtype)          # [1, H, Q, K]: sliced on the query axis
    return q, k, v, [mask_bias, tb]


def call(fn, args):
    q, k, v, b = args
    bc = [x.clone() for x in b]
    o = fn(q, k, v, bc)
    if not all(torch.equal(x, y) for x, y in zip(b, bc)):
        raise AssertionError("a bias was modified")
    return o


# 2a. below MIN: the original's own result
setsizes(4.0, 1.0)
w = A.make_attention(orig)
args = case(torch.float32)
ref = call(orig, args)
got = call(w, args)
same = torch.equal(ref, got)
print(f"below MIN (4 GiB): bitwise equal to the stock statement: {same}")
bad += not same

# 2b. above MIN: query blocks (fp32 and bf16), ragged and one-query blocks
for dtype, tol in ((torch.float32, 1e-5), (torch.bfloat16, 2 * 2 ** -7)):
    for gib, label in ((2e-4, "26-query blocks, ragged last"), (1e-6, "1-query blocks")):
        setsizes(1e-3, gib)
        A._STATE["blocked"] = 0
        w = A.make_attention(orig)
        args = case(dtype)
        ref = call(orig, args).float()
        got = call(w, args)
        diff = (got.float() - ref).abs().max().item()
        nd = int((got.float() != ref).sum())
        ok = A._STATE["blocked"] == 1 and tuple(got.shape) == tuple(ref.shape) and got.dtype == dtype and diff <= tol
        bad += not ok
        print(f"{str(dtype):15s} {label:30s} blocked={A._STATE['blocked']} max|diff|={diff:.3g} (tol {tol:.3g}) "
              f"differing={nd}/{ref.numel()}  {'OK' if ok else 'WRONG'}")

# 2c. the refiner's square case, one row: Q == K
setsizes(1e-3, 3e-4)
w = A.make_attention(orig)
args = case(torch.float32, rows=1, H=12, Q=290, K=290, Dh=32, seed=1)
diff = (call(w, args) - call(orig, args)).abs().max().item()
print(f"square one-row case (H=12, Q=K=290): max|diff|={diff:.3g}  {'OK' if diff <= 1e-5 else 'WRONG'}")
bad += diff > 1e-5

# 2d. bad sizes refused
for mn, gb in (("4", "0"), ("4", "8"), ("-1", "1"), ("4", "x")):
    os.environ["ODDE_ATTN_QBLOCK_MIN_GIB"], os.environ["ODDE_ATTN_QBLOCK_GIB"] = mn, gb
    try:
        A.make_attention(orig)
        bad += 1
        print(f"NOT REFUSED: MIN={mn} GIB={gb}")
    except ValueError:
        pass
os.environ.pop("ODDE_ATTN_QBLOCK_MIN_GIB", None)
os.environ.pop("ODDE_ATTN_QBLOCK_GIB", None)
print("wrappers: " + ("as specified" if not bad else "FAILED"))

# 3. hook6, fresh interpreters
PROBE = r'''
import opendde.model.triangular.layers as L
from opt_core.mem import torch_alloc as TA
import opt_core.mem.rowpair.trimul as T
print("RESULT attnq=%s native=%s ablock=%s" % (getattr(L._attention, "_odde_attn_qblock", False),
      getattr(TA.conf_for, "_odde_native_alloc", False), getattr(T.ablock_auto_cap, "_odde_ablock", False)))
'''
BASE = {"ODDE_TRUNK_PATCH": "1", "ODDE_TP_BLOCKS": "1", "ODDE_NATIVE_ALLOC": "1", "ODDE_ABLOCK_WHOLE_FRAC": "0.60"}
DROP = ("ODDE_TRUNK_PATCH", "ODDE_CHUNK_FLOOR", "ODDE_TP_BLOCKS", "ODDE_NATIVE_ALLOC", "ODDE_ABLOCK_WHOLE_FRAC",
        "ODDE_ATTN_QBLOCK", "ODDE_ATTN_QBLOCK_MIN_GIB", "ODDE_ATTN_QBLOCK_GIB", "ODDE_MASK_PATCH",
        "PYTORCH_CUDA_ALLOC_CONF", "PYTORCH_HIP_ALLOC_CONF")
cases = (
    ({**BASE, "ODDE_ATTN_QBLOCK": "1"}, 0, "attnq=True native=True ablock=True"),
    (dict(BASE), 0, "attnq=False native=True ablock=True"),
    ({**BASE, "ODDE_ATTN_QBLOCK": "1", "ODDE_ATTN_QBLOCK_GIB": "8"}, 3, None),
)
for flags, want_rc, expect in cases:
    env = {k: v for k, v in os.environ.items() if k not in DROP}
    env.update(flags)
    env["PYTHONPATH"] = os.path.join(D, "patches", "hook6") + os.pathsep + os.path.join(D, "scripts", "hooks")
    r = subprocess.run([os.path.join(D, "venv", "bin", "python"), "-c", PROBE], env=env,
                       capture_output=True, text=True, timeout=900)
    res = next((l for l in r.stdout.splitlines() if l.startswith("RESULT")), "no RESULT")
    ok = r.returncode == want_rc and (expect is None or expect in res)
    bad += not ok
    tag = " ".join(f"{k}={v}" for k, v in flags.items() if k.startswith("ODDE_ATTN"))
    print(f"[{tag or 'no ODDE_ATTN flag'}] rc={r.returncode} (want {want_rc}) {res}  {'OK' if ok else 'WRONG'}")
    for l in r.stderr.splitlines():
        if "odde-attn-qblock" in l:
            print("   " + l[:220])

print("ATTN_QBLOCK_RC=%d" % (1 if bad else 0))
sys.exit(1 if bad else 0)
