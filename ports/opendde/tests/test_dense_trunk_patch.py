#!/usr/bin/env python3
"""Bitwise equivalence of BOTH direct window constructions vs stock OpenDDE.

  rearrange_qk_to_dense_trunk(compute_mask=True)  -> the window mask
  rearrange_to_dense_trunk(attn_bias=None)        -> the window padding bias,
                                                     in bf16, fp16 and fp32
  rearrange_to_dense_trunk(attn_bias=<dense>)     -> must pass through to stock

Compares every returned tensor (values, dtype, shape) and every padding length.
CPU only; the stock side is the O(N_atom^2) path, so n is kept modest.
"""
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "patches"))
from opendde.model.modules import primitives as Pm            # noqa: E402
import odde_dense_trunk_patch as P                             # noqa: E402

stock_qk = Pm.rearrange_qk_to_dense_trunk
stock_dense = Pm.rearrange_to_dense_trunk
p_qk = P.make_patched_qk(stock_qk)
p_dense = P.make_patched_dense(stock_dense)

def same(a, b):
    if torch.is_tensor(a):
        return torch.is_tensor(b) and a.dtype == b.dtype and a.shape == b.shape and torch.equal(a, b)
    if isinstance(a, (list, tuple)):
        return len(a) == len(b) and all(same(x, y) for x, y in zip(a, b))
    if isinstance(a, dict):
        return a.keys() == b.keys() and all(same(a[k], b[k]) for k in a)
    return a == b

sizes = sorted(set(list(range(1, 130)) + [255, 256, 257, 1000, 1023, 1024, 1025, 4096, 5011, 8191, 8192, 8193]))
bad = checked = 0
for n in sizes:
    # 1. the mask
    r = torch.randn(n, 3); u = torch.randint(0, 7, (n,)).float()
    a = stock_qk([r, u], [r, u], [-2, -1], [-2, -1], n_queries=32, n_keys=128, compute_mask=True)
    b = p_qk([r, u], [r, u], [-2, -1], [-2, -1], n_queries=32, n_keys=128, compute_mask=True)
    checked += 1
    if not same(a, b):
        bad += 1; print(f"MASK MISMATCH n={n}")
    # 2. the padding bias, in each dtype the atom transformer can run in, with a head dim
    for dt in (torch.bfloat16, torch.float16, torch.float32):
        q = torch.randn(1, 4, n, 16).to(dt); k = torch.randn(1, 4, n, 16).to(dt); v = torch.randn(1, 4, n, 16).to(dt)
        a = stock_dense(q, k, v, 32, 128, attn_bias=None)
        b = p_dense(q, k, v, 32, 128, attn_bias=None)
        checked += 1
        if not same(a, b):
            bad += 1; print(f"BIAS MISMATCH n={n} dtype={dt}: " + ", ".join(
                f"{i}:{tuple(x.shape)}/{x.dtype} vs {tuple(y.shape)}/{y.dtype}" for i, (x, y) in enumerate(zip(a, b)) if torch.is_tensor(x)))
# 3. a caller's own dense bias must reach the stock path untouched
n = 300
q = torch.randn(1, 4, n, 16); k = torch.randn(1, 4, n, 16); v = torch.randn(1, 4, n, 16)
ab = torch.randn(1, 4, n, n)
checked += 1
if not same(stock_dense(q, k, v, 32, 128, attn_bias=ab), p_dense(q, k, v, 32, 128, attn_bias=ab)):
    bad += 1; print("PASSTHROUGH MISMATCH")

n = 68400   # the atom count at 8,192 tokens, reported without building the stock side
nt = -(-n // 32); E = n + 48 + int((nt - 0.5) * 32 + 64 - n + 0.5)
print(f"checked {checked} cases ({len(sizes)} sizes x mask + bias in bf16/fp16/fp32, + passthrough): "
      f"{checked - bad} identical, {bad} mismatched")
print(f"at n={n} atoms (8,192 tokens): stock bias scratch {2 * nt * 32 * E * 2 / 2**30:.2f} GiB per call (bf16, two copies) "
      f"-> direct {nt * 32 * 128 * 2 / 2**20:.1f} MiB")
print("TRUNK_PATCH_RC=%d" % (1 if bad else 0))
sys.exit(1 if bad else 0)
