#!/usr/bin/env python
"""F0 (2026-10-07): a CPU unit test of perf_launch.pin_pairlogits_rows against the fork's stock diffusion Transformer.
  pl_test.py PATH/TO/perf_launch.py
For the official-parameters branch (of3_weights False) and the OF3 branch: the levered class's init gives the same parameter tree (names,
shapes) as stock's, and applied with stock's parameters its outputs equal stock's -- at N <= rows (one call), N = 3 x rows (equal chunks)
and an uneven N (rows, rows, rest). A second install must be refused. One line per check; exit 1 on any failure."""
import importlib.util
import os
import sys

os.environ.setdefault("JAX_PLATFORMS", "cpu")
import haiku as hk
import jax
import numpy as np
from alphafold3.model import model_config
from alphafold3.model.network import diffusion_transformer as DT

spec = importlib.util.spec_from_file_location("perf_launch_under_test", sys.argv[1])
PL = importlib.util.module_from_spec(spec)
spec.loader.exec_module(PL)
STOCK = DT.Transformer
fails = 0


def transformed(cls, of3):
    gc = model_config.GlobalConfig(of3_weights=of3, bfloat16="none", flash_attention_implementation="xla")
    cfg = DT.Transformer.Config(num_blocks=4, super_block_size=2, attention=DT.SelfAttentionConfig(num_head=4))
    return hk.transform(lambda a, m, s, p: cls(cfg, gc)(a, m, s, p))


def check(name, ok, detail=""):
    global fails
    fails += 0 if ok else 1
    print(f"{'PASS' if ok else 'FAIL'}  {name}  {detail}", flush=True)


for of3 in (False, True):
    for n, rows in ((12, 16), (48, 16), (40, 16)):
        rng = np.random.default_rng(n)
        args = (rng.standard_normal((n, 64), dtype=np.float32), np.ones((n,), np.float32),
                rng.standard_normal((n, 32), dtype=np.float32), rng.standard_normal((n, n, 16), dtype=np.float32))
        DT.Transformer = STOCK
        fs = transformed(STOCK, of3)
        params = fs.init(jax.random.PRNGKey(0), *args)
        out_s = np.asarray(fs.apply(params, None, *args))
        rep, ok = PL.pin_pairlogits_rows(rows)
        check(f"install of3={of3} N={n} rows={rows}", ok and DT.Transformer is not STOCK)
        fl = transformed(DT.Transformer, of3)
        pl_params = fl.init(jax.random.PRNGKey(0), *args)
        same_tree = jax.tree_util.tree_structure(pl_params) == jax.tree_util.tree_structure(params) and all(
            a.shape == b.shape for a, b in zip(jax.tree_util.tree_leaves(pl_params), jax.tree_util.tree_leaves(params)))
        check(f"params  of3={of3} N={n} rows={rows}", same_tree, f"{len(jax.tree_util.tree_leaves(params))} leaves")
        out_l = np.asarray(fl.apply(params, None, *args))
        d = float(np.max(np.abs(out_l - out_s)) / (np.max(np.abs(out_s)) + 1e-30))
        check(f"values  of3={of3} N={n} rows={rows}", d <= 1e-6, f"max rel diff {d:.2e}; {rep.report()}")
        _, again = PL.pin_pairlogits_rows(rows)
        check(f"refuse  of3={of3} N={n} rows={rows}", not again)
DT.Transformer = STOCK
print(f"{'ALL PASS' if fails == 0 else f'{fails} FAILURES'}", flush=True)
sys.exit(1 if fails else 0)
