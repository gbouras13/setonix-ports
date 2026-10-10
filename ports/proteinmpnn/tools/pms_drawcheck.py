#!/usr/bin/env python
"""pms_drawcheck.py -- which uniform transform does this torch build's multinomial use, and does a fused draw kernel reproduce it?

    MPNN_DIR=... python pms_drawcheck.py --kit-worker <kit>/proteinmpnn/opt/forward/mpnn_exact_worker/addon/mpnn_worker2.py [--cases N]

1. exponential_ against a numpy Philox4x32-10 reference: q = torch.empty(n).exponential_(1, generator=g) at (seed, offset) for float64 and
   float32, element i = Philox subsequence i at offset/4 (one thread per element, n <= 256), against -log(u) with u from
   (a) curand's transforms and (b) rocRAND's. Counts exact bit matches (and matches within 1 ulp) per transform.
2. torch.multinomial(p, 1, generator=g_b) for K generators at distinct offsets, many steps, B in {1,2,3,4,8,16,32}, float64 and float32,
   probabilities with exact zeros and one-hot rows (as the worker's probe builds them), against
   (a) the kit's _fused_draw_kernel (curand transforms; loaded from the kit worker file) and (b) pms_worker_boot's ROCm kernel.
   Also the Philox increment one multinomial call consumes, per B and dtype.
One line per check; exit 0 always (a report, not a gate).
"""
import argparse
import importlib.util
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

M0, M1, W0, W1 = 0xD2511F53, 0xCD9E8D57, 0x9E3779B9, 0xBB67AE85
MASK = np.uint64(0xFFFFFFFF)


def philox(seed, subseq, off):
    """numpy Philox4x32-10: arrays of the four 32-bit outputs for counter (off//4 lo, hi, subseq lo, subseq hi), key = seed."""
    n = len(subseq)
    q4 = np.uint64(off // 4)
    c0 = np.full(n, q4 & MASK, np.uint64); c1 = np.full(n, (q4 >> np.uint64(32)) & MASK, np.uint64)
    c2 = (subseq.astype(np.uint64) & MASK); c3 = (subseq.astype(np.uint64) >> np.uint64(32)) & MASK
    k0 = np.uint64(seed & 0xFFFFFFFF); k1 = np.uint64((seed >> 32) & 0xFFFFFFFF)
    for _ in range(10):
        p0 = np.uint64(M0) * c0; hi0 = p0 >> np.uint64(32); lo0 = p0 & MASK
        p1 = np.uint64(M1) * c2; hi1 = p1 >> np.uint64(32); lo1 = p1 & MASK
        c0, c1, c2, c3 = (hi1 ^ c1 ^ k0) & MASK, lo1, (hi0 ^ c3 ^ k1) & MASK, lo0
        k0 = (k0 + np.uint64(W0)) & MASK; k1 = (k1 + np.uint64(W1)) & MASK
    return c0, c1, c2, c3


def uniforms(x, y, kind, dtype):
    if dtype == np.float64:
        if kind == "curand":
            z = x ^ ((y << np.uint64(21)) & np.uint64(0xFFFFFFFFFFFFFFFF))
            return z.astype(np.float64) * 2.0 ** -53 + 2.0 ** -54
        z = x | ((y >> np.uint64(11)) << np.uint64(32))
        return 2.0 ** -53 + z.astype(np.float64) * 2.0 ** -53
    xf = x.astype(np.float32)
    if kind == "curand":
        return (xf * np.float32(2.0 ** -32) + np.float32(2.0 ** -33)).astype(np.float32)
    return (np.float32(2.0 ** -32) + xf * np.float32(2.0 ** -32)).astype(np.float32)


def check_exponential(dev, seeds, offsets, n=168):
    out = []
    for dt, npdt in ((torch.float64, np.float64), (torch.float32, np.float32)):
        tot = 0; exact = {"curand": 0, "rocrand": 0}; ulp1 = {"curand": 0, "rocrand": 0}
        for seed in seeds:
            for off in offsets:
                g = torch.Generator(device=dev); g.manual_seed(seed); g.set_offset(off)
                q = torch.empty(n, dtype=dt, device=dev).exponential_(1.0, generator=g).cpu().numpy()
                x, y, _, _ = philox(seed, np.arange(n), off)
                for kind in ("curand", "rocrand"):
                    u = uniforms(x, y, kind, npdt)
                    eps = np.finfo(npdt).eps
                    ref = np.where(u >= 1.0 - eps / 2, eps / 2, -np.log(u)).astype(npdt)
                    exact[kind] += int((ref == q).sum())
                    ulp1[kind] += int((np.abs(ref.astype(np.float64) - q.astype(np.float64)) <= np.spacing(np.abs(ref)).astype(np.float64)).sum())
                tot += n
        out.append("EXPONENTIAL %s: %d values; exact match curand=%d rocrand=%d; within 1 ulp curand=%d rocrand=%d" % (
            str(dt).replace("torch.", ""), tot, exact["curand"], exact["rocrand"], ulp1["curand"], ulp1["rocrand"]))
    return out


def load_kit_kernel(path):
    spec = importlib.util.spec_from_file_location("pms_kit_worker_draw", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod._fused_draw_kernel, mod._pow2


def check_draw(dev, kernels, pow2, Bs, dtypes, n_steps, K, cases):
    lines = []
    for dt in dtypes:
        for B in Bs:
            A = 21
            gp = torch.Generator(device=dev); gp.manual_seed(12345)
            torch.multinomial(torch.full((B, A), 1.0 / A, device=dev, dtype=dt), 1, generator=gp); inc = int(gp.get_offset())
            res = {name: [0, 0] for name in kernels}
            for case in range(cases):
                seed = 37 + 1000 * case
                offs = [4 * (10 + 997 * b + 31 * case) for b in range(K)]
                gcpu = torch.Generator().manual_seed(4321 + case)
                logits = torch.randn(n_steps, K * B, A, generator=gcpu) * 3
                logits[::3, :, :5] = -1e9; logits[::7, min(1, K * B - 1)] = -1e9; logits[::7, min(1, K * B - 1), 4] = 0.0
                probs_all = F.softmax(logits.to(dev).to(dt), -1).contiguous()
                gens = []
                for o in offs:
                    g = torch.Generator(device=dev); g.manual_seed(seed); g.set_offset(o); gens.append(g)
                ref = torch.zeros((n_steps, K * B), dtype=torch.int64, device=dev)
                for step in range(n_steps):
                    for b in range(K):
                        ref[step, b * B:(b + 1) * B] = torch.multinomial(probs_all[step, b * B:(b + 1) * B], 1, generator=gens[b])[:, 0]
                for name, kern in kernels.items():
                    seed_t = torch.full((K,), seed, dtype=torch.int64, device=dev); off_t = torch.tensor(offs, dtype=torch.int64, device=dev)
                    ndraw = torch.zeros(K, dtype=torch.int64, device=dev); active = torch.ones(K, dtype=torch.int64, device=dev)
                    out = torch.full((n_steps, K * B), -1, dtype=torch.int64, device=dev)
                    try:
                        for step in range(n_steps):
                            kern[(K,)](probs_all[step], out[step], seed_t, off_t, ndraw, active, B=B, BP=pow2(B), A=A, AP=32, INC=inc, F64=(dt == torch.float64))
                        bad = int((ref != out).any(dim=1).sum().item())
                    except Exception as e:
                        lines.append("DRAW %s B=%d %s: EXCEPTION %s: %s" % (str(dt).replace("torch.", ""), B, name, type(e).__name__, str(e).splitlines()[0][:160] if str(e) else ""))
                        bad = n_steps
                    res[name][0] += n_steps - bad; res[name][1] += n_steps
            lines.append("DRAW %s B=%2d K=%d inc=%d: steps matching torch.multinomial -> %s" % (
                str(dt).replace("torch.", ""), B, K, inc, "  ".join("%s %d/%d" % (n, r[0], r[1]) for n, r in res.items())))
    return lines


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kit-worker", required=True)
    ap.add_argument("--cases", type=int, default=4)
    ap.add_argument("--steps", type=int, default=24)
    ap.add_argument("--Bs", default="1,2,3,4,8,16,32")
    ap.add_argument("--K", type=int, default=4)
    ap.add_argument("--dtypes", default="float64,float32")
    ap.add_argument("--no-exponential", action="store_true")
    a = ap.parse_args()
    dev = torch.device("cuda:0")
    print("DEVICE %s cc=%s torch=%s hip=%s" % (torch.cuda.get_device_name(0), torch.cuda.get_device_capability(0), torch.__version__, torch.version.hip), flush=True)
    t0 = time.time()
    if not a.no_exponential:
        for ln in check_exponential(dev, seeds=[0, 37, 123456789, 2 ** 40 + 7], offsets=[0, 4, 40, 1000, 2 ** 20, 2 ** 33 + 8]):
            print(ln, flush=True)
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts"))
    kit_kernel, pow2 = load_kit_kernel(a.kit_worker)
    import pms_worker_boot
    kernels = {"kit(curand)": kit_kernel, "pms(rocrand)": pms_worker_boot._rocm_draw_kernel()}
    for ln in check_draw(dev, kernels, pow2, Bs=[int(x) for x in a.Bs.split(",")], dtypes=[getattr(torch, d) for d in a.dtypes.split(",")],
                         n_steps=a.steps, K=a.K, cases=a.cases):
        print(ln, flush=True)
    print("DRAWCHECK done in %.1f s" % (time.time() - t0), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
