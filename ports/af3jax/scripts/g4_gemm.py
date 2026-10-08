#!/usr/bin/env python
"""G4 (AF3JAX_PERF_PLAN.md, Status, 2026-10-04): are large GEMMs computed correctly on one GCD, and by which library?

G3 (job 50379642) found setup B's rocblas runs silently wrong at buckets 4,352 and 4,480 (rc 0, finite confidences, misfolded chains)
and clean at 3,968. This probe runs ONE case per process, under the GEMM settings in XLA_FLAGS (the job gives each lane its own):
  - 2d:  C = A @ B, A shaped like a flattened pair array (rows = tokens^2, or tokens^2 / 8 for setup C's per-GCD rows);
  - out/in: triangle multiplication's channel-batched contraction on (C, N, N) planes, as the fork states it
    (outgoing 'cik,cjk->cij', incoming 'ckj,cki->cij').
The inputs come from jax.random in fixed blocks (identical in every lane; no single random call is large). Per case:
  1. which implementation XLA compiled the product to (rocBLAS / hipBLASLt custom calls, Triton GEMM fusions);
  2. fingerprints of the inputs before and after the first product (a GEMM that writes outside its output can hit them);
  3. Freivalds' test over EVERY row on the GPU: C w against A (B w) for 4 random vectors, by reductions over row chunks small enough
     that no library call inside the check can overflow; a row is bad when the normalised difference exceeds 0.05 (clean bf16 rows
     sit near 1e-3);
  4. a float64 CPU reference on sampled rows: first, last, around every 2^30 / 2^31 / 2^32 element and byte boundary, random;
  5. the time of warm calls (after the checks, so stray writes while timing cannot spoil them).
One JSON line per case is appended to --out. --scale shrinks every case for a local CPU test; --corrupt_from zeroes rows (2d) or
planes (out/in) of C from that index, to show the detection finds them.
"""
import argparse
import json
import os
import re
import sys
import time
import traceback

import numpy as np

# id: (kind, dims, dtype in, dtype out, why).  2d dims = (M, K, N);  out/in dims = (channels, N)
CASES = {
    "c01": ("2d", (16711680, 128, 128), "bf16", "bf16", "A and C just below 2^31 elements (rows 2^24 - 2^16)"),
    "c02": ("2d", (16842752, 128, 128), "bf16", "bf16", "A and C just above 2^31 elements (rows 2^24 + 2^16)"),
    "c03": ("2d", (18939904, 128, 128), "bf16", "bf16", "4,352^2 rows (G3's failing bucket), 128 -> 128"),
    "c04": ("2d", (8323072, 128, 256), "bf16", "bf16", "C just below 2^31 (rows 2^23 - 2^16); A far below"),
    "c05": ("2d", (8454144, 128, 256), "bf16", "bf16", "C just above 2^31 (rows 2^23 + 2^16); A below"),
    "c06": ("2d", (8454144, 256, 128), "bf16", "bf16", "A just above 2^31; C below"),
    "c07": ("2d", (16711680, 128, 256), "bf16", "bf16", "C above 2^31, just below 2^32"),
    "c08": ("2d", (16842752, 128, 256), "bf16", "bf16", "C just above 2^32"),
    "c09": ("2d", (15745024, 128, 128), "bf16", "bf16", "3,968^2 rows (setup B's rocblas cap), 128 -> 128"),
    "c10": ("2d", (15745024, 128, 384), "bf16", "bf16", "3,968^2 rows, 128 -> 384 (merged projections)"),
    "c11": ("2d", (15745024, 512, 128), "bf16", "bf16", "3,968^2 rows, 512 -> 128 (pair transition, down)"),
    "c12": ("2d", (11943936, 128, 384), "bf16", "bf16", "3,456^2 rows (setup A's rocblas cap), 128 -> 384"),
    "c13": ("2d", (18939904, 128, 128), "bf16", "f32", "4,352^2 rows, bf16 in, f32 out"),
    "c14": ("2d", (18939904, 128, 128), "f32", "f32", "4,352^2 rows, f32"),
    "c15": ("2d", (10913792, 128, 384), "bf16", "bf16", "9,344^2 / 8 rows (setup C's per-GCD pair rows at the one-node max), 128 -> 384"),
    "c16": ("2d", (10913792, 128, 512), "bf16", "bf16", "9,344^2 / 8 rows, 128 -> 512"),
    "t01": ("out", (128, 3968), "bf16", "bf16", "triangle multiplication, outgoing, 3,968 (each operand 2.02e9 elements)"),
    "t02": ("out", (128, 4096), "bf16", "bf16", "outgoing, 4,096 (each operand exactly 2^31 elements)"),
    "t03": ("out", (128, 4352), "bf16", "bf16", "outgoing, 4,352 (each operand 2.42e9 elements)"),
    "t04": ("in", (128, 4352), "bf16", "bf16", "incoming, 4,352"),
    "m01": ("2d", (4456448, 64, 512), "bf16", "bf16", "MSA transition at 4,352 (1,024 x 4,352 rows, 64 -> 512; C 1.06 x 2^31)"),
}
BYTES = {"bf16": 2, "f32": 4}
THRESH = 0.05


def pow2_divisor(n, cap):
    return min(n & -n, cap)


def classify(hlo):
    """the GEMM implementations in compiled HLO text, and the first library call (for a bug report)"""
    n_rb = hlo.count('custom_call_target="__cublas$gemm"')
    n_lt = hlo.count('custom_call_target="__cublas$lt$matmul')
    n_tr = len(re.findall(r'"kind":"__triton[a-z_]*gemm', hlo))
    parts = [f"{k}x{n}" for k, n in (("rocblas", n_rb), ("hipblaslt", n_lt), ("triton", n_tr)) if n]
    if not parts:
        parts = [f"other(dot x{len(re.findall(r'[ =]dot[(]', hlo))})"]
    first = next((ln.strip()[:600] for ln in hlo.splitlines() if "__cublas$" in ln), "")
    return "+".join(parts), first


def library_detail(hlo):
    """the first library GEMM call in full: selected algorithm, gemm_backend_config (precision_config dropped) and operand layouts"""
    lines = hlo.splitlines()
    defs = {}
    for ln in lines:
        m = re.match(r"\s*(?:ROOT\s+)?%?([\w.\-]+)\s*=\s*(\(?[a-z0-9]+\[[0-9,]*\]\{[0-9,]*\})", ln)
        if m:
            defs[m.group(1)] = m.group(2)
    for ln in lines:
        if "__cublas$" not in ln:
            continue
        alg = re.findall(r'"selected_algorithm":"?(-?\d+)', ln)
        cfg, i = "", ln.find('"gemm_backend_config":')
        if i >= 0:   # parsed, not matched: compiles print the keys in different orders
            try:
                obj = json.JSONDecoder().raw_decode(ln[i + len('"gemm_backend_config":'):])[0]
                obj.pop("precision_config", None)
                cfg = json.dumps(obj, sort_keys=True)
            except ValueError:
                pass
        ops = re.search(r"custom-call\(([^)]*)\)", ln)
        lay = [defs.get(o.strip().lstrip("%").split()[-1], o.strip()) for o in ops.group(1).split(",")] if ops else []
        res = re.search(r"=\s*(\([^)]*\)|[a-z0-9]+\[[0-9,]*\]\{[0-9,]*\})", ln)
        return {"algorithm": int(alg[0]) if alg else None, "gemm_cfg": cfg[:1500], "operands": lay,
                "result": res.group(1) if res else None}
    return {}


def ranges(idx, cap=12):
    """contiguous runs of sorted indices, as [start, end] pairs (at most cap)"""
    if idx.size == 0:
        return []
    cut = np.flatnonzero(np.diff(idx) != 1)
    starts = np.concatenate(([idx[0]], idx[cut + 1]))
    ends = np.concatenate((idx[cut], [idx[-1]]))
    return [[int(s), int(e)] for s, e in zip(starts[:cap], ends[:cap])]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lane", required=True)
    ap.add_argument("--case", required=True, help="an id from CASES, or from --spec")
    ap.add_argument("--spec", help="G6: a JSON list of production GEMMs (scripts/g6_cases.json); --case picks one by id")
    ap.add_argument("--out", required=True)
    ap.add_argument("--scale", type=int, default=1, help="shrink the case (local CPU tests)")
    ap.add_argument("--corrupt_from", type=int, default=-1, help="test: zero C from this row (2d) or plane (out/in)")
    ap.add_argument("--timed", type=int, default=3)
    ap.add_argument("--tag", default="", help="a label for this run (G5: the forced algorithm)")
    a = ap.parse_args()
    rec = {"lane": a.lane, "case": a.case, "tag": a.tag, "xla_flags": os.environ.get("XLA_FLAGS", ""),
           "serialise": "blocking" if os.environ.get("HIP_LAUNCH_BLOCKING") == "1" else
                        ("copy" if os.environ.get("AMD_SERIALIZE_COPY") else "none")}
    t0 = time.time()
    try:
        if a.spec:
            sp = next(c for c in json.load(open(a.spec)) if c["id"] == a.case)
            run_case_hlo(rec, a, sp)
        else:
            if a.case not in CASES:
                raise KeyError(f"unknown case {a.case}")
            run_case(rec, a)
        rec["status"] = "ok"
    except Exception as e:  # out of memory, a library refusal, ... -- recorded, not fatal to the lane
        rec["status"] = "error"
        rec["error"] = f"{type(e).__name__}: {str(e)[:700]}"
        rec["traceback"] = traceback.format_exc()[-2000:]
    rec["wall_s"] = round(time.time() - t0, 1)
    with open(a.out, "a") as f:
        f.write(json.dumps(rec) + "\n")
    keys = ("lane", "case", "tag", "algorithm", "status", "impl", "gemm_ms", "inputs_changed_by_gemm", "full_bad", "full_first_bad", "spot_bad", "error")
    print("G4 " + json.dumps({k: rec.get(k) for k in keys if k in rec}), flush=True)


def run_case(rec, a):
    import jax
    import jax.numpy as jnp
    lax = jax.lax
    dt = {"bf16": jnp.bfloat16, "f32": jnp.float32}
    kind, dims, din, dout, why = CASES[a.case]
    if a.scale > 1:
        dims = (dims[0] // a.scale,) + tuple(dims[1:]) if kind == "2d" else (dims[0], max(8, dims[1] // int(a.scale ** 0.5)))
    rec.update(kind=kind, dims=list(dims), dtype_in=din, dtype_out=dout, why=why, jax=jax.__version__, device=str(jax.devices()[0]))
    key = jax.random.PRNGKey(20261004)
    kA, kB, kW = jax.random.split(key, 3)

    # ---- inputs (blocked generation: the array plus one block of temporaries) ----
    t = time.time()
    if kind == "2d":
        M, K, N = dims
        blk = pow2_divisor(M, 1 << 16)

        def gen_a():
            def body(i, x):
                v = jax.random.normal(jax.random.fold_in(kA, i), (blk, K), jnp.float32)
                return lax.dynamic_update_slice(x, v.astype(dt[din]), (i * blk, 0))
            return lax.fori_loop(0, M // blk, body, jnp.zeros((M, K), dt[din]))
        A = jax.jit(gen_a)()
        B = (jax.random.normal(kB, (K, N), jnp.float32) / np.sqrt(K)).astype(dt[din])
        elems = {"A": M * K, "B": K * N, "C": M * N}
        flops = 2.0 * M * K * N
    else:
        Cn, N = dims

        def gen_planes(k, scale):
            def body(c, x):
                v = jax.random.normal(jax.random.fold_in(k, c), (1, N, N), jnp.float32) * scale
                return lax.dynamic_update_slice(x, v.astype(dt[din]), (c, 0, 0))
            return lax.fori_loop(0, Cn, body, jnp.zeros((Cn, N, N), dt[din]))
        A = jax.jit(lambda: gen_planes(kA, 1.0))()
        B = jax.jit(lambda: gen_planes(kB, float(1.0 / np.sqrt(N))))()
        elems = {"A": Cn * N * N, "B": Cn * N * N, "C": Cn * N * N}
        flops = 2.0 * Cn * N * N * N
    jax.block_until_ready((A, B))
    rec["gen_s"] = round(time.time() - t, 1)
    rec["elements"] = elems
    rec["elements_vs_2^31"] = {k: round(v / 2 ** 31, 3) for k, v in elems.items()}

    def fingerprint(x):    # one f32 sum per block of the leading axis: any overwrite changes it
        if x.ndim == 2:
            ch = pow2_divisor(x.shape[0], 1 << 16)
            return x.astype(jnp.float32).reshape(-1, ch, x.shape[1]).sum(axis=(1, 2))
        return x.astype(jnp.float32).sum(axis=(1, 2))
    fp = jax.jit(fingerprint)
    fpA0, fpB0 = np.asarray(fp(A)), np.asarray(fp(B))

    # ---- the product: compile, inspect, one call ----
    if kind == "2d":
        f = lambda A, B: jnp.matmul(A, B, preferred_element_type=dt[dout])
    elif kind == "out":
        f = lambda A, B: jnp.einsum("cik,cjk->cij", A, B, preferred_element_type=dt[dout])
    else:
        f = lambda A, B: jnp.einsum("ckj,cki->cij", A, B, preferred_element_type=dt[dout])
    t = time.time()
    comp = jax.jit(f).lower(A, B).compile()
    rec["compile_s"] = round(time.time() - t, 1)
    hlo = comp.as_text()
    rec["impl"], rec["library_call"] = classify(hlo)
    rec.update(library_detail(hlo))
    t = time.time()
    C = comp(A, B)
    C.block_until_ready()
    rec["first_call_ms"] = round((time.time() - t) * 1e3, 2)
    if a.corrupt_from >= 0:
        C = C.at[a.corrupt_from:].set(0)
    fpA1, fpB1 = np.asarray(fp(A)), np.asarray(fp(B))
    chgA, chgB = np.flatnonzero(fpA1 != fpA0), np.flatnonzero(fpB1 != fpB0)
    rec["inputs_changed_by_gemm"] = int(chgA.size + chgB.size)
    if chgA.size or chgB.size:
        rec["inputs_changed_blocks"] = {"A": ranges(chgA), "B": ranges(chgB)}

    # ---- Freivalds over every row (GPU), then float64 on sampled rows (CPU) ----
    nv = 4
    W = np.asarray(jax.random.normal(kW, (nv, N), jnp.float32), np.float64)
    norm = np.sqrt(N) * np.sqrt((W ** 2).mean(axis=1))
    Wd, nd = jnp.asarray(W, jnp.float32), jnp.asarray(norm, jnp.float32)
    t = time.time()
    if kind == "2d":
        B64 = np.asarray(B.astype(jnp.float32), np.float64)
        Ud = jnp.asarray((B64 @ W.T).T, jnp.float32)           # B w_v, exact on the CPU
        ch = pow2_divisor(M, 1 << 16)

        def check(A, C):
            def body(i, err):
                c = lax.dynamic_slice(C, (i * ch, 0), (ch, N)).astype(jnp.float32)
                x = lax.dynamic_slice(A, (i * ch, 0), (ch, K)).astype(jnp.float32)
                e = jnp.zeros((ch,), jnp.float32)
                for v in range(nv):
                    d = jnp.abs(jnp.sum(c * Wd[v][None, :], axis=1) - jnp.sum(x * Ud[v][None, :], axis=1)) / nd[v]
                    e = jnp.maximum(e, jnp.where(jnp.isfinite(d), d, jnp.float32(1e30)))
                return lax.dynamic_update_slice(err, e, (i * ch,))
            return lax.fori_loop(0, M // ch, body, jnp.zeros((M,), jnp.float32))
        cj = jax.jit(check).lower(A, C).compile()
        err = np.asarray(cj(A, C))
    else:
        def check(A, B, C):
            def body(c, err):
                x = lax.dynamic_index_in_dim(A, c, 0, keepdims=False).astype(jnp.float32)
                y = lax.dynamic_index_in_dim(B, c, 0, keepdims=False).astype(jnp.float32)
                o = lax.dynamic_index_in_dim(C, c, 0, keepdims=False).astype(jnp.float32)
                e = jnp.zeros((N,), jnp.float32)
                for v in range(nv):
                    w = Wd[v]
                    y1 = jnp.sum(o * w[None, :], axis=1)                          # (C w)[i]
                    if kind == "out":    # C[i,j] = sum_k x[i,k] y[j,k]  ->  (C w)[i] = sum_k x[i,k] (sum_j y[j,k] w[j])
                        y2 = jnp.sum(x * jnp.sum(y * w[:, None], axis=0)[None, :], axis=1)
                    else:                # C[i,j] = sum_k x[k,j] y[k,i]  ->  (C w)[i] = sum_k y[k,i] (sum_j x[k,j] w[j])
                        y2 = jnp.sum(y * jnp.sum(x * w[None, :], axis=1)[:, None], axis=0)
                    d = jnp.abs(y1 - y2) / nd[v]
                    e = jnp.maximum(e, jnp.where(jnp.isfinite(d), d, jnp.float32(1e30)))
                return lax.dynamic_update_index_in_dim(err, e, c, 0)
            return lax.fori_loop(0, Cn, body, jnp.zeros((Cn, N), jnp.float32))
        cj = jax.jit(check).lower(A, B, C).compile()
        err = np.asarray(cj(A, B, C))
    rec["check_s"] = round(time.time() - t, 1)
    rec["check_impl"] = classify(cj.as_text())[0]
    bad = err > THRESH
    rec["full_rows"] = int(err.size)
    rec["full_bad"] = int(bad.sum())
    rec["full_noise_max"] = float(err[~bad].max()) if (~bad).any() else None
    if kind == "2d":
        bidx = np.flatnonzero(bad)
        rec["full_first_bad"] = int(bidx[0]) if bidx.size else None
        rec["full_last_bad"] = int(bidx[-1]) if bidx.size else None
        rec["full_bad_ranges"] = ranges(bidx)
        if bidx.size:
            r0 = int(bidx[0])
            rec["first_bad_offsets"] = {"row*K": r0 * K, "row*N": r0 * N, "row*K*bytes_in": r0 * K * BYTES[din],
                                        "row*N*bytes_out": r0 * N * BYTES[dout]}
    else:
        per = bad.sum(axis=1)
        planes = np.flatnonzero(per)
        rec["full_bad_planes"] = [[int(c), int(per[c]), int(np.flatnonzero(bad[c])[0])] for c in planes[:40]]
        rec["full_first_bad"] = [int(planes[0]), int(np.flatnonzero(bad[planes[0]])[0])] if planes.size else None
        if planes.size:
            rec["first_bad_offsets"] = {"plane*N*N": int(planes[0]) * N * N}

    rng = np.random.default_rng(7)
    if kind == "2d":
        rows = set(range(min(32, M))) | set(range(max(0, M - 32), M))
        bounds = {}
        for name, x in (("K", K), ("N", N)):
            for nb in sorted({1, BYTES[din], BYTES[dout]}):
                for ln, lim in (("2^30", 1 << 30), ("2^31", 1 << 31), ("2^32", 1 << 32)):
                    r = lim // (x * nb)
                    if 0 < r < M:
                        bounds[f"row*{name}*{nb}B = {ln}"] = r
                        rows |= set(range(max(0, r - 4), min(M, r + 5)))
        rows |= set(rng.integers(0, M, 256).tolist())
        R = np.array(sorted(rows))
        Rd = jnp.asarray(R, jnp.int32)
        AR = np.asarray(jnp.take(A, Rd, axis=0).astype(jnp.float32), np.float64)
        CR = np.asarray(jnp.take(C, Rd, axis=0).astype(jnp.float32), np.float64)
        ref = AR @ B64
        serr = np.abs(CR - ref).max(axis=1) / (np.abs(ref).max(axis=1) + 1e-30)
        sbad = R[serr > THRESH]
        rec["bounds"] = {k: [int(v), int(bad[max(0, v - 64):v].sum()), int(bad[v:v + 64].sum())] for k, v in bounds.items()}
    else:
        picks = sorted({c for c in (0, 1, 63, 64, 100, 112, 113, 114, 120, Cn - 1) if c < Cn})
        sbad, serr_all = [], []
        for c in picks:
            ii = sorted({0, 1, N // 2, N - 1} | set(rng.integers(0, N, 4).tolist()))
            plane = np.asarray((B if kind == "out" else A)[c].astype(jnp.float32), np.float64)
            vecs = np.asarray((A[c][jnp.asarray(ii)] if kind == "out" else B[c][:, jnp.asarray(ii)].T).astype(jnp.float32), np.float64)
            ref = vecs @ plane.T if kind == "out" else vecs @ plane
            got = np.asarray(C[c][jnp.asarray(ii)].astype(jnp.float32), np.float64)
            e = np.abs(got - ref).max(axis=1) / (np.abs(ref).max(axis=1) + 1e-30)
            serr_all.extend(e.tolist())
            sbad.extend([[c, i] for i, x in zip(ii, e) if x > THRESH])
        serr = np.array(serr_all)
    rec["spot_rows"] = int(serr.size)
    rec["spot_bad"] = int(len(sbad))
    rec["spot_bad_list"] = [x if isinstance(x, list) else int(x) for x in list(sbad)[:24]]
    rec["spot_noise_max"] = float(serr[serr <= THRESH].max()) if (serr <= THRESH).any() else None
    # every spot-bad row must be Freivalds-bad (two independent checks of the same output)
    if kind == "2d":
        rec["spot_bad_not_in_full"] = int(sum(1 for r in sbad if not bad[r]))
    else:
        rec["spot_bad_not_in_full"] = int(sum(1 for c, i in sbad if not bad[c, i]))

    # ---- timing, last ----
    del C
    times = []
    for _ in range(max(1, a.timed)):
        t = time.time()
        o = comp(A, B)
        o.block_until_ready()
        times.append(time.time() - t)
        del o
    rec["gemm_ms"] = round(float(np.median(times)) * 1e3, 2)
    rec["tflops"] = round(flops / float(np.median(times)) / 1e12, 1)
    fpA2, fpB2 = np.asarray(fp(A)), np.asarray(fp(B))
    rec["inputs_changed_after_timing"] = int((fpA2 != fpA0).sum() + (fpB2 != fpB0).sum())



def run_case_hlo(rec, a, sp):
    """G6: one production GEMM replayed exactly. Each operand is generated in its physical order (XLA layouts list dimensions minor to
    major) and transposed to its logical shape inside the jitted product, and the product is transposed to production's physical
    result order, so XLA can absorb both into the library call (the record says whether it did). The checks are run_case's, on
    canonical [batch, M, K] x [batch, K, N] views of the logical arrays."""
    import jax
    import jax.numpy as jnp
    lax = jax.lax
    dt = {"bf16": jnp.bfloat16, "f32": jnp.float32}
    L, R, O, dd = sp["lhs"], sp["rhs"], sp["result"], sp["dd"]
    lc, rc, lb, rb = dd["lc"], dd["rc"], dd["lb"], dd["rb"]
    prod = lambda xs: int(np.prod(xs)) if len(xs) else 1
    m2m = lambda t: list(reversed(t["layout"]))
    phys = lambda t: [t["dims"][i] for i in m2m(t)]
    to_log = lambda x, t: jnp.transpose(x, [m2m(t).index(i) for i in range(len(t["dims"]))])
    to_phys = lambda x, t: jnp.transpose(x, m2m(t))
    lfree = [i for i in range(len(L["dims"])) if i not in lc and i not in lb]
    rfree = [i for i in range(len(R["dims"])) if i not in rc and i not in rb]
    B, M = prod([L["dims"][i] for i in lb]), prod([L["dims"][i] for i in lfree])
    K, N = prod([L["dims"][i] for i in lc]), prod([R["dims"][i] for i in rfree])
    elems = {"A": prod(L["dims"]), "B": prod(R["dims"]), "C": prod(O["dims"])}
    rec.update(kind="hlo", why=sp.get("why", ""), dims={"lhs": L, "rhs": R, "result": O, "dd": dd, "bMKN": [B, M, K, N]},
               dtype_in=L["dtype"], dtype_out=O["dtype"], prod_alg=sp.get("prod_alg"), elements=elems,
               jax=jax.__version__, device=str(jax.devices()[0]))
    rec["elements_vs_2^31"] = {k: round(v / 2 ** 31, 3) for k, v in elems.items()}
    key = jax.random.PRNGKey(20261004)
    kA, kB, kW = jax.random.split(key, 3)

    def gen(k, t, scale):   # flat chunks of at most 2^24 values: no single random call is large
        ps = phys(t)
        n = prod(ps)
        ch = pow2_divisor(n, 1 << 24)

        def body(i, x):
            v = jax.random.normal(jax.random.fold_in(k, i), (ch,), jnp.float32) * scale
            return lax.dynamic_update_slice(x, v.astype(dt[t["dtype"]]), (i * ch,))
        return jax.jit(lambda: lax.fori_loop(0, n // ch, body, jnp.zeros((n,), dt[t["dtype"]])).reshape(ps))()
    t = time.time()
    A = gen(kA, L, 1.0)
    Bm = gen(kB, R, float(1.0 / np.sqrt(K)))
    jax.block_until_ready((A, Bm))
    rec["gen_s"] = round(time.time() - t, 1)

    def fingerprint(x):
        ch = pow2_divisor(x.size, 1 << 20)
        return x.reshape(-1, ch).astype(jnp.float32).sum(axis=1)
    fp = jax.jit(fingerprint)
    fpA0, fpB0 = np.asarray(fp(A)), np.asarray(fp(Bm))

    def f(lp, rp):
        o = lax.dot_general(to_log(lp, L), to_log(rp, R), ((lc, rc), (lb, rb)), preferred_element_type=dt[O["dtype"]])
        assert list(o.shape) == O["dims"], f"dot shape {o.shape} != production {O['dims']}"
        return to_phys(o, O)
    t = time.time()
    comp = jax.jit(f).lower(A, Bm).compile()
    rec["compile_s"] = round(time.time() - t, 1)
    hlo = comp.as_text()
    rec["impl"], rec["library_call"] = classify(hlo)
    rec.update(library_detail(hlo))
    rec["n_ops_compiled"] = len(re.findall(r"^\s*(?:ROOT\s+)?%?[\w.\-]+\s*=", hlo, re.M))
    t = time.time()
    C = comp(A, Bm)
    C.block_until_ready()
    rec["first_call_ms"] = round((time.time() - t) * 1e3, 2)
    if a.corrupt_from >= 0:
        C = C.at[a.corrupt_from:].set(0)
    fpA1, fpB1 = np.asarray(fp(A)), np.asarray(fp(Bm))
    rec["inputs_changed_by_gemm"] = int((fpA1 != fpA0).sum() + (fpB1 != fpB0).sum())

    def canon(lp, rp, op):
        l = jnp.transpose(to_log(lp, L), lb + lfree + lc).reshape(B, M, K)
        r = jnp.transpose(to_log(rp, R), rb + rc + rfree).reshape(B, K, N)
        return l, r, to_log(op, O).reshape(B, M, N)
    nv = 4
    W = np.asarray(jax.random.normal(kW, (nv, N), jnp.float32), np.float64)
    nd = jnp.asarray(np.sqrt(N) * np.sqrt((W ** 2).mean(axis=1)), jnp.float32)
    Wd = jnp.asarray(W, jnp.float32)
    ch = pow2_divisor(M, 1 << 16)

    def check(lp, rp, op):
        l, r, o = canon(lp, rp, op)

        def per_b(b, err):
            lb_ = lax.dynamic_index_in_dim(l, b, 0, keepdims=False)
            rb_ = lax.dynamic_index_in_dim(r, b, 0, keepdims=False).astype(jnp.float32)
            ob_ = lax.dynamic_index_in_dim(o, b, 0, keepdims=False)
            U = [jnp.sum(rb_ * Wd[v][None, :], axis=1) for v in range(nv)]

            def per_c(i, e):
                x = lax.dynamic_slice(lb_, (i * ch, 0), (ch, K)).astype(jnp.float32)
                y = lax.dynamic_slice(ob_, (i * ch, 0), (ch, N)).astype(jnp.float32)
                ee = jnp.zeros((ch,), jnp.float32)
                for v in range(nv):
                    d = jnp.abs(jnp.sum(y * Wd[v][None, :], axis=1) - jnp.sum(x * U[v][None, :], axis=1)) / nd[v]
                    ee = jnp.maximum(ee, jnp.where(jnp.isfinite(d), d, jnp.float32(1e30)))
                return lax.dynamic_update_slice(e, ee, (i * ch,))
            return lax.dynamic_update_index_in_dim(err, lax.fori_loop(0, M // ch, per_c, jnp.zeros((M,), jnp.float32)), b, 0)
        return lax.fori_loop(0, B, per_b, jnp.zeros((B, M), jnp.float32))
    t = time.time()
    cj = jax.jit(check).lower(A, Bm, C).compile()
    err = np.asarray(cj(A, Bm, C))
    rec["check_s"] = round(time.time() - t, 1)
    rec["check_impl"] = classify(cj.as_text())[0]
    bad = err > THRESH
    rec["full_rows"] = int(err.size)
    rec["full_bad"] = int(bad.sum())
    rec["full_noise_max"] = float(err[~bad].max()) if (~bad).any() else None
    flat = np.flatnonzero(bad.reshape(-1))
    rec["full_first_bad"] = [int(flat[0] // M), int(flat[0] % M)] if flat.size else None
    rec["full_bad_ranges"] = ranges(flat)
    if flat.size:
        r0 = int(flat[0])
        rec["first_bad_offsets"] = {"row*K": r0 * K, "row*N": r0 * N}

    rng = np.random.default_rng(7)
    rows = set(range(min(4, M))) | set(range(max(0, M - 4), M)) | {M // 4, M // 2, 3 * M // 4} | set(rng.integers(0, M, 12).tolist())
    for x in (K, N):
        for lim in (1 << 31, 1 << 32):
            r = lim // x
            if r < M:
                rows |= set(range(max(0, r - 2), min(M, r + 3)))
    rows = np.array(sorted(rows))
    pick = jax.jit(lambda lp, rp, op, b, rr: tuple(z[b][rr] if i != 1 else z[b] for i, z in enumerate(canon(lp, rp, op))))
    sbad, serr = [], []
    for b in sorted({0, B // 2, B - 1}):
        lr, rr_, orow = (np.asarray(z.astype(jnp.float32), np.float64) for z in pick(A, Bm, C, b, jnp.asarray(rows, jnp.int32)))
        ref = lr @ rr_
        e = np.abs(orow - ref).max(axis=1) / (np.abs(ref).max(axis=1) + 1e-30)
        serr.extend(e.tolist())
        sbad.extend([[b, int(r)] for r, x in zip(rows, e) if x > THRESH])
    serr = np.array(serr)
    rec["spot_rows"] = int(serr.size)
    rec["spot_bad"] = len(sbad)
    rec["spot_bad_list"] = sbad[:24]
    rec["spot_noise_max"] = float(serr[serr <= THRESH].max()) if (serr <= THRESH).any() else None
    rec["spot_bad_not_in_full"] = int(sum(1 for b, r in sbad if not bad[b, r]))

    del C
    times = []
    for _ in range(max(1, a.timed)):
        t = time.time()
        o = comp(A, Bm)
        o.block_until_ready()
        times.append(time.time() - t)
        del o
    rec["gemm_ms"] = round(float(np.median(times)) * 1e3, 2)
    rec["tflops"] = round(2.0 * B * M * K * N / float(np.median(times)) / 1e12, 1)
    fpA2, fpB2 = np.asarray(fp(A)), np.asarray(fp(Bm))
    rec["inputs_changed_after_timing"] = int((fpA2 != fpA0).sum() + (fpB2 != fpB0).sum())


if __name__ == "__main__":
    sys.exit(main())
