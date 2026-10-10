#!/usr/bin/env python
"""pms_worker_boot.py -- run the kit's staged ProteinMPNN worker with named gfx90a / ROCm patches applied in memory.

    python pms_worker_boot.py <staged kit/mpnn_worker2*.py> <the worker's own arguments ...>

scripts/pms_launch.py rewrites the kit's worker command line ([python, <stage>/.../kit/mpnn_worker2_lowmem.py, args...]) into this. Nothing in
the kit tree or the staged copy is edited: the staged file is loaded as a module under another name (its `if __name__ == "__main__"` stays
inert), named module globals are replaced, and its own main() runs with sys.argv exactly as the worker would have seen it. Its exit code,
lines and end-of-run record are the worker's.

Patches, by name, in PMS_WORKER_PATCHES (comma list; unset = "rocm_draw" on a ROCm torch, nothing on CUDA; "none" = no patch):

  rocm_draw     the worker's fused draw kernel (--fused_draw, `_fused_draw_kernel`) computes torch.multinomial(p, 1, generator) as torch's
                CUDA build computes it: Philox4x32-10 + curand's uniform transforms + ATen's exponential transform + quotient + argmax.
                torch's ROCm build draws through hiprand -> rocRAND: the same Philox4x32-10 (rounds, key schedule, counter layout and the
                (subsequence, offset) seeding are curand's), but rocRAND's uniform transforms differ (rocrand_uniform.h):
                    float  u = 2^-32 + v * 2^-32                      (curand: v * 2^-32 + 2^-33)
                    double u = 2^-53 + (x | (y >> 11) << 32) * 2^-53   (curand: (x ^ (y << 21)) * 2^-53 + 2^-54)
                This replaces the kernel with one that differs from the kit's in those two lines only. The kit's own start-up probe
                (Worker.decide_fused_draw: torch.multinomial on the device vs the kernel, the job's dtype) still judges it, and a FAIL
                still refuses the job by name.
  hybrid_rows=N the worker's --hybrid_gemm row ceiling for this card: HYBRID_GROUP_ROWS[(9, 0)] = N (gfx90a reports compute capability
                (9, 0), the kit's H100 class, which has no ceiling). The kit's own probe still judges every group cell.
  shard=i/n     data parallel (scripts/pms_dp.py): the worker still reads the whole input and computes every backbone's offset in the
                one-process stock stream, then designs only batches i, i+n, i+2n, ... of its own length-sorted plan (plan_batches)
  hybrid_force  NUMERICS-CHANGING (measurement only): the kit's --hybrid_gemm lever runs although its own on-device probe fails (on gfx90a
                the batched message GEMM differs from the per-backbone one by ~1e-6): Worker.decide_hybrid still probes and records the
                failure, then enables every probed cell. Needs --hybrid_gemm requested (pms_dp.py --hybrid).
Diagnostic patches (bisects; each changes WHEN or HOW work is issued, never what is computed):
  lanes=N       Worker.DECODE_LANES = N (the kit: 3 decode loops in flight, each on its own stream)
  no_graph      the CUDA-graph levers inert (Worker.graph_rng / single_graph read False): the eager decode loop and eager scoring forwards.
                NOTE the end-of-run record still shows the requested flags (it records args), so the kit's lever evidence does not see this.
  no_fwd_graph  the scoring forwards eager (Worker.scoring_forward = decoder_scores); the decode-step graphs stay
  one_stream    every stage on the current stream (Worker._stream -> no switch; events recorded on the current stream); use with lanes=1
  xall_minus=a+b  the worker's command line with --x_all replaced by its own expansion minus the named levers (e.g. fused_draw+single_graph);
                the end-of-run record then shows them off, so the kit's parent reports them partial (run with --allow-partial)

One line is printed first: `[pms-boot] patches=... worker=... torch=... hip=...`.
"""
import importlib.util
import os
import sys


def _patch_names():
    raw = os.environ.get("PMS_WORKER_PATCHES")
    if raw is None:
        try:
            import torch
            raw = "rocm_draw" if getattr(torch.version, "hip", None) else ""
        except Exception:
            raw = ""
    return [p.strip() for p in raw.split(",") if p.strip() and p.strip() != "none"]


def _rocm_draw_kernel():
    """The kit's _fused_draw_kernel with rocRAND's uniform transforms (see the module docstring); everything else is the kit's text."""
    import triton
    import triton.language as tl
    from triton.language.extra import libdevice as _tl_libdevice

    _M0 = tl.constexpr(0xD2511F53); _M1 = tl.constexpr(0xCD9E8D57); _W0 = tl.constexpr(0x9E3779B9); _W1 = tl.constexpr(0xBB67AE85)

    @triton.jit
    def _pms_mulhilo32(a, b):
        p = a.to(tl.uint64) * tl.full(a.shape, b, tl.uint64)
        return (p >> 32).to(tl.uint32), (p & 0xFFFFFFFF).to(tl.uint32)

    @triton.jit
    def _pms_fused_draw_kernel_rocm(probs_ptr, out_ptr, seed_ptr, off_ptr, ndraw_ptr, active_ptr, B: tl.constexpr, BP: tl.constexpr, A: tl.constexpr,
                                    AP: tl.constexpr, INC: tl.constexpr, F64: tl.constexpr):
        b = tl.program_id(0)
        act = tl.load(active_ptr + b)
        seed = tl.load(seed_ptr + b); off = tl.load(off_ptr + b) + tl.load(ndraw_ptr + b) * INC
        rows = tl.arange(0, BP)[:, None]; cols = tl.arange(0, AP)[None, :]
        valid = (cols < A) & (rows < B)
        sub = (rows * A + cols).to(tl.uint32)                                  # the element's index = its thread's (sub)sequence (one thread per element)
        q4 = off // 4
        c0 = tl.full([BP, AP], 0, tl.uint32) + (q4 & 0xFFFFFFFF).to(tl.uint32); c1 = tl.full([BP, AP], 0, tl.uint32) + ((q4 >> 32) & 0xFFFFFFFF).to(tl.uint32)
        c2 = sub; c3 = tl.zeros([BP, AP], tl.uint32)
        k0 = tl.full([BP, AP], 0, tl.uint32) + (seed & 0xFFFFFFFF).to(tl.uint32); k1 = tl.full([BP, AP], 0, tl.uint32) + ((seed >> 32) & 0xFFFFFFFF).to(tl.uint32)
        for _ in range(10):                                                    # Philox4x32-10 (rocRAND's ten_rounds = curand's)
            hi0, lo0 = _pms_mulhilo32(c0, _M0)
            hi1, lo1 = _pms_mulhilo32(c2, _M1)
            n0 = hi1 ^ c1 ^ k0; n1 = lo1; n2 = hi0 ^ c3 ^ k1; n3 = lo0
            c0, c1, c2, c3 = n0, n1, n2, n3
            k0 = k0 + tl.full([BP, AP], _W0, tl.uint32); k1 = k1 + tl.full([BP, AP], _W1, tl.uint32)
        if F64:
            zz = c0.to(tl.uint64) | ((c1 >> 11).to(tl.uint64) << 32)            # rocRAND uniform_distribution_double(x, y): (0, 1]
            u = 1.1102230246251565e-16 + zz.to(tl.float64) * 1.1102230246251565e-16
            eps64: tl.constexpr = 2.220446049250313e-16
            lg = tl.where(u >= 1.0 - eps64 / 2, -eps64 / 2, _tl_libdevice.log(u))        # ATen's exponential transformation, double: log
            q = -1.0 * lg
        else:
            u32 = 2.3283064365386963e-10 + c0.to(tl.float32) * 2.3283064365386963e-10    # rocRAND uniform_distribution(v): (0, 1]
            eps32: tl.constexpr = 1.1920928955078125e-07
            lg32 = tl.where(u32 >= 1.0 - eps32 / 2, -eps32 / 2, _tl_libdevice.log(u32))   # ATen's float transformation uses __logf (HIP 7.2: __builtin_logf); Triton's HIP libdevice has no fast_logf. Never reached by the worker (its probabilities are float64)
            q = -1.0 * lg32
        pr = tl.load(probs_ptr + (b * B + rows) * A + cols, mask=valid, other=0.0)
        w = tl.where(valid, pr / q, -1.0)
        idx = tl.argmax(w, axis=1)
        tl.store(out_ptr + b * B + tl.arange(0, BP), idx.to(tl.int64), mask=(tl.arange(0, BP) < B) & (act != 0))
        tl.store(ndraw_ptr + b, tl.load(ndraw_ptr + b) + (act != 0).to(tl.int64))

    return _pms_fused_draw_kernel_rocm


def main():
    if len(sys.argv) < 2 or not sys.argv[1].endswith(".py"):
        print("usage: pms_worker_boot.py <staged worker .py> <worker args...>", file=sys.stderr)
        return 2
    path = os.path.abspath(sys.argv[1])
    names = _patch_names()
    import torch
    print("[pms-boot] patches=%s worker=%s torch=%s hip=%s" % (",".join(names) or "none", os.path.basename(path), torch.__version__,
                                                               getattr(torch.version, "hip", None)), flush=True)
    argv = sys.argv[2:]
    for n in names:                                                            # command-line patches first: main() parses sys.argv
        key, _, val = n.partition("=")
        if key == "xall_minus" and "--x_all" in argv:
            drop = {"--" + x for x in val.split("+") if x}
            exp = ["--chunk_gemm", "--cache_enc_ctx", "--graph_rng", "--single_graph", "--fused_draw", "--analytic_offsets", "--stock_shape_enc"]
            i = argv.index("--x_all")
            argv = argv[:i] + [f for f in exp if f not in drop] + argv[i + 1:]
            print("[pms-boot] xall_minus: --x_all -> %s" % " ".join(f for f in exp if f not in drop), flush=True)
    sys.argv = [path] + argv
    sys.path.insert(0, os.path.dirname(path))                                  # what `python <path>` would have put first
    spec = importlib.util.spec_from_file_location("pms_kit_worker", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["pms_kit_worker"] = mod
    spec.loader.exec_module(mod)                                               # module level only: imports, the kernel definitions, lowmem's install
    for n in names:
        key, _, val = n.partition("=")
        if key == "rocm_draw":
            if not getattr(mod, "HAVE_TRITON", False):
                print("[pms-boot] rocm_draw: no Triton in this process: not applied (the kit's own probe reports it)", flush=True)
                continue
            mod._fused_draw_kernel = _rocm_draw_kernel()
        elif key == "hybrid_rows":
            mod.HYBRID_GROUP_ROWS[(9, 0)] = int(val)
        elif key == "shard":
            si, sn = (int(x) for x in val.split("/"))
            _plan = mod.plan_batches
            mod.plan_batches = lambda *a, **k: _plan(*a, **k)[si::sn]
            if "--out_folder" in sys.argv:                                     # a shard may write nothing: the worker lists <out>/seqs at its end
                os.makedirs(os.path.join(sys.argv[sys.argv.index("--out_folder") + 1], "seqs"), exist_ok=True)
        elif key == "xall_minus":
            pass                                                               # applied to sys.argv above
        elif key == "hybrid_force":
            _decide = mod.Worker.decide_hybrid
            def _forced(self, B, Ks, Kn, requested, _decide=_decide):
                cell = _decide(self, B, Ks, Kn, requested)
                if requested and not self.hybrid_active:
                    groups = {K: mod.hybrid_groups(K, B, self.hybrid_group_rows) for K in sorted(set(Ks))}
                    self.hybrid_cells_ok = frozenset((B, g, Kn) for gs in groups.values() for g in gs)
                    self.hybrid_active = True
                    cell["forced_by"] = "pms-boot hybrid_force (numerics-changing)"
                    print("[pms-boot] hybrid_force: batched message GEMMs ON although the probe failed: NOT bit-identical to stock", flush=True)
                return cell
            mod.Worker.decide_hybrid = _forced
        elif key == "lanes":
            mod.Worker.DECODE_LANES = int(val)
        elif key == "no_graph":
            for attr in ("graph_rng", "single_graph"):
                setattr(mod.Worker, attr, property(lambda self: False, lambda self, v: None))
        elif key == "no_fwd_graph":
            def _eager_scoring(self, hVb, hEb, Ib, Sb, maskb, chain_Mb, randnb, use_input_decoding_order=False, decoding_order=None):
                return self.decoder_scores(hVb, hEb, Ib, Sb, maskb, chain_Mb, randnb, use_input_decoding_order=use_input_decoding_order, decoding_order=decoding_order)
            mod.Worker.scoring_forward = _eager_scoring
        elif key == "one_stream":
            import contextlib
            mod.Worker._stream = lambda self, st: contextlib.nullcontext()
            def _event_here(self, st):
                if st is None:
                    return None
                ev = torch.cuda.Event(enable_timing=True); ev.record(); return ev
            mod.Worker._event = _event_here
        else:
            print("[pms-boot] unknown patch %r: refused" % n, file=sys.stderr, flush=True)
            return 2
    print("[pms-boot] applied: %s" % (", ".join(names) or "nothing"), flush=True)
    return mod.main()


if __name__ == "__main__":
    rc = main()
    sys.exit(rc if isinstance(rc, int) else 0)
