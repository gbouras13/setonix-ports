#!/usr/bin/env python3
"""A Triton row LayerNorm for the tp line's LayerNorm launches (env-gated; off by default). A rounding-order change, not bitwise.

WHY
  The trunk profile at 6,930 res / P=8 (job 50387728) puts LayerNorm at ~5.5% of a pairformer block. ATen's vectorized_layer_norm_kernel
  reaches 227 GB/s on the 128-wide pair rows and 115 GB/s on the 64-wide MSA / template rows of a gfx90a GCD.
  tools/lnbench/lnbench.py (job 50480300, one GCD, the production shapes) timed a multi-row Triton kernel with the kit's fastln arithmetic.
  At 16 rows per program and 4 warps it reached 1.27-1.35 TB/s:
    - 5.6-5.9x ATen at width 128;
    - 11.7x at width 64.
  It is max |y - aten| 1.9-2.9e-6 from ATen, with the same error class vs fp64 as ATen (1.1-1.5e-6 both), and identical run to run.
  Below ~10^4 rows (the single representation, 6,930 x 384) ATen is as fast or faster, so those stay on ATen.

WHAT
  ROWPAIR_LN_KERNEL=1: in tp_rowpair.model's LayerNorm guard (every openfold3 LayerNorm / torch.nn.LayerNorm forward of the process runs
  through it), a launch is served by the Triton kernel when all of these hold; every other launch runs the stock statement unchanged:
    - x is a contiguous CUDA fp32 tensor;
    - it is normalized over one dim of width 32 <= C <= 128;
    - it has >= ROWPAIR_LN_KERNEL_ROWS rows (default 32768);
    - autograd is off for it;
    - weight / bias are fp32 or absent.
  The arithmetic: mean = sum(x) / C; d = x - mean; var = sum(d * d) / C; rstd = div_rn(1, sqrt_rn(var + eps)); y = d * rstd [* w] [+ b].
  That is fp32 two-pass tree sums with int64 offsets, where ATen uses an online Welford. Each process logs its first served launch once.
  Unset = the stock statement for every launch, exactly as before.

USAGE  python3 patch_ln_kernel.py --apply | --check | --revert [path]
"""
import os, sys, pathlib

P = pathlib.Path('/scratch/pawsey1018/gbouras/af3-setonix/of3_opt_src/opt/openfold3_opt/tp_rowpair/model.py')
if len(sys.argv) > 2:
    P = pathlib.Path(sys.argv[2])
MARK = '# [ln_kernel]'
BACKUP = P.with_suffix('.py.pre_ln_kernel')

ANCHOR = "def ln_forward_guarded(orig):\n"
HELPERS = '''# [ln_kernel] ROWPAIR_LN_KERNEL=1: LayerNorm launches of contiguous CUDA fp32 operands over one dim of width 32..128 with >= ROWPAIR_LN_KERNEL_ROWS
# rows (default 32768), autograd off, fp32 or absent weight / bias, run a Triton row kernel (the kit's fastln arithmetic: fp32 two-pass tree-sum
# statistics, rstd = div_rn(1, sqrt_rn(var + eps)), 16 rows x 4 warps per program, int64 offsets) instead of ATen's vectorized kernel: 5.6-5.9x at
# width 128, 11.7x at 64 on gfx90a (job 50480300). Not bitwise vs ATen (max |dy| ~2e-6, the same error class vs fp64). Unset = stock everywhere.
_LN_KERNEL = os.environ.get("ROWPAIR_LN_KERNEL", "").strip().lower() in ("1", "on", "triton")
_LN_KERNEL_ROWS = int(os.environ.get("ROWPAIR_LN_KERNEL_ROWS", "").strip() or 32768)
_LN_KERNEL_STATE = {"kernel": None, "served": 0, "said": False}


if _LN_KERNEL:                                                            # [ln_kernel] defined at module level (Triton resolves `tl` in its globals)
    import triton
    import triton.language as tl

    @triton.jit
    def _ln_rows_fwd(X, W, B, Y, M, eps, N: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_M: tl.constexpr, HAS_W: tl.constexpr,
                     HAS_B: tl.constexpr):
        pid = tl.program_id(0)
        rows = pid * BLOCK_M + tl.arange(0, BLOCK_M)
        cols = tl.arange(0, BLOCK_N)
        cmask = cols < N
        mask = (rows < M)[:, None] & cmask[None, :]
        offs = rows.to(tl.int64)[:, None] * N + cols[None, :]
        x = tl.load(X + offs, mask=mask, other=0.0)
        mean = tl.sum(x, 1) / N
        d = tl.where(mask, x - mean[:, None], 0.0)
        var = tl.sum(d * d, 1) / N
        rstd = tl.div_rn(1.0, tl.sqrt_rn(var + eps))
        y = d * rstd[:, None]
        if HAS_W:
            y = y * tl.load(W + cols, mask=cmask, other=0.0)[None, :]
        if HAS_B:
            y = y + tl.load(B + cols, mask=cmask, other=0.0)[None, :]
        tl.store(Y + offs, y, mask=mask)

    _LN_KERNEL_STATE["kernel"] = _ln_rows_fwd


def _ln_fast(mod, t):
    """``mod``'s LayerNorm of ``t`` through the Triton row kernel when the launch qualifies, else None (the caller runs the stock statement). [ln_kernel]"""
    if not (torch.is_tensor(t) and t.is_cuda and t.dtype == torch.float32 and t.is_contiguous()) or (torch.is_grad_enabled() and t.requires_grad):
        return None
    shape = getattr(mod, "normalized_shape", None)
    if shape is None:
        shape = getattr(mod, "c_in", None)
    shape = tuple(shape) if isinstance(shape, (tuple, list, torch.Size)) else (shape,)
    if len(shape) != 1 or shape[0] is None:
        return None
    C = int(shape[0])
    if t.dim() < 1 or int(t.shape[-1]) != C or not (32 <= C <= 128):
        return None
    M = t.numel() // C
    if M < _LN_KERNEL_ROWS:
        return None
    w, b = getattr(mod, "weight", None), getattr(mod, "bias", None)
    for p in (w, b):
        if p is not None and (p.dtype != torch.float32 or p.numel() != C or not p.is_cuda or not p.is_contiguous()):
            return None
    import triton
    k = _LN_KERNEL_STATE["kernel"]
    y = torch.empty_like(t)
    BM = 16
    k[(triton.cdiv(M, BM),)](t, w if w is not None else y, b if b is not None else y, y, M, float(getattr(mod, "eps", 1e-5)),
                             N=C, BLOCK_N=triton.next_power_of_2(C), BLOCK_M=BM, HAS_W=w is not None, HAS_B=b is not None, num_warps=4)
    _LN_KERNEL_STATE["served"] += 1
    if not _LN_KERNEL_STATE["said"]:
        _LN_KERNEL_STATE["said"] = True
        C_comm_log(f"[ln_kernel] ROWPAIR_LN_KERNEL=1: fp32 LayerNorm launches of width 32..128 with >= {_LN_KERNEL_ROWS:,} rows run the Triton "
                   f"row kernel (16 rows x 4 warps per program); first: {tuple(int(v) for v in t.shape)} width {C}")
    return y


def C_comm_log(msg):
    try:
        C.comm().log(msg)
    except Exception:                                                     # noqa: BLE001 -- a log line never breaks a launch
        print(msg, flush=True)


'''
OLD_LN = ("        def ln(t):\n"
          "            pieces[0] += 1\n"
          "            return orig(self, t, *a, **kw)\n")
NEW_LN = ("        def ln(t):\n"
          "            pieces[0] += 1\n"
          "            if _LN_KERNEL and not a and not kw:                            # [ln_kernel] the Triton row kernel where the launch qualifies\n"
          "                y = _ln_fast(self, t)\n"
          "                if y is not None:\n"
          "                    return y\n"
          "            return orig(self, t, *a, **kw)\n")


def _apply(s):
    counts = {"anchor": s.count(ANCHOR), "ln": s.count(OLD_LN)}
    if any(n != 1 for n in counts.values()):
        raise SystemExit(f'ANCHORS NOT FOUND EXACTLY ONCE ({counts}) -- not patching')
    for need in ('\nimport os\n', '\nimport torch\n', ' core as C'):
        if need not in s:
            raise SystemExit(f'model.py lacks {need.strip()!r} -- not patching')
    new = s.replace(ANCHOR, HELPERS + ANCHOR, 1).replace(OLD_LN, NEW_LN, 1)
    compile(new, str(P), 'exec')
    return new


def _write(text):
    tmp = P.with_suffix('.py.ln_kernel_tmp')
    tmp.write_text(text)
    os.replace(tmp, P)                                     # atomic: importers never see a half-written file


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else '--check'
    s = P.read_text()
    applied = MARK in s
    if mode == '--check':
        print(f'ln_kernel: applied={applied}  backup={BACKUP.exists()}')
        return 0
    if mode == '--apply':
        if applied:
            print('already applied'); return 0
        BACKUP.write_text(s)
        _write(_apply(s))
        print(f'applied (backup: {BACKUP.name})'); return 0
    if mode == '--revert':
        if not BACKUP.exists():
            print('no backup', file=sys.stderr); return 1
        _write(BACKUP.read_text())
        print('reverted'); return 0
    print('usage: --apply | --check | --revert [path]', file=sys.stderr); return 2


sys.exit(main())
