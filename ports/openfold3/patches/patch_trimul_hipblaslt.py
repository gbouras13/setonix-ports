#!/usr/bin/env python3
"""Run the triangle multiplication's tile contraction on hipBLASLt (env-gated; off by default).

WHY
  OpenFold3 0.5.0 forces the rocBLAS backend on ROCm (entry_points/import_utils.py: preferred_blas_library("cublas"),
  "to match the numerics of NVIDIA-trained models"). The row-sharded trimul's contraction, T = A @ B per held ring block
  (opt_core mem/rowpair/trimul.py _tiles: bf16 [C, rows, N] @ [C, N, w]), then gets rocBLAS's small-tile
  Cijk_Ailk_Bljk_BBS_BH_MT16x32x64 kernel: 13.4 ms per call at 6,930 res / P=8 (job 50190304, ~17 TFLOP/s). The same
  shape on hipBLASLt (MT128x160x32) takes 3.2 ms (ring benchmark, jobs 50216013 / 50247399, with or without the
  deterministic recipe). On gfx90a both accumulate bf16 products in fp32, so the change is one of rounding order, not
  precision class -- but it is a numerics change, to be judged by a precision test.

WHAT
  ROWPAIR_TRIMUL_HIPBLASLT=1 (ROCm builds, CUDA tensors): around that one torch.matmul, the preferred BLAS backend is
  switched to "cublaslt" (= hipBLASLt) and restored to whatever it was (try/finally). Every other GEMM keeps upstream's
  rocBLAS. Unset = the same statement as before plus one bool test per tile.

USAGE  python3 patch_trimul_hipblaslt.py --apply | --check | --revert [path]
"""
import os, sys, pathlib

P = pathlib.Path('/scratch/pawsey1018/gbouras/af3-setonix/opt_core_src/opt_core/opt_core/mem/rowpair/trimul.py')
if len(sys.argv) > 2:
    P = pathlib.Path(sys.argv[2])
MARK = '# [trimul_hipblaslt]'
BACKUP = P.with_suffix('.py.pre_trimul_hipblaslt')

OLD_DEF = 'def _tiles(A, i0: int, i1: int, Bq, pieces: List[Tuple[int, int]], C: int, N: int, st: "ContractStats"):\n'
GATE = ('# [trimul_hipblaslt] ROWPAIR_TRIMUL_HIPBLASLT=1: the tile contraction below on hipBLASLt instead of the rocBLAS backend\n'
        '# OpenFold3 forces on ROCm (preferred_blas_library("cublas")); bf16 operands, fp32 accumulation in both libraries. Only\n'
        '# this matmul is switched (and restored); every other GEMM keeps upstream\'s choice.\n'
        '_TRIMUL_HIPBLASLT = os.environ.get("ROWPAIR_TRIMUL_HIPBLASLT", "") == "1" and getattr(torch.version, "hip", None) is not None\n'
        '\n\n')
OLD_MM = "        T = torch.matmul(Ap, Bq)                              # [C, rows, v]  full k\n"
NEW_MM = ("        if _TRIMUL_HIPBLASLT and Ap.is_cuda:                  # [trimul_hipblaslt]\n"
          "            _prev = torch.backends.cuda.preferred_blas_library()\n"
          "            torch.backends.cuda.preferred_blas_library(\"cublaslt\")\n"
          "            try:\n"
          "                T = torch.matmul(Ap, Bq)                      # [C, rows, v]  full k, on hipBLASLt\n"
          "            finally:\n"
          "                torch.backends.cuda.preferred_blas_library(_prev)\n"
          "        else:\n"
          "            T = torch.matmul(Ap, Bq)                          # [C, rows, v]  full k\n")


def _apply(s):
    counts = {"def": s.count(OLD_DEF), "matmul": s.count(OLD_MM)}
    if any(n != 1 for n in counts.values()):
        raise SystemExit(f'ANCHORS NOT FOUND EXACTLY ONCE ({counts}) -- not patching')
    for need in ('import os', 'from ._torch import torch'):
        if need not in s:
            raise SystemExit(f'trimul.py lacks "{need}" at module level -- not patching')
    new = s.replace(OLD_DEF, GATE + OLD_DEF, 1).replace(OLD_MM, NEW_MM, 1)
    compile(new, str(P), 'exec')
    return new


def _write(text):
    tmp = P.with_suffix('.py.trimul_hipblaslt_tmp')
    tmp.write_text(text)
    os.replace(tmp, P)                                     # atomic: importers never see a half-written file


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else '--check'
    s = P.read_text()
    applied = MARK in s
    if mode == '--check':
        print(f'trimul_hipblaslt: applied={applied}  backup={BACKUP.exists()}')
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
