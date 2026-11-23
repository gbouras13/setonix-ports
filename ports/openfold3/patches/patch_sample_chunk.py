#!/usr/bin/env python3
"""Let the tp line run more than one diffusion sample per pass (env-gated; off by default).

WHY
  The row-sharded line runs the diffusion roll-out and the confidence heads one sample at a time: openfold3_opt/sample_loop.py
  hard-codes CHUNK = 1 in plan_for(), so device memory does not grow with --num-diffusion-samples. The kit itself supports any
  chunk, including the one batched pass k == S (tp_rowpair/model.py rollout_rows: "k == S: the one batched pass ... no host round
  trip"), and its draw discipline (opt_core.mem.ckpt.StockOrderDraws) gives every sample stock's noise whatever the chunk. At
  2,310 res / P=4 a sample's 200 steps take ~27 s, now the largest stage (5 x 27 s of a 324 s prediction); batching the samples
  could fill the GPU better at such sizes. Batched == chunked bit for bit only where the kernels are batch-invariant (GEMM
  selection by batch size may differ), so a run with it is compared with one without before it is used.

WHAT  (named ROWPAIR_*: the tp line refuses every OF3TP_* name its ENV_MAP does not list, tp_rowpair/env.py check_known;
      job 50323469's first try as OF3TP_SAMPLE_CHUNK was refused at startup)
  ROWPAIR_SAMPLE_CHUNK=<k> runs k samples per pass (clamped to 1..S); =all runs all S in one pass. Unset = CHUNK (today's plan,
  unchanged statement for statement). Read once at import, like the kit's other words.

USAGE  python3 patch_sample_chunk.py --apply | --check | --revert [path]
"""
import os, sys, pathlib

P = pathlib.Path('/scratch/pawsey1018/gbouras/af3-setonix/of3_opt_src/opt/openfold3_opt/sample_loop.py')
if len(sys.argv) > 2:
    P = pathlib.Path(sys.argv[2])
MARK = '# [sample_chunk]'
BACKUP = P.with_suffix('.py.pre_sample_chunk')

OLD = ('def plan_for(n_samples: int):\n'
       '    """The tp line\'s plan: ``CHUNK`` samples per pass (source ``default``)."""\n'
       '    return core().plan(int(n_samples), None, default_chunk=CHUNK)\n')
NEW = ('# [sample_chunk] ROWPAIR_SAMPLE_CHUNK=<k> | all: k samples per pass (clamped to 1..S; all = the one batched pass, k == S).\n'
       '# Unset = CHUNK (one sample per pass), as before.\n'
       '_SAMPLE_CHUNK_ENV = os.environ.get("ROWPAIR_SAMPLE_CHUNK", "").strip().lower()\n'
       '\n'
       '\n'
       'def plan_for(n_samples: int):\n'
       '    """The tp line\'s plan: ``CHUNK`` samples per pass (source ``default``); ``ROWPAIR_SAMPLE_CHUNK`` gives another (source ``given``)."""\n'
       '    if _SAMPLE_CHUNK_ENV:                                                                # [sample_chunk]\n'
       '        n = int(n_samples)\n'
       '        k = n if _SAMPLE_CHUNK_ENV == "all" else max(1, min(n, int(_SAMPLE_CHUNK_ENV)))\n'
       '        return core().plan(n, k, default_chunk=CHUNK)\n'
       '    return core().plan(int(n_samples), None, default_chunk=CHUNK)\n')


def _apply(s):
    if s.count(OLD) != 1:
        raise SystemExit(f'ANCHOR NOT FOUND EXACTLY ONCE ({s.count(OLD)}) -- not patching')
    if '\nimport os\n' not in s:
        raise SystemExit('sample_loop.py lacks "import os" at module level -- not patching')
    new = s.replace(OLD, NEW, 1)
    compile(new, str(P), 'exec')
    return new


def _write(text):
    tmp = P.with_suffix('.py.sample_chunk_tmp')
    tmp.write_text(text)
    os.replace(tmp, P)                                     # atomic: importers never see a half-written file


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else '--check'
    s = P.read_text()
    applied = MARK in s
    if mode == '--check':
        print(f'sample_chunk: applied={applied}  backup={BACKUP.exists()}')
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
