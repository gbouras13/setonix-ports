#!/usr/bin/env python3
"""Stop triangle attention's last query block from swallowing the ragged tail.

WHY
  opt_core's attend_query_blocks() evaluates attention over query sub-blocks of
  `qblock` rows so the logits transient stays at 2 * rows * H * qblock * N (its
  docstring). But it partitions with blocks4(), which MERGES the ragged tail into
  the previous block, so the last block holds qblock + (Q % qblock) queries -- up
  to ~2x the documented bound. The transient therefore depends on N mod qblock,
  and fit is not monotonic in N.

  Measured: OF3 33-mer, N = 12,705 (tail 161), job 49903750. The MSA module's
  first triangle attention asked for 20.21 GiB = one fp32 logits block
  [256 rows x 4 heads x 417 queries x 12,705 keys] with 44.89 GiB allocated, and
  OOM'd on all 8 ranks. A 256-query block is 12.41 GiB. The larger 34-mer
  (tail 34) fits.

FIX
  ROWPAIR_QBLOCK_SPLIT_TAIL=1 makes the tail its own block, so no block exceeds
  qblock. Every query row's softmax still runs over all keys, so the per-query
  arithmetic is unchanged; only which rows share a core call changes. blocks4()
  itself is untouched (its other callers are row loops for LayerNorm / bias
  projection, and the attention row windows use zblocks(), which never merged).

GATED, DEFAULT OFF: with the variable unset the code path is byte-for-byte the
old one.

USAGE  python3 patch_qblock_tail.py --apply | --check | --revert
"""
import os, sys, pathlib

P = pathlib.Path('/scratch/pawsey1018/gbouras/af3-setonix/opt_core_src/opt_core/opt_core/mem/rowpair/triatt.py')
if len(sys.argv) > 2:                      # optional explicit path (local testing)
    P = pathlib.Path(sys.argv[2])
MARK = '# [qblock_tail]'
BACKUP = P.with_suffix('.py.pre_qblock_tail')

OLD_DEF = 'def attend_query_blocks('
NEW_DEF = ('_QBLOCK_TAIL_NOTED: set = set()                                      # [qblock_tail] (Q, qblock) pairs already logged\n'
           '\n'
           '\n'
           'def attend_query_blocks(')

OLD_LOOP = '    for j0, j1 in blocks4(Q, int(qblock)):\n'
NEW_LOOP = '''    # [qblock_tail] blocks4 merges the ragged tail into the previous block, so the last
    # query block would hold qblock + (Q % qblock) queries and the logits transient
    # (logits + softmax copy) would grow with Q % qblock, up to ~2x the bound above.
    # ROWPAIR_QBLOCK_SPLIT_TAIL=1 makes the tail its own block (every block <= qblock).
    _qb = int(qblock)
    if os.environ.get("ROWPAIR_QBLOCK_SPLIT_TAIL", "") == "1":
        _grid = [(j0, min(Q, j0 + _qb)) for j0 in range(0, Q, _qb)]
        if Q % _qb and Q > _qb and (Q, _qb) not in _QBLOCK_TAIL_NOTED:
            _QBLOCK_TAIL_NOTED.add((Q, _qb))
            print(f"[qblock_tail] attend_query_blocks Q={Q} qblock={_qb}: tail of {Q % _qb} queries is its own block "
                  f"(merged it would be {_qb + Q % _qb})", flush=True)
    else:
        _grid = blocks4(Q, _qb)
    for j0, j1 in _grid:
'''


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else '--check'
    s = P.read_text()
    applied = MARK in s
    if mode == '--check':
        print(f'qblock_tail patch applied: {applied}')
        print(f'anchors present (unpatched form): def={s.count(OLD_DEF) == 1} loop={s.count(OLD_LOOP) == 1}')
        return 0
    if mode == '--apply':
        if applied:
            print('already applied'); return 0
        if s.count(OLD_DEF) != 1 or s.count(OLD_LOOP) != 1:
            print(f'ANCHORS NOT FOUND EXACTLY ONCE (def={s.count(OLD_DEF)} loop={s.count(OLD_LOOP)}) -- upstream changed; not patching',
                  file=sys.stderr)
            return 1
        new = s.replace(OLD_DEF, NEW_DEF, 1).replace(OLD_LOOP, NEW_LOOP, 1)
        compile(new, str(P), 'exec')                       # syntax check before anything is written
        BACKUP.write_text(s)
        tmp = P.with_suffix('.py.qblock_tail_tmp')
        tmp.write_text(new)
        os.replace(tmp, P)                                 # atomic: importers never see a half-written file
        print(f'applied (backup: {BACKUP.name})'); return 0
    if mode == '--revert':
        if not BACKUP.exists():
            print('no backup', file=sys.stderr); return 1
        tmp = P.with_suffix('.py.qblock_tail_tmp')
        tmp.write_text(BACKUP.read_text())
        os.replace(tmp, P)
        print('reverted'); return 0
    print('usage: --apply | --check | --revert [path]', file=sys.stderr); return 2


sys.exit(main())
