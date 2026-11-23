#!/usr/bin/env python3
"""Let the fused fp16 triangle attention serve the confidence head too (env-gated; off by default).

WHY
  With ROWPAIR_TRIATT_OPERANDS=fp16 the fused kernel serves the trunk, but the confidence head's 120 calls per hexamer
  (4 blocks x 2 nodes x 5 samples x 3 row blocks) fall back to the eager fp32 core (fallback_by=unsupported:
  dtype_float32), and the confidence head did not get faster (68 s per sample at 5,772 res, 18% of the run). The
  diagnostic (patch_triatt_diag.py, job 50127065) shows those calls DO reach triatt_fns.attend with fp16 q/k/v. The
  refusal is the serve layer's: OpenFold3 runs the confidence pairformer under torch.amp.autocast(dtype=float32)
  (prediction_heads.py:224; the kit's confidence.py:90), and flash_triattn_serve._compute_dtype_name takes the
  compute dtype from the autocast state whenever autocast is on, so it reads float32 and refuses.

WHAT
  ROWPAIR_TRIATT_AUTOCAST_OFF=1: when attend has cast the operands to 16-bit and autocast is on, the dispatch runs
  under torch.autocast("cuda", enabled=False), so the serve layer sees the operands' own dtype. That is the state the
  trunk's calls are already served in. Only the dispatch is affected; the output is cast back to the pair stack's
  dtype as before, and the gating / output projection still run under OpenFold3's autocast. Calls that were not cast
  (N above ROWPAIR_TRIATT_FP16_MAX_N, or ROWPAIR_TRIATT_OPERANDS unset) are untouched. Unset = today's behaviour.
  The confidence head runs after diffusion, so this can change only the confidence outputs (pLDDT, PAE, PDE, pTM,
  ipTM, ranking), never the structures.

USAGE  python3 patch_triatt_autocast.py --apply | --check | --revert [path]
"""
import os, sys, pathlib

P = pathlib.Path('/scratch/pawsey1018/gbouras/af3-setonix/of3_opt_src/opt/openfold3_opt/tp_rowpair/pairstack.py')
if len(sys.argv) > 2:
    P = pathlib.Path(sys.argv[2])
MARK = '# [triatt_autocast]'
BACKUP = P.with_suffix('.py.pre_triatt_autocast')

OLD_BIND = '    _opd_max_n = int(os.environ.get("ROWPAIR_TRIATT_FP16_MAX_N", "15400") or 15400)\n'
NEW_BIND = (OLD_BIND +
            '    # [triatt_autocast] ROWPAIR_TRIATT_AUTOCAST_OFF=1: a cast call dispatches with autocast off, so the flash serve layer\n'
            '    # reads the operands\' 16-bit dtype instead of the confidence head\'s autocast(float32) (it would refuse dtype_float32).\n'
            '    _opd_autocast_off = os.environ.get("ROWPAIR_TRIATT_AUTOCAST_OFF", "") == "1"\n')
OLD_CALL = "        o = dispatch(q, k, v, [mask_bias, tb])\n"
NEW_CALL = ("        if _opd_autocast_off and _opd is not None and q.dtype == _opd and torch.is_autocast_enabled():   # [triatt_autocast]\n"
            "            with torch.autocast(device_type=\"cuda\", enabled=False):\n"
            "                o = dispatch(q, k, v, [mask_bias, tb])\n"
            "        else:\n"
            "            o = dispatch(q, k, v, [mask_bias, tb])\n")


def _apply(s):
    counts = {"bind": s.count(OLD_BIND), "call": s.count(OLD_CALL)}
    if any(n != 1 for n in counts.values()):
        raise SystemExit(f'ANCHORS NOT FOUND EXACTLY ONCE ({counts}) -- not patching')
    if '[triatt_operands v2]' not in s:
        raise SystemExit('the fp16 cast (patch_triatt_fp16.py v2) is not applied -- nothing to extend, not patching')
    for need in ('import torch', 'import os'):
        if need not in s:
            raise SystemExit(f'pairstack.py lacks "{need}" at module level -- not patching')
    new = s.replace(OLD_BIND, NEW_BIND, 1).replace(OLD_CALL, NEW_CALL, 1)
    compile(new, str(P), 'exec')
    return new


def _write(text):
    tmp = P.with_suffix('.py.triatt_autocast_tmp')
    tmp.write_text(text)
    os.replace(tmp, P)                                     # atomic: importers never see a half-written file


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else '--check'
    s = P.read_text()
    applied = MARK in s
    if mode == '--check':
        print(f'triatt_autocast: applied={applied}  backup={BACKUP.exists()}  v2_marker={"[triatt_operands v2]" in s}  '
              f'diag_marker={"[triatt_diag]" in s}')
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
